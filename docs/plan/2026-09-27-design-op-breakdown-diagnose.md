# 설계: `serve diagnose` — decode step의 연산 클래스 분해와 상한 판정 (1단계 축소판)

작성 2026-09-27. 선행: [2026-09-22-design-stage1-diagnostic-report.md](2026-09-22-design-stage1-diagnostic-report.md)(원안, 보류), [PLAN.md](../../PLAN.md) v3.
상태: **사용자 승인(범위 "최소")**, 구현 계획 작성 전. 출발점은 커밋 6ce775f(design-1-3).

---

## 0. 한 문단 요약

지금의 서빙 실험은 step 시간 중 attention만 잰다. 나머지 시간이 어디에 쓰이고 왜 커널 선택으로 줄일 수 없는지는 추정이다. 이 기능은 실제 decode step을 **8개 연산 클래스**(embed, norm, qkv_proj, rope, attention, o_proj, mlp, lm_head)로 CUDA event로 실측 분해하고, 클래스마다 최소 이동 바이트와 FLOPs를 계산해 측정된 상한(DRAM 952.6 GB/s, 텐서코어 168.6 TFLOPS, `machines/rtx4090.json`)에 대어 "메모리 상한 / 연산 상한 / 런치 지배 / 병렬성 한계 후보 / 원인 미상"을 판정한다. attention 행에는 선택 변형·최적 대안·손실·Amdahl 상한을 붙인다. 산출물은 최종 보고서의 그림 한 장과 표 하나, 그리고 "균일 배치에서 이득이 없는 이유는 나머지 연산이 이미 대역폭 상한에 있어서"라는 문장의 실측 근거다. **프로젝트 정체성(측정·시뮬레이션 기반 attention 커널 동적 선택 시스템)은 바꾸지 않는다.** 이 기능은 PLAN §1 ②(원인 설명)의 한 단계다.

## 1. 범위

**한다.**
- `OpTimer`: 연산 클래스별 CUDA event 영역. 기존 `AttentionTimer`는 attention만 기록하는 하위 클래스로 유지해 성능 캠페인의 `attn_us` 의미와 경로가 바뀌지 않게 한다.
- `Engine.run(..., ops_mode=None | "event")`: event 모드에서 step·prefill마다 클래스별 GPU 시간을 `RunResult.ops`로 반환. `STEP_COLUMNS` 불변.
- `kernelscope/diagnose/opmodel.py`: 클래스별 바이트·FLOPs 모델(decode·prefill).
- `MachineSpec.tc_tflops`(선택 필드)와 `ridge_flop_per_byte()`.
- `kernelscope/diagnose/report.py`: 실행 폴더 → `diagnosis.json` + `ops.csv`.
- `kernelscope/diagnose/figures.py`: 정책별 연산 클래스 누적 막대(PNG/SVG, matplotlib).
- CLI `serve diagnose`(GPU 실행 + 진단 생성)와 `serve diagnose-report <dir>`(재생성, GPU 불필요).
- GPU 실행 D1·D2·D3, 실험 노트, `kernelscope verify` 항목, PLAN.md·STATUS.md·README 갱신, `package_demo` 포함.

**하지 않는다.** torch.profiler 커널 귀속(trace 모드), 대시보드 탭, HF `generate()` hook, nsys·ncu 비교, attention 외 연산의 처방, D4(Llama). 문서에 프로파일러 우월성 표현을 쓰지 않는다. 커널 내부(stall, bank conflict)는 ncu의 영역이라고 명시한다.

## 2. 측정 설계

### 2.1 연산 클래스와 영역 경계

`_forward`의 실제 연산 순서를 따른다. 잔차 덧셈은 뒤따르는 projection에 포함한다. `_rope`의 cos/sin 준비는 forward 시작 시 한 번이며 어느 클래스에도 넣지 않는다(미귀속에 잡힘).

| op_class | 감싸는 코드 | layer |
|---|---|---|
| embed | `F.embedding` (decode, prefill 청크) | −1 |
| norm | input_layernorm, qk_norm(q, k), post_attention_layernorm | l |
| qkv_proj | q/k/v `F.linear` 세 개와 `view` | l |
| rope | `q*cos + rotate_half(q)*sin`, k 동일 | l |
| attention | `flash_attn_with_kvcache` 또는 `_cpu_attention` | l |
| o_proj | o_proj `F.linear` + 잔차 덧셈 | l |
| mlp | gate/up `F.linear`, silu·mul, down `F.linear` + 잔차 덧셈 | l |
| lm_head | `_logits` = final RMSNorm + lm_head `F.linear` + `.float()` | −1 |

### 2.2 바이트·FLOPs 모델 (`opmodel.op_costs`)

입력: `ModelConfig`, dtype 바이트 b, phase, 토큰 수 T(decode: B, prefill: 청크 토큰 수), `lens`(decode의 요청별 KV 길이, prefill의 경우 `(start+T,)`). 출력: 클래스별 `(bytes, flops)`, 레이어 연산은 n_layers 배. 이 모델은 **압축 불가 최소 트래픽**(가중치 1회 읽기 + 활성 읽기·쓰기)이며 실제 트래픽은 이보다 클 수 있으므로 `achieved_gbps`는 하한 추정이라고 리포트에 쓴다. H = hidden, I = intermediate, V = vocab, Hq·Hkv·d는 헤드 구성.

| op_class | bytes (per step) | flops |
|---|---|---|
| embed | 2·T·H·b | 0 |
| norm | n_layers·[2·(2·T·H·b + H·b) + (qk_norm ? 2·T·(Hq+Hkv)·d·b : 0)] | 0 |
| qkv_proj | n_layers·[H·(Hq+2Hkv)·d·b + T·H·b + T·(Hq+2Hkv)·d·b] | n_layers·2·T·H·(Hq+2Hkv)·d |
| rope | n_layers·2·T·(Hq+Hkv)·d·b | 0 |
| attention | n_layers·`analytic.attention_traffic(w)["total_bytes"]` | n_layers·`analytic.attention_flops(w)` |
| o_proj | n_layers·[Hq·d·H·b + T·Hq·d·b + 2·T·H·b] | n_layers·2·T·Hq·d·H |
| mlp | n_layers·[3·H·I·b + T·b·(3H + 6I)] | n_layers·2·T·3·H·I |
| lm_head | 2·T·H·b + H·b + V·H·b + T·V·4 | 2·T·V·H |

attention의 `w`: decode는 `Workload("decode", B, 1, max(lens), Hq, Hkv, d, dtype, kv_lens=lens)`, prefill 청크는 `Workload("prefill", 1, T, start+T, Hq, Hkv, d, dtype)`. dtype 문자열은 모델 dtype에서 `analytic.DTYPE_BYTES` 키로 변환한다.

### 2.3 판정 규칙

`ai = flops/bytes`, `ridge = tc_tflops·1e12 / (dram_gbps·1e9)`(4090 ≈ 177), `achieved_gbps = bytes / gpu_us · 1e-3`, `achieved_tflops = flops / gpu_us · 1e-6`, `pct_dram = achieved_gbps / dram_gbps · 100`, `pct_tc = achieved_tflops / tc_tflops · 100`, `share = gpu_us / step_us_mean`, `layer_mean_us = gpu_us / n_layers`(레이어 연산만), 임계 θ = 0.7(설정값, 리포트에 인쇄).

| 조건(위에서부터 첫 일치) | verdict |
|---|---|
| ai < ridge 이고 achieved_gbps ≥ θ·dram_gbps | `memory_bound` |
| ai ≥ ridge 이고 achieved_tflops ≥ θ·tc_tflops | `compute_bound` |
| 레이어당 평균 gpu_us < 5 µs (레이어 연산) 또는 gpu_us < 5 µs (embed·lm_head) | `launch_bound` |
| op_class == attention | `parallelism_candidate` (대안 표 참조) |
| 그 외 | `below_ceiling_unknown` |

### 2.4 정직성 장치

- `unattributed_pct = (mean step_us − Σ_class mean gpu_us) / mean step_us × 100`. 라벨 사이의 GPU 유휴·런치 지연이 여기 잡힌다.
- 같은 정책·시나리오를 타이머 없이 한 번 더 돌려(`control_000/`) `timer_overhead_pct = mean(step_us, event) / mean(step_us, control) − 1`을 기록한다.
- 진단 실행의 manifest는 `evidence_kind = "diagnostic_op_breakdown"`, `performance_claim = false`. 기존 성능 캠페인과 합산하지 않는다.
- control과 event 실행의 생성 토큰이 다르면 `tokens_consistent = false`로 기록하고 종료 코드 1(진단은 그대로 생성).

## 3. 구성 요소

| ID | 위치 | 내용 |
|---|---|---|
| C1 | `kernelscope/serve/model.py` | `OpTimer(device, classes=CLASSES)`: `region(op_class, layer) -> contextmanager`(추적하지 않는 클래스는 공용 `nullcontext` 반환), `rows() -> list[dict(layer, op_class, gpu_us)]`(CUDA면 event 동기화 후), `total_us(op_class="attention")`. `AttentionTimer(OpTimer)`: `classes=("attention",)`, 기존 `start(layer)/stop(layer)/total_us()` 시그니처 유지(테스트 더블 호환). `_forward(..., timer=None)`은 `reg = timer.region if timer is not None else _null_region`으로 한 번 바인딩한 뒤 §2.1 경계마다 `with reg(cls, layer):`. `decode(..., timer=None)`는 embed·lm_head 영역 추가. `prefill(..., chunk=4096, *, timer=None)` 키워드 전용 인자, 청크마다 `_forward`에 전달하고 embed·lm_head 영역 포함 |
| C2 | `kernelscope/serve/engine.py` | `OPS_COLUMNS = ["phase", "step", "rid", "layer", "op_class", "gpu_us"]`. `RunResult.ops: pd.DataFrame`(기본 빈 프레임). `Engine.run(..., ops_mode=None)`: `"event"`면 decode step마다 `OpTimer`(그 `total_us()`가 `attn_us`), prefill 호출에도 `OpTimer`를 넘겨 `phase="prefill", step, rid`로 행 수집. `None`이면 현재와 바이트 단위로 같은 경로 |
| C3 | `kernelscope/model/machine.py` | `tc_tflops: float | None = None` 필드(끝에 추가, `from_json`은 `d.get`), `ridge_flop_per_byte()`(없으면 `ValueError`) |
| C4 | `kernelscope/diagnose/opmodel.py` | §2.2. `op_costs(cfg, dtype_bytes, phase, tokens, lens) -> dict[str, OpCost(bytes, flops)]`, `dtype_bytes(torch_dtype_str)` |
| C5 | `kernelscope/diagnose/report.py` | `diagnose(run_dir, machine, table_csv, data_root, params=None, threshold=0.7) -> dict`, `write(run_dir, diagnosis)`(`diagnosis.json`, `ops.csv`). 정책마다: (a) TPOT 폭포 `tpot_us_mean`(serve.report.tpot_us), `decode_wall_us_mean`, (b) step 폭포 `policy_us_mean`, `step_us_mean`, `host_residual_us = decode_wall − step − policy`, (c) 클래스 표(decode): `gpu_us`(step 평균, 레이어 합), `share`, `bytes`, `flops`, `ai`, `achieved_gbps`, `achieved_tflops`, `pct_dram`, `pct_tc`, `verdict`, `layer_mean_us`, (d) prefill 표(같은 열, rid 합산 후 평균), (e) `unattributed_pct`, `timer_overhead_pct`, (f) attention 추가 열 §3.1. 임계·상한·입력 경로를 `ceilings`와 `inputs`에 기록 |
| C6 | `kernelscope/diagnose/figures.py` | `op_breakdown(diagnosis, out_png, out_svg)`: 정책별 가로 누적 막대(클래스 gpu_us ms + 회색 미귀속), attention 비중과 판정 주석. 색은 `dashboard.style.CATEGORICAL["light"]` |
| C7 | `kernelscope/diagnose/run.py` | `run_diagnose(args)`: `serve run`의 검증·preflight·모델 로드·시나리오 해석·manifest를 재사용(공용 헬퍼는 `serve/cli.py`에서 import). 정책마다 warmup 1회 → `control_000/`(ops_mode None) → `event_000/`(ops_mode "event", `ops.parquet` 추가) → `compare_results`로 토큰 대조. 끝에 C5·C6 실행. `run_report(args)`: 기존 폴더에서 C5·C6만 재실행 |
| C8 | `kernelscope/serve/cli.py` | `serve diagnose`(공용 인자 + `--table`(기본 `demo_data/dispatch_paged_cold.csv`), `--data`(기본 `demo_data`), `--threshold 0.7`; `--repeats` 무시, `--warmup-runs` 기본 1), `serve diagnose-report <dir> [--table --data --machine --params --threshold]`. 두 명령은 C7로 위임 |
| C9 | `scripts/package_demo.py` | 변경 불필요 확인(`serve_4090/**` 전체 복사). `ops.parquet`·`diagnosis.json`·`ops.csv`·그림이 포함되는지 테스트 |
| C10 | `kernelscope/verify.py` | `diagnose.*` 항목(§6) |
| C11 | docs | `docs/experiments/2026-09-27-op-breakdown.md`, `docs/STATUS.md` 항목, `PLAN.md` §1 ② 근거에 한 줄, `README.md`에 `serve diagnose` 블록, 그림 `docs/img/op_breakdown_ragged.png` |

### 3.1 attention 행의 추가 열

- `chosen_splits`: 정책이 고른 `num_splits`의 step별 값(대부분 상수). `chosen_variant = hybrid.variant_for_splits(n)`.
- `workload_key`: step의 (B, lens)로 만든 `Workload(...).key()`.
- `source`: key가 `data_root/hw_4090/*/summaries.jsonl`(paged, cold)에 있으면 `"measured"`, 없으면 `"model"`(params 필요; 없으면 `"unavailable"`).
- `best_alternative`, `best_us`, `chosen_us`, `regret = chosen_us / best_us − 1`: measured면 같은 key의 `fd_*_paged/fa2_paged/flashdecoding_paged` 실측에서, model이면 `predict.rank_variants`의 예측에서(리포트에 "예측"으로 표시).
- `amdahl_bound = amdahl_speedup(share_attention, chosen_us / best_us)`: "attention을 최적으로 바꿔도 step은 최대 X배".
- step마다 값이 다를 수 있으므로(arrivals) 표에는 step 가중 평균과 최빈 key를 쓰고, step별 원자료는 `ops.csv`가 아니라 `attention_steps.csv`에 남긴다.

## 4. 데이터 계약

- `event_000/ops.parquet`: `phase("decode"|"prefill"), step(int), rid(str; decode는 ""), layer(int; embed·lm_head는 −1), op_class(str), gpu_us(float)`.
- `control_000/`, `event_000/`: 기존 `steps/tokens/prefill.parquet` + `meta.json`(`ops_mode` 포함).
- `manifest.json`: `serve run`과 같은 필드 + `evidence_kind="diagnostic_op_breakdown"`, `performance_claim=false`, `ops_mode_runs=["control","event"]`, `tokens_consistent`, `threshold`, `table`, `data_root`.
- `diagnosis.json`: `{schema_version: 1, identity{evidence_kind, performance_claim, model, scenario, created_at}, ceilings{dram_gbps, tc_tflops, ridge_flop_per_byte, threshold, machine}, policies{<name>: {tpot_waterfall{}, step_waterfall{}, ops[], prefill_ops[], attention{}, unattributed_pct, timer_overhead_pct, n_steps}}}`.
- `ops.csv`: `policy, phase, op_class, gpu_us, share, bytes, flops, ai, achieved_gbps, achieved_tflops, pct_dram, pct_tc, verdict, layer_mean_us`.
- 폴더: `<out>/<policy>/control_000/`, `<out>/<policy>/event_000/`, `<out>/diagnosis.json`, `<out>/ops.csv`, `<out>/attention_steps.csv`, `<out>/op_breakdown.{png,svg}`.

## 5. GPU 실행

출력 루트 `../kernelscope/results/serve_4090/diagnose_20260927/`. 각 실행은 decode 64 step 이하, 모델 로드 포함 1분 안쪽. 유휴 확인은 `serve doctor`.

| 실행 | 모델 | 시나리오 | 정책 | 인자 |
|---|---|---|---|---|
| D1 | Qwen3-4B | `graduation_ragged` (32K 1 + 512 31) | heuristic, table | `--kv-gib 10` |
| D2 | Qwen3-4B | `graduation_uniform` | heuristic, table | `--kv-gib 10` |
| D3 | Qwen3-4B | `heldout_text_ragged` (12K 1 + 384 27) | heuristic, model | `--machine machines/rtx4090.json --params models/rtx4090.json --kv-gib 10` |

실행 뒤 `scripts/package_demo.py`로 `demo_data/serve_4090/diagnose_20260927/`에 복사한다.

## 6. 합격 기준

| # | 기준 | 측정 |
|---|---|---|
| A1 | D1 heuristic `unattributed_pct ≤ 10` | diagnosis.json |
| A2 | `timer_overhead_pct ≤ 5` 세 실행 모두. 넘으면 리포트 경고 + STATUS에 원인 기록(실패 아님) | control 대조 |
| A3 | D1의 attention 행 `chosen_variant`, `best_alternative`, `regret`가 `dispatch_paged_cold.csv`의 `best_kernel`·`heuristic_regret`와 일치, D3는 `source="model"` | diagnosis.json vs CSV |
| A4 | D1에서 qkv_proj·o_proj·mlp·lm_head의 `pct_dram`이 (0, 100] 범위이고 verdict가 계산되며 attention만 `parallelism_candidate`. 수치가 높을 것을 요구하지 않는다. 예상과 다르면 그대로 보고 | diagnosis.json |
| A5 | D1의 `amdahl_bound`(heuristic 행)와 control 실행의 step 배율(heuristic/table)이 같은 방향이고 실측이 상한을 넘지 않음 | 두 값 비교 |
| A6 | control vs event 토큰 일치(세 실행 모두) | manifest `tokens_consistent` |
| A7 | 기존 CPU 테스트 전부 통과 + 새 테스트(OpTimer 행·합계, opmodel 손계산, machine tc_tflops, report fixture, CLI CPU tiny 실행, figures 렌더, package_demo 포함) | `make test` |
| A8 | 성능 경로 회귀 없음: `serve run` tiny GPU 시나리오 heuristic `step_us`가 변경 전후 2% 이내(각 3회 중앙값) | 변경 전 측정값을 먼저 기록 |
| A9 | `make verify` 전부 PASS(기존 40 + 신규), 문서 3종 갱신, README·graduation.md·STATUS.md·실험 노트에 프로파일러 우월성 표현 0건 | verify, grep |

신규 verify 항목(값은 실행 후 채움): `diagnose.ragged_attention_share_heuristic`, `diagnose.ragged_attention_share_table`, `diagnose.ragged_unattributed_pct_heuristic`(≤ 10 상한 검사), `diagnose.ragged_mlp_pct_dram_heuristic`, `diagnose.ragged_amdahl_bound_heuristic`, `diagnose.uniform_attention_share_heuristic`, `diagnose.timer_overhead_max_pct`(≤ 5). 각 항목은 `demo_data/serve_4090/diagnose_20260927/`의 parquet에서 C5로 재계산한다.

## 7. 테스트 전략

CPU에서 tiny 모델(`tests/test_serve_model.py`의 `TINY`)로 OpTimer 행이 8클래스 × 레이어를 덮고 `AttentionTimer.total_us()`와 `OpTimer.total_us("attention")`이 같은 정의임을 확인한다. 엔진은 `region()`을 지원하는 테스트 더블로 event 모드 행 수집을, `ops_mode=None`에서 빈 `ops`와 불변 `STEP_COLUMNS`를 확인한다. opmodel은 작은 구성에서 손계산과 비교한다. report는 합성 parquet 폴더 fixture(2 step, 2 레이어)로 판정·미귀속·오버헤드·attention 행(작은 표 CSV)을 확인한다. CLI는 `--device cpu --model tiny-random --scenario scenarios/tiny.yaml`로 전체 경로를 실행한다(CPU 시간은 perf_counter, `evidence_kind="cpu_functional"`). GPU 합격 기준 A1~A6은 실행 후 `diagnosis.json`을 읽는 스크립트로 확인한다.

## 8. 위험과 대응

| 위험 | 대응 |
|---|---|
| event 약 300개/step의 host 비용이 GPU 유휴를 만들어 미귀속·오버헤드가 커짐 | A2로 수치화. 넘으면 `--op-granularity group`(4레이어 묶음) 옵션을 후속으로 |
| 활성 트래픽을 단순화한 바이트 모델이 achieved를 과소평가 | "하한 추정" 표기. 가중치 항과 활성 항을 `ops.csv`에 분리 기록(`weight_bytes`, `act_bytes`) |
| heldout key가 표에 없어 attention 대안이 예측에 의존 | `source` 열로 구분, 예측을 실측처럼 쓰지 않음 |
| 판정이 `below_ceiling_unknown`으로 많이 나옴 | 그것이 정직한 결과. 문서는 attention 행의 처방 사례를 먼저 보여 준다 |
| `prefill(..., timer)` 추가로 기존 호출 깨짐 | 키워드 전용 인자, 기본 None. 기존 테스트가 회귀를 잡는다 |
