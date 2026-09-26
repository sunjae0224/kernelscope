# 설계: 1단계 — decode/prefill 진단 리포트 (TPOT → step → 연산 → 커널 → 한계 자원)

작성 2026-09-22. 선행: [2026-09-19-design-surrogate-dispatcher.md](2026-09-19-design-surrogate-dispatcher.md)(방향 1·3 설계), [graduation.md](../graduation.md), [demo.md](../demo.md).
상태: **보류(2026-09-26).** 중간보고서에서 정체성을 '측정·시뮬레이션 기반 attention 커널 동적 선택 시스템'으로 고정하면서, 진단 리포트는 독립 정체성이 아니라 향후 과제로 남긴다. 이 문서는 코드 변경 없이 작성했다. 커밋 5f678ea(design-1-3)가 출발점이다.

---

## 0. 한 문단 요약

kernelscope의 정체성을 "attention 커널 선택 시스템"에서 한 층 올려 **"실측으로 귀속하고, 모델로 설명하고, 시뮬레이션으로 반사실을 보여 주고, 고칠 수 있는 것은 고쳐서 검증하는 LLM 추론 진단기"**로 고정한다. 1단계는 그중 **귀속 층**을 만든다. 실제 decode 루프의 한 step을 TPOT → decode wall → step → 연산 클래스 → (선택) 커널로 내려가며 분해하고, 연산 클래스마다 옮긴 바이트와 FLOPs를 측정 상한(DRAM 952.6 GB/s, fp16 tensor core 168.6 TFLOPS, `machines/rtx4090.json`)에 대어 "무엇에 묶였는가"를 판정한다. attention 행에는 기존 실측표·대리 모델의 대안과 손실을 붙인다. 이것이 문제의식 "거시 지표(TTFT/TPOT)는 병목을 요청·연산·자원에 귀속시키지 못한다"에 대한 직접 답이다. 범용 프로파일러는 만들지 않는다. 우리 decoder(Llama/Qwen3) 위에서만 동작한다.

## 1. 이 설계가 답해야 하는 질문과, 답하지 않는 것

| 질문 | 1단계 답 | 근거 층 |
|---|---|---|
| 이 요청의 TPOT는 어디에 쓰였나 | TPOT = decode wall + 합류 대기(다른 요청 prefill) 로 분해; decode wall = step(GPU) + 정책 CPU + host 나머지 | 실측(CUDA event + host clock) |
| step 안에서 어느 연산이 몇 ms인가 | embed / norm / qkv_proj / rope / attention / o_proj / mlp / lm_head 8개 클래스 + 미귀속 잔여 | 실측(CUDA event, 레이어별) |
| 그 연산은 연산 한계인가 대역폭 한계인가 | 클래스별 bytes·FLOPs → 상한 대비 % → 판정 | 분석 모델 + 측정 상한 |
| 무엇을 바꾸면 되나 | attention만: 최적 대안, 손실, Amdahl 상한. 나머지: "상한에 붙음, 커널 선택으로 얻을 것 없음" 또는 "상한 아래, 원인은 1단계 범위 밖" | 실측표 / 대리 모델 |
| 어느 커널이 실제로 돌았나 | 선택 옵션: N step 창을 torch.profiler로 떠서 커널을 라벨에 귀속 | CUPTI activity (root 불필요) |
| TTFT는 어디에 쓰였나 | 큐 대기(admitted − arrival) + prefill; prefill도 같은 8클래스로 분해 | 실측 |

**하지 않는 것(1단계 비목표).** 외부 프레임워크·HF generate 부착(2단계), nsys·torch.profiler 나란히 비교 그림(3단계), attention 외 연산의 처방, 하드웨어 카운터(ncu), CUDA Graph, 다중 GPU, HTTP. "더 나은 프로파일러"라는 표현은 문서 어디에도 쓰지 않는다. 커널 내부(stall, bank conflict, L1)는 ncu의 영역이라고 명시한다. 기준선으로 이름을 밝힐 대상은 nsys, ncu, torch.profiler(`key_averages`), Holistic Trace Analysis 네 가지다.

## 2. 측정 설계

### 2.1 두 가지 측정 모드

| 모드 | 기전 | 산출 | 오버헤드 | 언제 |
|---|---|---|---|---|
| **event** (진단 실행 기본) | `_forward` 안의 연산 영역마다 CUDA event 쌍 기록, step 끝에 한 번 동기화. 기존 `AttentionTimer`의 일반화 | `ops.parquet`: (phase, step 또는 rid, layer, op_class, gpu_us) | host에서 event 기록 약 500개/step. GPU 측 거의 0 | 모든 진단 실행 |
| **trace** (옵션 `--trace-steps N`) | 프로세스당 **한 번**의 torch.profiler 세션이 연속 N step을 덮음. 연산 영역은 `record_function` 라벨로 감쌈. chrome trace를 저장 | `kernels.parquet`: (step, layer, op_class, kernel_name, dur_us, grid, block, regs, smem_bytes) | 세션 동안 step이 느려짐. 성능 수치로 쓰지 않음 | 커널 드릴다운이 필요한 대표 실행 1개 |

event 모드의 정직성 장치 두 가지. (a) `unattributed_us = step_us − Σ op gpu_us`를 항상 계산해 리포트에 "미귀속"으로 표시한다. 라벨 사이의 GPU 유휴(호스트 런치 지연 포함)가 여기 잡힌다. (b) 같은 시나리오를 타이머 없이 한 번 더 돌려 `step_us` 차이를 `timer_overhead_pct`로 manifest에 기록한다. 진단 실행은 `evidence_kind = "diagnostic_op_breakdown"`, `performance_claim = false`로 표시하고 기존 성능 캠페인과 절대 합산하지 않는다(`research.py`의 IDENTITY 규약 그대로).

### 2.2 연산 클래스와 경계

`_forward`의 실제 연산 순서를 따라 8개 클래스로 자른다. 잔차 덧셈은 뒤따르는 projection에 포함한다.

| op_class | 포함 | 바이트 모델(decode, 토큰 수 T = B) | FLOPs 모델 |
|---|---|---|---|
| embed | `F.embedding` | T·hidden·b | 0 |
| norm | input/post/final RMSNorm, qk_norm | 활성 읽기·쓰기 2·T·hidden·b (+가중치) | 무시 |
| qkv_proj | q/k/v `F.linear` | 가중치 hidden·(H_q+2H_kv)·d·b + 활성 | 2·T·hidden·(H_q+2H_kv)·d |
| rope | cos/sin 곱과 rotate_half | 활성 T·(H_q+H_kv)·d·b·2 | 무시 |
| attention | `flash_attn_with_kvcache` | 기존 `analytic.attention_traffic(Workload(kv_lens=lens))` | 기존 `attention_flops` |
| o_proj | o_proj `F.linear` + 잔차 | hidden·H_q·d·b + 활성 | 2·T·hidden·H_q·d |
| mlp | gate/up/silu·mul/down + 잔차 | 3·hidden·intermediate·b + 활성 | 2·T·3·hidden·intermediate |
| lm_head | final norm 뒤 `F.linear(lm_head)` | vocab·hidden·b + 활성 | 2·T·vocab·hidden |

b = dtype 바이트(bf16/fp16 = 2). prefill은 T = 청크 토큰 수로 청크마다 계산한 뒤 (rid, layer, op_class)로 합산한다. 나머지 공식 동일. 이 모델은 **압축 불가 최소 트래픽**(가중치 1회 읽기 + 활성)이며 실제 트래픽은 이보다 클 수 있다. 따라서 `achieved_gbps`는 하한 추정이고 리포트에 그렇게 쓴다. 가중치 총량(4B 모델 약 8 GB)이 L2(72 MiB)를 훨씬 넘으므로 decode의 가중치 스트리밍은 DRAM 상한과 비교한다(설계 F8과 같은 논리).

### 2.3 판정 규칙(연산 클래스별)

`ai = flops / bytes`, `ridge = tc_tflops·1e12 / (dram_gbps·1e9)` ≈ 177 FLOP/byte (4090 측정값).

| 조건 | 판정 문구 | 해석 |
|---|---|---|
| ai < ridge 이고 achieved_gbps ≥ 0.7·dram_gbps | 메모리 상한에 붙음 | 커널 선택으로 얻을 것 없음. 배치·양자화·캐시 문제 |
| ai ≥ ridge 이고 achieved_tflops ≥ 0.7·tc_tflops | 연산 상한에 붙음 | 같음 |
| 둘 다 아니고 레이어당 평균 gpu_us < 5 µs | 런치·지연 지배 | 커널 융합 대상, 1단계 범위 밖 |
| 둘 다 아니고 attention | 병렬성·스케줄링 한계 후보 → 대안 표 참조 | 방향 3의 영역 |
| 둘 다 아니고 그 외 | 상한 아래, 원인 미상 | 정직하게 "모름"으로 표시 |

임계 0.7은 설정값이며 리포트에 함께 인쇄한다. decode B=32 GEMM의 ai ≈ B ≪ ridge, prefill 12K 토큰의 ai ≈ 12K ≫ ridge라 두 phase가 서로 다른 판정을 받는 것이 이 표의 존재 이유다.

### 2.4 attention 행의 추가 열

- `workload_key`: `Workload("decode", B, 1, max(lens), H_q, H_kv, d, dtype, kv_lens=tuple(lens))`.
- `chosen_variant`: 정책이 고른 num_splits N을 변형 이름으로 옮긴다. N=0 → `flashdecoding_paged`, N=1 → `fa2_paged`, N≥2 → `fd_s{N}_paged`. 이어서 `best_alternative`, `regret`, `source ∈ {table, model}`: key가 `demo_data/dispatch_paged_cold.csv`(또는 지정 표)에 있으면 실측표, 없으면 대리 모델 예측. 모델이면 "예측"이라고 표시하고 검증표 링크를 단다.
- `amdahl_bound`: `data.amdahl_speedup(attention_share, chosen/best)`로 "attention을 최적으로 바꿔도 step은 최대 X배"를 계산한다. Serving 탭의 실측 배율과 나란히 놓는다.

### 2.5 trace 모드의 커널 귀속

chrome trace에서 `cat == "kernel"` 이벤트(기존 `kernel_events_from_chrome_trace`)를 `args.correlation`으로 같은 correlation의 `cuda_runtime` 런치 이벤트에 잇고, 그 런치 이벤트의 (tid, ts)를 포함하는 가장 안쪽 `user_annotation` 범위의 라벨을 붙인다. 라벨은 `ks:{phase}:{step}:{layer}:{op_class}` 형식의 한 문자열이다. `gpu_user_annotation` 이벤트가 있으면 교차 검증에만 쓴다. 어느 라벨에도 못 붙은 커널은 `op_class = "unlabeled"`로 남긴다. 프로파일러 세션은 프로세스당 한 번, 연속 창 하나다(이 저장소에서 확인된 torch 2.8/CUPTI의 세션 재개 실패 회피, `require_kernel_events`로 빈 캡처를 즉시 실패시킨다).

## 3. 구성 요소

| ID | 위치 | 내용 | 의존 |
|---|---|---|---|
| S1-1 | `kernelscope/serve/model.py` | `OpTimer`(레이어·클래스별 CUDA event 영역, `rows()` → DataFrame, `total_us(op_class)`) 도입. `AttentionTimer`는 attention 클래스만 켠 `OpTimer`의 별칭으로 유지해 `engine.attn_us` 값이 바뀌지 않게 한다. `_forward(..., ops=None)`: `ops`가 None이면 no-op 스코프. `prefill`도 `ops`를 받는다 | 없음 |
| S1-2 | `kernelscope/serve/model.py` | trace 모드용 `record_function` 라벨은 `ops.trace=True`일 때만 감싼다. 성능 캠페인 경로(`ops=None`)의 오버헤드는 no-op 컨텍스트 매니저 호출뿐 | S1-1 |
| S1-3 | `kernelscope/serve/engine.py` | `Engine.run(..., ops_mode=None|"event"|"trace", trace_window=(start_step, n))`. event 모드에서 step마다 `ops.rows()`를 모아 `RunResult.ops`(phase, step, rid, layer, op_class, gpu_us)로 반환. prefill 호출도 같은 표에 phase="prefill", rid 포함. trace 모드에서는 Engine이 `start_step`에 torch.profiler 세션을 열고 n step 뒤 닫아 chrome trace를 저장한다(프로세스당 1회). STEP_COLUMNS는 불변 | S1-1 |
| S1-4 | `kernelscope/diagnose/opmodel.py` | §2.2 바이트·FLOPs 모델. 입력: `ModelConfig`, dtype, phase, T, lens. attention은 `analytic` 재사용. 단위 테스트는 손계산과 비교 | 없음 |
| S1-5 | `kernelscope/model/machine.py` | `MachineSpec`에 `tc_tflops: float | None` 추가(`from_json`은 키가 없으면 None). `ridge_flop_per_byte()` 제공 | 없음 |
| S1-6 | `kernelscope/diagnose/trace.py` | §2.5 귀속. 입력 chrome trace JSON → `kernels` DataFrame. 합성 trace fixture로 테스트 | 없음 |
| S1-7 | `kernelscope/diagnose/report.py` | 실행 폴더 → `diagnosis.json` + `diagnosis.csv`. 내용: (a) 요청별 TPOT 폭포(decode wall 평균, 합류 대기), (b) step 폭포(정책 CPU, GPU step, host 잔여), (c) 연산 클래스 표(gpu_us, share, bytes, flops, ai, achieved_gbps, achieved_tflops, pct_dram, pct_tc, verdict), (d) attention 행 추가 열(§2.4), (e) 미귀속 비율, 타이머 오버헤드, (f) prefill 표. 임계값·상한·출처 경로를 같이 기록 | S1-3, S1-4, S1-5, S1-6 |
| S1-8 | `kernelscope/diagnose/figures.py` | 폭포 그림(matplotlib PNG/SVG): 왼쪽 TPOT→wall→step 막대, 오른쪽 연산 클래스 누적 막대 + 상한 대비 % 주석. dataviz 규약(색·라벨)은 기존 `dashboard/figures.py`와 맞춘다 | S1-7 |
| S1-9 | `kernelscope/serve/cli.py` | `serve diagnose`: `serve run`과 같은 인자 + `--table <csv>`(기본 `demo_data/dispatch_paged_cold.csv`) `--trace-steps N --trace-start S --no-overhead-check`. 기본 repeats 1, warmup 1. 정책마다 (a) event 모드 `repeat_XXX/`(ops.parquet), (b) 타이머 없는 대조 `control_000/`(steps/tokens만), (c) 요청 시 `trace_000/`(ops + kernels.parquet + trace.json.gz)를 만든 뒤 `diagnosis.*`를 생성. 대조·trace 실행의 토큰은 event 실행과 일치해야 하며 불일치는 실패로 기록. `serve diagnose-report <dir>`는 재생성만 | S1-3, S1-7, S1-8 |
| S1-10 | `dashboard/app.py`, `kernelscope/dashboard/report.py` | 새 첫 탭 **"00 Report"**: 진단 실행 선택 → 폭포 그림 → 연산 표(판정 색상) → attention 행 클릭 시 해당 workload_key로 Diagnose/What-if 탭 상태를 채움. 기존 4탭 유지. 데이터는 `demo_data/serve_4090/diagnose_*`에서도 읽힘 | S1-7 |
| S1-11 | `scripts/package_demo.py` | 진단 실행 폴더(ops/kernels/diagnosis, trace.json.gz 제외)를 번들에 포함 | S1-9 |
| S1-12 | docs | `graduation.md` 문제 진술 §을 "귀속 공백" 문장으로 교체, 정체성 한 문장 추가, 기준선 네 가지와 비목표 명시. `demo.md` 0:00 구간을 Report 탭으로 교체. `STATUS.md` 항목. `README.md` 상단에 `serve diagnose` 한 블록 | 전부 |

새 패키지 `kernelscope/diagnose/`는 `serve`와 `model`에만 의존하고 대시보드는 `diagnose.report`의 산출물(JSON/CSV)만 읽는다. Codex 영역(`backends/accelsim`)은 건드리지 않는다.

## 4. 데이터 계약

`ops.parquet` 열: `phase("decode"|"prefill"), step(int, prefill은 −1), rid(str, decode는 ""), layer(int, lm_head/embed는 −1), op_class(str), gpu_us(float)`. `kernels.parquet` 열: `step, layer, op_class, kernel_name, dur_us, grid_x, grid_y, grid_z, block_x, block_y, block_z, regs, smem_bytes, correlation`. `diagnosis.json` 최상위 키: `identity(연구 IDENTITY 필드), ceilings{dram_gbps, tc_tflops, ridge, threshold}, tpot_waterfall, step_waterfall, ops[], prefill_ops[], attention{...}, unattributed_pct, timer_overhead_pct, trace{steps, labeled_kernel_time_pct}`. manifest에 `evidence_kind="diagnostic_op_breakdown"`, `performance_claim=false`, `ops_mode`, `trace_steps`. 폴더 규약: `<policy>/repeat_XXX/`(event), `<policy>/control_000/`(타이머 없음, 오버헤드 대조), `<policy>/trace_000/`(프로파일러 창). `timer_overhead_pct = mean(step_us, repeat) / mean(step_us, control) − 1`.

## 5. 1단계에서 실제로 돌릴 GPU 실행

| 실행 | 모델 | 시나리오 | 정책 | 반복 | trace |
|---|---|---|---|---|---|
| D1 | Qwen3-4B | graduation_ragged (32K 1 + 512 31) | heuristic, table | 2 | heuristic 1회, 4 step |
| D2 | Qwen3-4B | graduation_uniform | heuristic, table | 2 | 없음 |
| D3 | Qwen3-4B | heldout_text_ragged (12K 1 + 384 27) | heuristic, model | 2 | 없음 |
| D4 (선택) | Llama-8B | heldout_text_ragged | heuristic, model | 1 | 없음 |

각 실행은 decode 64 step 이하라 GPU 시간은 모델 로드를 포함해 실행당 1분 안쪽이다. 유휴 GPU 확인은 기존 `serve doctor`. 결과는 `../kernelscope/results/serve_4090/diagnose_20260922/`에 두고 `package_demo`로 `demo_data/`에 복사한다.

## 6. 합격 기준

| # | 기준 | 측정 방법 |
|---|---|---|
| A1 | D1 heuristic에서 `unattributed_pct ≤ 10 %` | diagnosis.json |
| A2 | `timer_overhead_pct ≤ 5 %` (step_us 기준). 넘으면 리포트에 경고를 띄우고 원인을 STATUS에 적는다 | 대조 실행 |
| A3 | trace 창에서 라벨된 커널 시간 ≥ 95 % | kernels.parquet |
| A4 | attention 행의 `chosen`, `best_alternative`, `regret`가 D1에서 실측표 값과 일치(표에 있는 key), D3에서 `source="model"`로 표시 | diagnosis.json vs dispatch 표 |
| A5 | D1에서 decode GEMM 클래스(qkv/o/mlp/lm_head)의 판정이 계산되고, `pct_dram`이 0 < x ≤ 100 % 범위이며 attention 행만 "병렬성 한계 후보"로 분류됨. 수치가 높을 것을 요구하지 않는다. 결과가 예상과 다르면 그대로 보고 | diagnosis.json |
| A6 | D1의 `amdahl_bound`와 Serving 탭의 실측 step 배율(table/heuristic)이 같은 방향이고, 실측이 상한을 넘지 않음 | 두 값 비교 |
| A7 | Report 탭이 GPU 없이 `demo_data`에서 렌더되고 폭포 그림이 `docs/img/`에 저장됨 | `make test` + 스크린샷 |
| A8 | 기존 CPU 테스트 전부 통과 + 새 테스트(OpTimer 합계, opmodel 손계산, trace 귀속 fixture, report fixture, 대시보드 스모크). 성능 경로 회귀: `serve run` tiny 시나리오 heuristic step_us가 변경 전 대비 2 % 이내 | pytest, 대조 실행 |
| A9 | 문서 3종 갱신, "더 나은 프로파일러" 표현 0건 | grep |

## 7. 위험과 대응

| 위험 | 대응 |
|---|---|
| CUDA event 500개/step의 host 비용이 작은 B에서 GPU 유휴를 만들어 미귀속이 커짐 | A2 대조 실행으로 수치화. 필요하면 레이어 묶음(예: 4레이어마다 event) 옵션 `--op-granularity layer|group` |
| torch.profiler 세션이 커널을 못 잡음(알려진 문제) | 프로세스당 1세션, `require_kernel_events`, 실패 시 event 모드 결과는 보존하고 trace만 재실행 |
| 활성 트래픽을 무시한 바이트 모델이 achieved를 과소평가 | "하한 추정"이라고 표시. 활성 항을 넣되 가중치 항과 분리해 기록 |
| heldout key가 표에 없어 attention 대안이 예측에 의존 | `source` 열로 구분, 검증표 링크. 예측을 실측처럼 쓰지 않음 |
| 리포트가 "모름" 판정을 많이 내면 데모가 약해 보임 | 그것이 정직한 결과다. 데모 첫 클릭은 D1 ragged로 고정해 attention 행이 처방까지 이어지는 사례를 먼저 보여 준다 |

## 8. 이후 단계(참고, 이 스펙 범위 밖)

2단계: HF `generate()`에 forward hook으로 같은 `ops.parquet`를 만들어 "우리 엔진이 아니어도 귀속은 된다"는 증거 1개(처방 없음). 3단계: 같은 step의 nsys 타임라인·torch.profiler 표·kernelscope 리포트 나란히 그림, 대시보드 탭 순서 재편. 발표까지 남은 기간을 확인한 뒤 결정한다.
