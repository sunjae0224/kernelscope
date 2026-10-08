# 설계: 라이브러리 무관 선택표 — FlashInfer 변형을 후보에 넣는 테이블 정책

작성 2026-10-08. 선행: [TODO §1 3번](../../TODO.md), [문제 범위 노트 §2](../experiments/2026-10-06-problem-scope.md)(두 라이브러리의 정적 기본값), [PLAN §2 8번](../../PLAN.md). 분류: 기존 `TablePolicy`와 엔진의 `attention` 백엔드 배관을 넓히는 **bounded** 변경.
상태: 사용자 부재 중 TODO 문구대로 진행. TODO에 없는 결정은 §2.2의 **선택 비용 반영 옵션(plan_us)** 하나이며, 기본값(0)은 TODO가 정의한 동작(커널 시간만 보는 전체 최선)과 같다.

---

## 0. 한 문단 요약

지금 `table` 정책은 `dispatch_paged_cold.csv`의 `best_kernel`(FA2 분할 변형만)을 읽고 분할 수 하나를 돌려준다. 엔진은 정책의 `attention` 속성을 `model.decode`에 넘기지만, 그 속성은 정책마다 고정이다(`flashinfer`/`flashinfer_cudacore` 정책은 백엔드만 고르고 분할 수는 고르지 않는다). 이 작업은 `static_default_losses`가 만드는 표(FA2 분할 + FlashInfer tensor-core/CUDA-core 중 **전체 최선**과 후보별 커널 시간)를 같은 정책이 읽고, step마다 **(분할 수, 백엔드)를 함께** 고르게 한다. 32K 격자 211셀에서 전체 최선은 CUDA-core 148·tensor-core 45·FA2 18셀(혼합 162셀 중 FA2는 2셀)이지만, 혼합 셀에서 CUDA-core가 최선 FA2 분할보다 빠른 폭은 커널 호출당 중앙값 9.8 µs(최대 39 µs)라 36층이면 step당 약 350 µs인 반면 FlashInfer의 `plan()`은 step당 370~430 µs(10월 7일 로그)다. 즉 커널 시간만 보는 표는 이득이 상쇄될 가능성이 크므로, 선택 비용을 더해 비교하는 옵션을 같이 두고 둘 다 실생성으로 잰다.

## 1. 범위

**한다.**
- `scripts/package_demo.py`: 6개 격자(`{uniform,ragged}_s1_{paged,flashinfer,flashinfer_cudacore}`)의 cold 행으로 `static_default_losses` 표를 만들어 `demo_data/dispatch_paged_cold_any.csv`로 쓴다(`complete` 행만). 기존 `dispatch_paged_cold.csv`는 그대로.
- `kernelscope/serve/dispatch.py`: `TablePolicy(csv, backends=("fa2",), plan_us=0.0, name="table")` 확장, `make_policy`에 `table_any:<csv>[:<plan_us>]` 추가. `HybridPolicy` 안의 표는 FA2 전용 유지.
- `kernelscope/serve/engine.py`: `STEP_COLUMNS`에 `attention` 열 추가(그 step에 decode에 넘긴 백엔드 이름), 정책에 `n_layers` 주입, d=128 guard에 새 정책 포함.
- 테스트(CPU), GPU 캠페인 1회(§4), verify `anytable.*`, 노트 §5, PLAN/STATUS/TODO 갱신.

**하지 않는다.** 모델·혼합 정책에 FlashInfer 후보 추가(대리 모델은 FA2 커널만 학습), 커널 코드 변경, `serve/generate.py`(단일 요청 글 데모)의 백엔드 전환, 대시보드 변경(새 열은 무시된다), 64K 격자 표(32K 표로 외삽하는 기존 규약 유지).

## 2. 설계

### 2.1 표 형식

`kernelscope.analysis.dispatch.static_default_losses`의 출력 열을 그대로 쓴다. 정책이 읽는 열: `workload_key`, `best_kernel`, `best_us`, `fa2_best_kernel`, `fa2_best_us`, `flashinfer_tc_us`, `flashinfer_cc_us`, `complete`. 옛 표(`dispatch_table` 출력)는 `workload_key`, `best_kernel`만 있고 FA2 변형뿐이므로 그대로 읽힌다.

### 2.2 `TablePolicy`

- 생성: `TablePolicy(csv_path, backends=("fa2",), plan_us=0.0, name="table")`. `backends`는 `("fa2", "flashinfer", "flashinfer_cudacore")`의 부분집합.
- 행마다 후보 목록 `[(backend, kernel, num_splits, kernel_us)]`:
  - `fa2` → (`fa2_best_kernel`가 있으면 그것, 없으면 `best_kernel`; `parse_variant(...).num_splits`; `fa2_best_us` 또는 `best_us`). FA2 전용인데 커널이 FlashInfer 변형이면 `ValueError`.
  - `flashinfer` → (`flashinfer_paged`, 0, `flashinfer_tc_us`), `flashinfer_cudacore` → (`flashinfer_paged_cudacore`, 0, `flashinfer_cc_us`). 시간이 NaN이면 그 후보를 뺀다. `complete`가 있으면 False 행은 뺀다.
  - 후보가 하나도 없는 행은 `ValueError`.
- 비용: `cost = n_layers × kernel_us + (plan_us if backend != "fa2" else 0)`. `plan_us == 0`이면 `n_layers`와 무관하게 커널 최선 = `best_kernel`(TODO의 정의). `plan_us > 0`인데 `n_layers`가 `None`이면 `choose`에서 `ValueError`.
- `choose(lens, n_heads, n_kv_heads)`: 기존과 같은 특징·거리로 최근접 셀 → 그 셀의 후보 중 비용 최소 → `self.attention = backend`를 갱신하고 `num_splits`를 돌려준다. `self.last = {"workload_key", "distance", "kernel", "attention", "cost_us"}`. 캐시 키는 기존(page 양자화)과 같고 캐시 값에 `attention`도 넣는다.
- `n_layers`: 인스턴스 속성, 기본 `None`. `Engine.__init__`가 `policy.n_layers is None`이면 `model.cfg.n_layers`를 넣는다. 정책 지연 벤치(`scripts/benchmark_policy_latency.py`)와 대시보드 결정 카드(`dashboard/demo.py`)는 `plan_us == 0` 표만 쓰므로 영향 없음.
- `name`: `table`(기존), `table_any`(plan_us 0), `table_any_p<plan_us:g>`(예: `table_any_p400`). 한 실행 안에서 정책 이름이 유일해야 하므로 plan_us가 다른 두 표를 함께 돌릴 수 있다.

### 2.3 `make_policy`

`table:<csv>` 불변. `table_any:<csv>` → `TablePolicy(csv, backends=모두, plan_us=0, name="table_any")`; `table_any:<csv>:<plan_us>` → `plan_us=float`, `name=f"table_any_p{plan_us:g}"`. 파싱은 `hybrid:`와 같은 `rpartition(":")` 규칙(경로에 `:`가 없다고 가정, 기존과 동일). `HybridPolicy`는 `TablePolicy(csv_path)`(FA2 전용)를 그대로 쓴다 — any 표를 주어도 `fa2_best_kernel`로 FA2만 고른다.

### 2.4 엔진

- `STEP_COLUMNS = [..., "num_splits", "attention", "attn_us", ...]`. 값은 `getattr(policy, "attention", "fa2")`를 `choose` 뒤에 읽은 것(지금도 그 순서). 기존 열의 의미는 바꾸지 않는다(`attn_us`·`policy_us` 규약 동결).
- `Engine.__init__`: `if getattr(policy, "n_layers", "absent") is None: policy.n_layers = model.cfg.n_layers`.
- guard: `name in {"model", "table", "hybrid"} or name.startswith("table_any")`.
- `check_policy_numerics.py`는 `model.decode(**kwargs)`를 감싸므로 `attention` 인자가 그대로 전달된다(확인 후 테스트 없음).

### 2.5 표 생성

`package_demo.py`의 기존 `dispatch_paged_cold.csv` 생성 옆에: 같은 방식으로 6개 그룹의 `summaries.jsonl`에서 `status == ok`, `cache_state == cold` 행을 모아 `static_default_losses(frame)`를 만들고 `complete` 행만 `dispatch_paged_cold_any.csv`로 쓴다(격자가 없으면 건너뜀). `run.sh`에 `TABLE_ANY` 변수.

## 3. 테스트 (CPU, `CUDA_VISIBLE_DEVICES=`)

- `tests/test_serve_dispatch.py`: any 표에서 CUDA-core가 최선인 셀 → `choose` 0, `attention == "flashinfer_cudacore"`, `last["kernel"] == "flashinfer_paged_cudacore"`; 같은 표에 `plan_us`를 크게 주고 `n_layers`를 넣으면 `fa2_best_kernel`의 분할과 `"fa2"`; `plan_us > 0`인데 `n_layers None`이면 `ValueError`; FA2 전용 정책이 any 표에서 `fa2_best_kernel`을 쓴다; `fa2_best_kernel` 없는 표의 `best_kernel`이 FlashInfer면 `ValueError`; `make_policy("table_any:p")`·`("table_any:p:400")` 이름; 캐시 히트 시 `attention`이 복원된다.
- `tests/test_serve_engine.py`: `steps.columns == STEP_COLUMNS`(기존 단언이 자동 검증), `attention` 열이 `"fa2"`로 채워짐; `attention` 인자를 받는 더블과 백엔드를 바꾸는 가짜 정책으로 step별 값이 기록되고 decode에 전달됨; `n_layers` 주입.
- `tests/test_package_demo.py`: 6개 그룹이 있을 때 `dispatch_paged_cold_any.csv`가 생기고 `complete` 행만 있다.

## 4. GPU 캠페인 (규약은 10월 2일·6일과 동일)

```bash
PLAN=<FlashInfer 캠페인 혼합 조건의 flashinfer_cudacore policy_us_per_step, 정수 반올림>
for s in ragged uniform arrivals; do
  .venv/bin/python -m kernelscope.cli serve run --model Qwen/Qwen3-4B-Instruct-2507 --scenario scenarios/graduation_$s.yaml \
    --policy heuristic --policy table:demo_data/dispatch_paged_cold.csv \
    --policy table_any:demo_data/dispatch_paged_cold_any.csv --policy table_any:demo_data/dispatch_paged_cold_any.csv:$PLAN \
    --machine machines/rtx4090.json --params models/rtx4090.json --kv-gib 10 --warmup-runs 1 --warmup-steps 2 --repeats 3 --seed 0 \
    --policy-cache keep --out ../kernelscope/results/serve_4090/anytable_20261008/$s || true   # 종료 코드 1 = 토큰 비동일(정상)
done
.venv/bin/python -m scripts.check_policy_numerics --scenario scenarios/graduation_ragged.yaml --reference heuristic \
  --policy table_any:demo_data/dispatch_paged_cold_any.csv --policy table_any:demo_data/dispatch_paged_cold_any.csv:$PLAN \
  --machine machines/rtx4090.json --params models/rtx4090.json --steps 64 --logit-steps 64 --kv-gib 10 \
  --out ../kernelscope/results/serve_4090/anytable_20261008/numerics/ragged
CUDA_VISIBLE_DEVICES= .venv/bin/python -m scripts.classify_divergence --campaign ../kernelscope/results/serve_4090/anytable_20261008
```

기대: `table_any`는 혼합 조건에서 CUDA-core를 고르고 커널 이득(≈350 µs/step)이 `plan()` 비용(≈400 µs/step)과 상쇄되어 `table`과 같거나 조금 느리다; `table_any_p<plan>`은 대부분 FA2 분할로 돌아가 `table`과 같다. 어느 쪽이든 "외부 분할 선택이 커널 교체 이득을 거의 전부 얻는다"(PLAN §2 4번)의 엔진 수준 확인이다. 반대 결과(any 표가 뚜렷이 빠름)면 §0의 산수가 틀린 것이므로 step별 `attention`·`policy_us`로 원인을 적는다.

## 5. 산출

결과 `../kernelscope/results/serve_4090/anytable_20261008/`, 번들 `demo_data/serve_4090/anytable_20261008/`, verify `anytable.*`(TPOT·배율·step별 백엔드 분포·plan 비용·분기 사건), [문제 범위 노트](../experiments/2026-10-06-problem-scope.md) §5, PLAN §1 ⑦·§2 8번, STATUS, TODO.
