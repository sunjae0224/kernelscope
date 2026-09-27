# TODO — 다른 세션에서 바로 이어서 하기 위한 작업 목록

> 2026-09-27 기준. 브랜치 `design-1-3`. 계획의 단일 원본은 [PLAN.md](PLAN.md), 진행 기록은 [docs/STATUS.md](docs/STATUS.md).
> 호스트 구분과 불변 규칙은 [CLAUDE.md](CLAUDE.md). **커밋·푸시는 사용자가 요청할 때만.**
> 연구실 4090 호스트: 저장소 `/home/skkai/AI_Accelerator/kernelscope-design`(worktree), 파이썬 `.venv/bin/python`
> (= `/home/skkai/miniforge3/envs/gradkernel/bin/python`). 원본 결과는 리포 밖 `../kernelscope/results/`.

## 0. 지금 상태 한 줄

`serve diagnose`(decode step을 8개 연산 클래스로 CUDA event 분해 + DRAM/텐서코어 상한 판정) 구현 계획 10개 Task 중 **Task 1~5 완료·커밋**,
Task 6~9 남음. 중간보고서 v4(`docs/report/midterm/중간보고서_이선재_v4.pdf`) 생성 완료.

- 스펙: [docs/plan/2026-09-27-design-op-breakdown-diagnose.md](docs/plan/2026-09-27-design-op-breakdown-diagnose.md)
- 계획(Task별 코드·테스트 포함): [docs/plan/2026-09-27-plan-op-breakdown-diagnose.md](docs/plan/2026-09-27-plan-op-breakdown-diagnose.md)
- 완료: `OpTimer`/`AttentionTimer`(kernelscope/serve/model.py), `Engine.run(ops_mode="event")`·`RunResult.ops`(serve/engine.py),
  `MachineSpec.tc_tflops`·`ridge_flop_per_byte()`(model/machine.py), `kernelscope/diagnose/opmodel.py`, `kernelscope/diagnose/report.py`,
  테스트 5개 파일. CPU 스위트 518 통과(`make test`), `make verify` 40/40.

## 1. 즉시 이어서 할 것 (순서대로)

### 1-1. Task 5 리뷰 지적 3건 수정 (`kernelscope/diagnose/report.py`, `tests/test_diagnose_report.py`)
- (a) `decode_table`/`_mean_costs`: steps가 비어 있으면 `IndexError`. `prefill_table`처럼 빈 입력 가드 추가(빈 표 반환).
- (b) `diagnose()`: 정책 폴더를 `ops.parquet` 존재로만 고르고 `steps.parquet`이 없으면 `TypeError`. 파일명을 담은 `FileNotFoundError`로.
- (c) `attention_steps`의 `source="model"` 폴백(테이블에 헤드 구성이 없고 `machine`·`params`가 있을 때 `rank_variants` 사용)에 테스트가 없음.
  `models/rtx4090.json` + `machines/rtx4090.json`으로 테스트 추가.
- 확인: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_diagnose_report.py`

### 1-2. Task 6 — `kernelscope/diagnose/figures.py` (정책별 연산 클래스 누적 막대, matplotlib/Agg)
계획 문서 Task 6의 코드·테스트를 그대로 사용. 테스트 `tests/test_diagnose_figures.py`.

### 1-3. Task 7 — `kernelscope/diagnose/run.py` + CLI `serve diagnose`, `serve diagnose-report`
- 계획 문서 Task 7의 코드 그대로. `kernelscope/serve/cli.py`의 `register_parser` 루프에 `("diagnose", _diagnose)` 추가, `--table/--data/--threshold` 인자, `diagnose-report` 서브커맨드.
- CPU 확인: `tests/test_serve_cli.py`의 diagnose 테스트(tiny-random, `--device cpu --kv-gib 0.002`).
- 끝나면 `make test` 전체 통과 확인.

### 1-4. Task 8 — GPU 실행 (연구실 4090, 유휴일 때만: `.venv/bin/python -m kernelscope.cli serve doctor`)
```bash
cd /home/skkai/AI_Accelerator/kernelscope-design && export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
R=../kernelscope/results/serve_4090/diagnose_20260927
# D1 혼합 길이 / D2 균일
for s in ragged uniform; do .venv/bin/python -m kernelscope.cli serve diagnose --model Qwen/Qwen3-4B-Instruct-2507 \
  --scenario scenarios/graduation_$s.yaml --policy heuristic --policy table:demo_data/dispatch_paged_cold.csv \
  --kv-gib 10 --warmup-runs 1 --warmup-steps 2 --out $R/$s; done
# D3 자연어 혼합 (model 정책)
.venv/bin/python -m kernelscope.cli serve diagnose --model Qwen/Qwen3-4B-Instruct-2507 --scenario scenarios/heldout_text_ragged.yaml \
  --policy heuristic --policy model --machine machines/rtx4090.json --params models/rtx4090.json \
  --kv-gib 10 --warmup-runs 1 --warmup-steps 2 --out $R/heldout_ragged
.venv/bin/python scripts/package_demo.py --results ../kernelscope/results   # demo_data/serve_4090/diagnose_20260927/ 생성
```
- 합격 기준 A1~A6은 계획 문서 Task 8 Step 5의 스니펫으로 출력(미귀속 ≤10%, 타이머 오버헤드 ≤5%, attention 행이 dispatch 표와 일치, attention만 `parallelism_candidate`, Amdahl 상한 ≥ 실측 배율, control/event 토큰 일치).
- **A8(성능 경로 회귀 ≤2%)**: 변경 전 기준선은 아직 못 쟀음(GPU가 다른 프로세스에 점유돼 있었음). 변경 전 코드는 커밋 `6ce775f`.
  `git archive 6ce775f | tar -x -C <scratch>` 로 사본을 만들고 `PYTHONPATH=<scratch>`로 `serve run --scenario scenarios/graduation_uniform.yaml --policy heuristic --repeats 2 --warmup-runs 1 --warmup-steps 2 --kv-gib 10`을
  before/after 연달아 실행해 `steps.parquet`의 `step_us` 중앙값을 비교.

### 1-5. Task 9 — verify 항목 7개 + 문서
- `kernelscope/verify.py`에 `diagnose.*` 7개(계획 문서 Task 9 Step 3 코드; 값은 D1~D3 결과로 채움). `tests/test_verify.py`에 테스트 1개.
- `docs/experiments/2026-09-27-op-breakdown.md`(표·그림·A1~A8 결과·한계), `docs/img/op_breakdown_ragged.png`,
  `PLAN.md` §1 ② 근거 한 줄, `docs/STATUS.md` 항목, `README.md`에 `serve diagnose` 블록.
- 최종 확인: `make test`, `make verify`(47개 PASS 기대), 문서에 프로파일러 우월성 표현 0건.

## 2. PLAN.md §2의 GPU 필요 작업 (그대로 유효)
1. 혼합 정책 실생성 검증 — `serve run --policy hybrid:demo_data/dispatch_paged_cold.csv:0.2`, Qwen3-4B 3조건 × 3회.
2. 불일치 지표 교체 — 캠페인마다 `scripts/check_policy_numerics.py`(교사 강제 진단) 실행, 분기 사건을 `tie_1ulp/tie_2ulp/clear`로 분류.
3. Accel-Sim L2 파티션 해시 변경 후 재검증.
4. FlashInfer 비교 — [docs/plan/2026-09-26-plan-flashinfer-comparison.md](docs/plan/2026-09-26-plan-flashinfer-comparison.md) (11월).
5. **vLLM 재현(신규, 중간보고서 표 8에 추가됨)** — 4090의 vLLM FlashAttention 백엔드에서 혼합 길이 배치의 동일 손실이 나는지 측정만(통합 아님). 11월.

## 3. 보고서
- v4 산출물: `docs/report/midterm/중간보고서_이선재_v4.{pdf,docx}`. 소스 `docs/report/midterm/src/`(docx-js).
  재빌드: `cd docs/report/midterm/src && npm install && REPORT_FONT="Noto Sans CJK KR" REPORT_MONO="DejaVu Sans Mono" node build.js ../중간보고서_이선재_v4.docx && soffice --headless --convert-to pdf --outdir .. ../중간보고서_이선재_v4.docx`
  (Windows/맑은 고딕 환경이면 REPORT_FONT 생략).
- 선택: 이전 판에 있던 "원안 대비 변경 사항과 사유" 절(전환 대상 구체화·시뮬레이터 활용 방식 변경·측정 방법 변경)을 3장에 짧게 복원.
- 최종 보고서에서 답할 것: 시뮬레이터가 선택에 기여한 바(대리 모델 what-if와 시뮬레이터 민감도 방향 일치 또는 가상 하드웨어 선택 데모), 출력 검증 기준 변경 사유, FlashInfer 격차.

## 4. 작업 메모
- 셸이 zsh라 `git diff -- $FILES`처럼 따옴표 없는 변수의 단어 분리가 되지 않는다. 파일 목록은 명시적으로 나열.
- GPU 측정 전 `serve doctor`로 다른 프로세스(예: vLLM EngineCore) 점유 여부 확인. 점유 중이면 preflight가 막는다.
- `kernelscope/diagnose/opmodel.py`, `report.py`는 torch를 import하지 않는다(GPU 없는 노트북의 verify 경로). 유지할 것.
- 서브에이전트로 구현할 때는 계획 문서의 Task 브리프를 그대로 넘기고, `AttentionTimer.start(layer)/stop(layer)/total_us()`·`STEP_COLUMNS`·`attn_us` 의미는 동결.
