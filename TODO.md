# TODO — 다른 세션에서 바로 이어서 하기 위한 작업 목록

> 2026-09-28 기준. 브랜치 `design-1-3`. 계획의 단일 원본은 [PLAN.md](PLAN.md), 진행 기록은 [docs/STATUS.md](docs/STATUS.md).
> 호스트 구분과 불변 규칙은 [CLAUDE.md](CLAUDE.md). **커밋·푸시는 사용자가 요청할 때만.**
> 연구실 4090 호스트: 저장소 `/home/skkai/AI_Accelerator/kernelscope-design`(worktree), 파이썬 `.venv/bin/python`
> (= `/home/skkai/miniforge3/envs/gradkernel/bin/python`). 원본 결과는 리포 밖 `../kernelscope/results/`.

## 0. 지금 상태 한 줄

`serve diagnose`(decode step을 8개 연산 클래스로 CUDA event 분해 + DRAM/텐서코어 상한 판정) **Task 1~9 전부 완료**.
GPU 실행 D1~D3(2026-09-27 22:35~22:38, Qwen3-4B)까지 끝났고 번들은 `demo_data/serve_4090/diagnose_20260927/`,
실험 노트는 [docs/experiments/2026-09-27-op-breakdown.md](docs/experiments/2026-09-27-op-breakdown.md). `make test` 527 통과, `make verify` 47/47.

- 스펙: [docs/plan/2026-09-27-design-op-breakdown-diagnose.md](docs/plan/2026-09-27-design-op-breakdown-diagnose.md)
- 계획: [docs/plan/2026-09-27-plan-op-breakdown-diagnose.md](docs/plan/2026-09-27-plan-op-breakdown-diagnose.md)
- 핵심 수치: 혼합 길이 attention 비중 70.9%(휴리스틱)→37.5%(테이블), 균일 19.2%; GEMM 클래스 DRAM 63~90%; A1~A8 전부 기준 안(A8 성능 경로 회귀 +0.04%).

### Task 5 수정~Task 9 변경 — 2026-10-02 커밋(코드·테스트 / 번들·그림 / verify·문서, 3개 커밋)
`kernelscope/diagnose/{figures,run}.py`(신규), `kernelscope/diagnose/report.py`, `kernelscope/serve/cli.py`, `kernelscope/verify.py`,
`tests/test_diagnose_{figures,report}.py`, `tests/test_serve_cli.py`, `tests/test_verify.py`,
`docs/experiments/2026-09-27-op-breakdown.md`(신규), `docs/img/op_breakdown_ragged.png`(신규), `docs/STATUS.md`, `PLAN.md`, `README.md`, `TODO.md`,
`demo_data/serve_4090/diagnose_20260927/**`(신규 75개), `demo_data/provenance.json`(재색인). 작성자: sunjae0224 <sunjae0224@gmail.com>.

## 1. 다음에 할 것 (PLAN.md §2, GPU 필요 — 유휴일 때만: `.venv/bin/python -m kernelscope.cli serve doctor`)
1. 혼합 정책 실생성 검증 — `serve run --policy hybrid:demo_data/dispatch_paged_cold.csv:0.2`, Qwen3-4B 3조건 × 3회.
2. 불일치 지표 교체 — 캠페인마다 `scripts/check_policy_numerics.py`(교사 강제 진단) 실행, 분기 사건을 `tie_1ulp/tie_2ulp/clear`로 분류.
3. Accel-Sim L2 파티션 해시 변경 후 재검증.
4. FlashInfer 비교 — [docs/plan/2026-09-26-plan-flashinfer-comparison.md](docs/plan/2026-09-26-plan-flashinfer-comparison.md) (11월).
5. vLLM 재현(중간보고서 표 8) — 4090의 vLLM FlashAttention 백엔드에서 혼합 길이 배치의 동일 손실이 나는지 측정만(통합 아님). 11월.

선택(작음): 실험 노트의 control step 비율(D1 2.139, D3 1.523)에 `verify` 항목을 추가하면 노트의 모든 수치가 검사 대상이 된다(현재는 헤드라인 7개만).

## 2. 보고서
- 중간보고서 v4 산출물: `docs/report/midterm/중간보고서_이선재_v4.{pdf,docx}`. 소스 `docs/report/midterm/src/`(docx-js).
  재빌드: `cd docs/report/midterm/src && npm install && REPORT_FONT="Noto Sans CJK KR" REPORT_MONO="DejaVu Sans Mono" node build.js ../중간보고서_이선재_v4.docx && soffice --headless --convert-to pdf --outdir .. ../중간보고서_이선재_v4.docx`
  (Windows/맑은 고딕 환경이면 REPORT_FONT 생략).
- 최종 보고서에 넣을 `serve diagnose` 근거: "균일 배치에서 이득이 없는 이유"(attention 19.2%, 이미 memory_bound), 혼합 배치에서 attention 외 클래스는 DRAM 상한 근처라 선택 여지가 작음, Amdahl 추정 대 실측(2.10 vs 2.14). 그림 `docs/img/op_breakdown_ragged.png`.
- 최종 보고서에서 답할 것: 시뮬레이터가 선택에 기여한 바, 출력 검증 기준 변경 사유, FlashInfer 격차.
- 선택: 이전 판의 "원안 대비 변경 사항과 사유" 절을 3장에 짧게 복원.

## 3. 작업 메모
- 셸이 zsh라 `git diff -- $FILES`처럼 따옴표 없는 변수의 단어 분리가 되지 않는다. 파일 목록은 명시적으로 나열.
- GPU 측정 전 `serve doctor`. preflight는 `rerun` 외의 다른 GPU 프로세스가 하나라도 있으면 막는다(STiTy vLLM 벤치가 자주 점유). 유휴를 기다릴 때는 `nvidia-smi --query-gpu=memory.used`를 30초 간격으로 폴링.
- `kernelscope/diagnose/opmodel.py`, `report.py`, `verify.py`는 torch를 import하지 않는다(GPU 없는 노트북의 verify 경로). 유지할 것.
- `AttentionTimer.start(layer)/stop(layer)/total_us()`·`STEP_COLUMNS`·`attn_us` 의미는 동결. `serve diagnose`의 `--policy table:`은 d=128 fp16/bf16 모델에서만 동작(tiny-random은 heuristic만).
- `serve diagnose-report <run_dir>`로 기록된 parquet에서 GPU 없이 diagnosis.json/ops.csv/그림을 재생성할 수 있다.
