# TODO — 다른 세션에서 바로 이어서 하기 위한 작업 목록

> 2026-10-25 아침 기준(야간 자동 작업 P1a~P4 뒤). 브랜치 `design-1-3`. 계획의 단일 원본은 [PLAN.md](PLAN.md), 진행 기록은 [docs/STATUS.md](docs/STATUS.md).
> 호스트 구분과 불변 규칙은 [CLAUDE.md](CLAUDE.md). **커밋·푸시는 사용자가 요청할 때만.**
> 연구실 4090 호스트: 저장소 `/home/skkai/AI_Accelerator/kernelscope-design`(worktree), 파이썬 `.venv/bin/python`
> (= `/home/skkai/miniforge3/envs/gradkernel/bin/python`). 원본 결과는 리포 밖 `../kernelscope/results/`.

## 0. 지금 상태 한 줄

2026-10-24 밤 야간 작업으로 최종 보고의 논리 빈틈을 메우는 근거가 생겼다 — 전부 [docs/experiments/2026-10-24-final-evidence.md](docs/experiments/2026-10-24-final-evidence.md)에, 요약은 [docs/handoff/2026-10-25-overnight-report.md](docs/handoff/2026-10-25-overnight-report.md): (1) **고정 분할 수 기준선**(§1): 커널 격자에서 분할 8 하나가 최선 FA2 분할 대비 중앙값 1.010·최악 1.127이고, 엔진 4시나리오 실생성에서 고정 8·16의 TPOT가 table의 ±1% 안(혼합 1.811/1.804 대 1.813) → 동적 선택의 실측 이점은 균일 길이의 출력 보존(table 0곳 대 고정 6곳)뿐. (2) **문제 크기의 실측 검증**(§2·§6·§7): conv ×1의 13.5%(모델 예측)는 커널 표본 실측 8.9%(6.3–10.9%)로 과대 예측이지만 attention 시간 비는 실측 1.144 > 예측 1.119; 트레이스 구동 e2e에서 table 1.035배, model·hybrid는 캐시가 빈 첫 반복에서 0.86배. (3) **vLLM step 분해**(§8): 혼합−균일 차이의 97.6%가 attention GPU 시간, **10월 6일 "백엔드 무관 0.997배"는 FLASHINFER 실행이 FLASH_ATTN이었으므로 근거 없음**. (4) 64K 자연어 분기 0곳(§9), GSM8K 네 정책 정확도 동일(§10). (5) 관련 연구 초안 `docs/report/final/related_work_additions.md`, 발표 배경 파트 `docs/presentation/background/`. `make verify` 944/944, `make test` 718 passed(`CUDA_VISIBLE_DEVICES=`). 10월 8일 TODO §1의 1번(vLLM 프로파일)은 끝났다. 미실행: Mooncake(§3), 임계 표(§4), δ 민감도(§5).

### 커밋 상태
- origin/design-1-3 = **bdbc2c4**(2026-10-08 밤). 그 위의 **미커밋 변경**(사용자가 요청하면 작성자 sunjae0224 <sunjae0224@gmail.com>으로 커밋; 묶음 제안은 아침 보고서 §7):
  - 10월 8일 밤(야간 작업 전, 사용자 작업): 수정 `.gitignore`, `TODO.md`(v5 줄), `docs/report/midterm/src/{build.js,lib.js}`; 신규 `docs/report/midterm/src/{build_memo.js,content1_v5.js,content2_v5.js,make_figs_v5.py,md2docxjs.py,memo_v5.js,package-lock.json}`, `docs/report/midterm/중간보고서_이선재_v5.{docx,pdf}`, `docs/report/midterm/중간보고서_v5_검토메모.{md,pdf}`, `docs/report/fig/fig{8,9,10,11}_*.png`.
  - 야간 작업(10월 24~25일): 수정 `PLAN.md`, `TODO.md`, `docs/STATUS.md`, `demo_data/provenance.json`, `kernelscope/analysis/{dispatch,traffic}.py`, `kernelscope/results/store.py`(parquet 파일명 태그 120자 제한 — 대기열 수리), `kernelscope/verify.py`(+413항목), `scripts/traffic_replay.py`(`--fixed-splits`), `tests/test_{store,traffic}.py`;
    신규 `kernelscope/analysis/{accuracy,validation,vllm_steps}.py`, `scripts/{analyze_gsm8k_policies,bundle_traffic_sweep,gsm8k_accuracy,sample_replay_cells,trace_scenario,vllm_profile_step}.py`, `scenarios/{demo_text_ragged_64k,gsm8k_ragged32k_n200,trace_azure_conv_i550_n300}.yaml`, `tests/test_{fixed_split,gsm8k_accuracy,p3_analysis,sample_replay_cells,trace_scenario,vllm_profile_step}.py`,
    문서 `docs/experiments/2026-10-24-final-evidence.md`, `docs/handoff/2026-10-25-overnight-report.md`, `docs/report/final/related_work_additions.md`, `docs/presentation/background/{story.md,make_figs.py,prep_length_hist.py,fig_b1_length_dist.png,fig_b2_load_sweep.png,fig_b3_kernel_regret.png,fig_b5_cta_imbalance.png}`,
    번들 `demo_data/background/`, `demo_data/hw_4090/trace_validation_20261024/`, `demo_data/serve_4090/{fixedsplit,gsm8k,longctx_text,trace_e2e}_20261024/`, `demo_data/trace_e2e/`, `demo_data/traffic_sweep/`, `demo_data/vllm_4090/`.
- 리포 밖 결과(야간): `../kernelscope/results/{serve_4090/{fixedsplit,trace_e2e,longctx_text,gsm8k}_20261024, hw_4090/trace_validation_20261024, vllm_4090/profile_20261024(+ _failed_20261025T004829), traffic_sweep_20261024(+ _failed_20261024T2230), datasets/{traces_20261024,azure_llm_inference_2023,gsm8k}, overnight_20261024/}`. 모든 캠페인 manifest의 `git.status`가 더럽다(미커밋 상태에서 잼).
- 환경: 변화 없음(패키지 설치 없음).

## 1. 다음에 할 것 (우선순위 순)
1. **사용자 결정 7건**(아침 보고서 §5): ① 보고서·PLAN §0의 13.5%에 실측 8.9% 병기, ② 고정 분할 대비 동적 선택의 정당화 문구(TPOT는 같고 근거는 출력 보존), ③ vLLM "백엔드 무관 0.997배" 철회 문구(PLAN §2 7번·중간보고서 표 8·2026-10-02 노트 §9), ④ FlashInfer "GQA ≥ 4 tensor-core 권장" → SGLang 규칙으로 고치기(PLAN §0·보고서 2장), ⑤ 트레이스 e2e 보고 규약(첫 반복 분리·"선택된 구간"), ⑥ GSM8K 주 지표 사후 변경 표기, ⑦ 고정 8 혼합 반복 1 이상치 재측정 여부(GPU 약 3분).
2. **보고서 1·2장** — PLAN §0 개정 정의 + 문제 범위 노트 요약 + 이번 밤 근거(§1 표 ⑧~⑩)를 옮긴다. 관련 연구 초안의 `[R*]` 번호를 `[23]~`로 재배정해 2장에 넣는다(`content1_v5.js` 23줄 수정 포함).
3. **발표 자료** — 배경 파트는 `docs/presentation/background/story.md` + 그림 4장. 해결·검증 파트는 아직 없다. 도식(슬라이드 2)과 `fig_b4`(임계 표)는 미작성.
4. **선택 GPU 작업**(필요하면): 고정 8 혼합 재측정(3분), 트레이스 e2e 분기 9곳의 교사 강제 분류(`check_policy_numerics --logit-steps` 확대), GSM8K 분기 분류, vLLM CUDA graph 모드의 실제 `num_splits` 확인.
5. **미실행 절**: Mooncake 긴 문맥 트레이스(§3, 공개 데이터 다운로드 필요), 모델×GPU 분할 없음 임계 표(§4, 식 B ≥ 0.8·2·SM/H_kv는 있음), 신뢰구간·δ 민감도(§5). 중단 조건에 따라 생략 가능.
6. 도구 정리(측정 경로 아님): `scripts/vllm_profile_step.py analyze`의 `summary` 집계가 iteration 종류를 섞음(보정은 `kernelscope/analysis/vllm_steps.py`) + `engine_reported_backend` 정규식이 0.31 로그 문구(`Using AttentionBackendEnum.X backend.`)를 놓침; README의 "531개 검사" → 944. PLAN §2 3(b) Accel-Sim L2 해시 재검증은 그대로.
7. Demo 페이지 후속(다른 세션): 10월 8일 항목 그대로(실행 버튼 생존, 결과 폴더 새로고침, 장면 2·3에 표 추가, `attention` 열 읽기).

## 2. 보고서
- 중간보고서 v4 산출물: `docs/report/midterm/중간보고서_이선재_v4.{pdf,docx}`. 소스 `docs/report/midterm/src/`(docx-js).
  재빌드: `cd docs/report/midterm/src && npm install && REPORT_FONT="Noto Sans CJK KR" REPORT_MONO="DejaVu Sans Mono" node build.js ../중간보고서_이선재_v4.docx && soffice --headless --convert-to pdf --outdir .. ../중간보고서_이선재_v4.docx`
- **중간보고서 v5(2026-10-08 개정판)**: `docs/report/midterm/중간보고서_이선재_v5.{pdf,docx}` — 10월 2~8일 작업(문제 범위·두 라이브러리 기본 설정·64K·하이브리드 실생성·FlashInfer 통합·분기 분류·vLLM) 반영, 29쪽. 소스 `src/content{1,2}_v5.js`(v4 소스는 그대로), 새 그림 `docs/report/fig/fig{8,9,10,11}_*.png`(`src/make_figs_v5.py`). 재빌드: `REPORT_CONTENT=_v5 REPORT_DATE="2026 년   10 월   8 일" REPORT_FONT="Noto Sans CJK KR" REPORT_MONO="DejaVu Sans Mono" node build.js ../중간보고서_이선재_v5.docx`. 파트별 논리 점검·보강 실험 제안은 `docs/report/midterm/중간보고서_v5_검토메모.{md,pdf}`(`src/build_memo.js` + `src/md2docxjs.py`로 빌드).
- 최종 보고서에 넣을 근거: 문제의 크기(트레이스 재생 표), 두 라이브러리의 정적 기본값 표, 64K 추세와 실생성 2.4배, "균일 배치에서 이득이 없는 이유"(`serve diagnose`, attention 19.2%), 혼합 정책 실생성과 선택 비용의 한계, 출력 검증 기준(분기 사건 + 동점 분류 + 커널 탐침 — 이제 FA2 분할·FlashInfer 두 백엔드·64K 세 경우 모두 "커널 오류 아님"), FlashInfer 격차(커널 격자 + 엔진 캠페인 §4), 라이브러리 무관 선택표의 +1%와 plan() 한계(§5).

## 3. 작업 메모
- 셸이 zsh라 따옴표 없는 변수의 단어 분리가 되지 않는다. 파일 목록은 명시적으로 나열. `rm -rf "$VAR"/*` 꼴은 도구가 막는다 — 리터럴 경로를 쓴다. `pkill -f <패턴>`은 자기 셸도 죽일 수 있다(패턴을 `[0]` 꼴로 비켜 쓴다).
- GPU 측정 전 `serve doctor`. preflight는 `rerun` 외의 다른 GPU 프로세스가 하나라도 있으면 막는다. 유휴를 기다릴 때는 `nvidia-smi --query-compute-apps`를 30초 간격으로 폴링하고 연속 8회 유휴를 확인한 뒤 시작(ollama `llama-server`가 수시로 뜬다). 측정 중에는 30초 간격 nvidia-smi 로그를 남겨 나중에 간섭 여부를 확인한다. **`make test`도 CUDA 컨텍스트를 만든다** — `CUDA_VISIBLE_DEVICES= make test`.
- **`serve run`의 종료 코드 1 = 토큰 strict equivalence 실패(FlashInfer·`table_any` 정책에서는 정상 결과)**이고 결과 폴더는 완전하다(manifest `status: complete`). 대기열 스크립트는 종료 코드로 실패를 판단하지 말고 manifest의 status를 보며, 실패한 시도는 지우지 말고 `<폴더>_failed_<시각>`으로 옮긴다(`package_demo.py`가 `*_failed_*`를 뺀다). `--out`이 이미 있으면 `serve run`이 거부한다.
- 디스크가 HDD라 다른 사용자의 대용량 다운로드가 돌면 8 GB 체크포인트 적재가 10분 넘게 걸린다(`/proc/<pid>/io`로 확인). 두 번째부터는 페이지 캐시 덕에 1분 안쪽.
- **실행 전 GPU 없는 사전 점검**: 새 캠페인은 기록된 길이로 정책 선택을 재생해 정책들이 갈리는지 보고, 분류 도구가 필요한 열을 기록하는지 정적으로 확인한다. 트레이스 재생(`scripts/traffic_replay.py`)도 GPU 없이 1~2분이라 조건 설계에 먼저 쓴다. 분류(`scripts/classify_divergence.py`)는 시나리오마다 `numerics/<시나리오>`가 있어야 사건을 분류한다 — 없으면 `unclassified`로 남으므로 캠페인마다 교사 강제 진단(`scripts/check_policy_numerics.py`, 조건당 1~2분)을 같이 돌린다.
- 측정 규약: `--policy-cache fresh`(기본) / `keep`. manifest `policy_cache`에 기록. 캠페인 manifest의 `git.status`가 비어 있어야 clean 커밋에서 잰 것이다(10월 6~8일 캠페인은 모두 미커밋 상태에서 쟀다).
- FlashInfer: 커널 플러그인(`flashinfer_paged[_cudacore]`)과 엔진 백엔드(`serve/attention.py`)가 따로 있다. JIT는 cu128 사전 빌드 캐시에 의존하므로 새 head 차원·dtype이면 `tests/test_flashinfer_plugin.py -m gpu`로 먼저 확인. 엔진 백엔드의 `plan()`은 step마다 한 번, 층마다 `run()`; prefill은 FA2. 엔진 안 plan() 비용은 B 32에서 280~430 µs/step.
- 정책: `table:<csv>`(FA2 분할만), `table_any:<csv>[:<plan_us>]`(FA2 + FlashInfer 두 변형; 표는 `static_default_losses` 형식 = `demo_data/dispatch_paged_cold_any.csv`; `plan_us`를 주면 이름이 `table_any_p<plan_us>`), `hybrid:<csv>:<δ>`(안의 표는 FA2 전용). `Engine`이 `policy.n_layers`를 모델에서 채운다. **`steps.parquet`에 `attention` 열이 생겼다**(`num_splits` 다음; 10월 8일 이전 기록에는 없으므로 읽는 쪽은 있을 때만 쓴다). 기존 열의 의미(`attn_us`·`policy_us`·`STEP_COLUMNS` 순서)는 그대로.
- 커널 탐침 `scripts/probe_split_kernel.py`: 인자 없이 돌리면 10월 2일 대조군(기본값)이고, `--campaign/--run/--target-step/--target-rid/--candidate-splits/--candidate-backend/--kv-gib/--out`으로 다른 사건을 잰다(64K: §3c, FlashInfer: §4b의 명령). 판정 기준: 층별 attention 출력 차이가 그 층 최대 출력의 bf16 간격 1개 안이면 구현 오류 아님; 거기에 같은 KV 상태의 단일 step argmax 변화 여부를 함께 적는다.
- `kernelscope/diagnose/opmodel.py`, `report.py`, `verify.py`, `serve/divergence.py`, `analysis/traffic.py`는 torch를 import하지 않는다(GPU 없는 노트북의 verify 경로). 유지할 것. verify 항목 중 `fiengine.*`는 스펙 표에서 생성되므로 수가 많다(148).
- `AttentionTimer.start(layer)/stop(layer)/total_us()`·`attn_us` 의미는 동결. FlashInfer plan 비용은 새 열이 아니라 `policy_us`에 더한다.
