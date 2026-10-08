# TODO — 다른 세션에서 바로 이어서 하기 위한 작업 목록

> 2026-10-08 저녁 기준. 브랜치 `design-1-3`. 계획의 단일 원본은 [PLAN.md](PLAN.md), 진행 기록은 [docs/STATUS.md](docs/STATUS.md).
> 호스트 구분과 불변 규칙은 [CLAUDE.md](CLAUDE.md). **커밋·푸시는 사용자가 요청할 때만.**
> 연구실 4090 호스트: 저장소 `/home/skkai/AI_Accelerator/kernelscope-design`(worktree), 파이썬 `.venv/bin/python`
> (= `/home/skkai/miniforge3/envs/gradkernel/bin/python`). 원본 결과는 리포 밖 `../kernelscope/results/`.

## 0. 지금 상태 한 줄

10월 6일 TODO §1의 1~3번이 끝났다 — 전부 [docs/experiments/2026-10-06-problem-scope.md](docs/experiments/2026-10-06-problem-scope.md)에: (1) **FlashInfer 엔진 캠페인 재측정**(§4): 혼합 길이에서 CUDA-core 정책 1.833배 ≈ FA2 테이블 1.810배(커널은 빠르지만 step당 plan() 0.37 ms), tensor-core 기본값 1.257배; 균일 네 정책 ±1%; 도착 1.181/1.092배. FlashInfer 정책의 출력은 휴리스틱과 비동일(39~223토큰)이고 분기 사건은 tie 위주, clear는 한 자리(요청 8 위치 41 = 10월 2일 대조군과 같은 사건)로 두 백엔드 모두 커널 탐침 통과(§4b). (2) **64K clear 탐침**(§3c): 층별 출력 bf16 간격 1개 안 → 커널 오류 아님, 다만 같은 KV 상태의 단일 step에서 argmax가 바뀜(한 step 안 증폭, 10월 2일과 다른 점). (3) **라이브러리 무관 선택표** `table_any`(설계 [docs/plan/2026-10-08-design-anytable-policy.md](docs/plan/2026-10-08-design-anytable-policy.md), §5): 커널 시간만 보는 표는 혼합 길이 매 step CUDA-core를 골라 FA2 표보다 +1.15%(1.831배, 3회 모두), 도착 +0.36%(편차 수준), 균일 동률; plan() 비용(371 µs/step)을 더해 비교하면 FA2로 돌아가 `table`과 같다 → 라이브러리까지 바꾸는 선택의 추가 이득은 약 1%, 한계는 step당 plan() 비용. `make verify` 531/531, `make test` 667 통과(`CUDA_VISIBLE_DEVICES=`). 시연은 `./run.sh`. Demo 페이지·Policy lab 탭은 다른 세션 담당(b8e1707까지 푸시됨).

### 커밋 상태
- origin/design-1-3 = **b8e1707**(2026-10-06 저녁). 그 위의 **미커밋 변경**(10월 6일 오후~8일 저녁; 사용자가 요청하면 작성자 sunjae0224 <sunjae0224@gmail.com>으로 커밋):
  수정 `PLAN.md`, `README.md`, `TODO.md`, `docs/STATUS.md`, `docs/demo.md`, `docs/experiments/2026-10-02-hybrid-validation.md`(탐침 명령 형식), `docs/plan/2026-10-06-review-problem-definition.md`(§5), `kernelscope/analysis/dispatch.py`, `kernelscope/serve/{dispatch,engine,model}.py`, `kernelscope/verify.py`, `run.sh`, `scripts/{package_demo,probe_split_kernel}.py`, `tests/test_{serve_dispatch,serve_engine,verify}.py`;
  신규 `docs/experiments/2026-10-06-problem-scope.md`, `docs/plan/2026-10-08-design-anytable-policy.md`, `grids/{ragged,uniform}_s1_64k.yaml`, `scenarios/graduation_ragged_64k.yaml`, `kernelscope/analysis/traffic.py`, `kernelscope/serve/attention.py`, `scripts/traffic_replay.py`, `tests/test_{traffic,traffic_replay_script,static_defaults,serve_attention,serve_attention_gpu,package_demo,probe_split_kernel}.py`;
  번들 `demo_data/dispatch_paged_cold_any.csv`, `demo_data/hw_4090/{ragged,uniform}_s1_64k_{paged,flashinfer,flashinfer_cudacore}/`, `demo_data/traffic/*/summary.json`(20개), `demo_data/serve_4090/{longctx_20261006,flashinfer_20261006,anytable_20261008}/**`(실생성 + numerics + 탐침 + divergence.csv), `demo_data/serve_4090/demo_text_20261006.doctor.json`(다른 세션의 doctor 출력이 번들에 딸려 들어옴 — 빼려면 `../kernelscope/results/serve_4090/`의 원본을 지우고 `package_demo.py` 재실행), `demo_data/provenance.json`.
- 환경: 변화 없음(flashinfer-python 0.6.13 + cu128 JIT 캐시, vLLM 전용 venv). 트레이스 원본·재다운로드 방법은 10월 6일 항목과 같다(`./run.sh traffic`이 Azure conv를 받는다). 리포 밖 결과: `../kernelscope/results/{hw_4090/*_64k_*, traffic/, traffic_before_review_fixes/, traffic_every10_prelim/, serve_4090/{longctx_20261006,flashinfer_20261006,anytable_20261008}}`.

## 1. 다음에 할 것 (우선순위 순)
1. **vLLM step 프로파일링** — 10월 6일 노트 §9 결과(혼합 길이 step 79~81 ms가 백엔드·모드·블록 크기와 무관)의 원인 분리: vLLM 전용 venv(`~/.venvs/kernelscope-vllm`)에서 torch profiler 또는 nsys로 decode step 하나를 분해해 attention 시간을 떼어낸다.
2. **보고서** — PLAN §0 개정 정의와 문제 범위 노트의 요약(이제 §1~§5 다섯 줄)을 1·2장에 옮긴다. 새 근거: 엔진 안 FlashInfer 비교표(§4), 라이브러리 무관 선택표의 +1%와 plan() 한계(§5), 두 탐침 판정(§3c·§4b, "커널은 맞고 출력 불안정은 모델 쪽"). 검토 문서 §5의 표와 아티팩트 https://claude.ai/artifact/Avo8pWyMs6ngr7KX6UohCr 가 그 근거.
3. PLAN §2 3(b) Accel-Sim L2 파티션 해시 재검증 — 그대로.
4. Demo 페이지 후속(다른 세션): 실행 버튼의 백그라운드 작업 세션 밖 생존, 결과 폴더 자동 새로고침. Demo 장면 2·3에 §1 트래픽 표·§2 정적 기본값 표·§4 엔진 비교표를 넣을 수 있다(`collect_summaries`, `static_default_summary`, `kernelscope.serve.report.summarize`). 장면 3의 결정 카드는 `steps.parquet`의 새 `attention` 열을 아직 안 읽는다(FlashInfer 백엔드를 고른 step은 "분할 0"으로만 보임).
5. 오늘 드러난 열린 질문(필요하면): (a) FlashInfer `plan()`을 step마다 새로 하지 않고 재사용할 수 있으면 `table_any`의 손익이 바뀐다 — 범위 밖, 보고서엔 한계로만 적는다. (b) tensor-core의 단일 step 로짓 차이(1.97)가 CUDA-core(0.31)보다 큰데 층별 fp32 오차는 반대(0.120 대 0.167)인 이유는 가리지 않았다(§4b). (c) 64K 단일 step 불안정이 무작위 토큰 문맥 특유인지 자연어 64K 시나리오로 볼 수 있다(GPU 수 분, `scenarios/demo_text_ragged.yaml`을 64K로 늘린 시나리오 필요).
6. 정리: `kernelscope/serve/generate.py`(단일 요청 글 데모)의 d=128 guard는 `table_any`를 모른다(그 경로에서 쓸 일은 없음). 결과 폴더 `traffic_before_review_fixes/`, `traffic_every10_prelim/`는 검토 수정 전의 예비 재생이며 번들에 없다(규칙상 기록으로 둠).

## 2. 보고서
- 중간보고서 v4 산출물: `docs/report/midterm/중간보고서_이선재_v4.{pdf,docx}`. 소스 `docs/report/midterm/src/`(docx-js).
  재빌드: `cd docs/report/midterm/src && npm install && REPORT_FONT="Noto Sans CJK KR" REPORT_MONO="DejaVu Sans Mono" node build.js ../중간보고서_이선재_v4.docx && soffice --headless --convert-to pdf --outdir .. ../중간보고서_이선재_v4.docx`
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
