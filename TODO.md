# TODO — 다른 세션에서 바로 이어서 하기 위한 작업 목록

> 2026-10-06 기준. 브랜치 `design-1-3`. 계획의 단일 원본은 [PLAN.md](PLAN.md), 진행 기록은 [docs/STATUS.md](docs/STATUS.md).
> 호스트 구분과 불변 규칙은 [CLAUDE.md](CLAUDE.md). **커밋·푸시는 사용자가 요청할 때만.**
> 연구실 4090 호스트: 저장소 `/home/skkai/AI_Accelerator/kernelscope-design`(worktree), 파이썬 `.venv/bin/python`
> (= `/home/skkai/miniforge3/envs/gradkernel/bin/python`). 원본 결과는 리포 밖 `../kernelscope/results/`.

## 0. 지금 상태 한 줄

PLAN §2의 1(혼합 정책 실생성 검증)·2(불일치 지표 교체)는 2026-10-02에, 1의 후속(정책 캐시 유지 규약 재측정, 엔진 guard)과 4의 커널 부분(FlashInfer 격자 비교), 7(vLLM 재현 1차)은 **2026-10-06에** 끝났다.
결과 노트 [docs/experiments/2026-10-02-hybrid-validation.md](docs/experiments/2026-10-02-hybrid-validation.md)(§3b 캐시 유지 규약, §8 FlashInfer, §9 vLLM), 검토 문서 [docs/plan/2026-10-06-review-problem-definition.md](docs/plan/2026-10-06-review-problem-definition.md).
`make verify` 111/111(확인 명령은 §3). 대시보드에 **05 Policy lab** 탭(규약·백엔드·분기 사건 비교 + 측정 명령 조립·실행) 초안이 있고, 시연은 루트의 **`./run.sh`**(check · dashboard · verify · kernel · serve · diagnose · divergence · generate · gpu-all)로 한다.

- 핵심 수치: 요청 3조건에서 혼합 정책 = 테이블 선택(0/567 step), 혼합 길이 TPOT 1.793배(테이블 1.807배), 토큰 전부 일치. held-out 요청 도착: 캐시를 비우는 규약 1.020배 → 캐시 유지 규약 **1.090배**(테이블 1.086배, 모델 1.091배). FlashInfer: CUDA-core 변형 / 최선 FA2 분할 = 혼합 길이 162셀 중앙값 0.977배·최대 1.003배, tensor-core 변형 최악 2.401배.
- 불일치 지표: `scripts/classify_divergence.py`. 대조군 균일×fixed:8의 `clear` 1건은 커널 탐침(`scripts/probe_split_kernel.py`)으로 구현 오류 아님 확인.

### 커밋 상태
- 2026-10-02 커밋 3개(a25324c 코드·테스트, 6e8d13f 번들·그림, 195e9e1 verify·문서)는 로컬에만 있다 — **origin/design-1-3 푸시는 사용자 확인 후**.
- 그 위의 **미커밋 변경**(사용자가 요청하면 커밋): `kernelscope/serve/{divergence,engine,cli}.py`, `kernelscope/verify.py`, `kernelscope/plugins/builtin/{__init__,flashinfer}.py`, `kernelscope/dashboard/lab.py`(신규), `dashboard/app.py`,
  `scripts/{check_policy_numerics,classify_divergence,probe_split_kernel,vllm_reproduce}.py`(뒤 셋 신규),
  `tests/test_{divergence,policy_numerics,classify_divergence,verify,serve_cli,serve_engine,flashinfer_plugin,dashboard_lab,dashboard_app}.py`,
  `run.sh`(신규), `docs/experiments/vllm/*.json`(신규), `docs/experiments/2026-10-02-hybrid-validation.md`, `docs/plan/2026-10-06-review-problem-definition.md`, `docs/STATUS.md`, `docs/graduation.md`, `docs/experiments/2026-09-26-hybrid-policy.md`, `PLAN.md`, `README.md`, `TODO.md`,
  `demo_data/serve_4090/{hybrid_20261002,hybrid_keepcache_20261006}/**`, `demo_data/hw_4090/{ragged,uniform}_s1_flashinfer{,_cudacore}/`, `demo_data/provenance.json`. 작성자: sunjae0224 <sunjae0224@gmail.com>.
- 환경 변경(리포 밖): 프로젝트 env에 `flashinfer-python==0.6.13`, `flashinfer-cubin==0.6.13`, `flashinfer-jit-cache==0.6.13`(cu128 index) 설치; vLLM 전용 `~/.venvs/kernelscope-vllm`(uv, vllm 0.31.0, torch 2.13 cu130).

## 1. 다음에 할 것 (우선순위 순)
1. **FlashInfer 전체 모델 비교** — `serve run --attention flashinfer`(엔진의 `_flash_attention` 자리에 FlashInfer 어댑터: `append_paged_kv_cache` + 매 step `plan()`; `plan_us`를 정책 비용 자리에 기록). 커널 격자는 끝났으니 TPOT 격차만 남았다. GPU 필요.
2. **plan() 비용 측정** — 플러그인이 `inputs["plan_us"]`에 기록하지만 summaries에는 없다. 격자별 중앙값을 노트 §8에 추가.
3. **vLLM step 프로파일링** — §9 결과(혼합 길이 step 79~81 ms가 백엔드·모드·블록 크기와 무관)의 원인 분리: vLLM 전용 venv에서 torch profiler 또는 nsys로 decode step 하나를 분해해 attention 커널 시간과 나머지를 나눈다. 본 작품 엔진의 같은 배치는 51 ms(휴리스틱)/24 ms(테이블).
4. PLAN §2 3(b) Accel-Sim L2 파티션 해시 재검증 — 그대로.
5. 보고서: 검토 문서의 제안(문제 정의 수정 4가지, 제목의 "시뮬레이션 기반 선택" 완화, 트래픽 분포 모사로 문제 빈도 추정)을 최종 보고서 1·2장에 반영. 트래픽 분포 모사는 GPU 없이 가능.
6. ~~Policy lab 탭 다듬기~~ → 2026-10-06 **Demo 페이지**(dashboard/demo_page.py, kernelscope/dashboard/{demo.py,race.html}; 설계 docs/plan/2026-10-06-design-demo-page.md) 완료: 결론·경주·왜·어떻게·검증 다섯 장면, 라이브 측정·플러그인 벤치 패널, `run.sh demo-record`, 자연어 기록 `demo_text_20261006`. 남은 것: 실행 버튼의 백그라운드 작업 세션 밖 생존, 결과 폴더 자동 새로고침.

## 2. 보고서
- 중간보고서 v4 산출물: `docs/report/midterm/중간보고서_이선재_v4.{pdf,docx}`. 소스 `docs/report/midterm/src/`(docx-js).
  재빌드: `cd docs/report/midterm/src && npm install && REPORT_FONT="Noto Sans CJK KR" REPORT_MONO="DejaVu Sans Mono" node build.js ../중간보고서_이선재_v4.docx && soffice --headless --convert-to pdf --outdir .. ../중간보고서_이선재_v4.docx`
- 최종 보고서에 넣을 근거: "균일 배치에서 이득이 없는 이유"(`serve diagnose`, attention 19.2%), 혼합 정책 실생성 결과와 선택 비용의 한계(§1 1번), 출력 검증 기준 변경 사유(분기 사건 + 동점 분류, `clear` 사건의 커널 탐침), FlashInfer 격차(미실행).

## 3. 작업 메모
- 셸이 zsh라 따옴표 없는 변수의 단어 분리가 되지 않는다. 파일 목록은 명시적으로 나열.
- GPU 측정 전 `serve doctor`. preflight는 `rerun` 외의 다른 GPU 프로세스가 하나라도 있으면 막는다. 유휴를 기다릴 때는 `nvidia-smi --query-compute-apps`를 30초 간격으로 폴링하고, 다른 팀 배치는 항목 사이에 1분 미만의 빈틈이 있으니 연속 유휴를 확인한 뒤 시작. **`make test`도 CUDA 컨텍스트를 만들어 nvidia-smi에 잡힌다** — 남의 측정 중에는 `CUDA_VISIBLE_DEVICES= make test`.
- **실행 전 GPU 없는 사전 점검**(2026-10-06 검토의 교훈): 새 캠페인 설계는 먼저 기록된 길이로 정책 선택을 재생해 정책들이 실제로 갈리는지 보고(`scratchpad/replay_recorded.py` 방식, `make_policy` + steps.parquet의 `lens`), 분류·분석 도구가 필요한 열을 기록하는지 정적으로 확인한다.
- 측정 규약: `--policy-cache fresh`(기본, 실행마다 정책 캐시 초기화) / `keep`(warm-up 뒤 유지). manifest `policy_cache`에 기록. 캠페인 manifest의 `git.status`가 비어 있어야 clean 커밋에서 잰 것이다.
- FlashInfer JIT: 시스템 `/usr/bin/nvcc`는 CUDA 10.1, `/usr/local/cuda`는 13.3이라 cu128 torch와 맞는 nvcc가 없다. `flashinfer-jit-cache`(https://flashinfer.ai/whl/cu128/)로 사전 빌드 커널을 쓴다; 새 head 차원·dtype 조합이 캐시에 없으면 다시 실패하니 `tests/test_flashinfer_plugin.py -m gpu`로 먼저 확인.
- `kernelscope/diagnose/opmodel.py`, `report.py`, `verify.py`, `serve/divergence.py`는 torch를 import하지 않는다(GPU 없는 노트북의 verify 경로). 유지할 것.
- `AttentionTimer.start(layer)/stop(layer)/total_us()`·`STEP_COLUMNS`·`attn_us` 의미는 동결.
