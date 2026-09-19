# 설계: 대리 성능 모델(방향 1) + 출력 보존형 커널 디스패처(방향 3) — RTX 4090

작성 2026-09-19. 선행 문서: [2026-09-01-project-plan.md](2026-09-01-project-plan.md) (A100 기준 원 계획), [../setup/accelsim_4090_gate_report.md](../setup/accelsim_4090_gate_report.md) (Codex의 시뮬 트랙 이식 보고).
상태: **사용자 검토 대기**. 모든 수치는 이 4090에서 측정했고, 리뷰어(Fable)가 원자료 또는 독립 재현으로 확인한 항목은 표에 "검증"으로 표시했다.

---

## 0. 한 문단 요약

kernelscope는 지금 "커널 하나를 여러 방식으로 재는 하네스"다. 이 설계는 그 위에 두 개의 층을 올린다. **방향 1 — 대리 성능 모델**은 런치 지오메트리(grid·block·regs·smem), 커널이 옮기는 바이트, 머신 파라미터(SM 수·SM당 공유메모리·L2·DRAM 대역폭)로 커널 시간을 밀리초 안에 예측해서, 시뮬레이터로는 18시간 걸리는 what-if를 대시보드 슬라이더로 바꾼다. **방향 3 — 출력 보존형 디스패처**는 같은 attention을 계산하는 커널 변형(fa2, split-KV num_splits=N, SDPA) 중 decode step마다 하나를 고르는 최소 서빙 루프다. 두 층은 하나로 이어진다. 대리 모델의 argmin이 곧 디스패치 정책이고, 그래서 정책을 **가상의 하드웨어에서도** 계산할 수 있다. 라이브러리 휴리스틱은 이것을 못 한다.

## 1. 이 설계를 결정한 측정 사실 (2026-09-19, 이 4090에서)

출처: Codex 게이트 리포트, 소넷 에이전트 프로브 두 묶음과 리뷰어(Fable)의 독립 재현. 스크립트·CSV·로그는 `results/probes_2026-09-19/{probes,probes2,review}/`에 보존(git 무시 대상). "검증"은 원자료를 다시 열어 확인했다는 뜻이다.

| # | 사실 | 수치 | 근거 |
|---|---|---|---|
| F1 | split-KV 메인 커널은 CTA당 smem 80 KiB → Ada(100 KiB/SM)에서 **SM당 CTA 1개**. A100(164 KiB)에선 2개 | fa2 커널: 48 KiB, 255 regs → 2 CTA/SM (레지스터 한도) | `results/hw_4090_simtrack` profile 행, 검증 |
| F2 | flash-attn의 num_splits 휴리스틱은 `num_sm * 2` 슬롯(= SM당 2 CTA)을 가정하고, 시퀀스 길이로 **캐시 용량** `k_cache.size(1)`을 쓴다 | 4090에선 wave 계산이 2배 어긋남 | `flash_api.cpp:316-317`, 검증 |
| F3 | decode GQA는 q를 재배치해 grid = B × H_kv × splits | fa2 B=1 → 8 CTA / 128 SM | `flash_api.cpp:408`, 검증 |
| F4 | fa2 decode는 CTA당 순차 루프 지연에 묶인다: ≈ 4 µs + 56.5 ns/key | CTA당 ≈ 9 GB/s (스트리밍 커널의 1/3) | probe1 CSV, 검증 |
| F5 | DRAM 스트리밍 피크 958 GB/s (G ≥ 96–128에서 포화), CTA 1개는 DRAM 26–28 GB/s, L2 46–52 GB/s | L2 평탄부 ≈ 4.8 TB/s | probe2a |
| F6 | L2는 72 MiB에서 계단처럼 끊기지 않는다(LRU 아님) | 96 MiB 4.1 TB/s, 128 MiB 1.39 TB/s(역산 적중률 ≈ 39%), 512 MiB 이상 0.96 TB/s | probe2b, 해석은 리뷰어 |
| F7 | 기존 실측은 전부 warm 캐시(L2 flush 없음) → "achieved_gbps"가 DRAM 피크를 넘는다 | FD B16 L1K 1762 GB/s | 인벤토리, 검증 |
| F8 | **서빙에서 attention은 cold다.** 레이어 l의 attention 사이에 나머지 레이어 가중치 ~15 GB가 L2를 지나간다 | — | 추론(리뷰어). §2의 측정 규약을 결정 |
| F9 | 균일 배치에서 휴리스틱 손실(최적 고정 split 대비): **cold 중앙값 1.0%, 최대 15.7%** (98셀 중 1셀 > 10%); warm 중앙값 3.2%, 최대 40.4% (36셀 > 10%) | 잡음: p90/중앙값 −1 의 중앙값 0.3% | probe1, 검증 |
| F10 | fa2 고정(split 안 함)의 손실은 최대 49.7× (S2 B=1 L=32K: 1874 µs vs 37.7 µs) | — | probe1 |
| F11 | 같은 셀에서도 warm/cold 최적 변형이 다르다 | 98셀 중 42셀 | probe1 |
| F12 | SDPA: flash 백엔드는 같은 커널 계열(차이 ≤ 3%), cuDNN은 L_q=1 불가, efficient는 GQA 네이티브 불가(반복 시 3.6–3.8× 느림) | — | probe1 |
| F13 | 시뮬 트랙: sim/real FA2 B1 ≈ 1.05, FD 1.5–3.1, 원인은 캐시 상태 불일치(시뮬 cold vs 실측 warm), 런치 지연 5000 cycle, L2 파티션 해시 2배 불균형(`% 48`) | SM×2가 사이클을 늘리는 비물리적 결과 | Codex 리포트, 재현 확인 |
| F14 | **래깅 배치에서 휴리스틱 손실이 크다** (cold, S1): 긴 32K 2개 + 짧은 1K 30개(B=32) | 휴리스틱 3744 µs vs 최적 고정 split 500 µs → **7.49×**; 균일 대조군 16개는 전부 1.00–1.05× | probe3b(에이전트, profiler 7.23×), **리뷰어 독립 재현**(CUDA event) |
| F15 | 기전 ①: `B·H_kv ≥ 0.8 · 2 · n_sm`이면 휴리스틱이 즉시 split=1을 반환 → 긴 시퀀스의 CTA 16개가 256블록을 혼자 순차 처리. 문턱은 S1(H_kv 8)에서 B ≥ 26, S2(H_kv 4)에서 B ≥ 52 — 연속 배칭의 평범한 배치 크기다 | 같은 B의 균일 긴 배치는 1.01× (작업이 균등하면 문제없음) | `flash_api.cpp:264-265`, probe3c, 검증 |
| F16 | 기전 ②: 문턱 아래에선 휴리스틱이 "모든 시퀀스가 용량 길이"라고 보고 split을 적게 고른다 | 1×32K + 15×512: 휴리스틱 497 µs(split 2) vs split 16 251 µs → **1.98×** | probe3b, 리뷰어 재현 |
| F17 | 기전 ③: 균일 배치라도 캐시 용량 ≫ 실제 길이면 split이 부풀어 빈 CTA가 생긴다. **캐시 view를 `[:, :L]`로 자르면 공짜로 해결**(출력 비트 동일) | B=1 L=1K, 용량 32K: 2.28× | probe3a |
| F18 | 배치 전체에 **정수 하나(num_splits)만 바꿔도** 길이별 그룹 분할과 같은 효과 | 492 vs 492 µs. 임의 순서 재배치는 `cache_batch_idx`로(≈ +4%), 전체 용량 `index_select`는 오히려 10 ms | probe3b/3c |
| F19 | 래깅 배치의 긴 CTA는 혼자일 때보다 2배 느리다 → 같은 SM에 상주한 CTA끼리 처리량을 나눈다. 블록 번호가 인접한 긴 시퀀스 CTA가 같은 SM에 놓였을 가능성 | 3744 µs vs 단독 1861 µs | 리뷰어 재현. 블록→SM 배치는 `%smid` 프로브로 확인 예정(P0-9) |
| F20 | MPS는 이 GeForce에서 데몬이 뜨지 않음(exit 1, 로그 0바이트). 대신 **SM 블로커**(99 KiB smem, 32스레드, clock 스핀, 전역 메모리 접근 없음)가 SM을 1개씩 점유 | fp16 GEMM: 32/64/96 SM 차단 시 1.25/1.83/3.47× (이상치 1.33/2/4); DRAM 스트리밍: 변화 없음(128 CTA가 남은 32 SM에 함께 상주해 대역폭 유지) | probe4 |

**F8 + F9는 방향 3의 원래 명분을 약화시켰지만, F14–F16이 더 강한 명분을 준다.** 균일 배치에서는 cold 기준으로 휴리스틱이 이미 거의 최적이다. 그러나 긴 요청과 짧은 요청이 섞인 배치에서는 휴리스틱이 GPU 구조와 무관한 가정(모든 시퀀스가 같은 길이, SM당 2 CTA) 때문에 최대 7.5배 느린 선택을 한다. 이것이 §5.1의 논증이다.

## 2. 공통 기반 (Phase 0) — 실측 트랙 수정

모두 Fable/Claude 소유 영역(`backends/realhw`, `analytic.py`, `analysis/`, `bench/`, `plugins/`, `workload.py`). Codex 소유 영역(`backends/accelsim`, simsweep CLI 블록)은 건드리지 않는다.

| ID | 변경 | 이유 |
|---|---|---|
| P0-1 | 작업 브랜치 `design-1-3`을 `sim-track-4090`에서 git worktree로 분기 | Codex 체크아웃을 흔들지 않고 store.py 복원·4090 기본값을 이어받음 |
| P0-2 | `kprofile.props_from_torch`: `max_blocks_per_sm`를 compute capability 표에서(8.0→32, 8.6→16, 8.9→24, 9.0→32), 레지스터 할당 단위(warp당 256) 반영 | A100 값 32 하드코딩 버그 |
| P0-3 | **cache-state 모드**: `run_kernel --cache-state {warm,cold}`. cold는 매 반복 전 `4 × L2_cache_size` 버퍼를 전용 Triton 커널 `kernelscope_l2_flush`로 쓰고(L2가 LRU가 아니라 2배로는 이전 데이터가 일부 남을 수 있음, F6), 그 커널은 이름으로 합산에서 제외. warm 모드는 반복 사이에 표식 커널 `kernelscope_iter_marker`를 둔다. `cache_state`는 store `extra` 열로 기록, workload key는 불변 | F7, F8. 서빙 규약(cold)과 시뮬 비교(cold)가 같은 기준을 갖는다 |
| P0-4 | profiler 이벤트 유실 대응: 프로파일 블록 앞에 버림 호출을 두고, flush/표식 커널을 반복 경계로 삼아 이벤트를 반복 단위로 자른 뒤 런치 수가 어긋난 반복은 버리고 뒤쪽 창만 분석 | probe가 발견한 torch.profiler 첫 이벤트 누락 |
| P0-5 | `analytic`: cold 행만 `dram_util`을 낸다. warm 행의 `achieved_gbps`는 이름을 유지하되(roofline·report 소비자 호환) L2 적중분을 포함한 유효 대역폭임을 `cache_state` 열로 구분한다 | F7의 불가능한 수치 제거 |
| P0-6 | `report`: KEY에 `cache_state` 추가, sim 사이클→µs 클럭을 행의 `arch`로 결정(SM80_A100 1410, SM89_RTX4090 2520), `--clock-mhz`는 명시적 덮어쓰기만 | 1410 기본값 버그 |
| P0-7 | `Workload` 확장: (a) 래깅 길이 `kv_lens`(선택)를 key에 `Lkv32768x2+1024x30`처럼 압축 표기, 균일 key는 그대로; (b) 밑줄 있는 dtype(`float8_e4m3fn`) 허용. `analytic`·`reference`·`check`·flash 플러그인이 시퀀스별 길이(`cache_seqlens`)를 따르게 수정 | F14–F16의 래깅 워크로드를 하네스의 1급 시민으로. Codex 시뮬 트랙은 균일 key만 쓰므로 영향 없음 |
| P0-8 | `plugins/builtin/flash.py`에 `fd_s{N}` 고정 split 계열(N ∈ {2,4,8,16,32,64,128})을 생성하고, 각 변형의 **paged 판**(`block_table`, page 256)을 함께 등록. 기존 `fa2`(=N 1), `flashdecoding`(=휴리스틱) 유지. dense 캐시는 view를 실제 최대 길이로 자른다(F17) | 디스패치 표·모델 학습의 측정 격자. 서빙 루프는 24 GB에 래깅 배치를 담으려면 paged여야 하고, paged 경로는 `force_split_kernel`로 항상 split 커널을 써서 dense와 시간이 다르다 |
| P0-9 | `kernelscope ceilings` v2 → `machines/rtx4090.json` (§3.1의 MachineSpec). 측정 항목: DRAM 스트리밍 피크, CTA당 DRAM/L2 스트리밍 속도(G=1..8), L2 평탄 대역폭(반복마다 읽는 청크를 회전시켜 컴파일러가 load를 반복 루프 밖으로 끌어올리지 못하게 함 — 프로브의 1–32 MiB 이상치는 이 끌어올림의 산물로 판단), L2 적중률 곡선 h(W)(W = 16..512 MiB 순환 스윕), fp16 GEMM 피크, 디바이스 속성, **블록→SM 배치 규칙**(`%smid` 기록 커널로 인접 블록이 같은 SM에 가는지), **SM 블로커**(F20, `kernelscope/bench/sm_blocker`, CUDA 확장은 accelsim-build의 nvcc 12.9로 빌드) | F5, F6, F19, F20 |
| P0-10 | 측정 격자 → `results/hw_4090/`: (a) 균일 `grids/dispatch_s{1,2}.yaml`: decode, B ∈ {1,2,4,8,16,32,64}, L ∈ {512..32768}; (b) 래깅 `grids/ragged_s{1,2}.yaml`: B ∈ {16,24,26,32,48,64} × 긴 요청 수 {1,2,4} × L_long ∈ {8K,16K,32K} × L_short ∈ {512,1K,2K} (문턱 B=26 앞뒤를 포함); cold 기본 + warm 참고; dense·paged 둘 다; 플러그인 = fa2, flashdecoding, fd_s2..fd_s128, sdpa_flash(균일만) | 대리 모델 학습·검증, 디스패치 표, Codex 시뮬 검증 기준값 |
| P0-11 | **프로세스 내 배치 실행기** `kernelscope bench`: 플러그인 하나의 모든 (workload, cache_state)를 한 프로세스에서 측정하고 `sweep`과 같은 행 빌더·스키마로 기록, 셀별 예외 격리, `summaries.jsonl` 기반 재개. `sweep`(서브프로세스)은 ncu·실행 파일·격리가 필요할 때만 | 기존 `sweep`은 셀마다 서브프로세스 3개(셀당 약 10초)라 P0-10 격자(수천 셀)가 수 시간. 프로브는 같은 규모를 한 프로세스에서 2분에 끝냄 |
| P0-12 | `analysis/dispatch.py` + `kernelscope dispatch-table`: 측정 행에서 (workload, cache_state, 계열 dense/paged)별 최적 변형·휴리스틱 손실 표 | 프로브 수치 재현 확인(합격 기준), 계획 3의 디스패치 표 원천 |

Codex에게 넘길 조정 사항 두 가지(Codex 영역이라 요청만 한다): (a) 시뮬 비교 기준을 cold 실측으로 바꾸기, (b) profiler 커널 시간과 비교할 때 `-gpgpu_kernel_launch_latency 0` 규약 채택과 `IPOLY_MODULO`(mode 6) 인덱싱 검증.

## 3. 방향 1 — 대리 성능 모델 (`kernelscope/model/`)

### 3.1 입력

- **MachineSpec** (`machines/*.json`, 측정값 + 디바이스 속성): `n_sm, max_threads_sm, max_ctas_sm, regs_sm, smem_sm, l2_bytes, dram_gbps, l2_gbps, cta_dram_gbps, cta_l2_gbps, l2_hit_curve, tc_tflops, clock_mhz`. 가상 머신은 `spec.scaled(sm=0.5, dram=2, l2=2, smem=164/100)`처럼 필드 배율로 만든다.
- **지오메트리 제공자** (`model/geometry.py`): 워크로드 → 런치 목록. 각 런치 = (grid, block, regs, smem, CTA별 작업량 배열). 내장 flash 계열은 소스에서 확인한 규칙(F2, F3)을 파이썬으로 옮긴다: `grid = B × H_kv × splits`, 블록당 128 key, split 범위는 용량 기준, 휴리스틱 `num_splits_heuristic`도 그대로 포팅해서 **가상 머신에서 휴리스틱이 무엇을 고를지** 계산한다. regs·smem은 측정값(`results/hw_4090` profile 행)에서 가져온다. 제공자가 없는 외부 플러그인은 같은 머신 안에서 가장 가까운 측정 셀로 보간만 하고, what-if는 "지원 안 함"으로 표시한다.
- **바이트**: CTA별로 자기가 읽는 K/V 범위 × d × dtype, 쓰기는 출력(split이면 fp32 부분합 + LSE, combine이 다시 읽음). 기존 `analytic.attention_traffic`은 총량 검산에만 쓴다.

### 3.2 시간 모델 (런치 하나)

1. **상주 CTA 수**: `c = min(max_ctas_sm, ⌊threads_sm / T⌋, ⌊regs_sm / regs_alloc(T, R)⌋, ⌊smem_sm / S⌋)` — P0-2로 고친 점유율 함수를 그대로 쓴다.
2. **캐시 수준별 바이트 분할**: cold는 전부 DRAM. warm은 작업 집합 W에 대해 적중률 `h(W)`(측정 곡선, F6)로 L2/DRAM을 나눈다.
3. **CTA 하나의 작업 시간**(SM을 혼자 쓸 때): `t_cta = t0 + bytes_cta · (h / r_l2 + (1−h) / r_dram)`. 여기서 `r_l2, r_dram`은 **커널별 CTA당 처리율**(보정 파라미터, F4: fa2는 ≈ 9 GB/s로 스트리밍 커널의 1/3), `t0`는 CTA 고정 비용.
4. **병렬성 한계 시간** `t_par`: 이벤트 구동 스케줄러의 makespan. SM마다 슬롯 `c`개, CTA는 측정된 블록→SM 배치 규칙(P0-9)대로 블록 번호 순서로 배정한다. **같은 SM에 상주한 CTA는 SM 처리량을 나눠 쓴다**(프로세서 공유: 상주 k개면 각자 속도 × s(k), s(1)=1, s(2)는 보정 — F19에서 긴 CTA가 2배 느려진 현상). 래깅 작업량(F14)이 이 스케줄러에서 자연스럽게 긴 꼬리를 만든다. 수천 CTA도 numpy로 수 ms.
5. **대역폭 한계 시간** `t_bw = Σ bytes · (h / l2_gbps + (1−h) / dram_gbps)`.
6. **런치 시간** `t = t_launch0 + smoothmax(t_par, t_bw)` (p-노름, p≈4, 꺾임 대신 완만한 전이).
7. **커널 시간** = 런치 시간의 합(profiler 커널 시간과 같은 정의). 사용자 체감 지연 = 커널 시간 + 플러그인별 `event − kernel` 오프셋(측정 보정).

출력은 시간과 함께 **병목 분해**를 낸다: 어느 항이 이겼는지(병렬성/대역폭), 상주 한도가 무엇에 묶였는지(regs/smem/threads/CTA), wave 수, 빈 CTA 수. 대시보드의 "왜"는 이 분해에서 나온다.

보정 파라미터는 커널 종류별 5개(`r_dram, r_l2, t0, t_launch0, s(2)`), 로그 시간 최소제곱으로 적합한다. 머신 파라미터는 전부 측정값이라 적합하지 않는다.

### 3.3 검증 사다리 (모델 신뢰성의 근거)

| 단계 | 방법 | 합격 기준 |
|---|---|---|
| V1 보간 | 셀 일부로 적합, 나머지로 평가 (L 보류, B 보류 두 방식) | cold 기준 중앙 절대오차 ≤ 15%, p90 ≤ 30% |
| V2 정책 | 셀마다 모델 argmin num_splits를 실측 시간으로 평가 — 균일 격자와 **래깅 격자 모두** | 정책 손실 중앙 ≤ 5%, 최대 ≤ 15% (cold) — **이 기준이 MAPE보다 중요**: 디스패처가 쓰는 건 순위다 |
| V3 실HW what-if (SM 축) | SM 블로커(F20)로 32/64/96 SM을 차단하고 attention 변형을 실측. MPS는 이 GPU에서 불가 | 차단 수별 중앙 오차 ≤ 20%. 블로커 SM에 작은 커널(combine, smem ≤ 1 KiB)이 끼어들 수 있으므로 profiler로 겹침과 SM 배치를 확인 |
| V4 시뮬 what-if (BW·L2·smem 축) | Codex 시뮬 변형과 **민감도 방향·판정 일치**만 본다(절대 시간은 시뮬 미보정) | 보고만, 합격 관문 아님 |
| V5 교차 형상 | S1으로만 적합하고 S2(H_kv=4)를 예측 | V1 기준의 1.5배 이내 |
| V6 래깅 외삽 | 균일 격자로만 적합하고 래깅 격자를 예측 | V1 기준의 1.5배 이내. 통과하면 "표에 없는 길이 분포에서도 정책을 계산할 수 있다"는 §5.1 주장의 근거 |

DRAM 대역폭 축은 실HW로 바꿀 수 없다(메모리 클럭 고정은 root 필요). 사용자가 sudo로 `nvidia-smi --lock-memory-clocks`를 허용하면 V3과 같은 방식으로 추가 검증하고, 아니면 시뮬(V4)로만 제시한다.

## 4. 대시보드 (`dashboard/app.py`, Streamlit)

입력은 `results/**/*.parquet`, `machines/*.json`, 서빙 로그뿐. 대리 모델은 프로세스 안에서 호출한다(슬라이더 반응 < 100 ms 목표).

1. **진단**: 커널 × 워크로드 선택 → 실측 시간(cold/warm), 런치 지오메트리, 상주 한도의 원인, 모델 병목 분해.
2. **What-if**: SM 수 · DRAM 대역폭 · L2 · SM당 smem 슬라이더 → 모든 num_splits 변형의 예측 곡선과 최적점이 즉시 이동. 시뮬 점(Codex)과 실HW SM 제한 점(V3)을 겹쳐 신뢰 구간을 보여 준다. 예: "smem을 A100 수준(164 KiB)으로 올리면 split 커널이 SM당 2 CTA → 최적 split이 바뀐다".
3. **커널 맵**: (B, L) 히트맵에 최적 변형과 휴리스틱 손실, cold/warm 토글, 실측/모델 토글.
4. **라이브 서빙**: §5의 서빙 루프를 백그라운드 프로세스로 돌리고, 스텝별 선택 커널·attention 시간·요청별 TPOT를 정책별로 비교.

## 5. 방향 3 — 출력 보존형 디스패처 (`kernelscope/serve/`)

### 5.1 가치 논증 — 커널 수준의 head-of-line blocking

**문제.** 추론 모델 서빙에서는 수만 토큰짜리 추론 요청과 짧은 채팅 요청이 한 배치에서 함께 decode된다. flash-attn의 num_splits 휴리스틱은 배치 안의 모든 시퀀스가 같은 길이라고 가정하고, SM당 2 CTA가 상주한다고 가정한다. 배치가 크면(S1에서 B ≥ 26) "GPU가 이미 찼다"고 판단해 split을 끈다(F15). 그러면 긴 요청의 CTA 소수가 전체 KV를 순차로 읽는 동안 나머지 SM은 놀고, **배치의 모든 요청이 그 긴 요청 하나를 기다린다.** 측정된 attention 손실은 최대 7.5배다(F14).

**해법.** 디스패처는 스텝마다 실제 길이 분포를 보고 num_splits 정수 하나를 고른다(F18: 그룹 분할 불필요). 커널은 그대로이고 출력은 반올림 수준에서 같다(최대 절대차 2.4e-4). 선택 근거는 두 가지다. 측정 표(`table`)와 대리 모델의 예측(`model`)이다. 대리 모델은 래깅 작업량을 CTA 단위로 스케줄링하기 때문에(§3.2), 표에 없는 길이 분포와 가상의 하드웨어에서도 split을 고를 수 있다. 이것이 방향 1과 3을 잇는 고리다.

**E2E 효과 (추정, 측정 예정).** 8B급 모델, B=32(긴 32K 2개 + 1K 30개): attention이 레이어당 3.7 ms에서 0.5 ms로 줄면, 32 레이어 기준 스텝당 약 120 ms가 약 16 ms가 된다. 가중치 스트리밍(약 17 ms)을 더하면 **배치 전원의 TPOT가 약 137 ms에서 약 33 ms로** 줄어드는 규모다. 실제 값은 §5.2의 서빙 루프로 측정한다.

**정직한 위치 설정 (관련 연구).** 래깅 decode의 부하 불균형은 알려진 문제이고, FlashInfer는 커널 라이브러리 안의 plan 단계에서 split을 부하 균형 있게 계획해 이를 다룬다. 이 작품의 기여는 새 커널이 아니다. (a) 기존 FA2 휴리스틱이 Ada에서 왜 실패하는지를 하드웨어 수준에서 설명하고(F1, F2, F15, F19), (b) 커널을 바꾸지 않고 파라미터 선택만으로 격차를 회복하며, (c) 그 선택을 가상 하드웨어로 일반화하는 것이다. FlashInfer는 설치가 되면(`flashinfer-python`, glibc 2.31 호환성 확인 필요) 상한 기준선 플러그인으로 넣고, 반나절 안에 안 되면 정성적 비교로 대체한다.

**남는 한계.** 균일 배치에서는 이득이 거의 없다(F9). 문턱 부근의 이득은 B·H_kv에 민감하다. 따라서 평가는 길이 분포를 변수로 둔 워크로드에서 해야 하고, 균일 배치 결과도 "휴리스틱과 동등"으로 함께 보고한다.

### 5.2 구성

- **모델 로딩**: safetensors 직접 로드(bf16 → 연산 dtype), 토크나이저는 `tokenizers`. gradkernel에 `safetensors tokenizers`를 **추가 설치**(공유 env 규칙: 추가만). `transformers`는 테스트 오라클 전용으로 추가 설치.
- **데모 모델**: DeepSeek-R1-Distill-Llama-8B (로컬 캐시, H_q 32 / H_kv 8 / d 128 = 측정 격자 S1과 동일, 추론 모델). KV 여유 ≈ 57K 토큰이라 "긴 16K 1개 + 짧은 1K 31개"(≈ 48K 토큰, B=32로 문턱 위) 규모까지. 보조: Qwen3-4B-Instruct-2507 (헤드 구성 S1과 동일, 36 레이어, KV 여유 ≈ 105K 토큰) — "긴 32K 2개 + 1K 30개" 같은 큰 래깅 시나리오용.
- **decode 루프**: RMSNorm, RoPE(`flash_attn_with_kvcache`의 `rotary_cos/sin` 인커널 적용), GQA attention, SwiGLU. KV 캐시는 **paged**(`block_table`, page 256, 공용 블록 풀) — dense는 모든 시퀀스에 최대 길이 용량을 줘야 해서 래깅 배치가 24 GB에 들어가지 않는다. 새 k/v는 `k/v` 인자로 제자리 추가. CUDA graph 없음(eager) — 정책 간 비교는 같은 조건이라 공정하다.
- **prefill**: 긴 요청은 긴 문맥을 prefill해서 만든다(라이브로 32K 토큰을 생성하면 수십 분). prefill도 paged 캐시에 `flash_attn_with_kvcache`로 기록한다. 측정 대상은 decode 스텝뿐이다.
- **연속 배칭 워크로드 생성기**: 도착 과정과 요청 길이 분포(긴 추론·긴 문맥 요청 비율, 짧은 채팅 요청)로 스텝마다 배치 구성이 바뀌는 트레이스를 만든다. 같은 트레이스를 모든 정책에 재생한다. 요청 순서는 임의이므로 재배치가 필요하면 `cache_batch_idx`를 쓴다(F18).
- **정책**: `fa2`(고정), `heuristic`(num_splits=0 — **정직한 기준선**), `table`(측정 표 조회), `model`(대리 모델 argmin), `oracle`(오프라인 스텝별 최적, 평가 전용). 정책 입력 = 스텝의 (B, 시퀀스별 길이 배열, 용량).
- **로그** (`serve/*.parquet`, 스텝 단위): run_id, policy, step, B, 길이 통계, 선택 변형, attention 시간(레이어별 CUDA event 합), 스텝 시간, 요청별 토큰 타임스탬프.

### 5.3 "출력 보존"의 정확한 의미

split 수가 바뀌면 합산 순서가 바뀌어 로짓이 반올림 수준에서 달라진다. 긴 greedy 생성에서는 드물게 토큰이 갈라질 수 있다(서빙 엔진의 batch-invariance 문제와 같은 현상). 주장은 "수학적으로 동일, 수치적으로 반올림 이내"로 한정하고, 정책 간 스텝별 로짓 최대 절대차(≤ 1e-2)와 greedy 토큰 일치율을 **측정해서** 보고한다.

## 6. 단계와 일정 (가정: 시연까지 약 8주)

| 주 | 산출물 | 데모 가능 상태 |
|---|---|---|
| 1 | P0-1..P0-9, 4090 ceilings v2 | 진단 탭(실측만) |
| 2 | P0-10 측정 격자 전체(S1, S2, cold+warm) | 커널 맵(실측) |
| 3–4 | 대리 모델 + V1·V2·V5, (가능하면) V3 | What-if 탭 |
| 5–6 | 서빙 루프 + 정책 + 워크로드 생성기 + 로그 | 라이브 서빙 탭 |
| 7 | 대시보드 통합, V4(Codex 시뮬과 대조) | 전체 |
| 8 | 보고서·시연 리허설·버퍼 | — |

## 7. 테스트

- 모델: 합성 지오메트리로 각 영역(병렬성 한계, 대역폭 한계, smem 상주 한도)을 맞히는 CPU 단위 테스트, 휴리스틱 포팅이 소스와 같은 split을 내는지 C++ 결과(측정 grid.y)와 대조.
- 정책: 표 조회·argmin의 결정성, 경계 버킷.
- 서빙: 짧은 프롬프트에서 `transformers` 참조 구현 대비 로짓 일치(GPU 테스트), 정책 간 로짓 차.
- 실측 트랙: cold 모드에서 flush 커널이 합산에서 빠지는지, 이벤트 유실 대응, 래깅 workload key 왕복(`from_key(key())`)과 시퀀스별 길이를 따르는 reference 검사.

## 8. 위험

| 위험 | 대응 |
|---|---|
| paged 경로에서는 래깅 손실이 dense와 다름(항상 split 커널) | P0-10에서 paged로 다시 측정하고, §5.1의 수치는 paged 결과로 갱신. 손실이 작아지면 서빙 루프의 이득 주장도 그 수치로 낮춘다 |
| E2E에서 attention 비중이 추정보다 작음 | 스텝 시간 분해(attention vs 나머지)를 로그에 남겨 정직하게 보고. 이득은 긴 요청이 섞인 스텝에 집중됨을 보여 준다 |
| FlashInfer가 glibc 2.31에서 설치 안 됨 | 반나절 제한. 안 되면 관련 연구로 정성 비교 |
| SM 블로커에 작은 커널이 끼어듦 | combine 커널의 SM 배치를 `%smid`로 확인, 필요하면 블로커의 스레드 수를 늘려 상주 여지를 없앤다 |
| 커널별 4개 파라미터로 split 곡선의 U자 모양을 못 맞힘 | 파라미터에 wave 꼬리 효과(마지막 wave 부분 점유) 항 추가, 그래도 안 되면 V2 기준만 유지 |
| 8B 모델 + 긴 KV가 24 GB에 빠듯 | Qwen3-4B 보조 모델로 긴 래깅 시나리오 수행 |
| Codex 시뮬이 계속 미보정 | V4는 원래 관문이 아님. 대시보드에 시뮬 점은 "미보정" 배지로 표시 |
