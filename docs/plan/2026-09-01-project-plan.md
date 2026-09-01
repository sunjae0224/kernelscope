# 졸업작품 계획: 실측(ncu) + 시뮬레이션(Accel-Sim) 이중 트랙 GPU 커널 평가 시스템

## Context

사용자의 원안(*GPU 메모리 계층 프로파일링 기반 Transformer↔Mamba 동적 스위칭 시스템*)을 검토한 뒤, 사용자가 선택한 방향은:

- **방향 C — Accel-Sim what-if** (하드웨어 시뮬레이터를 실제로 섞는다)
- **Mamba는 선택적 분석 축** (라우팅 대상 아님, 시간 남으면 매트릭스에 1행 추가)
- **자작 커널은 만들지 않는다.** 대신 *"사람들이 개발한 기존 커널을 꽂으면 평가해 주는 시스템"* 을 만든다 (사용자 제안).

이 셋을 합치면 프로젝트는 다음이 된다:

> **기존 LLM 추론 커널(FA2, FlashDecoding, SDPA, flashinfer, Mamba scan …)을 플러그인으로 등록하면, (a) 실제 A100에서 ncu 하드웨어 카운터로, (b) Accel-Sim 시뮬레이터에서 A100 config 및 변형 config(L2 크기·HBM 대역폭·SM 수)로 평가하고, 두 결과를 대조·시각화하는 평가 프레임워크.**

"동적 스위칭"은 원안의 핵심이었지만 이 방향에서는 **평가 결과에서 파생되는 부산물**(워크로드별 추천 커널 표)로 격하한다. 만드는 것의 중심은 *평가 시스템*이다.

구현 위치: `/scratch/uceeeee/TinyPIM/graduation/` (빈 폴더). git은 AIMERS처럼 TinyPIM과 **완전히 별개의 독립 repo**.

### 이 머신에서 확인된 사실

| 항목 | 결과 | 의미 |
|---|---|---|
| GPU | A100-SXM4-80GB × 4, driver **595.71** | Accel-Sim 2.0 공식 `SM80_A100` config와 1:1 대조 가능. **driver 595는 NVBit 1.8 문서상 상한(≤575.xx)을 초과** → W1 게이트 |
| `ncu`/`nsys`/`nvcc` | `/usr/local/cuda/bin/` (ncu 2025.2) | 실측 트랙 준비됨 |
| 카운터 권한 | `NVreg_RestrictProfilingToAdminUsers=0` 설정됨 | **비root로 ncu 카운터 수집 가능** |
| conda env | `/scratch/uceeeee/conda_envs/` 다수 | 새 env 1개. `conda activate`가 잘못된 python을 잡는 머신 → 절대경로 python |
| 스토리지 | `/scratch`는 NFS로 느림 | 트레이스(.tracez)·모델 가중치는 `/var/tmp` NVMe에 두고 결과 parquet만 NFS로 |

### Accel-Sim 조사 결과 (sonnet 에이전트, 출처 포함)

- **Accel-Sim 2.0.0 — 2026-08-25 릴리스 (일주일 전)**. `tested-cfgs/`에 `SM80_A100`, `SM90_H100`, `SM90_H200` 공식 config 포함. 이전 release 브랜치는 RTX3070까지였음. — github.com/accel-sim/accel-sim-framework (2.0.0 tag)
- 트레이서: NVBit **1.8** (2026-04). NVBit README 요구사항: CUDA ≥12.0, **driver ≤ 575.xx**, gcc ≥ 8.5. — github.com/NVlabs/NVBit
- 2.0은 **PyTorch/vLLM 커널을 직접 훅하는 `torch_hook` 트레이서**와 FlashAttention-3 예제를 제공. 단 일주일 된 기능 — 제3자 검증 없음.
- 속도: 2.0 기준 **~27.5K warp-instructions/s** (1.x 대비 2.2×). 트레이스는 zstd `.tracez`로 20~30× 압축, 시뮬레이터가 페이지 단위 스트리밍(메모리 ~4 GB 고정).
- Tensor core: HMMA 모델링 있음. 2.0에서 Hopper TMA/WGMMA 추가(H100 34K 커널 대비 cycle 오차 13.4%, 상관 0.99 주장). A100 `mma.sync` 경로는 1.x부터 있던 것.
- 빌드 의존성: gcc/cmake, `libssl libxml2 boost zstd`, python3. 최근 이슈: CUDA 12.2/gcc 9.4 빌드 실패(#380), `libcudart.so.12` 경로(#413), PyTorch 트레이스가 시뮬에서 조용히 실패(#360).
- FlashAttention을 Accel-Sim에서 돌린 공개 선례: 2.0의 FA3 예제 외에는 없음(Sim-FA, GPU-Tile-Sim은 별도 시뮬레이터).
- `flash_attn_with_kvcache(num_splits=…)` 지원 확인. `pip install flashinfer-python` (+`flashinfer-cubin` cu129) 가능.

---

## 0. 동기: 왜 vLLM 위에서 테스트하지 않고 이 시스템인가 (서론의 뼈대)

vLLM은 "내 커널이 이 스택·이 GPU에서 지금 얼마나 빠른가"에 최선의 답을 준다. 이 시스템은 그 질문을 대체하지 않고, vLLM이 **원리적으로 못 답하는** 두 질문을 맡는다.

| 질문 | 도구 | vLLM에서 안 되는 이유 |
|---|---|---|
| 얼마나 빠른가 | vLLM / CUDA event | (된다 — 이 시스템의 영역 아님) |
| **왜** 느린가 | ncu, 통제된 (B, L) 마이크로벤치 | 엔진 별도 프로세스 + CUDA graph + 수천 런치 속 대상 커널 → ncu 격리 어려움; 스케줄러(chunked prefill, 버킷 패딩, prefix cache)가 배칭을 결정해 워크로드 통제 불가; attention 효과가 GEMM에 희석(8B, L=32K에서 attention 트래픽 ~20%) |
| **뭘 바꾸면** 나아지나 | Accel-Sim what-if (L2·HBM BW·SM 수·세대) | 실측은 가진 하드웨어에서만 가능 — 반사실적 하드웨어는 시뮬레이터만 답함 |

서론 예시(4.3에서 실제 수치로 채움): FlashDecoding vs FA2, B=1, L=32K — vLLM: split-KV 2.3× 빠름 / ncu: FA2는 grid 32 CTA → SM 30% → DRAM 25%, 병렬성 부족 / what-if: BW×2에 split-KV −40%, FA2 −5%; SM×2 무변화 → FA2의 문제는 HW가 아니라 커널.

**정직하게 명시할 시뮬레이터 한계 (5장 한계 절):** 속도 10⁴–10⁵× 느림(decode 위주); what-if는 재튜닝 없는 동일 SASS의 민감도이므로 하한 해석; A100 모델 cycle 오차 10–20% 통상 + 2.0은 미검증 → cuBLAS 캘리브레이션과 sim vs ncu 상관표 필수; GPU 설계 공간 한정(비-GPU accelerator는 별도 모델 필요).

**선택적 대조군 (W4 버퍼):** 기존 env 중 vllm이 있는 것을 재사용해 같은 워크로드에서 vLLM TPOT 개선폭 vs 커널 단독 개선폭을 나란히 — "희석"과 "통제 불가"를 데이터로 보임.

## 1. 원안 평가 (방향 결정 전 검토 — 보고서 서론/방법에 반영할 것)

### 유지할 것
- ① 거시 지표(TTFT/TPOT) → 미시 지표(카운터)로 병목 역추적 — 문제의식 그대로 유지. ncu 권한까지 확인됨.
- ③ ncu 기반 정량 분석 "프레임워크" — 이번 방향의 실측 트랙이 정확히 이것. 스윕 자동화 + 파싱 + 시각화를 진짜로 만든다.
- ④의 대시보드 — 산출물의 얼굴. 유지.

### 고친 것과 이유
- **(P1) Llama↔Mamba 모델 간 라우팅은 방어 불가.** 다른 모델로 보내면 지연시간이 아니라 *답*이 바뀐다. 서빙 최적화는 output-preserving이어야 한다. → 스위칭은 프로젝트 중심에서 빼고, 평가 결과의 파생물("이 워크로드엔 이 커널")로만 남긴다. Mamba는 라우팅 대상이 아니라 **평가 대상 커널 중 하나**로 격하 — 이러면 "다른 아키텍처의 커널이 메모리 계층을 어떻게 다르게 쓰는가"라는 원안 ②의 질문은 그대로 살아있다.
- **(P2) 범위 초과.** 5개 서브프로젝트 → 평가 시스템 하나 + 백엔드 2개(ncu, Accel-Sim)로 재구성. 시뮬레이터가 실패해도 ncu 트랙만으로 완결되는 구조.
- **(P3) "100K+"** → 실측은 512~32K(64K) 스윕 + 128K batch 1 단일 포인트. 시뮬레이션은 아래 예산표에 따라 decode 위주.
- **(P4) Paged KV / PagedAttention 소제목** → flashinfer paged decode 커널을 플러그인에 넣으면 살아남고, 아니면 관련연구 본문 언급으로 격하.
- **(P5) ncu 방법론 함정** → ncu는 클럭 고정·커널 직렬화라 시간값이 실측 지연이 아님. **지연시간은 CUDA event, 카운터는 ncu, 사이클은 시뮬레이터** — 세 값을 config 키로 join. 3장에 명시.

---

## 2. 시스템 설계

### 2.1 한 장 요약

```
kernels/  (플러그인: 기존 커널을 감싸기만 함)
   fa2, flashdecoding(num_splits), sdpa_efficient, sdpa_cudnn, flashinfer_decode,
   cublas_gemm(캘리브레이션용), [opt] mamba_scan / mamba_state_update
        │  공통 인터페이스: build_inputs(cfg) / run() / reference() / ncu_regex / trace_filter
        ▼
workload grid  (phase ∈ {prefill, decode}) × B × L × (H_q, H_kv, d)
        │
        ├─► backends/realhw   : CUDA event 지연 + ncu 카운터 10종  ──► results/*.parquet
        └─► backends/accelsim : NVBit 트레이스(대상 커널 1회만) → SM80_A100 시뮬
                                 → 변형 config {L2 ×½/×2, HBM BW ×½/×2, SM ×½/×2, H100} ──► results/*.parquet
        ▼
analysis/  실측 roofline · 시뮬 vs 실측 상관 · what-if 민감도
dashboard/ Streamlit 1페이지
```

### 2.2 커널 플러그인 인터페이스 (`kernels/base.py`) — 자체 커널도 같은 경로로 꽂힌다

**설계 원칙: 워크로드는 논리적으로 정의하고, 레이아웃은 플러그인이 소유한다.** 하네스는 `(phase, B, L_q, L_kv, H_q, H_kv, d, dtype, causal)`만 지정하고, 그걸 dense 텐서로 만들지 paged 블록으로 만들지 자체 포맷으로 만들지는 플러그인이 정한다. 그래야 레이아웃이 다른 커널(FA2 dense vs flashinfer paged vs 누군가의 자체 포맷)이 같은 논리 워크로드에서 비교된다.

```python
@dataclass
class Workload:            # 논리 워크로드 — 레이아웃 무관
    phase: str; B: int; L_q: int; L_kv: int; H_q: int; H_kv: int; d: int; dtype; causal: bool

class KernelPlugin:                       # (1) Python에서 호출 가능한 커널
    name: str
    phases: set[str]
    def build_inputs(self, w: Workload) -> dict        # 플러그인 소유 레이아웃으로 텐서 생성 (seed 고정)
    def run(self, inputs) -> Any                       # 커널 호출 1회 (여러 launch여도 됨)
    def to_dense_output(self, out) -> Tensor           # [B, L_q, H_q, d]로 변환 → 하네스가 SDPA math와 비교
    kernel_regex: str                                  # ncu -k / NVBit 필터 공용. 다중 launch면 전부 매치하는 regex
    num_launches: int = 1                              # split-KV처럼 attn+reduce 2 launch면 2 (ncu --launch-count, 시뮬 합산)

class ExecutablePlugin:                   # (2) torch에 안 묶인 순수 CUDA 바이너리
    name: str
    def command(self, w: Workload) -> list[str]        # 예: ["./my_attn", "--B", "1", "--L", "32768", ...]
    kernel_regex: str; num_launches: int
    # 일치 검사: 바이너리가 출력 파일을 쓰면 하네스가 읽어 비교 (선택)
```

**자체 커널을 등록하는 4가지 경로 (전부 코드 수정 없이 플러그인 파일 1개):**

| 커널 형태 | 플러그인 종류 | `run()`/`command()` | 비고 |
|---|---|---|---|
| torch C++/CUDA extension (`cpp_extension.load` 또는 pip .so) | KernelPlugin | 확장 함수 호출 | 가장 흔한 형태 |
| Triton 커널 | KernelPlugin | Triton 함수 호출 | ncu/NVBit 모두 Triton이 만든 SASS를 그대로 봄. 커널명은 `<fn>_0d1d…` 패턴 → regex |
| 순수 CUDA 바이너리 (torch 없음) | ExecutablePlugin | 바이너리 실행 | ncu·NVBit 모두 프로세스 단위 도구라 언어 무관 — 시뮬 백엔드는 SASS 트레이스만 보므로 사실상 이 경로가 가장 깨끗 |
| 라이브러리 호출(cuBLAS, cuDNN, vllm `_custom_ops`) | KernelPlugin | 함수 호출 | 클로즈드소스여도 SASS 트레이스는 됨 |

**자체 커널이 만족해야 하는 조건 (문서에 명시):**
- 하네스가 준 `Workload`로 입력을 스스로 만들 수 있어야 한다 (레이아웃 자유).
- `kernel_regex`로 대상 launch를 특정할 수 있어야 한다 (이름 없는 익명 커널은 안 됨).
- 시뮬 트랙: sm_80으로 컴파일된 SASS여야 하고, Accel-Sim SM80 opcode 맵이 아는 명령만 써야 한다 (`cp.async`/`ldmatrix`/`mma.sync` 지원 여부는 별도 확인 중 — §2.4). Hopper 전용(`wgmma`, TMA)은 A100에서 애초에 안 돎.
- 시뮬 예산: 실측 트랙의 `smsp__inst_executed.sum`에서 **시뮬 소요시간 = warp-inst / 27.5K 초**로 미리 예측 → 하네스가 예산 초과 셀은 자동 스킵/경고. (실측이 시뮬의 예산 게이트 역할 — 두 트랙의 자연스러운 결합점)

- 일치 검사: `to_dense_output()` 결과와 SDPA math를 max abs diff (fp16 < 1e-2)로 비교. 등록 시 자동 실행, 실패 시 결과에 `correctness=FAIL` 태그로 남기되 측정은 진행(디버깅용).
- 하네스가 **하지 않는 것**: vLLM 통합, 오토튜닝, 정확도 벤치. 커널 자체의 성능·병목·HW 민감도 평가만.

### 2.3 실측 백엔드 (`backends/realhw/`) — **Nsight-free가 기본** (2026-09-01 결정)

이 호스트는 `dcgm-exporter`가 `DCGM_FI_PROF_*` 카운터를 상시 폴링해 ncu가 시스템 전체에서 실패한다(권한 문제 아님, root 필요). 그래서 하드웨어 카운터 없이 같은 질문에 답하도록 설계를 바꿨다. ncu는 `--ncu <path>`로 켜는 **옵션**으로만 남긴다(카운터가 되는 호스트용).

| ncu가 주던 것 | Nsight-free 대체 | 구현 |
|---|---|---|
| 커널 시간 | torch.profiler(CUPTI activity API — DCGM 락 무관) per-launch `dur`, iters 중앙값 | `run_kernel --mode profile`, `kprofile.py` |
| grid/block, occupancy | profiler 이벤트의 grid/block/regs/smem → 1st-wave occupancy 추정 (`sm_coverage`, `warps_per_sm_device` — profiler의 "warps per SM"과 일치 확인) | `kprofile.occupancy_estimate` |
| DRAM 처리량 %, tensor pipe % | **해석적 모델**: compulsory 바이트(Q/KV/O, GQA·KV 확장 반영) ÷ 커널 시간, FLOPs(4·B·H·d·attended_pairs) ÷ 시간 → 실측 천장 대비 비율 | `analytic.py`, `bench/ceilings.py` (torch 마이크로벤치: HBM copy, L2 read, fp16 GEMM) |
| 명령 수, 명령 믹스 | 트레이서 `stats_ctx` total_insts(warp 단위) + 원본 `.trace.xz` opcode 히스토그램·global bytes requested | `analysis/trace_mix.py` |
| L2 hit rate | **실측 대체 없음** → Accel-Sim이 유일한 소스. 검증은 "sim cycles vs 실측 커널 시간"(첫 셀: FA2 1.3%, split-KV 31% 오차) | 5장 한계에 명시 |

- 지연시간은 두 종류를 모두 기록: CUDA event(launch 오버헤드 포함, 사용자 체감)와 profiler 커널 시간(시뮬 사이클과 비교 대상).
- 셀당 절차: check → latency → profile(launch 탐지 겸용) → analytic 파생(achieved GB/s·TFLOPS, ceilings 있으면 util) → (옵션) ncu.
- 첫 실측 근거(decode B=1 L=1K, fp16 32/8 heads): FA2는 GQA 때문에 grid = B·H_kv = **8 CTA**(SM 7%), 49.2 µs; split-KV 64 CTA, 13.6 µs. "병렬성 부족"이 카운터 없이 launch geometry만으로 드러남.

### 2.4 시뮬레이션 백엔드 (`backends/accelsim/`)

- 트레이스: 일반 NVBit 트레이서(`tracer_tool.so`, `CUDA_INJECTION64_PATH`) + `DYNAMIC_KERNEL_RANGE="<start>[-<end>]@<regex>"` 환경변수로 **대상 커널 launch만** 트레이스, `TERMINATE_UPON_LIMIT=1`로 범위 지나면 프로세스 종료 (PyTorch가 띄우는 나머지 커널 배제 — 안 하면 트레이스 폭발). 2.0에서 `KERNEL_BEGIN/END`는 사라졌음. `torch_hook`은 커널이 아니라 PyTorch *모듈* 단위 스코핑(`hook_nvbit_to_layer("model.layers.10.self_attn")`)이라 op를 직접 호출하는 이 하네스엔 맞지 않음 — E2E 모델 트레이스가 필요해질 때만 고려.
- **opcode 커버리지 (조사 완료, v2.0.0 `gpu-simulator/ISA_Def/ampere_opcode.h`):** `LDGSTS`(cp.async)·`LDGDEPBAR`·`DEPBAR`·`LDSM`(ldmatrix)·`HMMA`·`BAR`·`SHFL`·`REDUX`·`MUFU`·`BSSY/BSYNC`·`MEMBAR` 전부 매핑됨 → FA2 계열 자체 커널 시뮬 가능. 맵에 없는 opcode는 `trace_driven.cc`에서 `assert(0 && "undefined instruction")`로 **즉시 중단** → 하네스는 stderr의 `ERROR: undefined instruction : <op>`를 잡아 셀을 `sim_status=UNSUPPORTED_OPCODE:<op>`로 기록하고 계속 진행. 관련 이력: LDGSTS는 PR #250(2023)에서 추가, #451(2025, open) HMMA가 tensor-core 비활성 config에서도 실행되는 문제 — what-if에서 `gpgpu_tensor_core_avail`은 건드리지 않음.
- 시뮬: `SM80_A100/gpgpusim.config` 기준. what-if는 config 복사본에서 옵션만 바꿈:
  - L2: `-gpgpu_cache:dl2` (set/assoc), HBM 대역폭: DRAM 클럭 도메인(`-gpgpu_clock_domains`) 또는 `-gpgpu_n_mem`, SM 수: `-gpgpu_n_clusters`, 크로스체크용 `SM90_H100`.
- 결과 파싱: `gpu_tot_sim_cycle`, `L2_total_cache_accesses/misses`, `dram_bytes` 등 → 같은 long format parquet (`backend="sim:<cfgname>"`).
- **시뮬레이션 예산(27.5K warp-inst/s 가정)** — 어떤 셀을 시뮬할지 결정하는 기준:

  | 커널/워크로드 | 대략 warp-inst | 예상 시간 |
  |---|---|---|
  | decode attention, B=1, L=4K, GQA 8 KV heads | ~1M | ~1분 |
  | decode attention, B=1, L=32K | ~10M | ~6분 |
  | decode attention, B=16, L=32K | ~150M | ~1.5시간 |
  | prefill attention, L=4K, **head 1~4개만** | ~5–20M | 3–12분 |
  | prefill attention, L=4K, head 32개 전부 | ~150M | ~1.5시간 |
  | prefill L=16K 전 head | ~2.5B | ~1일 → **제외** |
  | cuBLAS GEMM 4096², B=1 (캘리브레이션) | ~2M | ~1분 |

  → **시뮬 트랙은 decode 중심**(메모리 계층 문제가 실제로 있는 곳이고 싸다). prefill은 L ≤ 4K, head 축소로 몇 포인트만. 셀당 what-if 6개 config이므로 시뮬 총량은 decode 격자(~30셀 × 7 config × 평균 5분 ≈ 18시간, 4 GPU/CPU 병렬 가능)로 관리.

### 2.5 분석·대시보드 (`analysis/`, `dashboard/`)

- 실측: roofline 산점도(SM% vs DRAM%), (B, L) 히트맵 — 커널별 최적 영역, occupancy/L2 hit vs L 추이.
- 검증: 커널·워크로드 전체에 대해 sim cycle vs ncu `gpu__time_duration`×clock, sim L2 miss rate vs `lts__t_sector_hit_rate`, sim DRAM bytes vs `dram__bytes` — 상관계수·MAPE 표. (**이 표가 시뮬레이터 트랙의 신뢰성 근거**)
- what-if: 커널×워크로드별 민감도 막대 — "HBM BW ×2 → −41%, L2 ×2 → −3%, SM ×2 → −2%" 식으로 병목 자원 판정을 자동 텍스트로.
- 파생물: 워크로드별 추천 커널 표(실측 지연 argmin) — 원안 ④의 흔적. 선택적으로 HF 모델 attention monkeypatch로 E2E TPOT 비교(W4 버퍼).
- Streamlit 1페이지, 입력 의존성은 `results/*.parquet` 하나.

### 2.6 Mamba (선택적)

`mamba_ssm.ops.selective_scan_fn`(prefill) / `selective_state_update`(decode)를 플러그인 2개로. 설치는 prebuilt wheel 우선, 컴파일은 `/var/tmp`에서. W3 이후, 실패해도 무관.

---

## 3. 원안 대비 추가 / 삭제

### 삭제
- 제목·4.4의 "Transformer↔Mamba 모델 간 동적 스위칭" (P1) → "이중 트랙 커널 평가 시스템 + what-if"
- "하이브리드 서빙 프록시" (두 모델 상주 불필요)
- "100K+" 명시 (P3)
- 자작 커널 (사용자 결정)

### 추가
- 커널 플러그인 인터페이스와 등록 커널 목록 (3.1)
- 세 가지 측정값(CUDA event / ncu / sim cycle)의 분리 측정·join 방법론 (3.2)
- Accel-Sim 백엔드: A100 config 검증 절차와 what-if 옵션 정의 (3.3)
- 시뮬 vs 실측 상관·MAPE 표 (4.x) — 시뮬레이터 신뢰성 근거
- what-if 민감도 → 병목 자원 판정 (4.x)
- 시뮬레이션 예산표 (부록)

### 목차 수정안
```
1. 서론 — 거시 지표의 한계; 실측 카운터만으로는 "어떤 HW가 바뀌면 나아지는가"에 답 못 함 → 시뮬레이터 결합
2. 배경 — LLM 추론 커널 계보(FA2/FlashDecoding/SDPA/flashinfer/[Mamba]), GPU 메모리 계층·occupancy·roofline,
          Accel-Sim 개요(트레이스 구동, SM80_A100 config)
3. 방법 — 3.1 커널 플러그인·워크로드 격자 / 3.2 실측 지표 및 분리 측정 방법론 / 3.3 시뮬 백엔드·what-if 정의 / 3.4 시뮬 예산과 셀 선정
4. 결과 — 4.1 실측: phase·B·L에 따른 HBM 포화·occupancy·L2 추이 / 4.2 시뮬 vs 실측 검증 /
          4.3 what-if 민감도와 병목 자원 판정 / 4.4 워크로드별 추천 커널 표·대시보드 (·E2E는 있으면)
5. 결론 — 시사점(어떤 커널은 HW가 바뀌어도 안 나아진다 등), 한계(시뮬 범위, 2.0 검증 부족), 향후(Mamba/hybrid, H100)
```

---

## 4. 4주 실행 계획

**W1 — 게이트 주간.** 시뮬 트랙이 이 머신에서 되는지 첫 이틀 안에 판정.
- D1–2: 새 env(절대경로 python, torch cu12x, flash_attn, flashinfer-python). conda로 boost/zstd/libxml2 확보 → Accel-Sim 2.0 빌드 → NVBit 1.8 설치 → **vector-add 트레이스 + SM80_A100 시뮬 스모크 테스트.** driver 595에서 트레이서가 죽으면:
  - (a) 공동연구자 H100 서버의 driver 버전 확인(≤575면 트레이스만 거기서 뜨고 시뮬은 여기서),
  - (b) NVBit 구버전(1.7.x)의 driver 상한 확인,
  - (c) 둘 다 안 되면 **시뮬 백엔드 제거, ncu 단일 트랙**으로 확정(시스템 구조는 동일하므로 손실은 백엔드 1개).
- D3–5: `kernels/base.py`(KernelPlugin + ExecutablePlugin) + fa2 / flashdecoding / sdpa 플러그인 + 일치 검사, `backends/realhw` 최소 격자(decode × B∈{1,16} × L∈{1K,8K,32K}), parquet. **"남의 커널 꽂기" 경로 검증:** Triton 튜토리얼 attention(외부 코드, 수정 없이)을 KernelPlugin으로, 간단한 CUDA 샘플 바이너리를 ExecutablePlugin으로 등록해 두 경로가 모두 ncu·트레이스를 통과하는지 확인.
- **데모:** 카운터 테이블 + occupancy vs L 그래프 1장 + (게이트 통과 시) decode L=1K 커널 1개의 sim cycle vs ncu 시간 한 점 + 외부 커널 2개가 플러그인 파일 1개로 등록되는 모습.

**W2 — 실측 전체 격자 + 시뮬 검증.** (Nsight-free: roofline 축은 `analytic` achieved GB/s·TFLOPS ÷ `ceilings`; occupancy 축은 `profile`의 sm_coverage/occupancy_device)
- 실측: 6커널 × 2phase × B 4단계 × L 6단계 (`grids/decode_full.yaml`, `grids/prefill.yaml`). roofline·히트맵.
- 시뮬: decode 격자 A100 기본 config로 전부(~30셀), prefill L≤4K 몇 점. 시뮬 vs 실측 상관표 = **sim cycles vs profiler kernel time** (+ sim DRAM bytes vs analytic bytes).
- **데모:** 히트맵 + 검증 산점도(sim vs real) + `kernelscope report` 표.

**W3 — what-if + 분석.**
- 6개 변형 config × decode 격자(4 GPU/CPU 병렬 큐). 민감도 막대, 병목 자원 판정 텍스트.
- 3장·4.1–4.3 초안. (시간 남으면 Mamba 플러그인 2개 추가.)
- **데모:** "FlashDecoding B=1 L=32K는 BW×2에 −40%, L2×2에 −3%" 류 그래프.

**W4 — 대시보드·보고서·버퍼.**
- Streamlit, 추천 커널 표, 보고서, (버퍼) HF 모델 E2E TPOT 비교, (버퍼) H100 config 크로스체크.

---

## 5. 실무 메모

- repo: `graduation/`에서 `git init`; TinyPIM과 무관. 계정 설정은 AIMERS 방식(URL 한정 로컬 설정) 참고.
- gitignore: `results/`, `traces/`, `*.tracez`, 빌드 산출물, 모델 가중치.
- `/var/tmp`에 Accel-Sim 빌드·트레이스·가중치; NFS엔 코드와 parquet만.
- ncu: 기본 clock-control 유지, 지연은 별도 프로세스. 시뮬 사이클→시간 환산은 config의 core clock(A100 1410 MHz)로.
- 트레이스 시 반드시 커널 필터 사용; 필터 없이 PyTorch 프로세스 전체를 트레이스하지 말 것.
- 단일 GPU로 실측, 시뮬은 CPU 작업이므로 4개 병렬 큐(간단한 `xargs -P` 또는 python 큐).

## 6. 검증

- 플러그인: `reference()` 대비 max abs diff < 1e-2 (fp16) — 등록 시 자동.
- 실측 스윕: 격자 셀 × 커널 수 = parquet 행 수, 결측 셀 리포트.
- 시뮬: 스모크(vector-add) → 캘리브레이션(cuBLAS GEMM sim vs ncu 오차 < 20%) → 본 커널. 셀별 sim/real 비율이 0.5–2 밖이면 플래그.
- what-if: 기본 config 재실행이 원 결과와 동일한지(결정성) 1셀 확인.
- 대시보드: parquet만으로 렌더.
