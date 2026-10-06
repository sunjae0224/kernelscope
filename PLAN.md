# 졸업작품 계획 v3 — GPU 프로파일링·시뮬레이션 기반 LLM 추론 attention 커널 동적 선택 시스템

> 2026-09-26 v3. **이 파일이 계획의 단일 원본이다** — 모든 호스트(로컬 WSL2, 연구실 4090)는 이 파일을 기준으로 삼고, 계획이 바뀌면 여기를 고쳐 커밋·푸시한다.
> v2(2026-07-23, GPGPU-Sim 축소 attention–SSM 비교)는 로컬 워크스페이스 `gpu_graduation/docs/plan-v2-gpgpusim-2026-07-23.md`(리포 밖)에 보관. 9월 1일부터 실작업은 이 리포(github.com/sunjae0224/kernelscope)이며, 이 문서는 그 방향을 한 장으로 고정한다.
> 원칙은 그대로다: **연구급 아님, 코어 스코프만으로 완성.** 중간보고서(2026-09-27 제출)는 로컬 워크스페이스 `gpu_graduation/midterm_report/`(리포 밖).

## 0. 한 줄 정의

LLM 서빙에서 길이가 다른 요청이 한 배치에 섞이면 FlashAttention의 KV 분할 수(num_splits) 휴리스틱이 GPU를 채우지 못한다. 이 문제를 **실측(프로파일링)으로 진단하고, 시뮬레이터·성능 모델로 설명·예측하며, 매 디코드 단계 분할 수를 동적으로 골라 실제 LLM 생성에서 검증하는 시스템**을 만든다. 정체성은 서빙 최적화 시스템이다. 프로파일러와 시뮬레이터는 제품이 아니라 이 파이프라인의 단계다.

원 신청서(Transformer↔Mamba 동적 스위칭 + 대시보드)와의 관계: 프로파일링 → 동적 전환 → 대시보드의 골격은 유지하고, 전환 대상을 "다른 모델"(답이 바뀜)에서 "같은 모델 안의 수학적으로 동등한 커널 설정"(답이 유지됨)으로 구체화했다.

## 1. 지금까지 확보한 것 (2026-10-02)

| 단계 | 상태 | 근거 |
|---|---|---|
| ① 커널 실측 | 완료 | RTX 4090, 6,801조건, 균일 길이 손실 중앙값 0.72% vs 혼합 길이 최악 12.53배 |
| ② 원인 설명·예측 | 진행 중 | CTA 작업량 분석(부하 불균형), 성능 모델(시간 오차 2.3~12.3%), Accel-Sim 4090(미보정, 1.05~3.06배), GPGPU-Sim 축소 커널로 기전 정성 재현(혼합 7.23배 vs 균일 2.25배); step 연산 분해(`serve diagnose`, 2026-09-27): 혼합 길이 attention 비중 70.9%→37.5%(휴리스틱→테이블), 나머지 GEMM 클래스는 DRAM 상한의 63~90%(o_proj·mlp·lm_head는 memory_bound, qkv_proj 63%는 임계 0.7 미만)로 attention 외에는 커널 선택의 여지가 작음 |
| ③ 선택 정책 | 완료 + 실생성 검증 | 휴리스틱/고정/테이블/모델 + **혼합 정책(δ=0.2)**: cold 검증 집합 최악 손실 1.97 / 7.54 / 10.90% (기준 15% 통과, GPU 없이 leave-one-out 평가); 실생성(2026-10-02): 요청 3조건은 측정 셀과 겹쳐 혼합 = 테이블 선택(0/567 step), held-out 요청 도착에서만 12/55 step이 모델 순위를 따라 attention 3.31→3.00 ms/step로 줄었으나 실행마다 정책 캐시를 비우는 규약의 첫 결정 비용(1.56 ms/step) 때문에 TPOT 1.020배(테이블 1.085배); 정책 캐시를 유지하는 규약(2026-10-06, `--policy-cache keep`)에서는 혼합 1.090배 ≈ 모델 1.091배 > 테이블 1.086배로 더 나은 커널 선택이 TPOT에도 드러남 |
| ④ 실제 LLM 검증 | 완료(2차) | Qwen3-4B 혼합 길이 TPOT 1.81배, 토큰 일치; 후속 108회, 두 모델 1.28배/1.18배; 혼합 정책 3조건×3회(2026-10-02): 혼합 길이 1.793배(테이블 1.807배), 요청 도착 1.146배, 균일 0.993배, 토큰 전부 일치, 9월 22일 수치 1.25% 이내 재현 |
| ⑤ 출력 보존 검증 | 지표 교체 완료 | 지표 = 분기 사건 수 + 동점 분류(`scripts/classify_divergence.py`, 1위 로짓 실측). 2026-10-02 캠페인: 요청 3조건 0건, held-out 요청 도착 정책별 2건 모두 tie_1ulp, 대조군 균일×fixed:8 6건 중 5건 tie_1ulp·1건 clear — 커널 탐침(같은 KV 상태, 36층, float32 참조)에서 분할 8은 분할 1과 bf16 간격 1개 안에서 일치하고 argmax 유지 → 구현 오류 아님, 각 실행이 쌓은 KV 반올림 차이의 증폭. "불일치는 로짓 동점에서만"은 더 이상 주장하지 않음 |
| ⑥ 재현·시각화 | 완료 | `kernelscope verify` 86개 항목 GPU 없이 재계산, 대시보드 4화면 |

## 2. 남은 작업 (우선순위 순)

1. ~~**혼합 정책의 실제 LLM 검증**~~ — 2026-10-02 완료, 2026-10-06 정책 캐시 유지 규약 재측정과 엔진 guard 추가까지 완료: [docs/experiments/2026-10-02-hybrid-validation.md](docs/experiments/2026-10-02-hybrid-validation.md) §3·§3b.
2. ~~**불일치 지표 교체**~~ — 2026-10-02 완료: `scripts/check_policy_numerics.py`(1위 로짓 기록) + `scripts/classify_divergence.py`. `clear`는 조사 대상이며 구현 오류 여부는 같은 KV 상태의 커널 탐침(`scripts/probe_split_kernel.py`)으로 판정한다(대조군 1건은 통과).
3. **시뮬레이터** — (a) 로컬 GPGPU-Sim 4.2(빌드·실행 확인, [experiments/gpgpusim/](experiments/gpgpusim/))로 직접 작성한 split-KV 커널(`splitkv_attn.cu`, 참조 대비 오차 2e-8)의 부하 불균형 기전을 정성 재현 — 스윕 결과([docs/experiments/gpgpusim-splitkv-sweep.csv](docs/experiments/gpgpusim-splitkv-sweep.csv)): 혼합 길이 배치는 S=1→16에서 7.23배, 균일 배치는 S=8에서 2.25배 후 포화 — 실측과 같은 방향의 정성 재현 완료; (b) 연구실 호스트에서 Accel-Sim L2 파티션 해시 설정 재검증.
4. **FlashInfer 비교** — 커널 격자 부분 완료(2026-10-06, [노트 §8](docs/experiments/2026-10-02-hybrid-validation.md)): FlashInfer CUDA-core 변형은 혼합 길이 162셀에서 최선 FA2 분할과 같은 수준(중앙값 0.977배, 최대 1.003배), tensor-core 변형은 최악 셀에서 2.401배 느림 → 외부 분할 수 선택이 커널 교체 이득을 거의 전부 얻는다. 남은 것: `serve run --attention flashinfer` 전체 모델 비교, plan() 비용 측정. [계획](docs/plan/2026-09-26-plan-flashinfer-comparison.md).
5. **최종 보고서·발표·데모** — 12월. 대시보드에 재현 검증 화면 추가(2026-10-06: 05 Policy lab 탭 초안 — 규약·백엔드·분기 사건 비교와 측정 명령 조립·실행). 2026-10-06 Demo 페이지 추가(기존 5탭은 Lab 페이지): 기록 재생 토큰 레이스 + 라이브 측정 패널, [설계](docs/plan/2026-10-06-design-demo-page.md).

7. **vLLM 재현(중간보고서 표 8)** — 별도 환경 `~/.venvs/kernelscope-vllm`(vllm 0.31, torch 2.13 cu130)과 `scripts/vllm_reproduce.py`(두 생성 길이의 차로 step당 decode 시간 추정, FLASH_ATTN·FLASHINFER 백엔드). 2026-10-06 실행 완료([노트 §9](docs/experiments/2026-10-02-hybrid-validation.md)): 혼합 길이 배치가 균일보다 step당 5.46배 느린 증상은 vLLM 0.31에서도 나지만 FLASH_ATTN/FLASHINFER 백엔드·CUDA Graph·KV 블록 크기와 무관(0.997배)해, 분할 수 휴리스틱이 원인이라는 본 작품의 기전은 이 엔진에서 확인되지 않았다. 다음: vLLM decode step 프로파일링으로 attention 시간 분리.

6. **step 연산 분해 진단(`serve diagnose`, 2026-09-27 시작)** — decode step을 8개 연산 클래스로 CUDA event 분해하고 DRAM/텐서코어 상한에 대어 판정, attention 행에 선택 변형·손실·Amdahl 상한. 정체성은 바꾸지 않는 원인 설명(§1 ②) 보강이며 "균일 배치에서 이득이 없는 이유"의 실측 근거가 목적. 스펙 [docs/plan/2026-09-27-design-op-breakdown-diagnose.md](docs/plan/2026-09-27-design-op-breakdown-diagnose.md), 계획 [docs/plan/2026-09-27-plan-op-breakdown-diagnose.md](docs/plan/2026-09-27-plan-op-breakdown-diagnose.md). Task 1~9 완료, 결과는 `demo_data/serve_4090/diagnose_20260927/`와 [docs/experiments/2026-09-27-op-breakdown.md](docs/experiments/2026-09-27-op-breakdown.md)에 있다.

**중단 조건**: GPU 접근이 11월 중순까지 없으면 1·2·4는 "설계·도구 완료, 미실행"으로 보고하고 3(a)와 재현 검증을 데모의 중심으로 둔다.

## 3. 환경

- 로컬: WSL2, 8코어, RAM 3.8GiB, GPU 없음. 리포 경로 `/root/sunjae/gpu_graduation/kernelscope`, `.venv`(CPU torch). CUDA 11.8 툴킷 + gcc-11, GPGPU-Sim 4.2 빌드됨(빌드 스크립트는 리포 밖 `gpu_graduation/build_gpgpusim.sh`, 실행은 `experiments/gpgpusim/run_sim.sh`). 무거운 프로세스는 한 번에 하나.
- 연구실: RTX 4090, `/home/skkai/...`(README의 절대 경로). Accel-Sim v2.0.0 빌드 있음. §2의 GPU 필요 작업(1·2·4, 3b)은 여기서 실행한다.
