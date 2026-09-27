# 졸업작품 계획 v3 — GPU 프로파일링·시뮬레이션 기반 LLM 추론 attention 커널 동적 선택 시스템

> 2026-09-26 v3. **이 파일이 계획의 단일 원본이다** — 모든 호스트(로컬 WSL2, 연구실 4090)는 이 파일을 기준으로 삼고, 계획이 바뀌면 여기를 고쳐 커밋·푸시한다.
> v2(2026-07-23, GPGPU-Sim 축소 attention–SSM 비교)는 로컬 워크스페이스 `gpu_graduation/docs/plan-v2-gpgpusim-2026-07-23.md`(리포 밖)에 보관. 9월 1일부터 실작업은 이 리포(github.com/sunjae0224/kernelscope)이며, 이 문서는 그 방향을 한 장으로 고정한다.
> 원칙은 그대로다: **연구급 아님, 코어 스코프만으로 완성.** 중간보고서(2026-09-27 제출)는 로컬 워크스페이스 `gpu_graduation/midterm_report/`(리포 밖).

## 0. 한 줄 정의

LLM 서빙에서 길이가 다른 요청이 한 배치에 섞이면 FlashAttention의 KV 분할 수(num_splits) 휴리스틱이 GPU를 채우지 못한다. 이 문제를 **실측(프로파일링)으로 진단하고, 시뮬레이터·성능 모델로 설명·예측하며, 매 디코드 단계 분할 수를 동적으로 골라 실제 LLM 생성에서 검증하는 시스템**을 만든다. 정체성은 서빙 최적화 시스템이다. 프로파일러와 시뮬레이터는 제품이 아니라 이 파이프라인의 단계다.

원 신청서(Transformer↔Mamba 동적 스위칭 + 대시보드)와의 관계: 프로파일링 → 동적 전환 → 대시보드의 골격은 유지하고, 전환 대상을 "다른 모델"(답이 바뀜)에서 "같은 모델 안의 수학적으로 동등한 커널 설정"(답이 유지됨)으로 구체화했다.

## 1. 지금까지 확보한 것 (2026-09-26)

| 단계 | 상태 | 근거 |
|---|---|---|
| ① 커널 실측 | 완료 | RTX 4090, 6,801조건, 균일 길이 손실 중앙값 0.72% vs 혼합 길이 최악 12.53배 |
| ② 원인 설명·예측 | 진행 중 | CTA 작업량 분석(부하 불균형), 성능 모델(시간 오차 2.3~12.3%), Accel-Sim 4090(미보정, 1.05~3.06배), GPGPU-Sim 축소 커널로 기전 정성 재현(혼합 7.23배 vs 균일 2.25배) |
| ③ 선택 정책 | 완료 + 개선 | 휴리스틱/고정/테이블/모델 + **혼합 정책(δ=0.2)**: cold 검증 집합 최악 손실 1.97 / 7.54 / 10.90% (기준 15% 통과, GPU 없이 leave-one-out 평가) |
| ④ 실제 LLM 검증 | 완료(1차) | Qwen3-4B 혼합 길이 TPOT 1.81배, 토큰 일치; 후속 108회, 두 모델 1.28배/1.18배 |
| ⑤ 출력 보존 검증 | 원인 규명 | 불일치는 bf16 간격 1~2개 안의 로짓 동점에서만 발생(분기 사건 25건, 교사 강제 진단과 4/4 일치). 지표를 "분기 사건 수 + 동점 분류"로 교체 |
| ⑥ 재현·시각화 | 완료 | `kernelscope verify` 40개 항목 GPU 없이 재계산, 대시보드 4화면 |

## 2. 남은 작업 (우선순위 순)

1. **혼합 정책의 실제 LLM 검증** — `serve run --policy hybrid:demo_data/dispatch_paged_cold.csv:0.2`, Qwen3-4B 3조건 × 3회. GPU 필요(연구실 4090 호스트).
2. **불일치 지표 교체** — 새 캠페인마다 교사 강제 진단을 실행해 분기 사건을 `tie_1ulp/tie_2ulp/clear`로 분류. `clear`가 나오면 구현 오류로 취급. GPU 필요(연구실 4090 호스트).
3. **시뮬레이터** — (a) 로컬 GPGPU-Sim 4.2(빌드·실행 확인, [experiments/gpgpusim/](experiments/gpgpusim/))로 직접 작성한 split-KV 커널(`splitkv_attn.cu`, 참조 대비 오차 2e-8)의 부하 불균형 기전을 정성 재현 — 스윕 결과([docs/experiments/gpgpusim-splitkv-sweep.csv](docs/experiments/gpgpusim-splitkv-sweep.csv)): 혼합 길이 배치는 S=1→16에서 7.23배, 균일 배치는 S=8에서 2.25배 후 포화 — 실측과 같은 방향의 정성 재현 완료; (b) 연구실 호스트에서 Accel-Sim L2 파티션 해시 설정 재검증.
4. **FlashInfer 비교** — [docs/plan/2026-09-26-plan-flashinfer-comparison.md](docs/plan/2026-09-26-plan-flashinfer-comparison.md). GPU 필요(연구실 4090 호스트).
5. **최종 보고서·발표·데모** — 12월. 대시보드에 재현 검증 화면 추가.

6. **step 연산 분해 진단(`serve diagnose`, 2026-09-27 시작)** — decode step을 8개 연산 클래스로 CUDA event 분해하고 DRAM/텐서코어 상한에 대어 판정, attention 행에 선택 변형·손실·Amdahl 상한. 정체성은 바꾸지 않는 원인 설명(§1 ②) 보강이며 "균일 배치에서 이득이 없는 이유"의 실측 근거가 목적. 스펙 [docs/plan/2026-09-27-design-op-breakdown-diagnose.md](docs/plan/2026-09-27-design-op-breakdown-diagnose.md), 계획 [docs/plan/2026-09-27-plan-op-breakdown-diagnose.md](docs/plan/2026-09-27-plan-op-breakdown-diagnose.md). Task 1~5(타이머·엔진·비용 모델·리포트) 완료, 6~9(그림·CLI·GPU 실행 D1~D3·verify/문서) 남음 — 이어서 할 순서는 [TODO.md](TODO.md).

**중단 조건**: GPU 접근이 11월 중순까지 없으면 1·2·4는 "설계·도구 완료, 미실행"으로 보고하고 3(a)와 재현 검증을 데모의 중심으로 둔다.

## 3. 환경

- 로컬: WSL2, 8코어, RAM 3.8GiB, GPU 없음. 리포 경로 `/root/sunjae/gpu_graduation/kernelscope`, `.venv`(CPU torch). CUDA 11.8 툴킷 + gcc-11, GPGPU-Sim 4.2 빌드됨(빌드 스크립트는 리포 밖 `gpu_graduation/build_gpgpusim.sh`, 실행은 `experiments/gpgpusim/run_sim.sh`). 무거운 프로세스는 한 번에 하나.
- 연구실: RTX 4090, `/home/skkai/...`(README의 절대 경로). Accel-Sim v2.0.0 빌드 있음. §2의 GPU 필요 작업(1·2·4, 3b)은 여기서 실행한다.
