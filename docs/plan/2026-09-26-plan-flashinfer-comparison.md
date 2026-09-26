# 계획: FlashInfer와의 같은 조건 비교 (GPU 필요, 미실행)

작성 2026-09-26. 상태: **연구실 RTX 4090 접근이 가능해질 때 실행.** 이 문서는 실험 설계만 담고 결과는 없다.

## 왜 비교하는가

FlashInfer[Ye et al., MLSys 2025]는 길이가 다른 요청이 섞인 배치의 부하 불균형을 커널 내부의 부하 분산 스케줄링(plan/run 구조)으로 해결한다. 본 작품은 커널을 바꾸지 않고 FlashAttention-2의 분할 수만 외부에서 고른다. 심사에서 "그냥 FlashInfer를 쓰면 되지 않는가"라는 질문에 답하려면 같은 입력·같은 KV 캐시 형식에서 두 접근의 attention 시간과 전체 생성 시간을 나란히 재야 한다. 목표는 이기는 것이 아니라 **격차의 크기와 방향을 재는 것**이다. FlashInfer가 더 빠르더라도, 본 작품의 기여(기존 라이브러리 위에서 설정만으로 얻는 개선과 그 원인 분석)는 유지된다.

## 실험 설계

| 항목 | 내용 |
|---|---|
| 커널 플러그인 | `kernelscope/plugins/builtin/flashinfer.py`(신규): `flashinfer.BatchDecodeWithPagedKVCacheWrapper`, page 256, bf16, GQA. `plan()`은 매 단계 호출하고 그 시간을 `plan_us`로 따로 기록 |
| 공정성 | 같은 `Workload`(B, 요청별 길이, 헤드 구성)와 같은 페이지 단위 KV 캐시 내용. 정확성은 기존 참조 구현과 비교. cold/warm 캐시 상태 구분 유지 |
| 커널 격자 | `grids/ragged_s1.yaml`(혼합 길이 162조건)과 `grids/dispatch_s1.yaml`(균일 49조건), `cache_state cold` — 기존 FlashAttention-2 변형 측정과 같은 조건 |
| 비교 대상 | (a) FlashAttention-2 기본 휴리스틱, (b) 측정 테이블 정책이 고른 변형, (c) 혼합 정책(δ=0.2)이 고른 변형, (d) FlashInfer |
| 전체 모델 | `serve run`에 attention 백엔드 선택 옵션 `--attention flashinfer` 추가. Qwen3-4B, `graduation_ragged`·`graduation_uniform`·`heldout_text_ragged`, 반복 3회, 생성 토큰 일치 검사 동일 적용 |
| 지표 | attention 커널 시간(µs), `plan_us`, 단계 시간, TPOT, 분기 사건 수(`scripts/analyze_mismatches.py`) |
| 예상 GPU 시간 | 커널 격자 약 15분, 전체 모델 약 30분 |

## 해석 규칙

- attention 시간은 (d)가 (b)/(c)보다 빠를 수 있다. 그 격차가 (b)/(c)가 (a) 대비 얻은 개선(최악 조건 3.84배, 페이지 단위 KV)보다 작으면 "설정 선택만으로 커널 교체 이득의 대부분을 얻는다"고 쓴다.
- FlashInfer의 `plan()` 비용은 정책 선택 비용(`policy_us`)과 같은 자리에서 비교한다.
- 생성 토큰 불일치는 두 접근 모두에서 bf16 동점으로 생길 수 있으므로 분기 사건 수와 동점 분류로 보고한다.

## 준비 사항

- flashinfer-python 0.6.x 휠(`env/requirements-lock.txt`에 기록된 버전)을 `gradkernel` 환경에 설치. CUDA 12.8 torch 2.8과 호환되는지 `serve doctor`로 확인.
- 이 노트북(GPU 없음)에서는 플러그인 import가 실패하므로, 플러그인은 `flash_attn`과 같은 방식으로 선택 의존성으로 등록한다(없으면 경고만).
