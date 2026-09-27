# 자연어·두 모델 후속 검증

RTX 4090 한 대에서 동일한 실측표·피팅 파라미터를 고정하고 두 모델, 세 시나리오, 두 시드를 측정했다. 각 조합은 세 정책을 세 번 반복하며 실행 순서를 회전한다. 입력은 직접 작성한 문단으로 구성했으며 운영 트래픽이나 표준 언어 품질 벤치마크가 아니다.

각 정책으로 전체 시나리오를 한 번 워밍업했다. 측정마다 새 정책 객체를 만들어 첫 선택 비용을 보존했다.

기존 측정 격자와 다른 배치 28, 짧은 문맥 384, 긴 문맥 12,288토큰을 사용했다. 두 모델의 attention head 구성은 같으므로 다른 head 구성이나 다른 GPU로의 일반화는 검증하지 않는다.

TPOT는 요청별 토큰 간 시간의 평균이며 도착 요청의 직렬 prefill과 정책 선택 비용을 포함한다. 배율은 같은 모델·시나리오·시드·반복의 휴리스틱 TPOT ÷ 해당 정책 TPOT이다. 생성 토큰 또는 스케줄이 다르면 배율을 성능 개선의 근거로 표시하지 않는다.

출력 검증을 통과한 정책·입력 조합은 **14/24개**다. 아래 표에는 불일치 사례도 모두 포함한다. 전체 캠페인의 종료 코드 1은 검증 불일치가 보존됐다는 뜻이며, 완료 여부는 각 manifest의 `status`로 확인한다.

| 모델 | 조건 | 시드 | 정책 | TPOT ms | 배율 | 출력 검증 |
|---|---|---:|---|---:|---:|---|
| Qwen 4B | uniform | 0 | model | 31.234 | 0.981× | 일치 |
| Qwen 4B | uniform | 0 | table | 30.652 | 1.000× | 일치 |
| Qwen 4B | uniform | 1 | model | 31.347 | 0.982× | 일치 |
| Qwen 4B | uniform | 1 | table | 30.815 | 0.999× | 일치 |
| Qwen 4B | ragged | 0 | model | 33.835 | 1.282× | 일치 |
| Qwen 4B | ragged | 0 | table | 33.812 | 1.283× | 일치 |
| Qwen 4B | ragged | 1 | model | 33.692 | 1.290× | 일치 |
| Qwen 4B | ragged | 1 | table | 33.749 | — | 불일치/미검증 |
| Qwen 4B | arrivals | 0 | model | 33.534 | — | 불일치/미검증 |
| Qwen 4B | arrivals | 0 | table | 31.759 | — | 불일치/미검증 |
| Qwen 4B | arrivals | 1 | model | 33.455 | — | 불일치/미검증 |
| Qwen 4B | arrivals | 1 | table | 31.669 | — | 불일치/미검증 |
| Llama 8B | uniform | 0 | model | 45.042 | 0.989× | 일치 |
| Llama 8B | uniform | 0 | table | 44.527 | 1.001× | 일치 |
| Llama 8B | uniform | 1 | model | 45.147 | 0.991× | 일치 |
| Llama 8B | uniform | 1 | table | 44.627 | 1.002× | 일치 |
| Llama 8B | ragged | 0 | model | 47.520 | 1.188× | 일치 |
| Llama 8B | ragged | 0 | table | 47.533 | — | 불일치/미검증 |
| Llama 8B | ragged | 1 | model | 47.765 | 1.184× | 일치 |
| Llama 8B | ragged | 1 | table | 47.720 | 1.185× | 일치 |
| Llama 8B | arrivals | 0 | model | 46.083 | — | 불일치/미검증 |
| Llama 8B | arrivals | 0 | table | 44.322 | — | 불일치/미검증 |
| Llama 8B | arrivals | 1 | model | 46.103 | — | 불일치/미검증 |
| Llama 8B | arrivals | 1 | table | 44.386 | — | 불일치/미검증 |

## 정책 선택 비용

아래 첫 호출·캐시 miss·hit는 서로 겹치지 않는 분류다. C 실행 경로의 컴파일/로드는 시작 단계에 수행하고 manifest의 `policy_setup_runs`에 따로 기록했다. 첫 호출에는 새 길이 구성의 전체 후보 예측이 포함된다.

| 모델 | 조건 | 시드 | 첫 호출 µs | 후속 miss µs | hit µs |
|---|---|---:|---:|---:|---:|
| Qwen 4B | uniform | 0 | 14928.2 | — | 6.5 |
| Qwen 4B | uniform | 1 | 15136.5 | — | 6.7 |
| Qwen 4B | ragged | 0 | 13644.2 | — | 6.8 |
| Qwen 4B | ragged | 1 | 13697.4 | — | 7.1 |
| Qwen 4B | arrivals | 0 | 1071.3 | 7138.8 | 6.0 |
| Qwen 4B | arrivals | 1 | 1052.0 | 7116.5 | 6.0 |
| Llama 8B | uniform | 0 | 15226.0 | — | 6.3 |
| Llama 8B | uniform | 1 | 15218.1 | — | 6.8 |
| Llama 8B | ragged | 0 | 13319.8 | — | 7.3 |
| Llama 8B | ragged | 1 | 13275.7 | — | 6.6 |
| Llama 8B | arrivals | 0 | 1013.0 | 7078.8 | 5.9 |
| Llama 8B | arrivals | 1 | 1012.6 | 7048.2 | 5.8 |

개별 반복의 시간은 [repetitions.csv](repetitions.csv), 반복 평균·표준편차, 95% paired bootstrap 구간, 비교 불가 사유와 입력·소스 식별자는 [summary.csv](summary.csv)에 보존한다. 이는 세 번 반복한 해당 실험의 변동이며 서비스 전체에 대한 보장 구간이 아니다.

![모델과 시나리오별 TPOT](tpot.png)
