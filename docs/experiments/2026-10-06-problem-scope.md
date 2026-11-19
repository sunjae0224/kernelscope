# 문제의 범위: 실제 트래픽에서의 빈도, 두 라이브러리의 정적 기본값, 64K 긴 요청 (2026-10-06)

> 검토 문서([2026-10-06-review-problem-definition.md](../plan/2026-10-06-review-problem-definition.md))가 지적한
> "문제가 좁아 보인다"에 대한 세 가지 측정. 실행 호스트는 연구실 RTX 4090(커널·실생성)과 그 CPU(트레이스 재생).
> 원본은 리포 밖 `../kernelscope/results/{hw_4090/*_64k_*, traffic/*, serve_4090/longctx_20261006, serve_4090/flashinfer_20261006, serve_4090/anytable_20261008}`,
> 번들은 `demo_data/hw_4090/*_64k_*`, `demo_data/traffic/<run>/summary.json`, `demo_data/serve_4090/{longctx,flashinfer}_20261006/`, `demo_data/serve_4090/anytable_20261008/`, `demo_data/dispatch_paged_cold_any.csv`.
> 이 문서의 수치는 전부 `kernelscope verify`의 `traffic.*`, `defaults.*`, `longctx.*`, `fiengine.*`, `anytable.*` 항목이 번들에서 다시 계산한다.

## 요약

1. **빈도(§1).** 공개 요청 트레이스를 연속 배치로 재생하고 매 decode step의 길이 분포를 성능 모델로 채점하면, Azure 채팅 트레이스를 4090 한 장이 받는 부하에서는 전체 step의 **13.5%** 가 분할 휴리스틱 때문에 attention 시간을 1.25배 이상 잃고, 전체 attention 시간의 **1.119**분의 1만 쓰면 되는 step 선택기가 있다(10.6% 절감). 짧은 프롬프트 위주의 BurstGPT에서는 **0.3%** 로 사실상 없다. 즉 드문 문제가 아니라 **트래픽의 긴 프롬프트 비중과 배치 크기에 달린 조건부 문제**다.
2. **두 라이브러리(§2).** 같은 혼합 길이 격자에서 FlashInfer의 기본 변형(tensor-core)도 전체 최선 대비 중앙값 **1.017배**, 90퍼센타일 **1.489배**, 최악 **2.485배** 느리고(162셀 중 **36셀** 이 1.25배 초과), 균일 길이에서는 최악 **1.053배** 로 멀쩡하다. FA2 휴리스틱(중앙값 **1.457배**)과 같은 방향의 같은 현상이며, FlashInfer CUDA-core 변형(최악 **1.014배**)과 FA2 분할만 고르는 본 작품의 선택기(최악 **1.063배**)는 둘 다 최선에 붙어 있다.
3. **64K(§3).** 긴 요청을 64K로 늘리면 같은 조건(길이마다 18셀)에서 휴리스틱 손실 중앙값이 8K **1.545** → 16K **1.933** → 32K **2.558** → 64K **2.839**, 최악 **5.183** 으로 단조 증가하고, FlashInfer tensor-core 기본값도 **1.126** → **1.634** 로 커진다. 문맥이 길어질수록 문제가 커진다.
4. **엔진 안(§4).** Qwen3-4B 혼합 길이 실생성에서 FlashInfer CUDA-core 정책은 **1.833배**로 FA2 테이블(**1.810배**)과 같은 수준이고 tensor-core 기본값은 **1.257배**에 그친다(attention은 더 빠르지만 step마다 `plan()` 370.8 µs). 균일 길이는 네 정책이 ±1% 안이다. FlashInfer 정책의 출력 분기는 대부분 tie_1ulp이고, clear 한 자리(요청 8 위치 41)는 10월 2일 대조군과 같은 사건으로 두 백엔드 모두 커널 탐침을 통과했다(층별 차이 bf16 간격 1개 안, 같은 KV 상태의 단일 step에서 argmax 유지).
5. **라이브러리 무관 표(§5).** 정책이 step마다 FA2 분할과 FlashInfer 변형을 함께 고르게 하면(커널 시간만 보는 표) 혼합 길이 실생성은 모든 step에서 FlashInfer CUDA-core를 골라 **1.831배**로 §4의 고정 CUDA-core 정책(**1.833배**)과 같은 자리에 서고, FA2 분할만 고르는 테이블(**1.810배**)보다 **1.15%** 빠르다. 대가는 출력이다: FA2 기준과 토큰이 **65개**(요청 도착 **39개**) 갈리고, 그 사건은 §4의 CUDA-core 정책과 같다. `plan()` 비용 371 µs를 더해 고르는 표는 FA2 분할 16으로 돌아가 테이블을 재현한다(**1.806배**). 균일 길이는 세 표가 모두 FA2 기본값을 골라 TPOT 차이가 **0.07%** 이내이고, 요청 도착은 순서가 같되 이득(**0.36%**)이 측정 편차와 구분되지 않는다. 즉 분할 수를 밖에서 고르는 것이 커널 교체 이득을 거의 다 얻고, 라이브러리까지 고르게 해서 더 얻는 것은 이 GPU의 혼합 길이에서 조금이다.

## 1. 실제 트래픽에서 얼마나 자주 생기는가 — 트레이스 재생 (GPU 없이)

### 방법

`scripts/traffic_replay.py`([모듈](../../kernelscope/analysis/traffic.py)). 공개 트레이스의 (도착 시각, 프롬프트 토큰, 출력 토큰)을 엔진과 같은 규칙의 연속 배치 스케줄러에 넣는다: FIFO admission, 배치 상한 64, 요청은 admission 때 `ceil((prompt + output) / 256)` 페이지를 끝까지 예약(KV 예산 10 GiB = Qwen3-4B bf16 **72,817 토큰**, 4090 설정), 첫 토큰은 prefill에서 나오므로 `output − 1`번 decode. 단순화는 두 가지다: prefill은 즉시(도착 다음 step에 합류), step 시간은 배치와 무관하게 30 ms 고정. 트레이스의 도착 간격은 `rate_scale`로 압축해 GPU 한 장의 부하를 만든다.

매 step의 KV 길이 벡터를 (1) FA2 휴리스틱의 분할 수(`geometry.num_splits_heuristic` 포팅; S1 헤드 32/8에서 B×8 ≥ 0.8×2×128 → **B ≥ 26이면 분할 없음**), (2) 실측으로 맞춘 대리 성능 모델(`models/rtx4090.json`, cold)로 휴리스틱 변형과 분할 2/4/8/16 변형의 시간을 예측해 **손실 = 휴리스틱 예측 시간 / 최선 예측 시간** 으로 채점한다(페이지 단위 길이 구성마다 한 번 예측, 모든 step 채점). 모델 오차가 2.3~12.3%이므로 1.25배 문턱을 주된 지표로 쓴다.

트레이스: Azure LLM Inference Trace 2023(conv 19,366건 1시간, code 8,819건 1시간; `AzurePublicDataset`), BurstGPT v2(`BurstGPT_without_fails_1.csv`, 첫 7일 17,893건; GPT-4만 14일 14,265건). 길이 통계: Azure conv 프롬프트 중앙값 1,020·최대 14,050, 출력 중앙값 129; code 중앙값 1,469·최대 7,437, 출력 중앙값 13; BurstGPT 프롬프트 중앙값 502·최대 5,087. **어느 트레이스도 32K 요청을 담고 있지 않다** — 본 작품의 최악 셀(32K×1+512×31)은 이 트레이스들에는 나오지 않는 극단이다.

### 결과

| 실행 | 평균 배치 | 분할 없음 step | 손실 ≥1.1 | 손실 ≥1.25 | 손실 ≥1.5 | 손실 p90 / 최대 | attention 시간 비 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Azure conv ×0.5 (평균 배치 17) | 17.4 | 6.2% | 18.1% | **2.4%** | 0.2% | 1.131 / 1.832 | 1.054 |
| **Azure conv ×1** (19,366건, 117,035 step) | **평균 배치 34.8** | **89.5%** | **46.5%** | **13.5%** | **0.6%** | **1.278** / **2.192** | **1.119** |
| Azure conv ×1, KV 436K 토큰(80 GB급 가정) | 34.8 | 89.3% | 45.9% | **13.7%** | 0.6% | 1.278 / 2.192 | 1.117 |
| Azure conv ×2 (평균 배치 47) | 47.0 | 98.8% | 41.2% | **4.1%** | 0.1% | 1.202 / 2.179 | **1.092** |
| Azure conv ×1, step 50 ms | 46.9 | 98.6% | 41.2% | 4.1% | 0.1% | 1.201 / 2.179 | 1.092 |
| Azure code ×4 | 11.7 | 14.6% | 29.1% | 5.2% | 0.1% | 1.201 / 1.842 | 1.091 |
| Azure code ×8 | 17.8 | 35.0% | 37.6% | 8.0% | 0.1% | 1.231 / 1.592 | 1.111 |
| **Azure code ×16** (평균 배치 27) | 27.4 | 70.1% | 55.0% | **14.1%** | 0.0% | 1.280 / 1.509 | **1.136** |
| BurstGPT 7일 ×50 | 14.1 | 14.2% | 14.0% | 0.3% | 0.0% | 1.122 / 1.382 | 1.033 |
| **BurstGPT 7일 ×100** (평균 배치 27) | 26.5 | 50.1% | 9.7% | **0.3%** | 0.0% | 1.099 / 1.469 | **1.018** |
| BurstGPT GPT-4 14일 ×300 (평균 배치 30) | 29.6 | 66.7% | 11.5% | **1.5%** | 0.0% | 1.110 / 1.666 | 1.030 |

"attention 시간 비"는 전체 step의 휴리스틱 attention 예측 시간 합 / 최선 분할 예측 시간 합이다. Azure conv ×1에서 손실 1.25배 이상인 step이 attention 시간의 **16.2%** 를 차지한다.

what-if(트레이스를 바꾼 것이므로 별도 표기): 프롬프트 길이를 ×2, ×4로 늘린 재생.

| what-if | 평균 배치 | 분할 없음 | 손실 ≥1.25 | attention 시간 비 |
|---|---:|---:|---:|---:|
| Azure conv ×1, 프롬프트 ×2, KV 10 GiB | 27.1 | 70.8% | **25.7%** | **1.156** |
| Azure conv ×1, 프롬프트 ×4, KV 10 GiB | **14.6** | 0.1% | **2.9%** | 1.064 |
| Azure conv ×1, 프롬프트 ×2, KV 436K 토큰 | 34.8 | 89.3% | **29.3%** | **1.179** |
| Azure conv ×1, 프롬프트 ×4, KV 436K 토큰 | 34.8 | 89.3% | **39.5%** | **1.222** |
| Azure code ×16, 프롬프트 ×2, KV 10 GiB | 15.0 | 1.7% | 1.6% | 1.060 |
| Azure code ×16, 프롬프트 ×2, KV 436K 토큰 | 31.3 | 48.4% | 4.4% | 1.104 |
| BurstGPT 7일 ×100, 프롬프트 ×4, KV 436K 토큰 | 26.5 | 48.5% | 3.5% | 1.069 |

### 읽는 법

- **빈도는 배치 크기와 길이 편차의 곱으로 정해진다.** 분할 없음 조건(B ≥ 26)이 성립하는 부하(Azure conv ×1, code ×16)에서 step의 13~14%가 1.25배 이상을 잃고, 부하가 낮아 B < 26이면 휴리스틱이 스스로 분할하므로 손실이 2~5%로 준다. 반대로 B ≈ 47(×2, step 50 ms)에서는 CTA가 376개라 긴 요청 하나의 비중이 희석되어 4.1%로 다시 준다(커널 격자에서 B=64의 손실 중앙값이 B=26보다 작은 것과 같은 이유).
- **프롬프트가 길어지면 두 방향이 겹친다.** 프롬프트 ×2는 손실 step을 25.7%, attention 절감을 15.6%로 키우지만, ×4는 10 GiB KV 안에서 배치가 14.6으로 쪼그라들어 B < 26이 되고 손실이 2.9%로 사라진다. KV 예산을 436K 토큰(80 GB급)으로 두면 배치가 34.8로 유지되어 프롬프트 ×2에서 **29.3%**(attention 시간 비 **1.179**), ×4에서 **39.5%**(**1.222**, 손실 1.25배 이상 step이 attention 시간의 47%)까지 커진다. 즉 긴 문맥 서빙에서 이 문제가 커지는 조건은 **긴 프롬프트 + 배치를 26 이상으로 유지할 KV 예산** 이다.
- **트래픽에 긴 프롬프트가 없으면 문제도 없다.** BurstGPT(프롬프트 중앙값 502)에서는 어느 부하에서도 1.5% 이하다.
- 한계: 성능 모델의 예측이지 실측이 아니다(실측은 §3·§4의 실생성). step 시간 고정·즉시 prefill은 배치 구성을 단순화한다. 트레이스의 토큰 수는 각 서비스의 토크나이저 기준이다.

## 2. 두 라이브러리의 정적 기본값 — 같은 격자, 같은 현상

### 방법

`kernelscope.analysis.dispatch.static_default_losses`: 페이지 KV cold 격자의 셀마다 측정된 모든 변형(FA2 분할 1/휴리스틱/2~128, FlashInfer tensor-core, FlashInfer CUDA-core)의 최선을 기준으로 **각 정적 기본값의 손실**(기본값 시간 / 전체 최선 시간)을 센다. `fa2_best`는 FA2 분할만 고를 수 있는 본 작품 선택기의 상한이다. 32K 격자는 `demo_data/hw_4090/{ragged,uniform}_s1_{paged,flashinfer,flashinfer_cudacore}`(2026-09-19·10-06 측정), 64K 격자는 §3.

### 결과 (손실 = 기본값 / 전체 최선)

| 격자 | 셀 | 기본값 | 중앙값 | p90 | 최악 | 1.25배 초과 셀 |
|---|---:|---|---:|---:|---:|---:|
| 32K 혼합 길이 | 162셀 | FA2 휴리스틱(`num_splits=0`) | **1.457배** | 2.519배 | 4.106배 | **118셀** |
| | | FlashInfer tensor-core(GQA 기본) | **1.017배** | **1.489배** | **2.485배** | **36셀** |
| | | FlashInfer CUDA-core | 1.000배 | 1.003배 | **1.014배** | 0 |
| | | FA2 분할만 고르는 선택기 | **1.025배** | 1.043배 | **1.063배** | 0 |
| 32K 균일 길이 | 49셀 | FA2 휴리스틱 | 1.025배 | 1.092배 | **1.171배** | 0 |
| | | FlashInfer tensor-core | 1.000배 | 1.035배 | **1.053배** | 0 |
| 64K 혼합 길이 | 18셀 | FA2 휴리스틱 | **2.861배** | 4.521배 | **5.265배** | 18셀 전부(2배 초과도 18셀 전부) |
| | | FlashInfer tensor-core | **1.634배** | 2.463배 | **2.989배** | **12셀** |
| | | FlashInfer CUDA-core | 1.000배 | 1.003배 | 1.014배 | 0 |
| | | FA2 분할만 고르는 선택기 | 1.023배 | 1.044배 | **1.049배** | 0 |
| 64K 균일 길이 | 2셀 | FA2 휴리스틱 | 1.003배 | 1.003배 | **1.003배** | 0 |

### 읽는 법

- 혼합 길이에서 **두 라이브러리의 정적 선택이 모두 틀린다.** FA2는 분할 수를 길이를 보지 않고 정하고, FlashInfer는 head 그룹 크기(≥4)만 보고 tensor-core 변형을 기본으로 고르는데, 그 변형은 긴 요청 하나가 섞인 배치에서 CUDA-core 변형보다 최대 2.5배(32K)·3.0배(64K) 느리다. 균일 길이에서는 둘 다 1.2배 안쪽이다. 따라서 문제 정의는 "FA2의 버그"가 아니라 **"혼합 길이 decode 배치에서 attention 라이브러리의 정적 기본값이 틀리고, 그 선택은 길이 분포를 봐야 한다"** 로 넓어진다.
- 본 작품의 선택기는 FA2 인자만 바꾸는데도 전체 최선(FlashInfer 포함)에 최악 1.063배(32K)·1.049배(64K)로 붙는다. 같은 선택기가 FlashInfer 두 변형까지 후보에 넣으면(§4의 엔진 통합) 라이브러리 무관 선택기가 된다.
- 32K 혼합 162셀 중 전체 최선 커널은 FlashInfer CUDA-core 139셀, FlashInfer tensor-core 21셀, FA2 분할 2셀이다. 커널 자체는 FlashInfer가 조금 빠르지만(중앙값 2.5%), 이득의 대부분은 **분할 수(= 부하 분배)** 에서 나온다.

## 3. 긴 요청을 64K로 — 길이에 따른 단조 증가

### 방법

`grids/ragged_s1_64k.yaml`(B 26/32/64 × 긴 요청 1·2개 × 짧은 512/1024/2048 = 18셀, 긴 요청 65,536)과 `grids/uniform_s1_64k.yaml`(B 16/32 × 65,536)을 FA2 페이지 변형 9개와 FlashInfer 두 변형으로 cold 측정(`kernelscope bench`, warm-up 10·반복 30, 2026-10-06, 오류 0). 32K 격자와 **같은 조건의 셀만** 골라(B ∈ {26, 32, 64}, 긴 요청 1~2개, 짧은 512~2048 → 길이마다 18셀) 긴 요청 길이별로 비교한다.

### 결과

| 긴 요청 길이 | 휴리스틱 / 최선 FA2 분할, 중앙값 | 최악 | FlashInfer tensor-core / 전체 최선, 중앙값 | 최악 | FlashInfer CUDA-core 최악 |
|---:|---:|---:|---:|---:|---:|
| 8K | **1.545** | 2.295 | **1.126** | 1.586 | 1.009 |
| 16K | **1.933** | 2.991 | 1.269 | 1.906 | 1.010 |
| 32K | **2.558** | 3.981 | 1.443 | 2.485 | 1.001 |
| 64K | **2.839** | **5.183** | **1.634** | 2.989 | **1.014** |

64K 대표 셀(65536+512×31, B=32): FA2 휴리스틱 **1957.0** µs → 최선 FA2 분할 16 **421.9** µs(4.64배), FlashInfer CUDA-core **414.7** µs, FlashInfer tensor-core **1239.7** µs. 64K 최악 셀은 65536+512×25(B=26) 5.183배. 균일 64K(2셀)는 휴리스틱 손실 1.003배.

### 읽는 법

긴 요청이 길어질수록 그 CTA 하나의 일이 배치 전체를 지배하므로 손실이 단조 증가한다(중앙값 1.5 → 1.9 → 2.6 → 2.8, 최악 2.3 → 3.0 → 4.0 → 5.2). FlashInfer tensor-core 기본값도 같은 추세(1.13 → 1.63)이고, CUDA-core 변형은 길이와 무관하게 최선에 붙어 있다. 문맥 길이가 늘어나는 워크로드(RAG, 에이전트)에서 §1의 빈도 수치가 커지는 방향이다 — 단, §1이 보였듯 배치가 26 이상으로 유지될 KV 예산이 있을 때다.

### 3b. 64K 전체 모델 실생성 (Qwen3-4B, 2026-10-06 16:20~16:36)

`scenarios/graduation_ragged_64k.yaml`(65536×1 + 512×31, 64 토큰, KV 13 GiB, 배치 64), 정책 heuristic·table·model·hybrid, 캐시 유지 규약(`--policy-cache keep`), warm-up 1회 뒤 3회 반복, 정책 순서 회전. 원본 `serve_4090/longctx_20261006/ragged_64k`. 테이블 정책은 64K 셀이 없으므로 가장 가까운 32K 셀의 분할 수를 쓴다(외삽), 모델 정책은 기하에서 예측한다.

| 정책 | TPOT(ms) | 배율 | attention ms/step | attention 비중 | 선택 비용 µs/step | 분할 수 |
|---|---:|---:|---:|---:|---:|---|
| heuristic | **96.93** | 1.000 | **71.70** | **82.8%** | 5.5 | 1(분할 없음) |
| table | **40.06** | **2.419배** | 14.88 | 49.8% | 9.4 | 63 step 전부 분할 8 |
| model | 39.36 | **2.463배** | 14.58 | 49.7% | 8.4 | 분할 8 |
| hybrid | 39.35 | **2.463배** | **14.55** | **49.6%** | 18.6 | 분할 8 |

32K 조건(1.79~1.81배)보다 이득이 커졌고(2.42~2.46배), 커널 격자의 추세(§3)가 전체 모델에도 그대로 나타난다. 휴리스틱에서는 decode step의 82.8%가 attention이라 Amdahl 상한이 높다.

**토큰 동일성: 분기 사건 1건.** 세 정책 모두, 세 반복 모두 같은 자리에서 갈린다 — 64K 요청(rid 0)의 위치 27에서 참조 토큰 220 대 후보 15, 이후 2,048 토큰 중 4개가 다르고 다시 참조 이력에 합류한다(cascade 아님). 세 정책이 같은 커널 설정(분할 8)을 실행하므로 출력도 서로 같다.

### 3c. 64K 분기 사건의 분류 (2026-10-07 01:05, GPU 대기열)

교사 강제 진단(`numerics/ragged_64k`, 64 step × 32 요청, 세 정책 6,048행 중 argmax 불일치 0.05%)으로 분류하면 그 사건은 **clear 1건**이다: 참조 로짓은 토큰 220이 후보 토큰 15보다 0.6875 높고(1위 로짓 12.81, bf16 간격 0.0625), 여유는 **11 ulp**, 교사 강제 실행도 같은 토큰을 내어 **재현**됐다(10월 2일 대조군의 clear는 70 ulp). 규칙대로 조사 대상이며, 판정은 같은 KV 상태의 커널 탐침(`scripts/probe_split_kernel.py --campaign ../kernelscope/results/serve_4090/longctx_20261006 --run ragged_64k --target-step 26 --target-rid 0 --candidate-splits 8 --kv-gib 13`, 36층 fp32 참조, 2026-10-08 16:25)으로 했다. 위치 27 뒤 4개 토큰이 다르고 다시 합류하는 꼴은 10월 2일 사건과 같다.

**탐침 결과 — 구현 오류 아님.** 분할 1과 분할 8의 층별 attention 출력은 fp32 참조 대비 최대 0.123(출력 최대 31.8인 34번째 층; 그 크기의 bf16 간격은 0.125), 두 분할 간 차이도 최대 0.125로 **36층 모두 그 층 최대 출력의 bf16 간격 1개 안**이다(기준 넘는 층 0개; 10월 2일 대조군은 0.134·0.0625). 10월 2일과 다른 점은 같은 KV 상태에서 분할 8로 계산한 **단일 step**에서도 64K 요청의 argmax가 바뀐다는 것이다(220 → 15, 참조 1위 로짓 12.81·여유 0.6875 = 11 ulp, 최대 |Δlogit| 8.99, 두 로짓 벡터의 코사인 0.10); 짧은 요청 31개의 로짓은 두 분할에서 완전히 같다(차이 0). 즉 이 사건은 실행이 쌓은 KV 반올림의 누적이 아니라 **한 step 안에서 층별 1 ulp 차이가 36층을 지나며 증폭된 것**이며, 65K 무작위 토큰 문맥에서 참조 분포가 평평해(여유 11 ulp) 한 step의 반올림만으로 순위가 바뀐다. 커널은 맞고, 긴 무작위 문맥에서 모델 출력이 그만큼 불안정하다는 뜻이다. 기록 `longctx_20261006/numerics/ragged_64k/kernel_probe/`(verify `serve64k.probe_*`).


## 4. FlashInfer를 엔진 안에서 — 같은 규약의 실생성 비교 (Qwen3-4B, 2026-10-08 16:08~16:21)

`kernelscope/serve/attention.py`의 `FlashInferDecode`가 decode step에서 `flash_attn_with_kvcache` 자리에 들어간다(새 K/V를 페이지 풀에 쓰고, step마다 `plan()`, 층마다 `run()`; prefill은 FA2). 정책 `flashinfer`(tensor-core)·`flashinfer_cudacore`는 분할 수 대신 이 백엔드를 고르고, `plan()` 시간은 `policy_us`에 기록된다. GPU 테스트(`tests/test_serve_attention_gpu.py`: flash-attn과 로짓 2e-2 안, argmax 동일, plan 시간 기록)는 통과했다.

### 방법

§3b와 같은 규약이다: 정책 순서를 반복마다 회전하고, `--policy-cache keep`, warm-up 뒤 **반복 3회**, KV 풀 **KV 10 GiB**. 세 시나리오 `graduation_ragged`(32768×1 + 512×31, 64 토큰), `graduation_uniform`(512×32), `graduation_arrivals`(16384×1에 짧은 요청들이 step 0·8·24에 나뉘어 합류)를 네 정책으로 돌렸다: `heuristic`, `table:demo_data/dispatch_paged_cold.csv`(FA2 분할만 고르는 §3b와 같은 측정 테이블), `flashinfer`(tensor-core 변형; vLLM이 GQA에서 고르는 기본), `flashinfer_cudacore`. 배율은 반복별 TPOT 평균의 비(휴리스틱 / 정책)이고, FlashInfer 정책의 `policy_us`는 그 step의 `plan()` 시간이다. 원본은 `serve_4090/flashinfer_20261006/{ragged,uniform,arrivals}`.

조건. 코드는 커밋 b8e1707에 미커밋 변경이 얹힌 상태다(manifest의 `git.status`가 비어 있지 않다). 실행 동안 이 캠페인 말고 GPU에 올라 있던 것은 상주 rerun 뷰어(436 MiB) 하나뿐이다(30초 간격 nvidia-smi 기록). 출력 비교는 `scripts/classify_divergence`가 휴리스틱과 토큰이 처음 갈린 자리마다 교사 강제 진단 로짓으로 분류한다(규칙은 §3c와 같다). 교사 강제 진단은 혼합 길이는 2026-10-07 02:50의 것이고, 균일·도착은 같은 정책으로 2026-10-08 16:28~16:33에 새로 돌려 다시 분류했다(진단의 참조 토큰이 이번 실행의 것과 다르면 분류하지 않고 `unclassified`로 남기는데, 그런 사건은 없었다). 2026-10-07 밤의 첫 실행은 결과 폴더가 지워져 번들에 없다. 로그에 남은 요약(혼합 길이의 FlashInfer 두 정책 배율과 plan 비용 범위)은 이번 값과 반올림 수준에서 일치했다. 그 원본은 없으므로 verify는 이번 캠페인만 다시 계산한다.

### 결과

step당 시간. 배율은 휴리스틱 대비이고, "정책 비용"은 휴리스틱·table에서는 선택 비용, FlashInfer 정책에서는 step마다의 `plan()` 시간(µs/step)이다.

혼합 길이(`graduation_ragged`):

| 정책 | TPOT(ms) | 배율 | attention ms/step | attention 비중 | 정책 비용 µs/step |
|---|---:|---:|---:|---:|---:|
| heuristic | 60.99 | 기준 | 36.78 | 71.9% | 3.7 |
| table | 33.70 | 1.810배 | 9.23 | 38.5% | 6.7 |
| flashinfer_cudacore | 33.28 | 1.833배 | 8.69 | 36.8% | 370.8 |
| flashinfer | 48.51 | 1.257배 | 23.87 | 61.6% | 432.4 |

균일 길이(`graduation_uniform`):

| 정책 | TPOT(ms) | 배율 | attention ms/step | attention 비중 | 정책 비용 µs/step |
|---|---:|---:|---:|---:|---:|
| heuristic | 27.45 | 기준 | 3.58 | 19.9% | 3.4 |
| table | 27.43 | 1.001배 | 3.58 | 19.9% | 6.5 |
| flashinfer_cudacore | 27.66 | 0.992배 | 3.47 | 19.0% | 291.0 |
| flashinfer | 27.57 | 0.996배 | 3.40 | 18.7% | 337.7 |

요청 도착(`graduation_arrivals`):

| 정책 | TPOT(ms) | 배율 | attention ms/step | attention 비중 | 정책 비용 µs/step |
|---|---:|---:|---:|---:|---:|
| heuristic | 59.18 | 기준 | 10.58 | 43.0% | 3.0 |
| table | 50.46 | 1.173배 | 5.04 | 26.3% | 7.4 |
| flashinfer_cudacore | 50.12 | 1.181배 | 4.78 | 25.0% | 284.2 |
| flashinfer | 54.18 | 1.092배 | 7.09 | 33.1% | 315.5 |

출력(휴리스틱과의 토큰 비교). table 정책은 세 시나리오 모두 불일치 토큰 0개, 분기 사건 0건이다. FlashInfer 정책은 휴리스틱과 토큰이 같지 않다(다른 커널의 반올림). 사건 수는 서로 다른 위치를 센 것이고, "끝까지 갈라진 사건"은 처음 갈린 뒤 생성이 끝날 때까지 다시 합류하지 않은 사건이다. 불일치 토큰은 반복 3회 중 최대값이다.

| 시나리오 · 정책 | 불일치 토큰 | 비교한 토큰 | 분기 사건 | tie_1ulp | tie_2ulp | clear | 끝까지 갈라진 사건 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ragged · flashinfer_cudacore | 65 | 2,048 | 4 | 3 | 0 | 1 | 1 |
| ragged · flashinfer | 92 | 2,048 | 4 | 3 | 0 | 1 | 0 |
| uniform · flashinfer_cudacore | 223 | 2,048 | 7 | 4 | 2 | 1 | 2 |
| uniform · flashinfer | 92 | 2,048 | 4 | 3 | 0 | 1 | 0 |
| arrivals · flashinfer_cudacore | 39 | 1,192 | 6 | 6 | 0 | 0 | 1 |
| arrivals · flashinfer | 53 | 1,192 | 7 | 7 | 0 | 0 | 0 |

여섯 쌍 모두 세 반복에서 사건의 자리가 같다. 미분류·미재현·토큰 누락·사건 밖 불일치는 모두 0이다.

### 읽는 법

- **혼합 길이: 엔진 안에서도 "분할 수를 밖에서 고르면 커널 교체의 이득을 거의 다 얻는다".** CUDA-core 변형 정책은 TPOT 33.28 ms로 FA2 table(33.70 ms)과 같은 수준이다(1.833배 대 1.810배). attention 시간은 8.69 대 9.23 ms/step으로 FlashInfer 쪽이 낮지만, 거기에 step마다 `plan()` 370.8 µs가 얹혀 TPOT 이득은 그만큼 줄어든다. 커널 격자의 결론(§2, [10월 2일 노트](2026-10-02-hybrid-validation.md) §8, PLAN §2 4번)을 전체 모델에서 확인한 셈이다. 반면 FlashInfer의 기본값(tensor-core)은 1.257배에 그친다 — table이 얻은 1.810배의 이득 대부분을 잃고, attention이 23.87 ms/step으로 step의 61.6%를 차지한다. §2의 "정적 기본값이 혼합 길이에서 틀린다"가 전체 모델 TPOT에서 이렇게 나타난다.
- **균일 길이: 네 정책이 ±1% 안에 있다.** table 1.001배, CUDA-core 0.992배, tensor-core 0.996배. FlashInfer 커널은 step당 attention이 조금 빠르지만(3.47·3.40 ms 대 3.58 ms) `plan()` 비용(291.0·337.7 µs/step)이 그 절감보다 커서 배율이 1 아래로 내려간다. 이 측정에서 FlashInfer 쪽 이득은 길이가 섞인 배치에서만 나타나고, 균일한 배치에서는 `plan()` 비용이 남는다. FlashInfer의 `plan()`은 매 step 수백 µs로, 분할 수만 고르는 정책의 선택 비용(몇 µs)과 자릿수가 다르다.
- **요청 도착: 순서가 혼합 길이와 같다.** CUDA-core 1.181배 ≈ table 1.173배, tensor-core 1.092배.
- **출력은 같지 않지만 갈림은 대부분 동점 수준이다.** FlashInfer 정책의 분기 사건은 대부분 tie_1ulp(참조 1위와 후보의 로짓 차이가 bf16 간격 1개 이내)이고, 나머지는 균일 길이 CUDA-core의 tie_2ulp 2건(간격 2개)과 아래 4b의 clear 한 자리(혼합·균일 길이 두 정책)다. 불일치 토큰 수(최대 223)는 사건이 끌고 간 꼬리를 센 것이라 사건 수가 비교 단위로 낫다: 균일 길이 CUDA-core의 223개에서는 사건 7건 가운데 끝까지 갈라진 2건(둘 다 tie_2ulp)의 꼬리가 큰 몫을 차지한다. table 정책은 이 세 시나리오에서는 토큰이 모두 같았다(64K에서는 갈렸다, §3b). 이 비교는 합성 무작위 프롬프트의 greedy 생성에서의 것이고, 실제 질의에서의 품질 차이는 측정하지 않았다.

### 4b. clear 사건 한 자리와 FlashInfer 커널 탐침

세 시나리오·두 백엔드를 통틀어 clear는 이 한 자리뿐이다: 요청 8 위치 41, 참조 토큰 25 대 후보 토큰 198, 참조의 1위 로짓 33.75에서 후보까지 여유 17.5 (= 70 ulp). 교사 강제 진단의 로짓 최대 차이는 혼합 길이 tensor-core 22.9 · CUDA-core 22.3, 균일 길이 tensor-core 22.9 · CUDA-core 21.6이다. 이 사건은 10월 2일 대조군의 `clear`와 같은 사건이다([노트](2026-10-02-hybrid-validation.md) §6) — 같은 요청·위치·두 토큰·여유이고, 요청 8의 합성 프롬프트는 같다(세 시나리오와 대조군의 해시·길이가 일치). 그래서 혼합 길이와 균일 길이에서 모두 나타나고, 도착 시나리오의 요청 8은 24 토큰만 생성하므로 이 위치가 없다. 해석도 10월 2일 노트 §6과 같다: 1·2위 로짓이 가까운 동점이 아니라, 앞 문맥의 어느 자리를 이어 쓸지 정하는 attention 점수의 동점이 반올림 차이로 갈린 것으로 읽는다(이 캠페인에서 새로 검증한 것은 아니다). 규칙대로 조사 대상이므로, 같은 KV 상태의 커널 탐침으로 판정했다(`scripts/probe_split_kernel.py --campaign ../kernelscope/results/serve_4090/flashinfer_20261006 --run ragged --target-step 40 --target-rid 8 --candidate-backend <백엔드> --kv-gib 10`, 36층 fp32 참조, 2026-10-08 16:37·16:41). 사건이 난 step의 KV 상태에서 층마다 FA2 분할 1과 FlashInfer 백엔드의 attention 출력을 fp32 참조와 비교하고, 그 한 step만 FlashInfer로 계산한 로짓을 비교한다.

| 항목 | flashinfer_cudacore | flashinfer(tensor-core) |
|---|---:|---:|
| 탐침한 층 수 | 36 | 36 |
| attention 출력의 fp32 참조 대비 최대 오차 | 0.167 | 0.120 |
| FA2 분할 1의 같은 오차 | 0.134 | 0.134 |
| 분할 1과의 층별 최대 차이 | 0.25 | 0.25 |
| 그 차이가 최대인 층 | 34 | 34 |
| 그 층의 최대 출력 | 33.13 | 33.13 |
| 차이가 그 층 출력의 bf16 간격 1개를 넘는 층 | 0 | 0 |
| 단일 step에서 비교한 요청 수 | 32 | 32 |
| 요청 8의 참조 1·2위 로짓 차이 | 11.875 | 11.875 |
| 요청 8의 argmax 토큰 | 25 | 25 |
| argmax가 바뀐 요청 수 | 0 | 0 |
| 요청 8의 로짓 최대 차이 | 0.3125 | 1.9688 |
| 요청 8의 로짓 코사인 | 0.9998 | 0.9903 |
| 배치 전체의 로짓 최대 차이 | 2.625 | 2.969 |
| 요청별 로짓 최대 차이의 중앙값 | 0.117 | 0.125 |

**탐침 결과 — 두 백엔드 모두 커널 오류 아님.** 분할 1과의 층별 차이는 최대 0.25로, 그 층의 최대 출력 33.13의 bf16 간격(0.25) 하나이고 36층 모두 기준 안이다(넘는 층 0개). fp32 참조 대비 최대 오차는 CUDA-core 0.167, tensor-core 0.120으로 FA2 분할 1의 0.134와 같은 크기다. 같은 KV 상태에서 그 한 step만 FlashInfer로 계산하면 요청 8의 argmax는 25로 유지되고(참조 1·2위 로짓 차이 11.875) 32개 요청 모두 argmax가 바뀌지 않는다. 즉 두 커널은 같은 입력에서 FA2와 같은 답을 낸다. 그렇다면 교사 강제 진단의 22.9 · 22.3 · 21.6은 한 step의 차이(요청 8에서 0.3125 · 1.9688)가 아니라, 각 실행이 그 위치까지 자기 KV 캐시에 쌓은 반올림 차이가 이 step에서 증폭된 것이다 — 10월 2일 대조군과 같은 구도다. tensor-core의 한 step 로짓 차이가 더 큰 것(1.9688 대 0.3125)은 층별 최대 오차의 순서(0.120 대 0.167)와 반대인데, 이 탐침으로는 그 까닭을 가리지 않았다. §3c의 64K 사건과는 다르다: 거기서는 단일 step 자체가 argmax를 뒤집었고, 여기서는 뒤집지 않는다.

한계: 한 모델·한 GPU·합성 무작위 프롬프트(긴 요청 하나가 끼는 극단 배치)의 측정이고, 도착 시나리오는 논리 step 기준 도착이다. FlashInfer 정책의 TPOT에는 `plan()` 비용이 포함된다.

## 5. 라이브러리 무관 선택표 — 정책이 분할 수와 백엔드를 함께 고를 때 (Qwen3-4B, 2026-10-08 16:44~16:55)

§4의 FlashInfer 정책은 백엔드를 통째로 고정했다. 여기서는 선택표 정책이 FA2 분할 수와 FlashInfer 변형을 한 후보 목록에 놓고 step마다 함께 고르게 했다([설계](../plan/2026-10-08-design-anytable-policy.md)). `TablePolicy`는 가장 가까운 셀에서 후보(그 셀의 최선 FA2 분할, FlashInfer tensor-core, FlashInfer CUDA-core)의 비용을 비교하고, 엔진은 고른 백엔드를 `steps.parquet`의 `attention` 열에 기록한다. 원본은 `serve_4090/anytable_20261008/{ragged,uniform,arrivals}`, 번들은 `demo_data/serve_4090/anytable_20261008/`.

### 방법

**표.** `demo_data/dispatch_paged_cold_any.csv`는 FA2 분할만 담은 기존 표와 달리 §2의 여섯 32K 격자(혼합·균일 × FA2 분할·FlashInfer tensor-core·FlashInfer CUDA-core, cold)를 `static_default_losses`로 합친 것이고, 번들의 격자에서 다시 계산한 표와 같다. 세 변형을 모두 잰 211셀 각각에 후보별 커널 시간과 전체 최선이 있다. 전체 최선은 CUDA-core 148셀, tensor-core 45셀, FA2 분할 18셀이고, 혼합 길이 162셀만 보면 CUDA-core 139셀, tensor-core 21셀, FA2 2셀이다. 그 CUDA-core 최선 셀에서도 최선 FA2 분할보다 빠른 폭은 커널 호출당 중앙값 9.8 µs, 최대 39 µs다.

**정책.** 비교하는 정책은 넷이다: `heuristic`, `table:demo_data/dispatch_paged_cold.csv`(§4와 같은 FA2 전용 표), `table_any:demo_data/dispatch_paged_cold_any.csv`(후보의 커널 시간만 보는 전체 최선; 비용 = 층 수 × 커널 시간), `table_any:demo_data/dispatch_paged_cold_any.csv:371`(이름 `table_any_p371`; FlashInfer 후보에는 step마다 `plan()` 시간 371 µs를 더한다). 371은 §4 혼합 길이에서 CUDA-core 정책이 잰 step당 `plan()` 370.8 µs를 반올림한 값이다. `plan()` 비용을 더하는 옵션은 TODO에 없던 것을 이 작업에서 보탠 것이고, 기본값 0은 TODO의 정의(커널 시간만 보는 전체 최선)와 같다.

격자 숫자로 미리 계산하면 이득이 작다. 혼합 길이 실생성이 쓰는 셀(32768×1 + 512×31)에서는 최선 FA2 분할(16)이 279.4 µs, CUDA-core가 269.9 µs로 호출당 9.4 µs 차이이고, Qwen3-4B는 36층이라 step당 340 µs다. 이는 `plan()` 비용에 못 미치므로, 비용을 더한 표는 이 셀에서 FA2 쪽을 step당 31 µs 싸게 본다.

**조건.** 규약은 §4와 같다: 정책 순서를 반복마다 회전하고, `--policy-cache keep`, warm-up 뒤 반복 3회, KV 10 GiB. 시나리오는 `graduation_ragged`·`graduation_uniform`·`graduation_arrivals`(§4와 같다)이고 정책은 위 넷이다. 코드는 커밋 b8e1707에 미커밋 변경이 얹힌 상태이고, 실행 동안 이 캠페인 말고 GPU에 올라 있던 것은 §4와 같은 상주 rerun 뷰어(436 MiB) 하나뿐이다(30초 간격 nvidia-smi 기록). 출력은 §4와 같은 규칙으로 `scripts/classify_divergence`가 분류한다. 교사 강제 진단은 혼합 길이와 요청 도착에서 `table_any`의 두 정책에 대해서만 돌렸다: 균일 길이와 `table`은 토큰이 휴리스틱과 같아 분류할 사건이 없다. 정책의 결정은 CPU에서 재현된다: 기록한 step 길이로 세 table 정책을 다시 돌리면 기록된 백엔드·분할과 다른 step이 0개다.

### 결과

step당 시간. 열은 §4와 같고, `table_any`의 정책 비용에는 step마다의 `plan()` 시간이 들어 있다. 배율은 휴리스틱 대비다.

혼합 길이(`graduation_ragged`):

| 정책 | TPOT(ms) | 배율 | attention ms/step | attention 비중 | 정책 비용 µs/step |
|---|---:|---:|---:|---:|---:|
| heuristic | 60.90 | 기준 | 36.81 | 72.0% | 3.9 |
| table | 33.64 | 1.810배 | 9.23 | 38.6% | 7.5 |
| table_any | 33.26 | 1.831배 | 8.69 | 36.9% | 382.2 |
| table_any_p371 | 33.71 | 1.806배 | 9.23 | 38.6% | 7.1 |

균일 길이(`graduation_uniform`):

| 정책 | TPOT(ms) | 배율 | attention ms/step | attention 비중 | 정책 비용 µs/step |
|---|---:|---:|---:|---:|---:|
| heuristic | 27.44 | 기준 | 3.58 | 19.9% | 3.6 |
| table | 27.40 | 1.001배 | 3.58 | 19.9% | 6.7 |
| table_any | 27.42 | 1.001배 | 3.58 | 19.9% | 6.6 |
| table_any_p371 | 27.42 | 1.001배 | 3.58 | 19.9% | 6.8 |

요청 도착(`graduation_arrivals`):

| 정책 | TPOT(ms) | 배율 | attention ms/step | attention 비중 | 정책 비용 µs/step |
|---|---:|---:|---:|---:|---:|
| heuristic | 59.08 | 기준 | 10.56 | 43.1% | 2.9 |
| table | 50.32 | 1.174배 | 5.04 | 26.3% | 7.8 |
| table_any | 50.14 | 1.178배 | 4.79 | 25.0% | 291.8 |
| table_any_p371 | 50.42 | 1.172배 | 5.04 | 26.3% | 7.8 |

step별 선택. `steps.parquet`의 `attention`·`num_splits`로 센 step 수다(반복당 63 step이고 세 반복이 같다). FA2 분할 0은 FA2 자체 휴리스틱이고, CUDA-core로 돈 step의 `num_splits`는 쓰이지 않는다.

| 시나리오 · 정책 | CUDA-core step | FA2 step | FA2 분할 0 | FA2 분할 8 | FA2 분할 16 |
|---|---:|---:|---:|---:|---:|
| ragged · table | 0 | 63 | 0 | 0 | 63 |
| ragged · table_any | 63 | 0 | 0 | 0 | 0 |
| ragged · table_any_p371 | 0 | 63 | 0 | 0 | 63 |
| uniform · table | 0 | 63 | 63 | 0 | 0 |
| uniform · table_any | 0 | 63 | 63 | 0 | 0 |
| uniform · table_any_p371 | 0 | 63 | 63 | 0 | 0 |
| arrivals · table | 0 | 63 | 0 | 41 | 22 |
| arrivals · table_any | 63 | 0 | 0 | 0 | 0 |
| arrivals · table_any_p371 | 0 | 63 | 0 | 41 | 22 |

출력(휴리스틱과의 토큰 비교). `table`과 `table_any_p371`은 세 시나리오 모두 불일치 토큰 0개, 분기 사건 0건이다. `table_any`는 FA2가 아닌 커널을 고른 시나리오에서만 갈린다. 열의 뜻은 §4와 같고 불일치 토큰은 반복 중 최대값이다.

| 시나리오 · 정책 | 불일치 토큰 | 비교한 토큰 | 분기 사건 | tie_1ulp | tie_2ulp | clear | 끝까지 갈라진 사건 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ragged · table_any | 65 | 2,048 | 4 | 3 | 0 | 1 | 1 |
| uniform · table_any | 0 | 2,048 | 0 | 0 | 0 | 0 | 0 |
| arrivals · table_any | 39 | 1,192 | 6 | 6 | 0 | 0 | 1 |

### 읽는 법

- **혼합 길이: 커널 시간만 보는 표(`table_any`)는 §4의 고정 CUDA-core 정책과 같은 자리에 선다.** 모든 step에서 CUDA-core를 골랐고, TPOT는 §4의 CUDA-core 정책(33.28 ms, 1.833배)과 같은 수준이며 FA2 `table`보다 1.15% 짧다. attention 시간은 `table`보다 0.54 ms/step 짧은데 `plan()`이 얹히고(정책 비용 열), TPOT 이득 0.38 ms는 attention 절감에서 `plan()` 시간을 뺀 값 0.16 ms보다 크다. 이 여분이 `plan()` 시간이 step 시간에 그대로 더해지지 않아서인지, FlashInfer 경로에서 attention 밖의 시간이 짧아서인지는 이 측정으로 가리지 않았다. 세 반복 모두 `table_any`가 `table`보다 빨랐다.
- **`plan()` 비용을 더한 표(`table_any_p371`)는 FA2로 돌아가 `table`을 재현한다.** 혼합 길이 셀에서 격자가 본 커널 이득이 `plan()` 비용에 못 미쳐 모든 step에서 FA2 분할 16을 골랐고, 이는 `table`과 같은 선택이며 토큰도 같다. TPOT는 `table`보다 0.21% 길 뿐인데, 두 정책이 같은 커널·분할을 골랐으므로 그 차이를 이 측정의 편차 크기로 읽는다. 다만 이 셀에서 비용 모델의 판정은 틀렸다: FA2 쪽이 싸다고 보았지만 실측은 `table_any`가 앞섰고, 엔진 안의 attention 절감은 층당 15.1 µs로 격자가 본 호출당 차이보다 컸다. 세 정책의 TPOT 차이는 1.36% 이내다.
- **균일 길이: 세 표가 모두 FA2 기본값을 고른다.** 가장 가까운 셀(B=32, 길이 512)의 전체 최선이 FA2 기본값이라 `table_any`도 `table_any_p371`도 모든 step에서 FA2 기본값을 골랐다. 그 셀의 커널 시간은 FA2 103.6 µs, tensor-core 104.2 µs, CUDA-core 105.6 µs다. 세 정책의 TPOT 차이는 0.07% 이내다.
- **요청 도착: 순서가 혼합 길이와 같다.** `table_any`는 모든 step에서 CUDA-core를 골라 `table`보다 0.36% 빠르고 세 반복 모두 앞섰지만, 같은 선택을 한 `table_any_p371`과 `table`의 차이가 0.19%이므로 이 이득은 측정 편차와 구분하기 어렵다. `table_any_p371`은 `table`과 step마다 같은 분할을 골라 `table`을 재현한다. 세 정책의 TPOT 차이는 0.55% 이내다.
- **출력: 라이브러리를 바꾼 시나리오에서만 토큰이 갈리고, 그 갈림은 §4의 것과 같다.** `table_any`의 토큰 이력은 혼합 길이와 요청 도착에서 §4의 `flashinfer_cudacore` 정책과 반복 3회 모두 한 토큰도 다르지 않고(같은 커널, 같은 입력, 결정적), 분기 사건의 자리·토큰·분류도 같다. 그래서 혼합 길이의 clear 한 자리는 §4b의 그 사건이다: 요청 8 위치 41, 여유 70 ulp, 교사 강제 로짓 최대 차이 22.3. 그 사건의 커널 탐침이 이미 구현 오류가 아님을 보였으므로(§4b) 여기서는 다시 탐침하지 않았다. 균일 길이의 `table_any`는 FA2를 골라 토큰이 휴리스틱과 같고, `table_any_p371`의 토큰 이력은 세 시나리오에서 `table`과 한 토큰도 다르지 않다. 미분류·미재현·토큰 누락·사건 밖 불일치는 모두 0이고 사건의 자리는 세 반복에서 같다.
- **PLAN §2 4번·8번에는 이렇게 읽힌다.** 분할 수를 밖에서 고르는 정책(`table`, `table_any_p371`)이 커널 교체 이득을 거의 다 얻는다. 정책이 라이브러리까지 고르게 하면 혼합 길이에서 이득이 조금 더 생기고 요청 도착에서는 편차와 구분되지 않으며, 그 대가는 FA2 기준과의 토큰 동일성이다. 이득을 가르는 것은 `plan()` 비용이다: 커널 이득과 같은 자릿수라 비용 모델이 어느 쪽이 나은지 가르지 못했고, 한 번 계획한 결과를 step 사이에 재사용하는 방법은 시도하지 않았다(성립하면 균형이 바뀐다).
- **설계의 기대와 비교.** [설계 §4](../plan/2026-10-08-design-anytable-policy.md)의 기대는 절반만 맞았다. `table_any_p371`이 FA2 분할로 돌아가 `table`과 같아진다는 것은 맞았다. 커널 이득과 `plan()` 비용이 상쇄되어 `table_any`가 `table`과 같거나 조금 느리리라는 것은 틀렸다: 혼합 길이에서 빨랐고, 설계 §0의 산수는 엔진 안의 attention 절감을 격자보다 작게 보았다.

한계: 한 모델·한 GPU·합성 무작위 프롬프트의 측정이고, 표는 32K 격자의 가장 가까운 셀을 쓴다. 371은 혼합 길이에서 잰 `plan()` 시간 하나이고 시나리오마다 다르다(§4 표의 정책 비용 열). 반복이 세 번뿐이라 같은 선택을 한 정책 사이의 차이보다 작은 차이는 구분하지 못한다.

## 재현 명령

```bash
# §1 트레이스 재생(GPU 없이, 한 구성 1~2분). 트레이스는 공개 저장소에서 받는다.
curl -L -o traces/AzureLLMInferenceTrace_conv.csv \
  https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv
PYTHONPATH=. .venv/bin/python -m scripts.traffic_replay --trace traces/AzureLLMInferenceTrace_conv.csv \
  --rate-scale 1 --out ../kernelscope/results/traffic/azure_conv_x1
# 다른 구성: --rate-scale 0.5|2, --kv-tokens 436900, --step-ms 50, --prompt-scale 2|4(what-if),
#   BurstGPT: --trace BurstGPT_without_fails_1.csv --window-s 604800 --rate-scale 100 [--model GPT-4 --window-s 1209600 --rate-scale 300]
# §2 정적 기본값 표(GPU 없이, 번들에서)
PYTHONPATH=. .venv/bin/python -c "from kernelscope.results.store import load_dirs; from kernelscope.analysis.dispatch import *; \
  import pathlib; d=pathlib.Path('demo_data/hw_4090'); t=static_default_losses(load_dirs([d/g for g in ('ragged_s1_paged','ragged_s1_flashinfer','ragged_s1_flashinfer_cudacore','uniform_s1_paged','uniform_s1_flashinfer','uniform_s1_flashinfer_cudacore')])); print(static_default_summary(t))"
# §3b 64K 실생성(GPU, KV 13 GiB)과 §3c 교사 강제 진단
.venv/bin/python -m kernelscope.cli serve run --model Qwen/Qwen3-4B-Instruct-2507 --scenario scenarios/graduation_ragged_64k.yaml \
  --policy heuristic --policy table:demo_data/dispatch_paged_cold.csv --policy model --policy hybrid:demo_data/dispatch_paged_cold.csv:0.2 \
  --machine machines/rtx4090.json --params models/rtx4090.json --kv-gib 13 --warmup-runs 1 --warmup-steps 2 --repeats 3 --seed 0 \
  --policy-cache keep --out ../kernelscope/results/serve_4090/longctx_20261006/ragged_64k
.venv/bin/python -m scripts.check_policy_numerics --scenario scenarios/graduation_ragged_64k.yaml --reference heuristic \
  --policy table:demo_data/dispatch_paged_cold.csv --policy model --policy hybrid:demo_data/dispatch_paged_cold.csv:0.2 \
  --machine machines/rtx4090.json --params models/rtx4090.json --steps 64 --logit-steps 64 --kv-gib 13 \
  --out ../kernelscope/results/serve_4090/longctx_20261006/numerics/ragged_64k
.venv/bin/python -m scripts.classify_divergence --campaign ../kernelscope/results/serve_4090/longctx_20261006
# §3 64K 격자(GPU)
for g in ragged uniform; do
  .venv/bin/python -m kernelscope.cli bench --grid grids/${g}_s1_64k.yaml --cache-state cold --results ../kernelscope/results/hw_4090/${g}_s1_64k_paged \
    --plugins fa2_paged,flashdecoding_paged,fd_s2_paged,fd_s4_paged,fd_s8_paged,fd_s16_paged,fd_s32_paged,fd_s64_paged,fd_s128_paged
  .venv/bin/python -m kernelscope.cli bench --grid grids/${g}_s1_64k.yaml --cache-state cold --results ../kernelscope/results/hw_4090/${g}_s1_64k_flashinfer --plugins flashinfer_paged
  .venv/bin/python -m kernelscope.cli bench --grid grids/${g}_s1_64k.yaml --cache-state cold --results ../kernelscope/results/hw_4090/${g}_s1_64k_flashinfer_cudacore --plugins flashinfer_paged_cudacore
done
# §4 FlashInfer를 엔진 안에서(GPU, 유휴 확인 뒤 약 20분). serve run의 종료 코드 1은 FlashInfer 정책의 토큰이 휴리스틱과 달라
# strict equivalence가 실패한 정상 결과이므로 무시하고(|| true) 결과 폴더를 지우지 않는다. 교사 강제 진단은 시나리오마다 따로 돌린다.
for s in ragged uniform arrivals; do
  .venv/bin/python -m kernelscope.cli serve run --model Qwen/Qwen3-4B-Instruct-2507 --scenario scenarios/graduation_$s.yaml \
    --policy heuristic --policy table:demo_data/dispatch_paged_cold.csv --policy flashinfer --policy flashinfer_cudacore \
    --machine machines/rtx4090.json --params models/rtx4090.json --kv-gib 10 --warmup-runs 1 --warmup-steps 2 --repeats 3 --seed 0 \
    --policy-cache keep --out ../kernelscope/results/serve_4090/flashinfer_20261006/$s || true
  .venv/bin/python -m scripts.check_policy_numerics --scenario scenarios/graduation_$s.yaml --reference heuristic \
    --policy table:demo_data/dispatch_paged_cold.csv --policy flashinfer --policy flashinfer_cudacore \
    --steps 64 --logit-steps 64 --kv-gib 10 --out ../kernelscope/results/serve_4090/flashinfer_20261006/numerics/$s
done
.venv/bin/python -m scripts.classify_divergence --campaign ../kernelscope/results/serve_4090/flashinfer_20261006
# §4b clear 사건(요청 8의 위치 41 = decode step 40의 출력)의 커널 탐침, 백엔드마다
for b in flashinfer_cudacore flashinfer; do
  .venv/bin/python -m scripts.probe_split_kernel --campaign ../kernelscope/results/serve_4090/flashinfer_20261006 --run ragged \
    --target-step 40 --target-rid 8 --candidate-backend $b --kv-gib 10
done
# §5 라이브러리 무관 선택표(GPU, 유휴 확인 뒤 약 11분). 표(demo_data/dispatch_paged_cold_any.csv)는 scripts/package_demo.py가 여섯 32K 격자로 만든다.
# 371 = §4 혼합 길이 flashinfer_cudacore의 policy_us_per_step(370.8)을 반올림한 값. serve run의 종료 코드 1은 table_any의 토큰이 휴리스틱과 달라서이므로 무시한다.
for s in ragged uniform arrivals; do
  .venv/bin/python -m kernelscope.cli serve run --model Qwen/Qwen3-4B-Instruct-2507 --scenario scenarios/graduation_$s.yaml \
    --policy heuristic --policy table:demo_data/dispatch_paged_cold.csv \
    --policy table_any:demo_data/dispatch_paged_cold_any.csv --policy table_any:demo_data/dispatch_paged_cold_any.csv:371 \
    --machine machines/rtx4090.json --params models/rtx4090.json --kv-gib 10 --warmup-runs 1 --warmup-steps 2 --repeats 3 --seed 0 \
    --policy-cache keep --out ../kernelscope/results/serve_4090/anytable_20261008/$s || true
done
# 교사 강제 진단은 table_any가 FA2가 아닌 커널을 고른 혼합 길이와 요청 도착에서만 돌린다(균일 길이는 토큰이 휴리스틱과 같다).
for s in ragged arrivals; do
  .venv/bin/python -m scripts.check_policy_numerics --scenario scenarios/graduation_$s.yaml --reference heuristic \
    --policy table_any:demo_data/dispatch_paged_cold_any.csv --policy table_any:demo_data/dispatch_paged_cold_any.csv:371 \
    --steps 64 --logit-steps 64 --kv-gib 10 --out ../kernelscope/results/serve_4090/anytable_20261008/numerics/$s
done
CUDA_VISIBLE_DEVICES= .venv/bin/python -m scripts.classify_divergence --campaign ../kernelscope/results/serve_4090/anytable_20261008
```
