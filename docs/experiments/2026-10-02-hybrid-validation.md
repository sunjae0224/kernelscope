# 혼합 정책(δ=0.2)의 실제 LLM 생성 검증과 분기 사건 분류 (2026-10-02)

**결론.** 요청한 세 조건(`graduation_ragged`·`graduation_uniform`·`graduation_arrivals`, Qwen3-4B, 3회 반복)에서 혼합 정책은 **모든 decode step에서 측정 테이블 정책과 같은 분할 수를 골랐다**(다르게 고른 step 0/567). 세 조건이 측정 테이블의 셀과 정확히 겹치기 때문이다(이웃 셀 거리 ≤ 0.10). 따라서 attention 시간은 테이블 정책과 같고, TPOT 차이는 선택 비용뿐이다: 혼합 길이 **61.11 → 34.09ms(1.793배**; 테이블 33.82ms, 1.807배), 요청 도착 1.146배(테이블 1.164배), 균일 0.993배(테이블 1.004배). 생성 토큰은 세 조건 아홉 실행 모두 휴리스틱과 같았다(분기 사건 0건, 교사 강제 진단에서 뒤바뀐 위치 0개). 요청하지 않은 보충 실행(held-out 자연어 요청 도착)에서는 혼합 정책이 55 step 중 12 step에서 테이블과 다르게(모델 순위대로) 골라 attention 시간이 테이블보다 줄었지만(3.31 → 3.00 ms/step), 매 실행 정책 캐시를 비우는 측정 규약에서는 첫 결정 비용(1561.1 µs/step)이 그 이득을 넘어 TPOT 배율은 1.020배(테이블 1.085배, 모델 1.028배)에 그쳤다. **2026-10-06 추가(§3b):** 정책 캐시를 실행 간에 유지하는 규약(`--policy-cache keep`)으로 같은 조건을 다시 재면 첫 결정 비용이 사라져 혼합 정책 TPOT 31.62ms, **1.090배**(테이블 1.086배, 모델 1.091배)가 되어 결론이 바뀐다: 혼합 정책의 더 나은 커널 선택은 TPOT로도 드러나며, 10월 2일의 1.020배는 규약의 첫 결정 비용이었다. **FlashInfer 비교(§8):** 커널 격자에서 FlashInfer의 CUDA-core 변형은 최선 FA2 분할과 같은 수준(혼합 길이 162셀 중앙값 0.977배, 최대 1.003배)이고 기본 tensor-core 변형은 가장 어려운 셀에서 2.401배 느리다 — 외부에서 분할 수만 고르는 선택이 커널 교체의 이득을 거의 전부 얻는다. 토큰 불일치는 새 지표로 보고한다: held-out 사건 정책별 2건, 6건이 모두 `tie_1ulp`; 대조군(균일, fixed:8) 분기 사건 6건 중 5건 `tie_1ulp`, **`clear` 1건** — 같은 KV 상태의 커널 수준 탐침에서 분할 8 출력은 분할 1과 float32 참조에 bf16 간격 하나 안에서 일치했고 argmax도 유지되므로 구현 오류가 아니라 실행이 각자 쌓은 KV 반올림 차이의 증폭이다(§6). 이 문서의 수치는 `kernelscope verify`의 `hybrid.*`·`divergence.*`·`flashinfer.*`·`vllm.*` 항목 64개로 재계산된다.

## 1. 방법

- **대상.** Qwen/Qwen3-4B-Instruct-2507(bf16), RTX 4090, commit `195e9e1`(작업 트리 clean 상태에서 실행, manifest의 `git` 항목). 정책: `heuristic`(라이브러리 기본), `table:demo_data/dispatch_paged_cold.csv`, `hybrid:demo_data/dispatch_paged_cold.csv:0.2`(모델 순위 + 이웃 실측 셀, [설계](2026-09-26-hybrid-policy.md)). 요청한 캠페인은 `graduation_20260922`와 같은 규약(`--kv-gib 10 --warmup-runs 1 --warmup-steps 2 --repeats 3 --seed 0`, 반복마다 정책 순서 회전, 측정 실행마다 정책 캐시 초기화).
- **보충 실행 1 — held-out 자연어 요청 도착.** GPU 없이 기록된 step 길이로 세 정책을 재생해 보면 요청한 세 조건과 `heldout_text_ragged`·`heldout_text_uniform`에서는 혼합 정책이 모든 step에서 테이블과 같은 답을 내고, `heldout_text_arrivals`에서만 갈린다. 혼합 정책이 테이블과 실제로 다른 선택을 하는 조건이 요청 범위에 없었으므로 이 조건을 `followup_fullwarmup_20260922`의 규약(`--kv-gib 4 --warmup-runs 1`, 전체 시나리오 warm-up, seed 0)으로 `heuristic`·`table`·`hybrid`·`model` 네 정책 3회 실행했다.
- **보충 실행 2 — 분기 사건 분류의 대조군.** 9월 22일 캠페인에서 유일하게 토큰이 갈렸던 조합(`graduation_uniform` × `fixed:8`)을 같은 규약으로 다시 실행해, 새 지표가 실제 사건을 어떻게 분류하는지 확인했다.
- **불일치 지표.** 토큰 일치율 대신 **분기 사건**(요청별 첫 불일치 위치, `kernelscope.serve.divergence.divergence_events`)을 세고, 각 사건을 교사 강제 진단(`scripts/check_policy_numerics.py`, 휴리스틱의 토큰 이력을 후보 정책에 넣고 로짓 비교, `--steps 64 --logit-steps 64`)으로 분류한다(`scripts/classify_divergence.py`). 분류는 기준(휴리스틱) 1위 로짓과 **후보가 실제로 고른 토큰**의 로짓 차이를 기준 1위 로짓 크기의 bf16 간격 단위로 잰 값이다: 1간격 이내 `tie_1ulp`, 2간격 이내 `tie_2ulp`, 그 밖 `clear`. 이번 진단부터 1위 로짓 값을 기록하므로 [9월 26일 분석](2026-09-26-mismatch-analysis.md)이 가정했던 로짓 범위(16~32)가 필요 없다. 교사 강제 진단이 그 위치에서 같은 토큰을 재현하지 못한 사건, 진단 행이 없는 사건, 사건 밖 토큰 불일치, 토큰 누락은 각각 따로 표시되며 하나라도 있으면 `classify_divergence`는 종료 코드 1을 돌려준다.

## 2. 요청한 캠페인: 세 조건 × 3회

| 조건 | 정책 | TPOT(ms) | 배율 | attention(ms/step) | 선택 비용(µs/step) | 테이블과 다른 step | 토큰 |
|---|---|---:|---:|---:|---:|---:|---|
| 혼합 길이 32K×1 + 512×31 | heuristic | 61.11 | 1.000 | 36.80 | 3.7 | – | 기준 |
| | table | 33.82 | 1.807배 | 9.22 | 7.8 | – | 일치 |
| | hybrid | 34.09 | 1.793배 | 9.24 | 245.8 | 0/63 | 일치 |
| 균일 512×32 | heuristic | 27.63 | 1.000 | 3.58 | 3.3 | – | 기준 |
| | table | 27.52 | 1.004배 | 3.59 | 7.4 | – | 일치 |
| | hybrid | 27.83 | 0.993배 | 3.58 | 317.4 | 0/63 | 일치 |
| 요청 도착 16K×1 + 512×47 | heuristic | 59.33 | 1.000 | 10.58 | 3.0 | – | 기준 |
| | table | 50.97 | 1.164배 | 5.04 | 12.4 | – | 일치 |
| | hybrid | 51.79 | 1.146배 | 5.04 | 751.8 | 0/63 | 일치 |

배율은 반복별로 짝지은 TPOT 평균비(부트스트랩 95% 구간은 `summary.csv`). 혼합 정책이 테이블과 다르게 고른 step은 세 조건 아홉 실행을 합쳐 **0/567**이다. 혼합 정책의 선택 비용은 캐시 미스(새 페이지 구성에서 모델을 처음 평가하는 일)에서 나온다: 실행당 캐시 미스는 혼합·균일·요청 도착 조건에서 각각 **1·1·5회**이고 캐시 적중 시 선택은 10 µs대라, step당 평균 비용 245.8 µs(혼합 길이)와 751.8 µs(요청 도착)는 거의 전부 첫 결정의 비용이다. 이 규약은 측정 실행마다 정책 캐시를 비우므로 혼합 정책에 가장 불리한 쪽이다. 같은 실행의 휴리스틱·테이블 TPOT는 2026-09-22 캠페인과 최대 **1.25%** 차이로 재현됐다.

## 3. 보충 실행 1: held-out 자연어 요청 도착 (hybrid ≠ table)

| 정책 | TPOT(ms) | 배율 | attention(ms/step) | step GPU 시간(ms) | 선택 비용(µs/step) | 테이블과 다른 step | 토큰 |
|---|---:|---:|---:|---:|---:|---:|---|
| heuristic | 34.44 | 1.000 | 4.85 | 19.10 | 3.1 | – | 기준 |
| table | 31.74 | 1.085배 | 3.31 | 17.64 | 23.5 | – | 사건 2 |
| hybrid | 33.77 | 1.020배 | 3.00 | 17.34 | 1561.1 | 12/55 | 사건 2 |
| model | 33.49 | 1.028배 | 2.93 | 17.22 | 1553.2 | – | 사건 2 |

혼합 정책은 55 step 중 **12/55**에서 모델 순위를 따랐고(나머지 43 step은 테이블), GPU 없이 기록된 길이로 재생하면 같은 선택이 나온다. 그 결과 attention 시간은 테이블(3.31)보다 짧고 모델(2.93)에 가까운 **3.00** ms/step, step GPU 시간도 테이블보다 짧다. 그러나 이 조건은 요청이 네 번에 걸쳐 도착해 페이지 구성이 자주 바뀌므로 실행당 캐시 미스가 **13회**이고, 혼합·모델 정책의 선택 비용 1.5 ms/step이 attention에서 번 0.3 ms/step을 넘는다. 즉 **혼합 정책의 커널 선택은 테이블보다 낫지만, 첫 결정 비용을 매 실행 지불하는 규약에서는 TPOT로 드러나지 않는다.** 선택 캐시를 실행 간에 유지하거나 모델 평가를 배경에서 미리 하는 것이 다음 과제다(§7).

## 3b. 같은 조건을 정책 캐시를 유지하는 규약으로 재측정 (2026-10-06)

§3의 규약은 측정 실행마다 정책 객체를 새로 만들어 첫 결정 비용을 매번 낸다(비관적 경계). `serve run --policy-cache keep`(2026-10-06 추가)은 warm-up 전에 만든 정책을 반복 간에 그대로 써서, warm-up이 본 페이지 구성은 모두 캐시 적중이 된다(같은 배치 구성이 되풀이되는 서빙 정상 상태의 낙관적 경계). 그 밖의 설정은 §3과 같다(`followup_fullwarmup` 규약, seed 0, 3회). 번들 `demo_data/serve_4090/hybrid_keepcache_20261006/`. 이 실행의 코드는 195e9e1 위에 `--policy-cache` 옵션과 분류 도구가 더해진 미커밋 상태다(manifest의 `git.status`에 기록).

| 정책 | 규약 | TPOT(ms) | 배율 | attention(ms/step) | step GPU 시간(ms) | 선택 비용(µs/step) | 캐시 미스/실행 | 토큰 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| heuristic | fresh / keep | 34.44 / 34.46 | 1.000 | 4.85 / 4.86 | 19.10 / 19.09 | 3.1 / 3.4 | – | 기준 |
| table | fresh / keep | 31.74 / 31.73 | 1.085배 / 1.086배 | 3.31 / 3.30 | 17.64 / 17.60 | 23.5 / 6.1 | 13 / 0 | 사건 2 |
| hybrid | fresh / keep | 33.77 / **31.62ms** | 1.020배 / **1.090배** | 3.00 / 3.01 | 17.34 / 17.33 | 1561.1 / **14.95** | 13 / 0 | 사건 2 |
| model | fresh / keep | 33.49 / 31.58 | 1.028배 / 1.091배 | 2.93 / 2.93 | 17.22 / 17.22 | 1553.2 / 6.5 | 13 / 0 | 사건 2 |

캐시를 유지하면 혼합·모델 정책의 선택 비용은 15 µs/step 아래로 떨어지고(혼합 정책 캐시 미스 0회, 세 반복 합), attention·step GPU 시간은 §3과 같다. 그 결과 TPOT 순위가 뒤집힌다: 혼합 1.090배 ≈ 모델 1.091배 > 테이블 1.086배. 차이는 작지만(테이블 대비 0.1 ms/step) 방향은 attention 시간의 차이(3.31 → 3.01 ms/step)와 일치한다. 분기 사건은 규약과 무관하게 같다(캐시를 유지해도 사건 6건 모두 `tie_1ulp`, 교사 강제 재현). 두 규약은 실제 서빙의 양 끝이다: 요청 구성이 매 step 바뀌는 서비스에서는 캐시 미스가 fresh 쪽에, 같은 구성이 되풀이되는 재생 부하에서는 keep 쪽에 가깝다.

## 4. 분기 사건과 동점 분류 (`divergence.csv`)

| 실행 | 정책 | 불일치 토큰(최대) | 분기 사건 | tie_1ulp | tie_2ulp | clear | 미분류 | 교사 강제 미재현 | 반복마다 같은 위치 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 요청 3조건 | table, hybrid | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 예 |
| held-out 요청 도착 | table | 21/896 | 2 | 2 | 0 | 0 | 0 | 0 | 예 |
| | hybrid | 25/896 | 2 | 2 | 0 | 0 | 0 | 0 | 예 |
| | model | 25/896 | 2 | 2 | 0 | 0 | 0 | 0 | 예 |
| 대조군 균일 × fixed:8 | fixed8 | 114/2048 | 6 | 5 | 0 | 1 | 0 | 0 | 예 |

요청한 세 조건은 **분기 사건 0건**이고 교사 강제 진단에서도 **뒤바뀐 위치 0개**다. held-out 요청 도착의 사건은 **정책별 2건**(table: 요청 23 위치 25, 요청 24 위치 18; hybrid·model: 요청 21 위치 14, 요청 23 위치 25 — 9월 22일 후속 캠페인과 같은 위치)이며, 이번에 기록한 1위 로짓 값으로 분류하면 **6건이 모두 `tie_1ulp`**다. 1위 로짓이 31.25~38.75라 bf16 간격이 0.125~0.25인데 기준 1위와 후보 토큰의 차이가 정확히 간격 1개다(9월 26일 분석이 "간격 2개"로 분류했던 두 건은 로짓 범위 가정 때문이었고, 실측 범위에서는 1개다). 대조군은 **분기 사건 6건**으로 9월 22일과 위치가 같고(요청 2·4 위치 10, 요청 20·27 위치 9, 요청 31 위치 26, 요청 8 위치 41), 그중 **5건**은 `tie_1ulp`(차이 0 또는 간격 1개), 요청 8 위치 41의 1건이 **`clear` 1건**이다. 모든 사건이 교사 강제 진단에서 같은 토큰으로 재현됐다.

## 5. 교사 강제 진단 요약

| 실행 | 후보 정책 | 비교 위치 | argmax 뒤바뀜 | 로짓이 완전히 같은 위치 | 정책 간 로짓 최대 차이 |
|---|---|---:|---:|---:|---:|
| 혼합 길이 | table, hybrid(분할 16) | 2016 | 0 | 96.9% | 5.41 |
| 균일 | table, hybrid(분할 0 = 휴리스틱과 동일) | 2016 | 0 | 100% | 0 |
| 요청 도착 | table, hybrid(분할 8·16) | 1144 | 0 | 94.5% | 0.50 |
| held-out 요청 도착 | table, hybrid, model | 868 × 3 | 2 + 2 + 2 | 61.4% | 1.375 |
| 대조군 균일 | fixed8 | 2016 | 10 | 0% | 22.625 |

혼합 길이 조건의 로짓 최대 차이 **5.41**은 32K 토큰 요청(요청 0) 하나에서만, 그것도 뒤쪽 step(52~56)에서 나온다. 512토큰 요청 31개는 2016개 위치 중 **96.9%**에서 로짓이 비트 단위로 같고, 요청 0도 첫 step에서는 차이가 bf16 간격 1개(0.0625)다. 즉 분할 16 커널은 같은 KV 상태에서 같은 답을 내고, 차이는 두 실행이 각자 쌓은 KV 캐시의 반올림 차이가 36층과 수십 step을 거치며 증폭된 것이다. 합성 무작위 토큰 32K개 뒤에 같은 토큰을 반복 생성하는 이 요청은 입력 분포 밖의 상태라 민감도가 특히 크다. argmax는 모든 위치에서 같았다.

## 6. `clear` 사건의 조사 (대조군, 요청 8 위치 41)

사실관계. 기준(휴리스틱) 1위 토큰 25의 로짓은 33.75이고 2위와 11.9 차이인데, fixed:8은 기준에서 11위인 토큰 198을 골랐다. 기준 1위와 토큰 198의 로짓 차이는 17.5, 즉 bf16 간격 **70**개이고, 정책 간 로짓 최대 차이는 **22.625**(cosine 유사도 −0.17)다. 로짓 동점이 아니다. 교사 강제 진단에서 같은 사건이 재현됐고, 자유 생성 3회에서도 같은 위치에서 같은 토큰으로 갈렸다(결정적).

관찰. 같은 요청의 직전 step(위치 40)과 직후 step(위치 42)에서는 정책 간 로짓 차이가 0.40~0.41로 반올림 수준이고, 위치 41의 차이만 50배 크다. 같은 step의 다른 31개 요청도 argmax가 바뀌지 않았다. 요청 8의 생성문은 합성 무작위 프롬프트 뒤에 25개 토큰 주기로 같은 숫자열을 반복하는 상태(…, 51, 16, 17, **25**, 15, 15, 25, …)이고, fixed:8은 그 주기의 다른 자리(…, 57, **198**, 17, 15, …)로 건너뛰었다. 두 후보 모두 앞선 문맥에 실제로 등장한 이어쓰기다. 이는 1·2위 로짓이 가까운 동점이 아니라, **앞 문맥의 어느 자리를 복사할지 정하는 attention 점수의 동점**이 반올림 차이로 갈린 것으로 해석된다: 어느 쪽이 이기든 이긴 쪽의 로짓 여유는 크지만 선택 자체는 간격 하나에 달려 있다.

커널 수준 탐침(`numerics/control_uniform_fixed8/kernel_probe/`). 휴리스틱의 토큰 이력을 같은 KV 캐시 상태로 재생한 뒤 사건이 난 step에서 36개 층 각각의 attention 출력을 분할 1·분할 8·float32 참조로 비교했다. 두 분할의 출력은 float32 참조와 최대 **0.134** 차이(출력 크기 최대 33, 두 분할의 오차가 같음 — bf16 출력 반올림)이고 분할 1과 분할 8 출력끼리의 최대 차이는 **0.0625**로 bf16 간격 하나다. 같은 KV 상태에서 그 한 step만 분할 8로 계산한 로짓은 요청 8에서 최대 **1.08** 차이이고 argmax는 32개 요청 모두 유지된다(**바뀐 요청 0개**, 요청 8도 토큰 25). 즉 분할 8 커널은 같은 입력에서 같은 답을 내며, 22.625라는 차이는 fixed:8 실행이 40 step 동안 자기 KV 캐시에 쌓은 반올림 차이가 이 step에서 증폭된 것이다.

판정. 구현 오류가 아니다. 다만 "불일치는 로짓 동점에서만 난다"는 9월 26일의 서술은 이 사건으로 **틀렸다**: 불일치는 로짓 동점(`tie_1ulp`)뿐 아니라 로짓 위 단계의 동점(누적된 KV 차이가 attention의 선택을 바꾸는 경우)에서도 생기며, 후자는 `clear`로 보고된다. 규칙은 유지한다 — `clear`는 조사 대상이고, 구현 오류 여부는 같은 KV 상태의 커널 수준 탐침으로 판정한다. 이번에는 통과했다.

## 8. FlashInfer와의 커널 격자 비교 (2026-10-06)

[계획](../plan/2026-09-26-plan-flashinfer-comparison.md)의 커널 부분. 플러그인 `flashinfer_paged`(`BatchDecodeWithPagedKVCacheWrapper`, `use_tensor_cores=True`, 권장 설정)와 `flashinfer_paged_cudacore`(`use_tensor_cores=False`)를 같은 페이지 KV(page 256)·같은 입력으로 `ragged_s1`(혼합 길이 **162셀**)과 `dispatch_s1`(균일 **49셀**) 격자에서 cold 상태로 쟀다(flashinfer-python 0.6.13 + cu128 사전 빌드 JIT 캐시; 시스템 nvcc가 CUDA 10.1이라 JIT 컴파일은 실패했고 그 시도는 `hw_4090/*_failed_jit_20261006`에 남겼다). 참조 검사는 균일 격자 28셀에서 통과했고 큰 셀은 크기 제한으로 생략됐으며, 혼합 길이의 정확성은 GPU 테스트(`tests/test_flashinfer_plugin.py`)의 작은 혼합 셀로 확인했다. `plan()`은 셀마다 한 번 입력을 만들 때 호출되고 그 시간은 커널 시간에 들어가지 않는다 — plan 비용과 정책 선택 비용의 비교는 아직 하지 않았다.

| 격자 | FlashInfer tensor-core / 최선 FA2 분할 | FlashInfer CUDA-core / 최선 FA2 분할 | 더 빠른 변형 / 최선 FA2 | 휴리스틱 / 더 빠른 변형(최대) |
|---|---|---|---|---|
| 혼합 길이 162셀 | 중앙값 0.998배, 최대 2.401배 | 중앙값 **0.977배**, 최대 **1.003배** | 중앙값 0.975배 | 4.11배 |
| 균일 길이 49셀 | 중앙값 0.998배, 최대 1.053배 | 중앙값 1.004배, 최대 1.181배 | 중앙값 **0.998배** | 1.17배 |

최악 셀(32K×1 + 512×31, 페이지 KV): 휴리스틱 1071.8 µs, 최선 FA2 분할 `fd_s16_paged` 279.4 µs, FlashInfer CUDA-core **269.9** µs, FlashInfer tensor-core **670.9** µs. 즉 (1) 변형을 맞게 고른 FlashInfer는 모든 혼합 길이 셀에서 최선 FA2 분할과 같은 수준이고(최대 0.3% 느림, 중앙값 2.3% 빠름), (2) 라이브러리 기본 설정에 가까운 tensor-core 변형은 긴 요청이 끼고 배치가 큰 13셀에서 최선 FA2보다 1.5배 이상 느리므로 FlashInfer 쪽에도 "변형 선택"이라는 같은 종류의 문제가 있으며, (3) **분할 수만 밖에서 고르는 본 작품의 선택은 커널 교체(FlashInfer)의 이득을 사실상 전부 얻는다** — 계획 문서의 해석 규칙(격차가 외부 선택의 개선보다 작으면 "설정 선택만으로 커널 교체 이득의 대부분을 얻는다")을 만족한다. 전체 모델(`serve run`)에서의 FlashInfer 백엔드 비교와 plan 비용 측정은 남아 있다.

## 9. vLLM에서의 재현 — 측정만 (2026-10-06)

중간보고서 표 8의 "운영 엔진 재현" 항목. 프로젝트 환경과 분리한 vLLM 전용 환경(`~/.venvs/kernelscope-vllm`: vllm 0.31.0, torch 2.13 cu130)에서 `scripts/vllm_reproduce.py`로 같은 모델(Qwen3-4B, bf16)에 합성 토큰 프롬프트의 혼합 길이 배치(32K×1 + 512×31)와 균일 배치(512×32)를 넣고, 생성 길이 4와 64의 소요 시간 차로 step당 decode 시간을 추정했다(prefill 상쇄; prefix caching 끔; 측정 전 warm-up 1회; 3회 반복의 중앙값). 1차 시도(`docs/experiments/vllm/repro_20261006.json`)는 prefix cache와 warm-up을 처리하지 않아 첫 반복이 음수로 나왔고, 기록만 남긴다. 유효한 측정은 2차(eager), 3차(CUDA Graph 모드), 4차(eager, KV 블록 256)다.

| 실행 | 모드 | KV 블록 | 백엔드 | 혼합 길이 step(ms) | 균일 step(ms) |
|---|---|---|---|---:|---:|
| repro2 | eager | 16(기본) | FLASH_ATTN | **80.52** | **14.74** |
| repro2 | eager | 16 | FLASHINFER | **80.27** | 14.52 |
| repro3 | CUDA Graph | 16 | FLASH_ATTN | **78.94** | 14.12 |
| repro3 | CUDA Graph | 16 | FLASHINFER | 79.37 | 13.90 |
| repro4 | eager | 256 | FLASH_ATTN | **81.27** | 14.69 |
| repro4 | eager | 256 | FLASHINFER | 81.32 | 14.55 |

혼합 길이 배치는 균일 배치보다 step당 **5.46배** 느리지만(eager, FLASH_ATTN), 그 손실은 **attention 백엔드와 무관하다**: 같은 조건에서 FLASHINFER / FLASH_ATTN = **0.997배**이고, CUDA Graph 모드(vLLM이 분할 수 상한을 고정하는 경로)나 KV 블록 256도 바꾸지 못한다. 본 작품의 엔진에서 같은 배치의 step은 휴리스틱 51 ms, 테이블 24 ms이므로 vLLM의 79~81 ms는 attention 커널 선택으로 설명되지 않는 다른 비용이 지배한다. 즉 **vLLM에서 "혼합 길이 배치가 느리다"는 증상은 재현되지만, 그 원인이 분할 수 휴리스틱이라는 본 작품의 기전은 vLLM 0.31에서는 확인되지 않았다**(이 버전의 FlashAttention 백엔드가 어떤 경로를 쓰는지, 어디에 시간이 가는지는 프로파일링 전에는 알 수 없다). 반복 간 생성 토큰이 균일 조건에서도 달라진 점(`tokens_identical` false)은 vLLM의 반복 간 비결정성이며 별도 문제다. 다음 단계는 vLLM decode step 하나를 torch profiler로 분해해 attention 커널 시간을 분리하는 것이다(TODO §1).

## 7. 한계와 다음 일

- 요청한 세 조건은 측정 테이블의 셀과 겹쳐 혼합 정책과 테이블 정책을 구분하지 못한다. 구분되는 조건(held-out 요청 도착)은 보충 실행 하나뿐이고 자연어 입력 조건의 다른 시드·모델은 다루지 않았다.
- §3의 선택 비용은 "측정 실행마다 정책 캐시 초기화" 규약의 값이고 §3b가 반대쪽 경계를 쟀다. 실제 서비스의 캐시 적중률(요청 구성의 변화 빈도)은 둘 사이 어딘가이며 측정하지 않았다.
- 커널 수준 탐침은 사건이 난 한 step에서만 했다. 다른 step·다른 조건의 `clear`가 나오면 같은 탐침을 다시 돌린다(`kernel_probe` 재현 명령은 아래).
- (2026-10-06 해결) 혼합 정책의 d=128·fp16/bf16 조건 검사를 엔진에 추가했다(`Engine.run`).

## 재현 명령

```bash
# 요청한 캠페인(세 조건), 보충 실행은 스크립트 scratch 기록 참고
for s in ragged uniform arrivals; do
  .venv/bin/python -m kernelscope.cli serve run --model Qwen/Qwen3-4B-Instruct-2507 --scenario scenarios/graduation_$s.yaml \
    --policy heuristic --policy table:demo_data/dispatch_paged_cold.csv --policy hybrid:demo_data/dispatch_paged_cold.csv:0.2 \
    --machine machines/rtx4090.json --params models/rtx4090.json \
    --kv-gib 10 --warmup-runs 1 --warmup-steps 2 --repeats 3 --seed 0 --out ../kernelscope/results/serve_4090/hybrid_new/$s
done
# 교사 강제 진단(조건마다) 과 분기 사건 분류(GPU 없이)
.venv/bin/python -m scripts.check_policy_numerics --scenario scenarios/graduation_ragged.yaml --reference heuristic \
  --policy table:demo_data/dispatch_paged_cold.csv --policy hybrid:demo_data/dispatch_paged_cold.csv:0.2 \
  --machine machines/rtx4090.json --params models/rtx4090.json --steps 64 --logit-steps 64 --kv-gib 10 \
  --out ../kernelscope/results/serve_4090/hybrid_new/numerics/ragged
.venv/bin/python -m scripts.classify_divergence --campaign ../kernelscope/results/serve_4090/hybrid_new
# clear 사건의 커널 수준 탐침(같은 KV 상태에서 층별 attention 출력 비교; 스크립트는 scripts/probe_split_kernel.py)
.venv/bin/python -m scripts.probe_split_kernel --campaign ../kernelscope/results/serve_4090/hybrid_20261002   # 기본값 = control_uniform_fixed8, rid 8, step 40, 분할 8, KV 10 GiB
```

원본은 `demo_data/serve_4090/hybrid_20261002/`(실행별 parquet, `numerics/<조건>/teacher_forced_logits.csv`, `divergence.csv`)에 있다.
