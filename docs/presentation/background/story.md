# 발표 — 문제 정의·배경 파트 구성안 (6장)

> 2026-10-25 야간 단계 P3b. 그림은 `make_figs.py`가 `demo_data/`만 읽어 만든다(`.venv/bin/python docs/presentation/background/make_figs.py`).
> 숫자 옆의 `[verify: …]`는 `kernelscope verify`가 번들에서 다시 계산하는 항목 이름이다. **모델 예측**은 성능 모델(커널 시간 오차 2.3~12.3%)의 값, **실측**은 RTX 4090에서 잰 값이다.
> 이 파트가 쓰지 않는 것: Mooncake 긴 문맥 트레이스(P2b)와 모델×GPU 임계 batch 표(P2c)는 해당 단계가 사용량 한도로 실패해 데이터가 없다 — 슬라이드에 넣지 않았다(§끝 "빠진 것").
> 권장 흐름에서 바꾼 점: (5)의 "Mooncake·임계 표"를 뺐고, 대신 길이 what-if(프롬프트 ×2·×4)로 "문맥이 길어질수록" 근거를 보였다. 그 외 순서는 권장대로다.

---

## 슬라이드 1 — LLM 서빙의 한 batch에는 길이가 다른 요청이 섞인다

- **한 줄 메시지(제목):** continuous batching 때문에 한 decode batch에 짧은 요청과 긴 요청이 같이 들어간다.
- **화면:** `fig_b1_length_dist.png`(프롬프트 길이 CCDF 세 트레이스). 오른쪽 아래에 숫자 세 개만: "conv 중앙값 1,020 · p99 4,142 · 최대 14,050".
- **말할 내용**
  1. LLM 추론은 prefill 한 번 뒤에 토큰을 하나씩 만드는 decode가 이어진다. 응답이 길수록 decode step이 대부분이다(Azure conv 출력 중앙값 129토큰: 요청 하나당 prefill 1회 대 decode 128 step) — 일반 배경이고 이 작품이 측정한 값은 아니다.
  2. continuous batching은 요청이 끝나는 대로 새 요청을 batch에 넣는다. 그래서 batch의 KV 길이가 요청마다 다르다.
  3. 공개 트레이스의 프롬프트는 대부분 짧다(conv 중앙값 1,020, BurstGPT 중앙값 502, code 중앙값 1,469). 그러나 꼬리가 있다: conv p99 4,142, 최대 14,050. 정직하게 말할 점: **세 트레이스 모두 32K 이상 0건**이고 8K 이상은 conv 1건뿐이다. 우리 커널 실험의 32K·64K 요청은 장문 서비스를 가정한 stress 조건이다.
- **숫자 출처:** `[verify: bg.len.conv.p50 / bg.len.conv.p99 / bg.len.conv.max / bg.len.conv.n_ge_8192 / bg.len.code.p50 / bg.len.burst.p50 / bg.len.all.n_ge_32768]` (번들 `demo_data/background/prompt_length_hist.json`, 만든 스크립트 `prep_length_hist.py`). conv 요청 19,366건·code 8,819건·BurstGPT 7일 17,893건(10월 6일 문제 범위 노트 §1과 동일).
- **예상 질문과 답**
  - Q. 트레이스에 32K가 없으면 32K 실험은 왜 하나? A. 길이가 커질수록 문제가 커지는 방향을 보려는 극단 조건이고, 문제의 빈도는 별도로 trace replay로 본다(슬라이드 5). 실제 트레이스는 4~14K 요청이 짧은 요청 사이에 섞이는 경우가 문제다.
  - Q. decode 비중은 어디서 확인했나? A. 이 작품의 측정이 아니라 일반적인 서빙 구조다. 우리 측정은 한 decode step 안의 시간 분해(`serve diagnose`: 혼합 길이 batch에서 attention 비중 70.9%, 2026-09-27 노트)다.
  - Q. CCDF 곡선이 계단처럼 보이는 이유는? A. 로그 구간 40개짜리 히스토그램으로 그렸기 때문이다(최댓값 점은 정확한 값).

---

## 슬라이드 2 — decode attention은 memory-bound이고 병렬성은 batch × KV head × split 수다

- **화면:** 도식(새 그림 없음 — 아래 세 항목을 한 장짜리 도식으로 그린다: 요청별 KV 구간 → CTA 격자 → SM). 숫자: "batch 32 × KV head 8 = CTA 256개, RTX 4090 SM 128개".
- **말할 내용**
  1. decode는 query가 토큰 하나라 KV cache를 읽는 양이 연산량을 지배한다 — attention은 memory-bound다(일반 roofline 논거).
  2. FlashAttention-2 decode 경로는 (요청, KV head)마다 CTA 하나를 둔다: 분할이 없으면 CTA 수 = B × H_kv. Qwen3-4B는 KV head 8개이므로 batch 32에서 CTA 256개 `[verify: bg.cta.heuristic_ctas]`다.
  3. Flash-Decoding의 split-KV는 한 요청의 KV를 S조각으로 나눠 CTA 수를 S배로 늘려(분할 16이면 CTA 4,096개 `[verify: bg.cta.s16_ctas]`) 병렬성을 올린다. 조각 결과는 마지막에 합친다(수학적으로 동등).
- **숫자 출처:** CTA 수는 `kernelscope.model.geometry.build_launch`가 만든 launch 구성이다(예측이 아닌 기하학적 계산, 대표 셀 `decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8`).
- **예상 질문과 답**
  - Q. split을 늘리면 항상 좋은가? A. 아니다. 합치는 단계와 launch 비용이 있고 짧은 요청 위주 batch에서는 낭비다 — 고정 split 16은 BurstGPT류 트레이스에서 최선 대비 1.107~1.128배(모델 예측, 2026-10-24 노트 §2 `sweep.fixed.*`).
  - Q. 논문 근거는? A. Flash-Decoding(Dao 외, 2023 블로그)의 split-KV가 출발점이다. 인용 정리는 `docs/report/final/related_work_additions.md`(P2e).

---

## 슬라이드 3 — 라이브러리는 split 수를 길이를 보지 않고 정한다

- **화면:** FA2 휴리스틱을 한 줄로 요약한 코드 조각 + 계산 한 줄(아래). 오른쪽에 FlashInfer 한 줄.
  ```
  if (batch × KV head × m-block) ≥ 0.8 × (2 × SM 수):  num_splits = 1     # 요청별 길이는 입력이 아님
  4090·Qwen3-4B: 32 × ... → B × 8 ≥ 204.8  ⇒  B ≥ 26 이면 split 없음
  ```
- **말할 내용**
  1. flash-attn 2.8.x의 `flash_attn_with_kvcache`에서 `num_splits=0`(기본)이면 위 규칙으로 split 수를 고른다. 입력은 batch 크기·헤드 수·KV 용량이고 **요청별 실제 길이는 보지 않는다.** RTX 4090(SM 128)·KV head 8에서는 B ≥ 26이면 분할 없이 돈다. 실제로 conv ×1 replay의 step 중 89.5%가 이 조건이다 `[verify: sweep.conv_x1.no_split_frac]`.
  2. FlashInfer의 기본 변형(tensor-core 쪽)도 head 그룹 크기만 보고 고른다. 길이를 입력으로 받지 않는 점은 같다.
  3. 둘 다 균일 길이 batch에서는 충분하다. 문제는 길이가 섞일 때다(다음 장).
- **숫자 출처:** `geometry.num_splits_heuristic`(flash-attn 2.8.3 소스 포팅). 89.5%는 trace replay의 step 길이 벡터에 이 규칙을 그대로 적용해 센 비율이다(성능 모델의 시간 예측이 아니라 규칙의 판정).
- **주의(보고서 문구):** 중간 보고서·PLAN §0의 "FlashInfer GQA ≥ 4 tensor-core 권장" 문구는 FlashInfer가 아니라 SGLang의 규칙이라는 P2e 지적이 있다. 발표에서는 "FlashInfer 기본 변형이 길이를 보지 않는다"만 말하고 GQA 임계는 말하지 않는다.
- **예상 질문과 답**
  - Q. vLLM·SGLang은 이 문제가 없나? A. vLLM 0.31에서도 혼합 길이 batch가 균일보다 step당 5.46배 느린 증상이 있고 그 시간의 대부분이 attention GPU 시간임을 확인했다(`vllm_profile` 실측: FA2 5.5배, FlashInfer 2.5배, 2026-10-24 노트 §8). 다만 **분할 수 휴리스틱이 직접 원인이라는 증거는 아직 없다**: vLLM의 FA2 wrapper는 `num_splits`를 0/1만 허용해 외부에서 바꿀 수 없다. (10월 6일 "백엔드와 무관 0.997배" 문구는 철회 대상 — FLASHINFER 실행이 실제로 FLASH_ATTN이었다. 발표에서 쓰지 않는다.)
  - Q. 다른 GPU(SM 수가 다름)는? A. 임계 batch는 SM 수와 KV head 수로 정해진다(B ≥ 0.8·2·SM / H_kv). 모델×GPU 표 실측은 이번 야간 작업에서 못 만들었다(P2c 실패).

---

## 슬라이드 4 — 그 결과 load imbalance: 긴 요청의 CTA 8개가 시간을 정한다

- **화면:** 왼쪽 `fig_b5_cta_imbalance.png`, 오른쪽 `fig_b3_kernel_regret.png`(둘을 한 장에 넣기 어려우면 b5 → b3 두 장으로 나눈다).
- **말할 내용**
  1. 32K 요청 1개 + 512 요청 31개(batch 32)를 split 없이 돌리면 CTA 256개 중 긴 요청의 8개(KV head마다 하나)는 한 CTA가 KV 32,768개를 읽는다. SM 1개가 맡을 균등 몫(총 작업 ÷ 128)의 **10.8배**다 `[verify: bg.cta.heuristic_max_keys, bg.cta.heuristic_max_over_sm_share]`. 짧은 요청의 CTA가 먼저 끝나 다른 SM이 놀아도 그 CTA가 끝나야 kernel이 끝난다.
  2. split 16이면 가장 큰 CTA가 최대 2,048 `[verify: bg.cta.s16_max_keys]`로 SM 몫 이하라 꼬리가 사라진다.
  3. 실측 regret(= 휴리스틱 시간 / 최선 시간, RTX 4090, 32K 격자): 혼합 길이 162셀에서 중앙값 1.457배 `[verify: defaults.32k.fa2_heuristic_median]`, 최악 4.106배 `[verify: bg.kernel.fa2_heuristic_max]`, 1.25배 초과 118셀 `[verify: defaults.32k.fa2_heuristic_over_1.25]`. 균일 길이 49셀은 최악 1.171배다 `[verify: defaults.32k.uniform_fa2_heuristic_max]`. FlashInfer tensor-core 기본 변형도 혼합 길이에서 중앙값 1.017배, 최악 2.485배(36셀이 1.25배 초과) `[verify: defaults.32k.flashinfer_tc_median / flashinfer_tc_max / flashinfer_tc_over_1.25]`다.
  4. 긴 요청이 길수록 커진다(같은 18셀 조건, FA2 split 중 최선 대비): 중앙값 8K 1.545 → 16K 1.933 → 32K 2.558 → 64K 2.839, 64K 최악 5.183 `[verify: longctx.heuristic_median_8k / _16k / _32k / _64k, longctx.heuristic_max_64k]`.
- **숫자 출처:** b5는 `build_launch` 기하 계산, b3는 실측 커널 시간(`demo_data/hw_4090/*_paged`, `*_64k_*`; cold L2, 페이지 KV, fp16). 왼쪽 패널은 FlashInfer 포함 전체 최선과의 비, 오른쪽 패널은 FA2 split 중 최선과의 비이므로 같은 32K 중앙값이 1.457과 2.558로 **서로 다르다** — 비교 대상이 다르기 때문이며 화면에 축 이름으로 구분해 둔다(2.558은 긴 요청 1~2개·B 26/32/64의 18셀 부분집합).
- **예상 질문과 답**
  - Q. 왜 1.457과 2.558이 둘 다 "32K 중앙값"인가? A. 앞은 162셀 전체·분모가 FlashInfer까지 포함한 최선, 뒤는 18셀 부분집합·분모가 FA2 split 최선이다(위 출처 문단).
  - Q. 시간은 실제 서빙에서 얼마나 영향을 주나? A. attention이 step의 일부라 희석된다. 엔진 실측: Qwen3-4B 혼합 길이 batch에서 TPOT 1.81배(휴리스틱 → 테이블, 3회, `serve` 캠페인 2026-10-08)이고 균일 길이는 변화가 1% 안이다. 이 수치는 해결 파트에서 쓴다.
  - Q. 캐시 상태는? A. 커널 실측은 cold L2(매 반복 L2 flush)이고 엔진의 실제 L2 상태와 다를 수 있다 — 한계로 말한다.

---

## 슬라이드 5 — 얼마나 자주 생기나: 긴 프롬프트 비중 × batch 크기에 달린 조건부 문제

- **화면:** `fig_b2_load_sweep.png`(부하 스윕, B = 26 세로선). 오른쪽에 표 세 줄:
  | 조건 | regret ≥ 1.25배인 step 비율 |
  |---|---|
  | Azure conv ×1 | 13.5% (모델 예측) / 8.9% (실측 표본 추정, 95% 구간 6.3–10.9%) |
  | Azure conv ×4(과부하) | 4.2% |
  | BurstGPT 7일 | 0.3% 이하 |
- **말할 내용**
  1. trace replay: 공개 요청 트레이스의 도착 시각·프롬프트·출력 길이를 연속 batching 스케줄러(batch 상한 64, KV 10 GiB, step 30 ms 가정)에 넣고 매 step의 길이 벡터를 성능 모델로 채점했다. 도착률 배율을 21가지로 바꿔 평균 batch 7.5~50 구간을 훑었다.
  2. 곡선은 봉우리 모양이다. 왼쪽 경사는 휴리스틱이 B < 26에서 스스로 분할하기 때문, 오른쪽 경사는 batch가 커져 긴 요청 CTA 하나의 비중이 희석되기 때문이다(후자는 모델 예측의 배치별 regret으로 보인 것이며 별도 커널 실험으로 분리하지 않았다). conv는 ×1에서 13.5%로 최대이고 `[verify: sweep.conv_x1.loss_ge_1.25]` 과부하(×4, 평균 batch 47)에서 4.2% `[verify: sweep.conv_x4.loss_ge_1.25]`로 내려온다. 이 13.5%는 큐 대기 p99가 3.7 s인, 서버가 감당하는 부하의 끝 가장자리의 값이다.
  3. 짧은 프롬프트 트레이스(BurstGPT)는 어느 부하에서도 0.3% 이하다 `[verify: sweep.burstgpt.max_loss_ge_1.25]`. 드문 문제가 아니라 **긴 프롬프트 비중과 batch 크기에 달린 조건부 문제**다.
  4. 모델 예측을 실측으로 확인했다: conv ×1 표본 100셀을 같은 하네스로 재면 step 비율은 예측 13.5%에 대해 실측 8.9%(95% 구간 6.3–10.9%) `[verify: p3val.conv.w_pred_frac, p3val.conv.w_meas_frac, p3val.conv.w_meas_frac_lo, p3val.conv.w_meas_frac_hi]` — **예측이 구간 밖이다.** 반면 되찾을 attention 시간 비는 예측 1.119보다 실측 1.144로 크고 `[verify: p3val.conv.att_pred, p3val.conv.att_meas]`, 예측·실측 regret의 순위 상관은 0.959 `[verify: p3val.err.spearman]`다. 즉 방향과 되찾을 시간은 지지되고 1.25배 문턱 바로 아래의 셀에서 빈도는 과대 예측된다.
  5. 프롬프트가 길어지면(what-if, 트레이스를 바꾼 것): 프롬프트 ×2에서 25.7% `[verify: traffic.azure_conv_x1_prompt2_loss_ge_1.25]`, KV 예산을 436K 토큰(80 GB급)으로 두고 ×4에서 39.5% `[verify: traffic.azure_conv_x1_kv436k_prompt4_loss_ge_1.25]`까지 커진다. 10 GiB KV 안에서 ×4는 batch가 14.6으로 줄어 2.9%로 사라진다 `[verify: traffic.azure_conv_x1_prompt4_loss_ge_1.25]`. 조건은 "긴 프롬프트 + batch를 26 이상으로 유지할 KV 예산"이다.
- **숫자 출처:** 곡선은 `demo_data/traffic_sweep/sweep_table.csv`(모델 예측, 21개 replay), ◆는 `demo_data/hw_4090/trace_validation_20261024`(실측 표본 131셀 중 conv 100셀, cold L2). 2026-10-24 노트 §2·§6, 2026-10-06 노트 §1. **Mooncake 긴 문맥 트레이스는 이 발표 자료에 없다**(데이터 없음).
- **보고서 문구와의 정합:** P3 보고의 "사용자 결정 필요 1번"(13.5%에 실측 8.9%를 병기할지)이 아직 열려 있다. 이 슬라이드는 병기하는 안으로 만들었다. 병기하지 않기로 하면 ◆ 표기와 4번 항목을 지운다.
- **예상 질문과 답**
  - Q. 그래서 13.5%인가 8.9%인가? A. 모델 예측은 13.5%, 표본 실측 추정은 약 9%(셀 가중을 step 수로 바꾸면 10.2%이고 그때 구간 6.5–14.6%는 13.5%를 포함)다. 어느 쪽이든 "배치 ≥ 26인 부하에서 step의 약 1/10"이라는 크기 차수는 같다.
  - Q. 왜 도착률 ×1을 골랐나? A. 임의가 아니다. 부하 스윕에서 큐가 막 생기기 시작하는 지점(대기 > 1 s 요청 5.5%)이 봉우리 꼭대기와 겹친다. "SLO 근처 부하"의 임계값은 정해 두지 않았다(사용자 결정 필요).
  - Q. 성능 모델은 믿을 만한가? A. 커널 시간 오차 2.3~12.3%, 표본 131셀에서 regret 오차 중앙값 0.048 `[verify: p3val.err.abs_median]`, 최대 0.245 `[verify: p3val.err.abs_max]`. 그래서 1.25배 문턱 주변 빈도는 실측 구간과 함께 쓴다.
  - Q. step 시간이 30 ms 상수라는 가정은? A. 한계다. 큐 대기는 순서 정보로만 읽어야 한다.

---

## 슬라이드 6 — 문제 정의와 목표

- **한 줄 메시지(제목):** 라이브러리의 split 설정이 batch 안의 길이를 보지 않아, 긴 프롬프트가 섞이고 batch가 큰 부하에서 attention 시간을 잃는다.
- **화면:** 문제 정의 문장 + 목표 세 줄.
  - **문제:** 배치 안 KV 길이가 불균일하고 batch ≥ 26(4090, KV head 8)이면 decode attention의 split 설정이 맞지 않아 load imbalance가 생긴다. 32K 격자에서 regret 중앙값 1.46배·최악 4.1배(균일 길이는 1.17배 안), 공개 트레이스 replay에서는 모델 예측 step 13.5%·실측 표본 추정 약 9%가 1.25배 이상이다. 짧은 프롬프트 위주 트래픽에서는 0.3% 이하다.
  - **목표:** (1) 커널과 엔진을 수정하지 않고, (2) 출력의 의미를 유지하면서(분기 지점과 bf16 동점 여부로 보고), (3) 매 decode step마다 split 수·변형을 외부에서 고른다.
  - → 다음 파트(해결): 실측 테이블 + 성능 모델 기반 선택, 엔진 안 실생성 검증.
- **말할 내용**
  1. 조건부 문제임을 먼저 인정한다: 긴 프롬프트 비중과 batch ≥ 26에 달렸다.
  2. 해결 파트로 연결: 같은 모델·같은 수학의 다른 설정만 고르므로 출력이 유지된다는 점이 "다른 모델로 바꾸는 것"과의 차이다.
- **숫자 출처:** 슬라이드 4·5의 verify 항목 그대로.
- **예상 질문과 답**
  - Q. 그냥 split을 8로 고정하면 되지 않나? A. 맞는 지적이고 실측했다(2026-10-24 노트 §1): **고정 split 8·16은 엔진 TPOT로 table 정책과 같다**(혼합 길이 table 1.813배·고정 16 1.804배는 3회 평균, 고정 8 1.811배는 원인 불명 이상치 1회를 뺀 2회 평균이며 3회 평균은 1.644배). 고정 설정이 강한 기준선이다. 동적 선택의 근거는 TPOT가 아니라 ① 균일 길이에서 고정 8·16이 출력을 6곳 바꾸는 반면 table은 0곳이라는 출력 보존, ② 모델 replay에서 고정 8이 BurstGPT류에서 휴리스틱보다 최대 5.7% 손해라는 점이다(모델 예측, 격자 밖 조건은 엔진 미측정). 이 문구의 최종 표현은 "사용자 결정 필요 2번"이다.
  - Q. 이 문제는 이미 알려져 있나? A. 길이 불균형 decode는 알려진 문제이고(Flash-Decoding의 split-KV, FlashInfer의 load-balanced scheduling 등) 본 작업의 기여는 두 라이브러리의 기본 설정이 4090에서 얼마나 잃는지의 측정과 외부에서 고르는 방식의 한계 검증이다. 선행 연구 정리는 P2e 문서.
  - Q. 출력이 정말 같은가? A. 토큰 일치는 요청 3조건에서 0건 분기, 일부 조건에서 bf16 동점(tie_1ulp)과 1건의 clear 분기가 있었다. 해결 파트에서 다룬다.

---

## 그림 목록

| 파일 | 슬라이드 | 내용 | 데이터 |
|---|---|---|---|
| `fig_b1_length_dist.png` | 1 | 프롬프트 길이 CCDF, 32K·64K 선 | `demo_data/background/prompt_length_hist.json` |
| `fig_b2_load_sweep.png` | 5 | 평균 batch 대 regret ≥ 1.25 step 비율, B = 26, 실측 ◆ | `demo_data/traffic_sweep/`, `demo_data/hw_4090/trace_validation_20261024/` |
| `fig_b3_kernel_regret.png` | 4 | 혼합 vs 균일 regret CDF, 8K→64K 추세 | `demo_data/hw_4090/*_paged`, `*_64k_*` |
| `fig_b5_cta_imbalance.png` | 2·4 | 대표 셀의 CTA별 KV 작업량(split 1 대 16) | `kernelscope.model.geometry.build_launch`(코드, 데이터 없음) |
| (`fig_b4_threshold.png`) | — | **만들지 않음**: 모델×GPU 임계 batch 표(P2c)에 데이터 없음 | — |

## 빠진 것·열린 결정

- **Mooncake(P2b), 모델×GPU 임계 표(P2c):** 두 단계가 22:49 사용량 한도로 즉시 실패해 산출물이 없다. 필요하면 다시 돌려야 한다. 임계식(B ≥ 0.8·2·SM / H_kv)은 소스에서 도출했지만 다른 GPU·모델의 실측은 없다.
- **사용자 결정 필요(이 파트에 영향):** (a) 13.5%에 실측 8.9% 병기(슬라이드 5), (b) 동적 선택 정당화 문구(슬라이드 6), (c) vLLM 문구 철회(슬라이드 3).
- **그림 한계:** b1의 CCDF는 로그 구간 히스토그램이라 계단 모양이다; b2의 큐 대기 기준(과부하 표시: 요청의 절반 이상이 1 s 넘게 대기)은 이 문서에서 정한 표시 기준이며 SLO가 아니다; b5의 작업량은 KV 토큰 수이고 시간이 아니다.
