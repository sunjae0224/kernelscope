# 2026-10-25 아침 보고서 — 야간 자동 작업(2026-10-24 21:21 ~ 10-25 약 03:30) 검토 결과

> 검토자 P4(Claude Fable 5.1). 작업자 sonnet 단계 P1a·P1b·P2a·P2e·P3·P3b, 수리 2회, GPU 대기열 6항목. 원장 `../kernelscope/results/overnight_20261024/STATUS.md`, 단계 보고 `.../reports/*.md`, 실험 노트 [docs/experiments/2026-10-24-final-evidence.md](../experiments/2026-10-24-final-evidence.md).
> **커밋은 하지 않았다**(HEAD = bdbc2c4 그대로). 모든 변경은 미커밋이며 §7에 커밋 묶음을 제안한다.

## 1. 한 줄 결론과 주의 사항

**GPU 대기열 6항목이 모두 완료되어 §1·§2·§6~§10의 근거가 생겼고, 검토자가 절마다 3개 이상의 수치를 번들에서 독립 재계산해 모두 일치했다. 결론 중 가장 무거운 것 둘은 (a) 엔진 TPOT로는 고정 분할 8 하나가 table 정책과 같다(동적 선택의 실측 이점은 출력 보존만), (b) "13.5%"는 커널 실측으로 약 9%다.** `make verify` 944/944 PASS(10월 8일 531), `make test` 718 passed(667).

규칙 점검 결과(위반은 되돌리지 않고 기록만):
- `git log -1` = bdbc2c4, 삭제된 파일 없음(`git status`에 D 없음), 결과 폴더 모두 보존(실패 폴더 3개는 `_failed_<시각>`으로 남아 있음). **중대 위반 없음.**
- 주의 1: `kernelscope/results/store.py`(커널 벤치의 저장 경로)가 대기열 도중 수정됐다(00:04, parquet 파일명 태그 120자 제한). 02 시도 1이 파일명 255바이트 초과로 죽어 수리 단계가 고친 것으로 "대기열 수리 예외"에 해당하며, 그 전에 끝난 01 결과에는 영향이 없다(파일명만 바뀌고 내용은 같다).
- 주의 2: `gpu_queue/04_vllm_profile.sh`가 `.ready`(21:54) 뒤 00:47에 수정됐다(`--kv-cache-gib 8` 추가). ".ready 뒤에는 고치지 않고 새 번호로"에 어긋나지만, 항목이 실행 중이 아닐 때 수리 단계가 고친 것이고 재시도 2가 그 스크립트로 완료됐다.
- 주의 3: 04 시도 1(00:37~00:46) 중 다른 사용자의 `asr-nemotron` 파이썬이 GPU에 2분 겹쳤다. 그 폴더는 `_failed_`로 보존되고 §8 수치는 시도 2(간섭 없음)만 쓴다. 다른 모든 시도의 30초 nvidia-smi 기록에는 상주 rerun 뷰어와 항목 자신의 프로세스뿐이다.
- 주의 4: P2b(Mooncake)·P2c(임계 표)·P2d(δ 민감도)는 22:49 세션 사용량 한도(429)로 즉시 실패해 산출물이 없다. 노트 §3·§4·§5는 비어 있다.
- 주의 5: 작업 트리에는 야간 작업 전부터 있던 사용자 미커밋 변경(10월 8일 23:03, 중간보고서 v5 소스·산출물, `.gitignore`, `build.js`, `lib.js`)이 섞여 있다. §7에서 따로 묶었다.
- 주의 6: 모든 캠페인 manifest의 `git.status`가 더럽다(미커밋 상태에서 잼). 재현성 주장을 위해 커밋이 필요하다.

## 2. 작업별 결과

| 작업 | 상태 | 핵심 수치(예측/실측 표기) | 노트 |
|---|---|---|---|
| P1a 고정 분할 수 격자 분석 | 완료 | 실측(격자, cold): 211셀에서 분할 8 고정은 최선 FA2 분할 대비 중앙값 1.010·최악 1.127, 휴리스틱 최악 3.981; warm 균일에서는 분할 8 최악 1.712 | [§1](../experiments/2026-10-24-final-evidence.md) |
| 01 고정 분할 실생성(GPU) | 완료(1회 시도) | 실측 TPOT 배율: 혼합 table 1.813 / 고정 16 1.804 / 고정 8 1.811(이상치 1회 제외, 포함 1.644); 균일 0.998/0.990; 균일에서 고정 8·16만 6곳 분기(table 0) | §1 실생성 |
| P2a 부하 스윕 | 완료 | 모델 예측: conv regret≥1.25 비율 ×0.75 11.6% → ×1 13.5% → ×4 4.2%(봉우리), BurstGPT ≤0.3%; 고정 8은 BurstGPT에서 −5.7% | §2 |
| 02 trace replay 실측 검증(GPU) | 완료(2회 시도) | 실측 8.9%(95% 6.3–10.9%) 대 예측 13.5%; attention 시간 비 실측 1.144 대 예측 1.119; 순위 상관 0.959 | §6 |
| 03 트레이스 구동 e2e(GPU) | 완료(1회) | 실측: table 1.035배·처리량 +2.8%, attention 비 1.178(예측 1.187); model·hybrid 첫 반복 0.86배(결정 비용 6.5 ms/step), 캐시가 찬 뒤 1.035~1.037 | §7 |
| 04 vLLM step 분해(GPU) | 완료(2회 시도) | 실측: 혼합 81.0 대 균일 14.6 ms(5.5배), 차이의 97.6%가 attention(67.7 대 2.95 ms); FlashInfer 36.0 ms(2.5배). 10월 6일 "백엔드 무관 0.997배"는 근거 없음 | §8 |
| 05 64K 자연어(GPU) | 완료(1회) | 실측: table 2.449배, 분기 0곳(무작위 토큰 64K의 clear 1건 미재현) | §9 |
| 06 GSM8K 정확도(GPU) | 완료(1회) | 실측: 네 정책 `####` 정확도 156/200 = 0.780 동일; table·고정 8 토큰 동일, flashinfer_cudacore 140문항 생성 변경·답 변경 8(3↔3) | §10 |
| P2e 관련 연구 초안 | 완료 | 인용 19건; FlashInfer "GQA≥4 tensor-core 권장"은 SGLang 규칙(vLLM은 항상 True) | [related_work_additions.md](../report/final/related_work_additions.md) |
| P3b 발표 배경 파트 | 완료 | 6장 구성안 + 그림 4장(b4는 데이터 없음) | [story.md](../presentation/background/story.md) |
| P2b Mooncake / P2c 임계 표 / P2d δ 민감도 | **미실행** | 사용량 한도 429 | §3·§4·§5 비어 있음 |

**검토자가 독립 재계산한 항목(모두 일치).** §1: 격자 regret 표(fixed 1/8/16, 휴리스틱; 32K 혼합·균일·합집합·64K·warm) 전부, 실생성 분기 자리(균일 rid 2·4·8·20·27·31, 64K rid 0 위치 27·4토큰), table이 고른 분할(혼합 16, 64K 8, 균일 0). §2: conv ×1·×0.75·×4·BurstGPT ×400·code ×16의 비율·attention 비·S=8 비·배치 구간별 비율. §6: 가중 비율 8.86%, 부트스트랩 6.28–10.91%, 키 step 가중 10.18%, attention 비 1.144/1.119, precision 41/50, 층별 중앙값, 순위 상관 0.959, 오차 0.048/0.245, 회차 잡음 0.155/0.752%. §7: TPOT 반복값·1.035·0.97/0.972·결정 비용 6,479 µs·캐시 적중 65.7%·step 비율 43.1/46.1%·분기 10/10/8곳·921/740토큰. §8: iteration 행에서 attention 합/22 step = 67.7·24.0 ms, 균일 2.9 ms; FlashInfer 실행의 커널 이름(`BatchPrefillWithPagedKVCacheKernel`)과 로그의 `Using AttentionBackendEnum.FLASHINFER backend.`로 백엔드 전환 확인. §9: 배율·분기 0. §10: 156/200·79/200·140·8·3/3·13/15·필러 6/7. 발표: conv p50 1,020·p99 4,142·최대 14,050·≥8192 1건·출력 중앙값 129, CTA 256/4,096, 10.78배.

**검토자가 고친 것.** (1) 노트 §8에 "검토" 문단 추가: 매니페스트의 `engine_reported_backend`가 null인 이유(스크립트 정규식이 0.31 로그 문구를 놓침)와 커널 이름·로그 줄로 백엔드를 확인한 근거. (2) story.md 슬라이드 6의 "모두 3회 평균 … 제외한 값"을 "table·고정 16은 3회 평균, 고정 8은 이상치 1회를 뺀 2회 평균(3회 평균 1.644)"로 정확히 고침. (3) 관련 연구 초안의 R2(Ansor)를 arXiv Comments 줄로 OSDI 2020 확인, 5건 재확인 기록 추가(R11의 NeurIPS 2025는 arXiv 페이지에 없어 미확인 유지). (4) PLAN §1 ⑧~⑩ 행, §2 7번 갱신·9번 추가, docs/STATUS.md 항목, TODO.md 갱신. 수치 불일치는 없었다.

**참고(수정 아님).** 프로젝트의 TPOT 정의(`kernelscope/serve/report.py`의 `tpot_us`)는 요청별 토큰 간격의 평균이며 prefill 뒤 첫 토큰까지의 대기를 포함한다. decode step 시간만으로 배율을 내면 혼합 길이에서 2.13배(table)·2.12배(고정 16)·2.15배(고정 8, 이상치 제외)로 절대값이 다르지만 정책 순서와 "고정 ≈ table" 결론은 같다. 9월부터 쓴 정의라 바꾸지 않았다.

## 3. 발표 "문제 정의·배경"에 바로 쓸 숫자

| 숫자 | 뜻 | 종류 | verify 항목 |
|---|---|---|---|
| 1,020 / 4,142 / 14,050 | Azure conv 프롬프트 길이 p50 / p99 / 최대(19,366건); 32K 이상 0건 | 트레이스 통계 | `bg.len.conv.p50`, `bg.len.conv.p99`, `bg.len.conv.max`, `bg.len.all.n_ge_32768` |
| B ≥ 26 | 4090(SM 128)·KV 헤드 8에서 FA2 휴리스틱이 분할 없이 도는 배치 문턱(0.8·2·128/8); conv ×1 step의 89.5%가 해당 | 규칙 판정 | `sweep.conv_x1.no_split_frac` |
| 256 → 4,096 CTA, 10.78배 | 32K+512×31 배치의 CTA 수(분할 없음 → 분할 16)와 가장 큰 CTA / SM당 균등 몫 | 기하 계산 | `bg.cta.heuristic_ctas`, `bg.cta.s16_ctas`, `bg.cta.heuristic_max_over_sm_share` |
| 1.436 / 3.981 | 32K 혼합 162셀에서 FA2 휴리스틱의 regret 중앙값 / 최악(최선 FA2 분할 대비) | 실측 | `fixedsplit.ragged32k.heuristic.median`, `fixedsplit.ragged32k.heuristic.max` |
| 13.5% → 8.9%(6.3–10.9%) | conv ×1에서 regret ≥ 1.25 step 비율: 모델 예측 → 커널 표본 실측 | 예측 / 실측 | `sweep.conv_x1.loss_ge_1.25`, `p3val.conv.w_meas_frac`, `..._lo`, `..._hi` |
| 1.119 → 1.144 | 같은 조건의 attention 시간 비(휴리스틱/최선): 예측 → 실측 | 예측 / 실측 | `p3val.conv.att_pred`, `p3val.conv.att_meas` |
| 0.3% / 4.2% | BurstGPT 어느 부하에서나 / conv 과부하(×4) 포화 값 | 모델 예측 | `sweep.burstgpt.max_loss_ge_1.25`, `sweep.conv_x4.loss_ge_1.25` |

그림: `docs/presentation/background/fig_b1_length_dist.png`(길이 CCDF), `fig_b2_load_sweep.png`(부하 스윕 + 실측 ◆), `fig_b3_kernel_regret.png`(혼합 vs 균일 regret CDF, 8K→64K), `fig_b5_cta_imbalance.png`(CTA별 KV 작업량). 재생성 `.venv/bin/python docs/presentation/background/make_figs.py`. 정확한 항목 이름은 story.md의 `[verify: …]` 표기를 따른다.

## 4. 기존 결론을 바꾸거나 약화시킨 지점

1. **"동적 선택이 필요하다"의 TPOT 근거가 약해졌다.** 커널 격자와 엔진 실생성 모두에서 고정 분할 8(또는 16) 하나가 table 정책과 같은 TPOT 이득을 낸다(혼합 ±0.5%, 도착 조건에서는 고정 8이 0.4% 더 빠름). 남는 실측 근거는 (a) 균일 길이에서 고정 분할은 출력을 6곳 바꾸고 table은 바꾸지 않는다, (b) 모델 예측에서 고정 8은 짧은 프롬프트 트레이스(BurstGPT)에서 휴리스틱보다 최대 5.7% 손해다, (c) warm 균일 격자에서 고정 8이 최악 1.712배다(엔진 미측정). table이 혼합 길이에서 사실상 고정 16이었다는 점도 같이 적어야 한다.
2. **"13.5%"는 과대 예측이다.** 커널 실측 8.9%(가중 방식에 따라 10.2%). 모델이 1.25 문턱 바로 아래 셀(실측 1.197~1.250)을 문턱 위로 예측한다. 다만 되찾을 attention 시간은 실측이 더 크다(10.6% → 12.6%)고 방향·순위(0.959)는 지지된다. 트레이스 e2e 엔진 실측에서는 반대로 step 비율이 예측(31%)보다 크다(43%) — cold 커널 시간과 엔진 안 시간의 조건 차이로 보이나 원인은 가리지 않았다.
3. **vLLM 문장 철회.** 10월 6일 "FLASH_ATTN/FLASHINFER와 무관(0.997배)"은 같은 백엔드를 두 번 잰 것이다. 새 사실: attention GPU 시간이 지배(97.6%), FlashInfer는 2.5배로 덜 나쁘다, vLLM의 FA2 래퍼는 `num_splits` 0/1만 허용해 본 작품의 외부 선택은 vLLM에 그대로 적용되지 않는다. "분할 수 휴리스틱이 직접 원인"이라는 증거는 아직 정황뿐이다.
4. **FlashInfer 기본값 서술.** "GQA ≥ 4면 tensor-core 권장"은 FlashInfer 문서가 아니라 SGLang 규칙이고 vLLM은 항상 tensor-core다. PLAN §0의 "FlashInfer는 head 그룹 크기만 보고 tensor-core 변형을 고른다"와 중간보고서 2장 문구가 영향받는다.
5. **model·hybrid 정책의 첫 결정 비용.** 처음 보는 트래픽(캐시 적중 65.7%)에서 결정 비용 6.5 ms/step로 휴리스틱보다 16% 느리다. 10월 6일 "캐시 유지 규약"의 유리한 수치는 같은 트래픽을 한 번 본 뒤의 값임이 분명해졌다.
6. 약화되지 않은 것: 혼합 길이 TPOT 이득(1.8배/2.4배), 출력 보존(분할 수만 바꾸는 정책은 GSM8K 200문항 토큰 동일, 64K 자연어 분기 0), 64K 추세, 트레이스 재생의 조건부 문제 성격.

## 5. 사용자 결정이 필요한 것

1. **13.5% 표기**(보고서 1장·PLAN §0·§1·문제 범위 노트 요약): "step 13.5%(모델 예측; 커널 실측 표본 추정 약 9%, 95% 6.3–10.9%)"로 병기할지. 병기하지 않으면 story.md 슬라이드 5의 ◆와 4번 항목을 지운다.
2. **동적 선택 정당화 문구**: 제안 — "고정 분할 8은 커널·엔진 수준에서 강한 기준선이며, 동적 선택이 사는 것은 (휴리스틱이 맞는) 균일 길이에서의 출력 보존과 측정 격자 밖 조건(warm 캐시, 짧은 프롬프트 트래픽)에서의 안전성이다." 아울러 **hybrid·model 정책을 보고서의 주 정책으로 유지할지** — 세 트레이스·시나리오에서 table과 선택이 같거나(캐시가 찬 뒤) 결정 비용으로 손해(첫 반복)였다. table(+고정 8 기준선)로 단순화하는 안도 가능하다.
3. **vLLM 문구**: PLAN §2 7번·중간보고서 표 8·2026-10-02 노트 §9의 "백엔드 무관 0.997배" 철회와 §8의 새 문장으로 교체. PLAN §2 7번에는 검토자가 사실만 덧붙였고 원문은 지우지 않았다.
4. **PLAN §0 문장 두 곳(검토자가 고치지 않음, 제안만)**: (a) "FlashInfer는 head 그룹 크기만 보고 tensor-core 변형을 고른다" → "FlashInfer를 쓰는 엔진은 head 그룹 크기(SGLang) 또는 고정값(vLLM)으로 변형을 고른다"; (b) "decode step의 13.5%가 … 10.6%를 되찾을 수 있다" → 모델 예측임을 명시하고 실측 9%·12.6%를 병기.
5. **트레이스 e2e 보고 규약**: model·hybrid의 3회 평균(0.97)이 아니라 첫 반복(0.86)과 반복 1·2(1.035~1.037)를 따로 쓰고 캐시 규약을 병기; 트레이스 창은 예측 regret이 큰 쪽으로 고른 "선택된 구간"으로 적는다.
6. **GSM8K 주 지표 사후 변경**(마지막 숫자 → `####` 뒤 숫자): 두 지표를 모두 적되 사전 지정 지표의 한계(생성 이어짐)를 밝힐지.
7. **GPU 재측정 여부**: 고정 8 혼합 반복 1 이상치(약 3분), 트레이스 e2e 분기 9곳의 교사 강제 분류, vLLM CUDA graph 모드의 실제 `num_splits`.

## 6. 남은 작업(우선순위)과 예상 소요

1. 위 결정 7건 → 보고서 1·2장 문구 확정(사용자, 1~2시간).
2. 보고서 1·2장 갱신: PLAN §0 개정 정의, 문제 범위 노트 요약, 이번 밤 §1 표 ⑧~⑩, 관련 연구 초안 반영(`[R*]` → `[23]~` 재번호, `content1_v5.js` 23줄 수정). 반나절.
3. 발표 자료 해결·검증 파트(배경 파트는 있음) + 슬라이드 2 도식. 1~2일.
4. 선택 GPU 작업(결정 7에 따라): 합쳐서 30분 안.
5. 미실행 절(§3 Mooncake, §4 임계 표, §5 δ 민감도): 각 2~4시간, 중단 조건상 생략 가능.
6. 도구 정리: `vllm_profile_step.py analyze`의 집계·백엔드 정규식, README의 verify 수(531 → 944). 30분.

## 7. 커밋 제안(실행하지 않음; 작성자 `sunjae0224 <sunjae0224@gmail.com>`, 트레일러 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`)

1. **중간보고서 v5(야간 작업 전 사용자 변경)**: `.gitignore`, `docs/report/midterm/src/{build.js,lib.js,build_memo.js,content1_v5.js,content2_v5.js,make_figs_v5.py,md2docxjs.py,memo_v5.js,package-lock.json}`, `docs/report/midterm/중간보고서_이선재_v5.{docx,pdf}`, `docs/report/midterm/중간보고서_v5_검토메모.{md,pdf}`, `docs/report/fig/fig{8,9,10,11}_*.png`
   > `Add the 2026-10-08 midterm report v5: updated chapters, figures 8-11 and the review memo`
2. **분석 코드·스크립트·테스트(야간)**: `kernelscope/analysis/{dispatch,traffic,accuracy,validation,vllm_steps}.py`, `kernelscope/results/store.py`, `kernelscope/verify.py`, `scripts/{traffic_replay,analyze_gsm8k_policies,bundle_traffic_sweep,gsm8k_accuracy,sample_replay_cells,trace_scenario,vllm_profile_step}.py`, `scenarios/{demo_text_ragged_64k,gsm8k_ragged32k_n200,trace_azure_conv_i550_n300}.yaml`, `tests/test_{store,traffic,fixed_split,gsm8k_accuracy,p3_analysis,sample_replay_cells,trace_scenario,vllm_profile_step}.py`
   > `Add the fixed-split baseline, load sweep, trace-replay validation, trace-driven scenario, vLLM step profiler and GSM8K scoring with 413 new verify checks`
3. **증거 번들(야간)**: `demo_data/provenance.json`, `demo_data/background/`, `demo_data/hw_4090/trace_validation_20261024/`, `demo_data/serve_4090/{fixedsplit,gsm8k,longctx_text,trace_e2e}_20261024/`, `demo_data/trace_e2e/`, `demo_data/traffic_sweep/`, `demo_data/vllm_4090/`
   > `Add the 2026-10-24 evidence bundles: fixed-split campaign, 131-cell replay validation, trace e2e, vLLM profile, 64K text, GSM8K, load sweep`
4. **문서(야간)**: `PLAN.md`, `TODO.md`, `docs/STATUS.md`, `docs/experiments/2026-10-24-final-evidence.md`, `docs/handoff/2026-10-25-overnight-report.md`, `docs/report/final/related_work_additions.md`, `docs/presentation/background/`
   > `Record the overnight evidence in the note, PLAN, STATUS and TODO; add the related-work draft, the presentation background outline and the morning report`

커밋 2·3은 순서를 바꾸면 `make verify`가 번들 없이 실패하므로 2 → 3(또는 하나로 합침) 순서를 지킨다.
