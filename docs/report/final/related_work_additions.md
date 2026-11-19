# 관련 연구 보강 초안 (P2e) — 보고서 2장에 옮길 문단과 표 2 추가 행

> 작성 2026-10-24. 대상: 검토 메모 §2 W1(커널 튜닝·디스패치 계열 누락), W2(출력 동일성의 문헌 근거 누락), W3(FA3·FlashInfer 버전 한정), §1.2 W5(FlashInfer tensor-core 근거).
> 번호 `[R1]`~`[R22]`는 이 문서 안의 임시 번호다. 기존 보고서의 `[1]`~`[22]`와 겹치지 않도록 옮길 때 `[23]`부터 다시 매긴다.
> 모든 서지는 말미 "참고문헌"의 확인 상태를 따른다. `[미확인]` 표시는 원문에서 확인하지 못한 부분이다.

---

## A. 2장 가) 또는 나) 뒤에 넣을 문단 — 커널 자동 튜닝과 런타임 디스패치

(추가 소절 제목 제안: **"다) 커널 자동 튜닝과 런타임 디스패치"**, 기존 다)·라)는 라)·마)로 밀린다.)

**문단 A-1 (오프라인 튜닝 계열).** 입력 모양에 맞는 커널 구성을 탐색으로 고르는 접근은 딥러닝 컴파일러에서 오래 연구되었다. TVM[R1]은 학습 기반 비용 모델로 연산자의 저수준 코드를 자동 최적화하며, Ansor[R2]는 계층적으로 표현한 탐색 공간에서 프로그램을 표집하고 비용 모델이 이끄는 진화적 탐색으로 개선한다. Triton[R3]의 `triton.autotune`[R4]은 후보 구성(`configs`)을 두고 지정한 인자(`key`)의 값이 바뀔 때 모든 후보를 다시 벤치마크하며, 성능 모델로 후보를 걸러내는 `prune_configs_by` 옵션을 제공한다. 이들은 모두 연산자의 모양이 정해진 뒤 오프라인에서 탐색하거나, 모양이 바뀔 때 한 번 탐색한 결과를 재사용한다. 벤더 라이브러리도 같은 구조를 가진다. cuBLASLt는 `cublasLtMatmulAlgoGetHeuristic`으로 요청 개수만큼의 후보 알고리즘을 휴리스틱 결과로 돌려준다[R5].

**문단 A-2 (추론 엔진의 디코드 attention 디스패치).** 추론 엔진은 디코드 attention에도 비슷한 디스패치를 둔다. TensorRT-LLM 문서는 생성 단계의 MQA/GQA에 XQA 최적화가 기본으로 켜져 있고 XQA 커널과 masked MHA 커널 중 무엇을 쓸지 휴리스틱이 결정한다고 설명한다. 또한 masked MHA 커널에는 작업을 여러 스레드 블록으로 나누는 multi-block 모드가 0.13부터 기본으로 켜져 있으며, 문서가 제시하는 시험 기준은 `batch_size × num_heads`가 멀티프로세서 수보다 작은 경우이다[R6]. 즉 이 기준은 배치 크기와 헤드 수로 정해지며 요청별 KV 길이를 입력으로 언급하지 않는다. vLLM은 attention 백엔드를 지정하지 않으면 GPU 세대별 우선순위(Ampere·Hopper는 FlashAttention, Blackwell은 FlashInfer가 먼저)에 따라 처음 조건에 맞는 백엔드를 사용하며 `--attention-backend`로 고정할 수 있다[R7]. SGLang의 FlashInfer 백엔드는 초기화 시점에 헤드 수와 KV 자료형으로 디코드의 tensor-core 사용 여부를 정한다[R8]. 두 엔진 모두 선택은 시작 시점에 한 번 이루어지며 매 디코드 단계의 길이 분포를 입력으로 하지 않는다.

**문단 A-3 (라이브러리 내부의 입력 적응).** FlashInfer[R9]는 커널 템플릿의 query tile 크기를 배치의 평균 질의 길이(GQA에서는 그룹 크기와 융합한 값)에 맞춰 고르고, 부하 균형 스케줄러가 질의·KV 길이를 입력으로 받아 SM 유휴 시간을 줄이도록 작업을 CTA에 배정한다(논문 3.2.3절, 3.3.1절). FlashDecoding++[R10]는 입력 동적성에 따라 데이터플로를 휴리스틱으로 고르는 방식을 제안하며, 초록에서 단일 정적 데이터플로가 서로 다른 모양의 GEMM에서 50.25%의 성능 손실을 낼 수 있다고 보고하였다. 앞의 FlashInfer 스케줄러와 FlashDecoding++는 입력에 적응한다는 점에서 본 작품과 같은 방향이지만, 적응이 커널 또는 라이브러리 내부에서 이루어진다.

**문단 A-4 (본 작품과의 차이).** 정리하면 본 작품은 이 계열과 방법론이 같다. 입력 모양에 따른 실행 설정의 런타임 선택, 실측 테이블과 실측으로 맞춘 성능 모델, 후보 선택의 regret 평가가 그것이다. 차이는 세 가지이다. 첫째, 오프라인·모양별 튜닝은 모양이 같으면 같은 구성을 쓰지만, 본 작품은 같은 (배치 크기, 최대 길이)에서도 길이 분포가 다르면 다른 분할 수를 고르며 이를 매 디코드 단계에 수행한다. 둘째, 커널과 엔진을 수정하지 않고 `flash_attn_with_kvcache`의 분할 수 인자(`num_splits`)만 외부에서 지정한다. 셋째, 선택의 대가를 연산 시간뿐 아니라 단계당 선택 비용(FlashInfer의 `plan()` 0.37 ms/step 등)과 출력 변화까지 포함하여 평가한다. 반대로 TVM·Ansor의 탐색이나 FlashInfer의 스케줄러처럼 새 커널 구성을 만들어 내거나 커널 내부 배정을 바꾸는 이득은 본 작품의 범위 밖이며, 4장 라)의 FlashInfer 비교가 그 격차를 측정한다.

---

## B. 2장 또는 4장 라)(5) 앞에 넣을 문단 — 수치 비결정성과 배치 불변성

(추가 소절 제안: **"라) 수치 비결정성과 배치 불변성"**)

**문단 B-1 (문헌의 현상 규명).** 같은 입력에 대해 서빙 조건에 따라 출력이 달라지는 현상은 부동소수점 덧셈의 비결합성에서 비롯된다. Yuan 등[R11]은 평가 배치 크기, GPU 수, GPU 종류를 바꾸는 것만으로 생성 응답이 크게 달라질 수 있고, 특히 추론형 모델에서 초기 토큰의 사소한 반올림 차이가 사고 사슬의 분기로 이어진다고 보고하였다(bfloat16·greedy 디코딩에서 DeepSeek-R1-Distill-Qwen-7B의 정확도가 최대 9%, 응답 길이가 최대 9,000 토큰 변동). Thinking Machines Lab의 He[R12]는 서버 부하에 따라 배치 크기가 달라지는데 커널이 배치 불변이 아니면 같은 요청의 출력이 달라진다고 설명하고, RMSNorm·행렬 곱·attention의 세 연산에서 배치 불변 커널을 제시하였다. 이 글에 따르면 디코드 attention은 split-KV(FlashDecoding)로 reduction을 나누는데, 분할 방식이 요청당 처리하는 질의 토큰 수와 배치에 따라 달라지면 배치 불변성이 깨진다. 해법은 분할 수를 고정하는 것이 아니라 분할 크기를 고정하고 분할 수가 길이에 따라 변하게 하는 것이며(예: 길이 1000의 KV를 크기 256 분할 세 개와 232 분할 한 개로), KV 캐시를 먼저 갱신한 뒤 attention을 호출해 캐시된 KV와 현재 KV를 따로 reduce하지 않도록 한다. 같은 글은 FlashInfer의 균형 스케줄링이 GPU를 채우는 가장 큰 분할 크기를 고르므로 배치 불변이 아닌 예로 든다.

**문단 B-2 (엔진의 결정적 추론 모드).** 이 논의는 엔진 기능으로 이어졌다. SGLang은 `--enable-deterministic-inference` 플래그를 제공하며 FlashInfer·FlashAttention 3·Triton의 세 attention 백엔드에서만 지원한다[R13]. LMSYS 블로그[R14]는 Thinking Machines Lab의 배치 불변 연산자를 바탕으로 SGLang이 청크 프리필, CUDA Graph, radix cache와 함께 결정적 출력을 지원한다고 소개한다. vLLM은 `VLLM_BATCH_INVARIANT` 환경 변수로 배치 불변 모드를 켜며, 문서는 켜면 기본 모드보다 성능이 낮아질 수 있고 이는 재현성을 위한 의도된 절충이라고 설명한다[R15].

**문단 B-3 (본 작품과의 관계).** 본 작품이 고르는 분할 수는 이 문헌이 지목한 reduction 순서를 직접 바꾸는 설정이다. 따라서 정책이 선택한 분할 수가 배치 구성에 따라 달라지면, 같은 요청의 출력이 정책과 배치에 따라 달라질 수 있다. 배치 불변 커널은 분할 크기를 고정하여 이 분기를 원천적으로 없애는 대안이고, 본 작품은 분할 수를 성능을 위해 바꾸는 대신 분기가 얼마나 어디에서 생기는지를 측정하고 분류한다. 구체적으로 분기 지점마다 teacher forcing으로 같은 KV 상태에서 두 설정의 로짓을 비교하고, 1위 후보 사이의 차이를 bf16 ULP 단위로 분류하여 동점(tie)과 명확한 차이(clear)를 구분하며, clear는 층별 출력이 bf16 간격 안에서 일치하는지를 커널 단위로 다시 확인한다. 배치 불변 모드와 본 작품의 선택은 서로 배타적이지 않다. 배치 불변성이 필요한 서비스는 분할 크기를 고정하는 대가로 본 작품이 되찾는 attention 시간 이득의 일부를 포기해야 하며, 그 크기는 본 작품의 측정 범위 밖이다. [사용자 확인 필요: 위 R11·R12는 원인과 해법을 다루고 teacher forcing·ULP 분류를 "표준적 대응"으로 제시하지는 않는다. 따라서 보고서에서는 "표준적 대응"이라고 쓰지 말고 "문헌이 원인으로 지목한 reduction 순서 차이를 측정·분류한다"까지만 쓰는 것을 권한다.]

---

## C. 한정 문장과 각주 (버전·세대, FlashInfer tensor-core 근거)

**문장 C-1 (본문, 2장 나) 끝 또는 5장 한계).** 본 작품의 관찰은 flash-attn 2.8.x(실측 환경 2.8.3.post1), FlashInfer 0.6.13, 128개 SM을 가진 Ada(RTX 4090) 기준이다. Hopper용 FlashAttention-3[R16]는 비동기 실행과 저정밀도를 이용하는 별도 커널이며, 그 `flash_attn_with_kvcache`도 `num_splits=0`일 때 분할 수를 휴리스틱으로 정하지만[R17] 그 휴리스틱과 가변 길이 배치에서의 손실은 본 작품이 측정하지 않았다. FlashInfer 역시 버전에 따라 기본 변형과 스케줄링이 달라질 수 있어, vLLM 0.31이 번들하는 0.7.0.post1에서는 변형 선택 방식이 아래 각주와 같이 다르다. 따라서 "기본 설정이 틀린다"는 주장은 위 버전·세대에서 측정된 범위에 한정한다.

**각주 C-2 (FlashInfer의 tensor-core 선택 근거).** FlashInfer의 디코드 래퍼 `BatchDecodeWithPagedKVCacheWrapper`는 `use_tensor_cores`의 기본값이 `False`이며, 문서는 이 옵션이 "GQA의 그룹 크기가 클 때 더 빠를 것"이라고만 설명한다(설치본 flashinfer 0.6.13의 `decode.py` 744–746줄 docstring). "그룹 크기가 4 이상이면 tensor-core를 쓴다"는 기준은 FlashInfer 문서가 아니라 SGLang의 `should_use_tensor_core`에 있다. 이 함수는 KV 자료형이 fp16/bf16이면 `gqa_group_size >= 4`일 때 true를 돌려주고(fp8 KV 캐시는 항상 true), 주석으로 "FlashInfer에서는 헤드 그룹을 토큰 차원과 융합해 MMA로 처리하므로 GQA 그룹 크기가 최소 4여야 tensor core를 효율적으로 쓴다"고 적는다(`python/sglang/srt/layers/attention/flashinfer_backend.py` 2670–2716줄, 저장소 main, 2026-10-24 조회, 해당 파일의 최근 커밋 991b846). 한편 vLLM 0.31.0은 그룹 크기와 무관하게 디코드 래퍼를 `use_tensor_cores=True`로 만들며, 주석은 최신 GPU에서는 모든 attention 연산에서 CUDA core보다 성능이 같거나 낫다는 이유를 든다(`vllm/v1/attention/backends/flashinfer.py` 1260–1264줄, flashinfer-python 0.7.0.post1과 함께 설치). 본 작품의 Qwen3-4B는 질의 헤드 32개, KV 헤드 8개로 그룹 크기가 4이므로 SGLang 기준으로도 tensor-core 변형이 선택된다. 두 엔진이 같은 변형(tensor-core)을 고르는 이유는 다르다.

> **보고서 수정 제안(사용자 확인 필요).** 현재 2장 본문(`content1_v5.js` 23줄)의 "FlashInfer의 디코드 래퍼는 GQA 그룹 크기가 4 이상이면 tensor-core 변형을 권장하며"는 위 확인 결과와 맞지 않는다. 권장 수정: "FlashInfer의 디코드 래퍼는 `use_tensor_cores` 옵션을 사용자에게 맡기며(기본값 비활성), SGLang은 GQA 그룹 크기 4 이상에서, vLLM 0.31은 항상 tensor-core 변형을 선택한다. 어느 쪽도 배치의 길이 분포를 보지 않는다." PLAN §0의 "FlashInfer는 head 그룹 크기만 보고 tensor-core 변형을 고른다"도 같은 이유로 "FlashInfer를 쓰는 엔진은 head 그룹 크기(SGLang) 또는 고정값(vLLM)으로 변형을 고른다"로 고치는 것이 정확하다. FlashInfer 라이브러리 자체의 FA2 템플릿은 별도로 배치의 평균 질의 길이(GQA는 그룹 크기와 융합)로 query tile 크기를 고르는 휴리스틱을 가진다(논문 3.2.3절). PLAN.md는 이 단계에서 고치지 않았다.

---

## D. 표 2에 더할 행

기존 표 2("연구", "접근 방법", "본 작품과의 관계") 형식을 따른다. "FlashDecoding++" 행 아래에 넣는 것을 권한다.

| 연구 | 접근 방법 | 본 작품과의 관계 |
|---|---|---|
| TVM[R1]·Ansor[R2]\nTriton autotune[R3][R4] | 비용 모델·탐색 기반 오프라인 커널 구성 튜닝 | 같은 계열(실행 구성 선택). 모양별 오프라인 튜닝 대 매 디코드 단계의 길이 분포 기반 선택 |
| cuBLASLt[R5]\nTensorRT-LLM[R6] | 벤더 휴리스틱 후보 반환, XQA·multi-block 모드 선택 | 엔진 내장 디스패치. 문서의 기준은 배치×헤드 수이며 요청별 길이를 언급하지 않음 |
| vLLM[R7]·SGLang[R8] | 시작 시점의 attention 백엔드·변형 선택 | 같은 엔진에서 매 단계 재선택하지 않음. FlashInfer 변형은 시작 시점에 고정 |
| 배치 불변 추론\n(Thinking Machines Lab[R12], SGLang[R13], vLLM[R15]) | 분할 크기 고정 등으로 reduction 순서를 배치와 무관하게 고정 | 분기를 없애는 대안. 본 작품은 분할 수를 바꾸고 분기 지점을 측정·분류 |
| Yuan 등[R11] | 배치·GPU 구성에 따른 출력 변동의 수치적 원인 분석 | 분기 현상의 일반적 배경 |
| FlashAttention-3[R16] | Hopper용 비동기·저정밀도 attention 커널 | 본 작품의 측정 범위 밖(Ada, FA2 기준) |

---

## 참고문헌 (확인 상태 포함)

확인 상태: **원문** = 원문 페이지·소스를 직접 조회해 해당 내용을 확인, **검색** = 검색 결과 요약으로 서지만 확인(원문 본문 미조회), **부분** = 일부만 확인.

- [R1] T. Chen, T. Moreau, Z. Jiang, L. Zheng, et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI 2018. arXiv:1802.04799. https://www.usenix.org/conference/osdi18/presentation/chen — **검색**(arXiv 제출·OSDI 2018 게재, 초록 내용 확인; 저자 전체 목록은 원문 페이지에서 확인 필요)
- [R2] L. Zheng 외. "Ansor: Generating High-Performance Tensor Programs for Deep Learning." arXiv:2006.06762. https://arxiv.org/abs/2006.06762 — **원문**(P4 검토 2026-10-25: arXiv 초록 페이지의 Comments 줄이 "OSDI 2020"을 명시). OSDI 2020으로 인용한다.
- [R3] P. Tillet, H. T. Kung, D. Cox. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." MAPL 2019, pp. 10–19. https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf — **검색**
- [R4] Triton 문서 `triton.autotune`. https://triton-lang.org/main/python-api/generated/triton.autotune.html (2026-10-24 조회) — **원문**(configs, key, prune_configs_by 설명)
- [R5] NVIDIA cuBLASLt `cublasLtMatmulAlgoGetHeuristic` — **부분**: NVIDIA nvmath-python 바인딩 문서(https://docs.nvidia.com/cuda/archive/13.1.0/nvmath-python/0.2.1/bindings/generated/nvmath.bindings.cublasLt.matmul_algo_get_heuristic.html)가 요청 알고리즘 개수와 결과 배열을 받는 함수임을 보인다(검색 결과 요약). cuBLAS 공식 문서 본문(결과 정렬 규칙 포함)은 조회하지 못했으므로 "결과가 성능 순으로 정렬된다"는 서술은 쓰지 않았다 **[미확인]**.
- [R6] NVIDIA TensorRT-LLM 문서 "Multi-Head, Multi-Query, and Group-Query Attention". https://nvidia.github.io/TensorRT-LLM/legacy/advanced/gpt-attention.html (2026-10-24 조회, legacy 페이지) — **원문**(XQA 기본 활성·휴리스틱, multi-block 0.13부터 기본·`batch_size*num_heads < #SM` 시험 기준)
- [R7] vLLM 문서 "Attention Backend Feature Support"/설계 문서 `design/attention_backends`. https://docs.vllm.ai/en/latest/design/attention_backends/ (2026-10-24 조회) — **원문**(미지정 시 GPU 세대별 우선순위, `--attention-backend`)
- [R8] SGLang `flashinfer_backend.py`의 `FlashInferAttnBackend.__init__`·`should_use_tensor_core`. https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py — **원문**(소스 직접 조회, 2670–2716줄, 파일 최근 커밋 991b846). SGLang 시스템 논문: L. Zheng 외, "SGLang: Efficient Execution of Structured Language Model Programs," NeurIPS 2024, arXiv:2312.07104 — **검색**
- [R9] Z. Ye 외(L. Chen, R. Lai, W. Lin, Y. Zhang, S. Wang, T. Chen, B. Kasikci, V. Grover, A. Krishnamurthy, L. Ceze). "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving." MLSys 2025. arXiv:2501.01005 (v2, 2025-04-21). https://arxiv.org/abs/2501.01005 — **원문**(arXiv 초록 페이지와 HTML v2 본문에서 3.2.3절 tile 크기 휴리스틱, 3.3.1절 부하 균형 스케줄링 확인)
- [R10] K. Hong 외. "FlashDecoding++: Faster Large Language Model Inference on GPUs." arXiv:2311.01282. https://arxiv.org/abs/2311.01282 — **원문**(초록의 heuristic dataflow·50.25% 문구). 게재 학회는 확인하지 못했다 **[미확인]**(arXiv 번호로만 인용).
- [R11] J. Yuan, H. Li, X. Ding 외. "Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference." arXiv:2506.09501 (v2, 2025-10-24), NeurIPS 2025 구두 발표. https://arxiv.org/abs/2506.09501 — **검색**(초록 요약: 배치·GPU 수·GPU 종류에 따른 변동, bf16 greedy에서 정확도 최대 9%·길이 9,000 토큰 변동, LayerCast 완화). 본문은 조회하지 못했다.
- [R12] Horace He 외(Thinking Machines Lab). "Defeating Nondeterminism in LLM Inference." 2025-09-10. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ — **원문**(split-KV 분할 크기 고정 vs 분할 수 고정, KV 캐시 갱신 후 attention 호출, FlashInfer 균형 스케줄링이 배치 불변이 아니라는 서술). 블로그 글이며 학술 논문이 아니다. 저자 표기는 조회 결과(Horace He, "with others")에 따랐다.
- [R13] SGLang 문서 "Deterministic Inference". https://docs.sglang.io/advanced_features/deterministic_inference.html — **원문**(`--enable-deterministic-inference`, 지원 백엔드 3종)
- [R14] LMSYS Org. "Towards Deterministic Inference in SGLang and Reproducible RL Training." 2025-09-22. https://lmsys.org/blog/2025-09-22-sglang-deterministic/ — **검색**(검색 요약만; 오버헤드 수치는 원문 미조회이므로 본문에 쓰지 않았다)
- [R15] vLLM 문서 "Batch Invariance". https://docs.vllm.ai/en/latest/features/batch_invariance/ — **원문**(성능 영향 문구). `VLLM_BATCH_INVARIANT` 변수명은 검색 결과 요약에서 확인했다. 이 문서는 attention의 split-KV 처리를 설명하지 않는다.
- [R16] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, T. Dao. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." NeurIPS 2024. arXiv:2407.08608. https://arxiv.org/abs/2407.08608 — **검색**
- [R17] Dao-AILab/flash-attention `hopper/flash_attn_interface.py`의 `flash_attn_with_kvcache` docstring("If num_splits == 0, we use a heuristic to automatically determine the number of splits"). https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py — **원문**(2026-10-24 조회)
- 기존 보고서 [20] LeanAttention: R. Sanovar 외, arXiv:2405.10480. 이번에 서지를 다시 확인했다. arXiv v1 초록은 FlashAttention-2 대비 평균 2.6배, 512k 문맥에서 최대 8.33배를 보고하고, MLSys 2025 게재판은 검색 요약 기준 FlashDecoding 대비 평균 1.73배(256k에서 최대 2.18배)로 보고한다 — **검색**(v1은 원문, MLSys판 수치는 요약). 본 작품이 직접 측정하지 않았으므로 "보고하였다" 수준으로 서술하고, 인용할 판(v1/MLSys 2025)을 정해 수치와 비교 대상을 맞춰 쓴다. 현재 보고서 2장은 수치를 쓰지 않는다.
- 기존 보고서 [4] Flash-Decoding: PyTorch 블로그 "Flash-Decoding for long-context inference," https://pytorch.org/blog/flash-decoding/ — **원문**(KV를 chunk로 나눠 병렬 계산 후 log-sum-exp로 결합). 이 페이지 사본에서 저자 표기를 확인하지 못했다 **[미확인]**.

## 확인 방법 메모

- 서지: WebSearch로 제목·저자·학회/연도·arXiv 번호를 대조했고, 본문 서술에 쓴 항목은 WebFetch로 원문 페이지(논문 HTML·문서·블로그·GitHub raw 소스)를 직접 조회해 문구를 확인했다. 검색 결과 요약에만 의존한 항목은 위에 "검색"으로 표시했다.
- 소스 코드: SGLang은 `curl`로 받은 `flashinfer_backend.py`에서 직접 줄을 확인했다(2026-10-24). vLLM은 설치본 `~/.venvs/kernelscope-vllm/lib/python3.12/site-packages/vllm/v1/attention/backends/flashinfer.py`(vllm 0.31.0, flashinfer-python 0.7.0.post1), FlashInfer 0.6.13은 `kernelscope-design/.venv/lib/python3.11/site-packages/flashinfer/decode.py`에서 확인했다. 실측 환경의 flash-attn은 2.8.3.post1이다.
- 확인하지 못한 것: cuBLASLt 공식 문서의 결과 정렬 규칙, Ansor의 학회명, FlashDecoding++의 학회명, Flash-Decoding 블로그의 저자, LMSYS 블로그의 오버헤드 수치, vLLM 문서의 `VLLM_BATCH_INVARIANT` 줄 직접 인용(검색 요약만). 이 항목들은 본문에서 단정하지 않았다.
- P4 검토(2026-10-25, arXiv 초록 페이지 직접 조회): R2(Ansor, OSDI 2020 확인), R9(FlashInfer, 저자 Ye·Chen·Lai…Ceze, v2 2025-04-21, "Accepted by MLSys 2025" 확인), R10(FlashDecoding++, 초록의 50.25% 문구 확인), R11(제목·저자 Yuan·Li·Ding, v2 2025-10-24, 9%·9,000 토큰 문구 확인; **NeurIPS 2025 구두 발표는 arXiv 페이지에 없어 미확인으로 남김**), R16(FA3 제목·저자 6인 확인; NeurIPS 2024는 arXiv 페이지에 없어 검색 근거만). 다섯 건 모두 존재하고 서지가 본문과 일치한다.
- 범위 주의: R11·R12는 현상의 원인과 해법을 다루며, 본 작품의 분기 지점 분석 방법(teacher forcing, bf16 ULP 분류)이 "표준"이라는 근거는 이 문헌들에서 찾지 못했다. B-3의 괄호 안 권고를 따른다.
