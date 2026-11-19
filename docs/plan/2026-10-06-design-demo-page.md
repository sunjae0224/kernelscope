# 설계: 시연용 Demo 페이지 — 고정 순서 장면 + 토큰 레이스

작성 2026-10-06. 선행: [2026-09-19-plan-p3-dashboard.md](2026-09-19-plan-p3-dashboard.md)(대시보드 원 계획), [docs/demo.md](../demo.md)(5분 시나리오), 루트 `run.sh`.
상태: **사용자 승인**(접근 1 "기존 Streamlit 위에 Demo 페이지 추가, 기존 5탭은 Lab으로 유지", 범위 "주요 기능과 결과만"). 시연 환경은 발표 노트북 + 연구실 4090 원격, 1순위 메시지는 **"같은 답, 더 빠른 토큰"**.

---

## 0. 한 문단 요약

지금 대시보드는 셀렉트박스가 많은 탐색용 연구실이라 발표 흐름이 클릭 순서에 달려 있고, 핵심 메시지인 "출력은 같고 토큰은 더 빨리 나온다"를 보여 주는 화면이 없다. 이 작업은 같은 Streamlit 앱에 **Demo 페이지**를 추가해 결론 → 경주 → 왜 → 어떻게 → 검증의 다섯 장면을 고정 순서로 두고, 기존 5탭은 **Lab 페이지**로 남겨 Q&A의 drill-down에 쓴다. 경주 장면은 Chatbot Arena식 A/B 두 패널이며, 같은 GPU에서 **순차 측정한 기록을 타임스탬프대로 동시에 재생**한다(동시 실행은 측정을 오염시키므로 하지 않는다). 재생 컨트롤은 작은 HTML+JS 컴포넌트 안에 두어 Streamlit의 재실행과 분리한다. 데이터 로더·그림·테스트는 그대로 재사용하고 새 스택은 없다. 조사한 패턴은 Nsight Compute의 "요약 → 상세 + baseline 덧씌우기", HTA의 "무엇 → 왜" 질문 순서, PyTorch Profiler의 한 줄 권고 패널, vLLM 블로그의 "속도 수치와 기전 설명 분리", speculative decoding 계열의 "출력 동일 명시"다.

## 1. 범위

**한다.**
- 진입점 `dashboard/app.py`를 `st.navigation` 두 페이지(Demo 기본, Lab)로 바꾸고 `st.set_page_config`는 여기서만 호출. 기존 본문은 `dashboard/lab_page.py`로 이동(내용 불변).
- `dashboard/demo_page.py`: 장면 0~4(§3).
- `kernelscope/dashboard/demo.py`: Streamlit·torch 없이 기록 파일 → 레이스 payload, 토큰 일치, CTA 작업량, 결정 카드, 라이브 시나리오 조립(§5). 단위 테스트 가능.
- `kernelscope/dashboard/race.html`: 두 패널 토큰 재생 컴포넌트(§4).
- `lab.compose_command`에 단일 워크로드 bench 옵션 추가(플러그인 벤치용).
- `scripts/export_race.py`: 기록 폴더 → `race.json`(토크나이저 없는 장비에서도 글이 보이게 사전 디코드).
- 시나리오 `scenarios/demo_text_ragged.yaml`(완료: 32768×1 + 512×31, 64토큰, 자연어 corpus_repeat)과 그 GPU 기록 1회(`run.sh demo-record`), 번들 `demo_data/serve_4090/demo_text_20261006/`.
- `run.sh`: `demo-record` 추가, `dashboard`는 그대로(이제 Demo가 첫 화면).
- 문서: `docs/demo.md`를 장면 순서로 재작성, README 데모 절 갱신, TODO/PLAN/STATUS 한 줄씩.

**하지 않는다.** 새 측정 코드, 두 정책 동시 실행, 임의 커널 파일 업로드(등록된 플러그인 선택만), 백그라운드 작업의 세션 밖 생존, Gradio·별도 정적 사이트, Lab 탭 내용 변경, 새 수치의 문서 인용(화면 수치는 전부 기록에서 계산하므로 verify 항목 추가 없음. 문서에 새 수치를 적을 때만 verify 항목을 추가한다).

## 2. 참고 패턴 → 적용

| 패턴 | 출처 | Demo 적용 |
|---|---|---|
| 결론 먼저, 상세는 drill-down | Nsight Compute Summary → Details | 장면 0 타일 3개, 나머지는 아래 장면과 Lab |
| baseline을 같은 화면에 덧씌움 | Nsight Compute baseline, HTA Trace Diff | 모든 수치 옆에 heuristic 기준값·배율 |
| 무엇 → 왜 | HTA Temporal → Kernel breakdown | 장면 1(무엇) → 장면 2(왜) |
| 한 줄 권고 패널 | PyTorch Profiler Performance Recommendation | 장면 3 결정 카드 |
| 속도 수치와 기전 설명 분리 | vLLM 블로그 | 장면 1 수치 / 장면 2 기전 |
| A/B 두 패널, 같은 입력 | Chatbot Arena | 장면 1 레이스 |
| 출력 동일을 문장으로 명시 | speculative decoding 계열 | 레이스 하단 "생성 토큰 일치 N/N 요청" |
| 입력을 바꾸면 결과가 바뀜 | Compiler Explorer | 라이브 측정(배치 구성 → 측정 → 재생) |

## 3. 화면 구조

진입 `dashboard/app.py`:
```python
st.set_page_config(page_title="KernelScope", page_icon="◈", layout="wide")
pages = [st.Page("demo_page.py", title="Demo", default=True), st.Page("lab_page.py", title="Lab")]
st.navigation(pages).run()
```
두 페이지 모두 `ROOT`를 `Path(__file__).resolve().parents[1]`로 잡고 `sys.path`에 넣는다(기존 방식). 테마 판별 코드는 두 페이지가 공유하도록 `kernelscope/dashboard/style.py`에 `current_theme(default)` 헬퍼로 올린다(순수 함수 아님, Streamlit import는 함수 안에서).

Demo 페이지는 위에서 아래로 한 흐름이며 탭이 없다. 각 장면은 제목 한 줄 + 주장 한 문장 + 시각물 하나(최대 둘) + 근거 라벨(실측/재생/예측). 셀렉트박스는 장면당 최대 하나이며 기본값으로 발표가 끝나야 한다.

**장면 0 · 한 줄 결론.** 대표 기록(§3.1)의 `summary.csv`에서 heuristic TPOT, 선택 정책 TPOT, 배율, `tokens_equivalent`를 읽어 큰 타일 3개(TPOT 전→후, 배율, 토큰 일치). 아래에 "모델 · 시나리오 · 반복 수 · 측정일" 한 줄.

**장면 1 · 경주.** 프리셋 라디오(혼합 길이 / 요청 도착 / 균일), 비교 정책 라디오(table / hybrid / model, 기록에 있는 것만). 레이스 컴포넌트(§4). 아래에 "순차 측정, 동시 재생" 라벨과 "생성 토큰 일치 N/N 요청" 문장(불일치가 있으면 요청·위치와 Lab 05 분류표로 가는 안내). 접힌 패널 "라이브 측정"(§5).

**장면 2 · 왜 느린가.** 최악 셀(`decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal`, paged, cold)에 대해 좌: 배치 구성 막대(`figures.workload_lengths`), 우: CTA 작업량(§6 `cta_work`) — 휴리스틱이 고르는 분할 수(라이브러리 규칙 `num_splits_heuristic`)와 테이블이 고른 분할 수에서 CTA별 KV 토큰을 정렬한 두 곡선, 지표 "가장 긴 CTA / 평균 CTA", "CTA 수 / SM 수". 그 아래 실측 커널 막대(`figures.variants_bar`, 모델 예측 없이 실측만)와 연산 분해(`diagnose_20260927/ragged/op_breakdown.png` + `diagnosis.json`의 attention 비중 두 값). 라벨: 커널 막대와 연산 분해는 실측, CTA 작업량은 길이와 분할 수에서 계산한 값.

**장면 3 · 어떻게 고르나.** 좌: 커널 지도 히트맵(`figures.regret_heatmap`, paged cold), 우: **결정 카드**(§6 `decision_card`) — 같은 배치에 대해 heuristic / table / hybrid / model이 고른 분할 수, 그 변형의 실측 커널 시간, heuristic 대비 배율, table은 일치한 셀과 거리, hybrid는 출처(table/model). 아래 "다른 커널은?" — `lab.backend_comparison`의 최악 셀 행으로 heuristic · 최선 FA2 · FlashInfer(tensor-core, CUDA-core) 막대. 접힌 패널 "플러그인 벤치": 등록 플러그인 다중 선택 → `bench --workload <최악 셀>` 명령 표시 → GPU 호스트면 실행(`lab.launch`), 끝나면 `summaries.jsonl`을 읽어 같은 막대에 추가.

**장면 4 · 검증.** 대표 캠페인의 조건별 표(혼합 길이 / 요청 도착 / 균일 × 정책: 배율, 반복, 토큰 일치) — 균일에서 이득이 없다는 대조군을 그대로 보여 준다. `lab.divergence_overview` 요약(사건 수와 tie 분류). 버튼 "문서 수치 재계산": `kernelscope.cli verify` 서브프로세스 실행 후 마지막 줄(`{"PASS": n}`)과 꼬리 출력.

### 3.1 대표 기록 선택

`demo.featured_runs(root) -> dict[str, Path]`: `data.find_dirs("serve", root)`의 시나리오 폴더에서 `manifest.json`을 읽어 `scenario_family`(없으면 시나리오 파일명에서 ragged/arrivals/uniform 추정)별로 하나를 고른다. 우선순위: 자연어(`prompt_kind != seeded_synthetic_token_ids`) > **혼합 길이 선택과 같은 모델** > 최신 `created_at`(구현 중 결정: 균일 프리셋이 Llama 후속 실험으로 가는 것을 막아 프리셋이 한 모델에 머물게 한다). 결과 루트에 없으면 `demo_data`도 본다(`data.results_root()` 규칙과 같음). 장면 1의 프리셋은 이 dict의 키이고, 접힌 "다른 기록" 셀렉트로 바꿀 수 있다.

장면 4의 "대표 캠페인"은 `demo.featured_campaign(root) -> Path | None`: 시나리오 폴더의 부모 중 가장 많은 배치 종류(ragged/arrivals/uniform)를 가진 캠페인, 동률이면 최신. 자연어 혼합 길이 기록(`demo_text_20261006`)은 조건이 하나뿐인 캠페인에 있고 균일 대조군은 `hybrid_20261002`에 있어서, 레이스의 기록과 검증 표의 캠페인을 분리했다.

## 4. 레이스 컴포넌트

**payload**(`demo.race_payload(run_dir, policy_a, policy_b, decode=None) -> dict`, JSON 직렬화 가능):
```
scenario: {name, B, lens: [...], max_new_tokens, natural_text: bool, model, repeats}
t0_rule: "all_admitted"            # 시계 0 = 마지막 prefill의 first_token_us(요청 도착형은 첫 prefill 끝); 음수는 0으로
policies: [ {name, label, color, repeat_index, tpot_ms, end_ms,
             steps: [{step, t_ms, num_splits, attn_ms}],
             requests: [{rid, prompt_len, featured: bool, tokens: [{t_ms, text}]}]}, ×2 ]
agreement: {identical: bool, requests_equal, requests_total, first_divergence: [{rid, position}]}
```
- 반복 선택: 정책별로 `end_ms`가 중앙값인 repeat(짝수면 작은 쪽). `repeat_index`를 기록.
- 토큰 글: `decode(ids[:i+1])`에서 `decode(ids[:i])`를 뺀 증분(다중 바이트 안전). `decode`가 None이거나 합성 토큰 기록이면 `text`는 빈 문자열이고 `natural_text=false` → 컴포넌트는 글 대신 토큰 수만 보여 준다.
- `featured`: 가장 긴 요청 하나와 가장 짧은 요청 중 rid가 가장 작은 하나.
- steps의 `t_ms`: 같은 step의 decode 토큰 `t_us`(steps.parquet에는 시각이 없다).
- 캐시: `demo.save_race(run_dir, payload)` / `demo.load_race(run_dir)` → `race.json`. 페이지는 race.json → 토크나이저 → 글 없는 payload 순으로 시도한다. `scripts/export_race.py --run <dir> --policies heuristic,table[,hybrid]`가 race.json을 쓴다(정책 쌍마다 하나의 파일이 아니라 모든 정책을 담은 하나의 파일; 컴포넌트에는 두 개만 넘긴다).

**표시**(두 패널, 같은 폭): 패널 제목(정책 라벨, 색), 경과 시계(ms), tokens/s(누적), 요청별 진행 격자(B칸, 토큰 수만큼 채움), 대표 요청 2개의 글(또는 토큰 수), step 띠(분할 수 숫자 + attention ms 막대), 종료 시 "완료 · N ms" 깃발. 패널 사이에 공통 컨트롤: ▶/⏸, ↺, 속도 0.25×/0.5×/1×(기본 0.5×). 하단에 `agreement` 문장. 모든 컨트롤·상태는 JS 안에 있고 Streamlit 위젯이 아니다(재실행 시 컴포넌트가 다시 마운트되는 것은 허용).

**색·테마**: 정책 색은 `style.policy_color`(heuristic 파랑, table 초록, model 주황, hybrid 슬롯 6 보라 — 기존 그림과 동일)로 Python에서 넣는다. 표면·글자색은 `style.SURFACE/TEXT`를 payload의 `theme`로 받는다. 글자색에 정책 색을 쓰지 않는다.

**구현**: 템플릿 파일의 `__PAYLOAD__`를 `json.dumps`로 치환해 `st.components.v1.html(html, height=…)`. 외부 스크립트 없음.

## 5. 라이브 측정과 플러그인 벤치(GPU 호스트)

- 입력: 긴 요청 길이(8192/16384/32768), 짧은 요청 수(7/15/31), 짧은 길이(384/512), 생성 토큰(32/64), 질문 한 줄(기본 "Read the notes above. Summarize one monitoring practice in a short sentence.\nAnswer:"), 비교 정책(table/hybrid/model).
- `demo.compose_scenario(long_len, n_short, short_len, max_new_tokens, question, corpus, dataset) -> dict`는 `scenarios.load_scenario`가 받는 구조를 만든다(corpus와 dataset은 `demo_text_ragged.yaml`에서 읽어 재사용). `demo.write_scenario(path, spec)`.
- 실행: `$KERNELSCOPE_RESULTS/serve_4090/demo_live_<UTC stamp>/scenario.yaml`에 쓰고 `lab.compose_command("serve_run", scenario=…, policies=["heuristic", <정책>], out=…/ragged, kv_gib=10, warmup_runs=1, warmup_steps=2, repeats=1, seed=0, policy_cache="keep")` → `serve doctor` 통과 시 `lab.launch`. 로그는 `lab.tail`. 완료(`poll()`이 None이 아니고 `summary.csv` 존재) 시 그 폴더를 장면 1의 기록 선택지에 넣고 선택한다. 소요 1~2분(모델 로드 포함)이라 발표 중에는 기록 재생을 쓰고 라이브는 Q&A·발표 직전용임을 화면에 적는다.
- 플러그인 벤치: `kernelscope.plugins` 레지스트리의 이름 중 최악 셀을 지원하는 것을 다중 선택. `lab.compose_command("bench", workload=<최악 셀>, plugins=…, results=…, cache_state="cold", warmup=5, iters=20)`(`workload=`가 있으면 `--grid` 대신 `--workload`). 결과 `summaries.jsonl`의 `status=="ok"` 행을 실측 막대에 추가하고 실패 행은 이유를 표시.
- GPU 없는 장비: 두 버튼 비활성, 명령만 표시(Policy lab과 같은 규칙).

## 6. 모듈·인터페이스

`kernelscope/dashboard/demo.py` (torch·streamlit import 없음):
- `WORST_KEY = "decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal"`
- `featured_runs(root) -> dict[str, Path]` (§3.1)
- `headline(run_dir, policy) -> dict`: `{heuristic_tpot_ms, policy_tpot_ms, speedup, tokens_equivalent, repeats, model, scenario_name, created_at}`
- `race_payload(run_dir, policy_a, policy_b, decode=None) -> dict`, `save_race`, `load_race`
- `token_agreement(run_dir, policy_a, policy_b) -> dict` (payload의 `agreement`; 반복은 payload와 같은 repeat)
- `cta_work(lens, num_splits, n_kv_heads) -> pd.DataFrame[cta, request, kv_tokens]`; `num_splits<=0`이면 `model.geometry.num_splits_heuristic`로 라이브러리 규칙을 적용(필요 인자: B·H_kv, n_sm, KV 블록 수 = ceil(max(lens)/256)? → 실제 라이브러리 규칙의 인자 정의는 `geometry.py`를 따른다)
- `decision_card(lens, n_heads, n_kv_heads, table_csv, machine, params, measured: pd.Series) -> list[dict]`: 정책별 `{policy, num_splits, kernel, kernel_us, speedup_vs_heuristic, note}`; `make_policy`로 heuristic/table/hybrid/model을 만들고 `last`에서 출처·셀·거리를 읽는다. `measured`는 최악 셀의 커널별 중앙값(µs). 모델 정책이 실패하면 그 행만 `note`에 사유.
- `compose_scenario(...)`, `write_scenario(path, spec)`, `load_demo_corpus(path) -> (corpus, dataset)`
- `verify_tail(output: str) -> dict`: 마지막 줄 `{"PASS": n}` 파싱.

`kernelscope/dashboard/figures.py` 추가: `cta_work_curves(work_a, work_b, labels, theme)`, `backend_bar(row, theme)`. 기존 함수 재사용: `workload_lengths`, `variants_bar`, `regret_heatmap`.

`kernelscope/dashboard/lab.py`: `compose_command(kind="bench", workload=None, warmup=None, iters=None, …)` 확장(기존 호출 호환).

`kernelscope/dashboard/style.py`: `current_theme(default="light") -> str`.

## 7. 테스트

- `tests/test_dashboard_demo.py`: 합성 기록(tmp_path에 manifest/summary/steps/tokens/prefill parquet 두 정책 × 2 repeat) → `race_payload` 구조·t0·repeat 선택·featured·증분 디코드(가짜 decode), `token_agreement`(일치/불일치 1건), `save/load_race`, `featured_runs`(자연어 우선·최신 우선), `headline`, `cta_work`(길이 합 보존, CTA 수 = B·H_kv·splits), `decision_card`(heuristic·table은 실제 `demo_data/dispatch_paged_cold.csv`로, model/hybrid는 machine/params 파일로; 예외 시 note), `compose_scenario` → `write_scenario` → `scenarios.load_scenario` 왕복(요청 수 = 1 + n_short), `verify_tail`, race.html 템플릿에 `__PAYLOAD__`가 한 번 있고 치환 결과가 유효 HTML 조각인지.
- `tests/test_dashboard_app.py` 갱신: 진입점 AppTest → Demo가 빈 루트와 기록 있는 루트에서 예외 없음, 장면 0 타일 존재; `switch_page`로 Lab → 기존 단언(5탭 등) 유지. `tests/test_dashboard_lab.py`에 `compose_command(workload=…)` 케이스 추가.
- 브라우저 검증(수동, 보고에 스크린샷): Demo 페이지 라이트 테마 전체, 레이스 재생 중·종료 후, Lab 전환. 레이스 종료 시각 비(패널 A end_ms / B end_ms)가 payload의 TPOT 비와 같은 방향인지 확인.
- `make test`(CUDA 숨김), `make verify` 111/111 유지.

## 8. 합격 기준

- A1 빈 결과 루트, `demo_data` 루트 둘 다에서 Demo·Lab 렌더 예외 없음.
- A2 레이스 payload가 합성 기록(`hybrid_20261002/ragged`)과 자연어 기록(`demo_text_20261006/ragged`)에서 만들어지고 `natural_text`가 각각 false/true.
- A3 `agreement.identical`이 해당 기록 `summary.csv`의 `tokens_equivalent`와 일치.
- A4 기존 Lab 테스트와 verify 111/111 통과, 새 테스트 포함 `make test` 전부 통과.
- A5 GPU 기록 1회: `demo_text_ragged`에서 heuristic·table·hybrid 3회 반복, 토큰 일치 여부와 배율을 결과 그대로 기록(수치는 미리 정하지 않는다).
- A6 `./run.sh dashboard`로 띄운 화면에서 레이스가 재생되고 두 패널의 종료 시각이 다르며, 하단 일치 문장이 보인다(스크린샷).

## 9. 문서·실행기

- `docs/demo.md`: 장면 순서 = 발표 순서(0:00 결론, 0:20 경주, 1:50 왜, 2:50 어떻게, 3:50 검증, 4:30 한계)와 Q&A 때 Lab으로 가는 안내. 기존 Q&A 문답은 유지.
- README "졸업 프로젝트 데모" 절: 첫 화면이 Demo임을 한 줄, `run.sh demo-record` 한 줄.
- `run.sh`: `demo-record`(demo_text_ragged, heuristic/table/hybrid, keep, REPEATS 기본 3, 끝나면 `scripts/export_race.py`), usage에 추가.
- TODO.md §1 6번을 "Demo 페이지 완료, 남은 것: 백그라운드 작업 세션 밖 생존"으로, PLAN §2 5에 한 줄, STATUS에 항목 하나.
