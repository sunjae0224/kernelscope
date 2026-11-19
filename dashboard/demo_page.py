"""Demo page: fixed-order scenes for the presentation. Lab (lab_page.py) keeps the exploration tabs.

Scenes: 0 headline tiles · 1 token race (sequential measurements replayed on one clock) + live
measurement panel · 2 why (batch shape, CTA work, measured kernels, op breakdown) · 3 how (kernel map,
decision card, other backends, plugin bench) · 4 verification (campaign table, divergence, verify)."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from kernelscope.analysis.dispatch import FAMILIES, dispatch_table
from kernelscope.dashboard import data, demo, figures, lab, style
from kernelscope.workload import Workload

theme = style.current_theme()
root = data.results_root()
PY = sys.executable
GPU_HOST = Path("/usr/bin/nvidia-smi").exists() or Path("/usr/local/bin/nvidia-smi").exists()
TABLE_CSV = ROOT / "demo_data" / "dispatch_paged_cold.csv"
MACHINE, PARAMS = ROOT / "machines" / "rtx4090.json", ROOT / "models" / "rtx4090.json"
STAMP = st.session_state.setdefault("demo_stamp", pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ"))

st.markdown(style.PAGE_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def featured(root_str: str) -> dict:
    return {family: str(path) for family, path in demo.featured_runs(Path(root_str)).items()}


@st.cache_data(ttl=300, show_spinner="기록을 읽는 중…")
def race(run_dir: str, policies: tuple) -> dict | None:
    cached = demo.load_race(run_dir)
    if cached and {p["name"] for p in cached["policies"]} >= set(policies):
        return cached
    decode = None
    try:  # without torch or the local model snapshot the race shows token counts instead of text
        from kernelscope.serve.hf import load_tokenizer, snapshot_dir
        tokenizer = load_tokenizer(snapshot_dir(data.load_manifest(run_dir).get("model")))
        decode = lambda ids: tokenizer.decode(list(ids), skip_special_tokens=True)
    except Exception:
        decode = None
    try:
        return demo.race_payload(run_dir, list(policies), decode=decode)
    except (FileNotFoundError, ValueError, OSError):
        return None


def chart(fig, key):
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displaylogo": False})


def fmt(value, suffix="", digits=1):
    return "—" if value is None or pd.isna(value) else f"{value:,.{digits}f}{suffix}"


def job_panel(prefix: str, argv, start_label: str, before_launch, done_check):
    """Shared launcher: command preview, doctor gate, background job, log tail. Returns True once done."""
    st.code(lab.shell_line(argv), language="bash")
    job, log_path = st.session_state.get(f"{prefix}_job"), st.session_state.get(f"{prefix}_log")
    running = job is not None and job.poll() is None
    if st.button(start_label, disabled=not GPU_HOST or running, key=f"{prefix}_start"):
        try:
            doctor = subprocess.run([PY, "-m", "kernelscope.cli", "serve", "doctor"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            doctor = None
            st.error("serve doctor가 2분 안에 끝나지 않았습니다.")
        if doctor is None:
            pass
        elif doctor.returncode != 0:
            st.error("serve doctor 실패 · 다른 GPU 프로세스가 있거나 모델·의존성이 준비되지 않았습니다.")
            st.code((doctor.stdout or doctor.stderr)[-1500:], language="text")
        else:
            log_path = before_launch()
            st.session_state[f"{prefix}_job"] = lab.launch(argv, log_path, cwd=ROOT)
            st.session_state["demo_stamp"] = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
            st.session_state[f"{prefix}_log"] = str(log_path)
            st.rerun()
    if log_path:
        st.caption(("실행 중 · " if running else f"종료 (코드 {job.returncode}) · " if job is not None else "기록 · ") + str(log_path))
        if job is not None and not running and not done_check():
            st.error("실행이 끝났지만 결과 파일이 없습니다. 아래 로그를 확인하세요.")
        st.code(lab.tail(log_path, 30) or "(아직 출력이 없습니다)", language="text")
        if st.button("로그 새로고침", key=f"{prefix}_refresh"):
            st.rerun()
        if not running and done_check():
            return True
    if not GPU_HOST:
        st.caption("이 장비에는 nvidia-smi가 없어 실행 버튼이 비활성입니다. 명령을 복사해 연구실 호스트에서 돌리세요.")
    return False


# ---------------------------------------------------------------- scene 0 · headline
runs = featured(str(root))
st.markdown('<div class="eyebrow">KERNELSCOPE · DEMO</div>', unsafe_allow_html=True)
st.title("같은 답, 더 빠른 토큰.")
st.markdown('<div class="hero-note">길이가 다른 요청이 한 배치에 섞이면 attention 커널이 GPU를 다 쓰지 못합니다. '
            'KernelScope는 매 decode step의 KV 분할 수를 골라, 출력은 그대로 두고 토큰 간격을 줄입니다.</div>',
            unsafe_allow_html=True)
if not runs:
    st.info("서빙 기록이 없습니다. 결과 루트(KERNELSCOPE_RESULTS) 또는 demo_data 아래 serve_4090/<campaign>/<scenario>/summary.csv가 필요합니다.")
    st.stop()
hero_family = "ragged" if "ragged" in runs else next(iter(runs))
hero_run = runs[hero_family]
hero_candidates = [p for p in ("table", "hybrid", "model", "fixed8") if p in demo.policy_dirs(hero_run)]
hero = demo.headline(hero_run, hero_candidates[0] if hero_candidates else "heuristic")
tiles = st.columns(3)
tiles[0].metric("토큰 간격 TPOT · 기본 → KernelScope", f"{fmt(hero['heuristic_tpot_ms'])} → {fmt(hero['policy_tpot_ms'])} ms",
                help="요청이 토큰을 받는 평균 간격. 왼쪽은 FlashAttention 휴리스틱, 오른쪽은 KernelScope 정책.")
tiles[1].metric("배율", fmt(hero["speedup"], "×", 2), help="heuristic TPOT ÷ 정책 TPOT, 같은 실험의 반복 평균.")
tiles[2].metric("생성 토큰 일치", {True: "일치", False: "불일치", None: "—"}[hero["tokens_equivalent"]],
                help="모든 반복에서 정책의 생성 토큰이 heuristic과 같았는지.")
st.caption(f"실측 · {hero['model']} · {demo.FAMILY_LABELS.get(hero_family, hero_family)} ({hero['scenario_name']}) · 정책 {hero['policy']} · "
           f"반복 {hero['repeats']}회 · 측정 {str(hero['created_at'] or '')[:10]} · {hero_run}")

# ---------------------------------------------------------------- scene 1 · race
st.divider()
st.subheader("1 · 경주 — 같은 배치, 같은 모델, 다른 커널 선택")
st.write("왼쪽은 FlashAttention의 기본 분할 수 규칙, 오른쪽은 KernelScope의 선택입니다. 같은 GPU에서 **순차로 측정한 기록**을 "
         "타임스탬프대로 **동시에 재생**합니다. 두 정책을 동시에 돌리면 측정이 서로를 오염시키므로 그렇게 하지 않습니다.")
pick = st.columns([1.2, 1.6, 2])
family = pick[0].radio("배치 종류", list(runs), format_func=lambda f: demo.FAMILY_LABELS.get(f, f), horizontal=True, key="race_family")
run_dir = runs[family]
with st.expander("다른 기록 고르기"):
    all_runs = sorted({str(d) for d in data.find_dirs("serve", root) if (d / "summary.csv").exists()}
                      | set(st.session_state.get("demo_live_runs", [])))
    run_dir = st.selectbox("기록 폴더", all_runs, index=all_runs.index(run_dir) if run_dir in all_runs else 0, key=f"race_run_{family}")
available = demo.policy_dirs(run_dir)
candidates = [p for p in ("table", "hybrid", "model", "fixed8", "fa2") if p in available]
if "heuristic" not in available or not candidates:
    st.info("이 기록에는 heuristic과 비교할 정책 쌍이 없습니다. 다른 기록을 고르세요.")
else:
    policy = pick[1].radio("KernelScope 정책", candidates, horizontal=True, key="race_policy")
    payload = race(str(run_dir), tuple(["heuristic", *candidates]))
    if payload is None:
        st.warning("이 기록의 토큰 parquet를 읽을 수 없습니다.")
    else:
        st.iframe(demo.render_race_html(payload, "heuristic", policy, theme), height=600)
        agreement = payload["agreement"].get(policy, {})
        if agreement.get("identical"):
            st.success(f"생성 토큰 일치 · {agreement['requests_equal']}/{agreement['requests_total']} 요청 — 재생한 반복에서 두 정책이 같은 토큰을 만들었습니다. "
                       f"전체 반복 기준: {'일치' if agreement.get('summary_tokens_equivalent') else '불일치 또는 미기록'}.")
        elif not agreement.get("requests_total"):
            st.caption("이 정책의 토큰 일치 정보가 없습니다.")
        else:
            first = agreement.get("first_divergence", [])
            st.warning(f"생성 토큰 불일치 {len(first)}건 · 요청 {', '.join(str(d['rid']) for d in first[:5])} — 분기 사건의 tie 분류는 Lab의 05 Policy lab에서 봅니다.")
        pick[2].caption(f"{payload['scenario']['B']}개 요청 · 프롬프트 {min(payload['scenario']['lens']):,}~{max(payload['scenario']['lens']):,} tokens · "
                        f"{payload['scenario']['max_new_tokens']}토큰 생성 · 반복 {payload['scenario']['repeats']}회 중 중앙값 실행을 재생")
        if not payload["scenario"]["natural_text"]:
            st.caption("이 기록은 합성 토큰 프롬프트이거나 토크나이저가 없어 글 대신 토큰 수로 재생합니다. 자연어 기록(demo_text_ragged)은 글이 보입니다.")
        st.caption("재생 · 시계 0 = 첫 요청 묶음의 prefill이 모두 끝난 시점. 띠는 step별 분할 수와 attention 시간(실측).")

with st.expander("라이브 측정 · 배치를 구성해 GPU 호스트에서 재고 바로 재생 (모델 로드 포함 1~2분 · 발표 중에는 기록 재생을 권함)"):
    c = st.columns(5)
    long_len = c[0].selectbox("긴 요청 길이", [8192, 16384, 32768], index=2, key="live_long")
    n_short = c[1].selectbox("짧은 요청 수", [7, 15, 31], index=2, key="live_n")
    short_len = c[2].selectbox("짧은 요청 길이", [384, 512], index=1, key="live_short")
    new_tokens = c[3].selectbox("생성 토큰", [32, 64], index=1, key="live_tokens")
    live_policy = c[4].selectbox("비교 정책", ["table", "hybrid", "model"], key="live_policy")
    question = st.text_input("긴 문서에 던질 질문 (프롬프트 끝에 붙습니다)", value=demo.DEFAULT_QUESTION, key="live_question")
    live_dir = root / "serve_4090" / f"demo_live_{STAMP}"
    spec_map = {"table": f"table:{TABLE_CSV}", "hybrid": f"hybrid:{TABLE_CSV}:0.2", "model": "model"}
    argv = lab.compose_command("serve_run", python=PY, scenario=live_dir / "scenario.yaml", policies=["heuristic", spec_map[live_policy]],
                               out=live_dir / "ragged", kv_gib=10, warmup_runs=1, warmup_steps=2, repeats=1, seed=0,
                               policy_cache="keep", machine=str(MACHINE), params=str(PARAMS))

    def _prepare_live():
        corpus, dataset = demo.load_demo_corpus(ROOT / "scenarios" / "demo_text_ragged.yaml")
        spec = demo.compose_scenario(int(long_len), int(n_short), int(short_len), int(new_tokens), question, corpus, dataset)
        demo.write_scenario(live_dir / "scenario.yaml", spec)
        st.session_state["demo_live_dir"] = str(live_dir / "ragged")
        return live_dir / "serve.log"

    def _live_done():
        done = st.session_state.get("demo_live_dir")
        return bool(done and (Path(done) / "summary.csv").exists())

    if job_panel("demo_live", argv, "GPU 호스트에서 측정", _prepare_live, _live_done):
        done = st.session_state["demo_live_dir"]
        if done not in st.session_state.get("demo_live_runs", []):
            st.session_state.setdefault("demo_live_runs", []).append(done)
        st.success("측정이 끝났습니다. 위 '다른 기록 고르기'에서 이 폴더를 고르면 재생됩니다: " + done)

# ---------------------------------------------------------------- scene 2 · why
@st.cache_data(ttl=300, show_spinner=False)
def hw_index(dirs: tuple) -> pd.DataFrame:
    return data.load_index([Path(d) for d in dirs])


@st.cache_data(show_spinner="정책을 계산하는 중…")
def decision(lens: tuple, n_heads: int, n_kv_heads: int, measured_items: tuple) -> list:
    return demo.decision_card(list(lens), n_heads, n_kv_heads, TABLE_CSV, MACHINE, PARAMS, pd.Series(dict(measured_items)))


@st.cache_data(show_spinner=False)
def plugin_names(workload_key: str) -> list:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from kernelscope.run_kernel import DEFAULT_REGISTRY, load_registry
            return sorted(p.name for p in load_registry(DEFAULT_REGISTRY).supporting(Workload.from_key(workload_key)))
    except Exception:  # plugins import torch / flash-attn; a laptop without them still shows the page
        return []


st.divider()
st.subheader("2 · 왜 느린가 — 긴 요청 하나가 블록 하나에 갇힌다")
w = Workload.from_key(demo.WORST_KEY)
n_sm = int(json.loads(MACHINE.read_text()).get("n_sm", 128)) if MACHINE.exists() else 128
index = hw_index(tuple(str(d) for d in data.find_dirs("hw", root)))
family = FAMILIES["paged"]
if not index.empty:
    states = index["cache_state"].fillna("warm") if "cache_state" in index else pd.Series("warm", index=index.index)
    cold = index[states == "cold"]
else:
    cold = index
cell = cold[(cold.workload_key == demo.WORST_KEY) & cold.kernel.str.match(family["members"])] if not cold.empty else cold
measured = cell.groupby("kernel").kernel_time_us.median().sort_values() if not cell.empty else pd.Series(dtype=float)
left, right = st.columns([1, 1.5])
with left:
    chart(figures.workload_lengths(w, theme), "demo_lengths")
    st.write(f"{w.B}개 요청 · 긴 요청 {w.L_kv:,} tokens 1개 + 짧은 요청 {min(w.lens()):,} tokens {w.B - 1}개 · 총 {sum(w.lens()):,} KV tokens")
with right:
    base = demo.cta_work(demo.WORST_KEY, "flashdecoding_paged", n_sm)
    best_kernel = str(measured.idxmin()) if not measured.empty else "fd_s16_paged"
    chosen = demo.cta_work(demo.WORST_KEY, best_kernel, n_sm)
    chart(figures.cta_work_curves([
        {"label": f"휴리스틱 · 분할 {base['splits']} · CTA {base['ctas']:,}개", "color": style.policy_color("heuristic", theme), "keys": base["keys"]},
        {"label": f"{best_kernel} · 분할 {chosen['splits']} · CTA {chosen['ctas']:,}개", "color": style.policy_color("table", theme), "keys": chosen["keys"]},
    ], theme), "demo_cta")
    m = st.columns(2)
    m[0].metric("가장 긴 CTA / 평균 CTA · 휴리스틱", f"{base['longest_over_mean']:.1f}×")
    m[1].metric("가장 긴 CTA / 평균 CTA · 선택", f"{chosen['longest_over_mean']:.1f}×")
    st.caption(f"계산 · 길이와 분할 수에서 구한 CTA 작업량(실측 아님). 휴리스틱은 CTA {w.B * w.H_kv}개 ≥ 0.8 × {2 * n_sm}이면 분할 {base['splits']}로 "
               f"조기 결정해, 긴 요청의 {w.L_kv:,} keys를 블록 하나가 처리하는 동안 {n_sm}개 SM 대부분이 빕니다.")
if measured.empty:
    st.info("최악 셀의 paged 커널 기록이 없습니다 (hw_4090/*/summaries.jsonl).")
else:
    chart(figures.variants_bar(pd.DataFrame(), measured, family["heuristic"], theme, "같은 입력, 다른 분할 수 · 커널 시간 실측 (paged KV, cold)"), "demo_variants")
    h = measured.get(family["heuristic"])
    if h is not None and pd.notna(h):
        st.caption(f"실측 · 라이브러리 휴리스틱 {h:,.0f} µs vs 가장 빠른 분할 {measured.idxmin()} {measured.min():,.0f} µs = {h / measured.min():.2f}× "
                   f"(attention 커널만). dense 레이아웃의 같은 셀은 Lab 01 Diagnose에서.")
diagnoses = sorted((root / "serve_4090").rglob("diagnosis.json")) if (root / "serve_4090").exists() else []
diagnoses = [p for p in diagnoses if p.parent.name == "ragged"] or diagnoses
if diagnoses:
    try:
        d = json.loads(diagnoses[-1].read_text())
        shares = {policy: next((r["share"] for r in p["ops"] if r["op_class"] == "attention"), None) for policy, p in d["policies"].items()}
    except (OSError, ValueError, KeyError, TypeError):
        shares = {}
    if shares:
        png = diagnoses[-1].with_name("op_breakdown.png")
        png = png if png.exists() else ROOT / "docs" / "img" / "op_breakdown_ragged.png"
        cols = st.columns([1.5, 1])
        if png.exists():
            cols[0].image(str(png), caption="실측 · decode step을 8개 연산 클래스로 분해 (CUDA event)")
        with cols[1]:
            for policy, share in shares.items():
                st.metric(f"attention 비중 · {policy}", "—" if share is None else f"{100 * share:.1f}%")
            st.caption("attention 외 연산은 이미 DRAM 상한 근처라 커널 선택의 여지가 작습니다. 분할 수를 바꾸면 attention 비중만 줄어듭니다.")

# ---------------------------------------------------------------- scene 3 · how
st.divider()
st.subheader("3 · 어떻게 고르나 — 실측표와 성능 모델이 같은 배치에 내리는 결정")
paged_cold = cold[cold.kernel.str.match(family["members"]) & cold.workload_key.str.contains(f"_Hq{w.H_q}_Hkv{w.H_kv}_d{w.d}_{w.dtype}_")] if not cold.empty else cold
table = dispatch_table(data.index_rows(paged_cold), "paged", "cold") if not paged_cold.empty else pd.DataFrame()
left, right = st.columns([1.3, 1])
with left:
    if table.empty:
        st.info("커널 지도를 그릴 paged cold 기록이 없습니다.")
    else:
        chart(figures.regret_heatmap(table, theme, "실측 · 라이브러리 휴리스틱의 선택 손실 (paged, cold)"), "demo_heatmap")
        st.caption("실측 · 손실 = heuristic 시간 / 최적 실측 시간 − 1. 균일 길이 행은 거의 0, 혼합 길이 행에서 커집니다.")
with right:
    st.markdown("**이 배치에 대한 결정 카드**")
    for row in decision(tuple(w.lens()), w.H_q, w.H_kv, tuple(measured.items())):
        split = "—" if row["num_splits"] is None else f"분할 {row['num_splits']}"
        us = "" if row["kernel_us"] is None else f" · {row['kernel_us']:,.0f} µs"
        gain = "" if row["speedup_vs_heuristic"] is None else f" · 휴리스틱 대비 {row['speedup_vs_heuristic']:.2f}×"
        st.markdown(f"**{row['label']}** — {split}{' · ' + row['kernel'] if row['kernel'] else ''}{us}{gain}  \n"
                    f"<span class='footnote'>{row['note']}</span>", unsafe_allow_html=True)
    st.caption("분할 수와 출처는 정책 코드가 지금 계산한 값(계산), 커널 시간은 그 변형의 실측 중앙값(실측).")
backend = lab.backend_comparison(index, "cold") if not index.empty else pd.DataFrame()
backend_row = backend[backend.workload_key == demo.WORST_KEY] if not backend.empty else backend
if not backend_row.empty:
    r = backend_row.iloc[0].to_dict()
    chart(figures.backend_bar(r, theme), "demo_backends")
    fi = r.get("flashinfer_us")
    if fi is not None and pd.notna(fi) and pd.notna(r.get("best_fa2_us")):
        st.caption(f"실측 · FlashInfer(더 빠른 변형) / 최선 FA2 분할 = {fi / r['best_fa2_us']:.3f}배. 분할 수를 밖에서 고르는 것으로 커널 교체 이득의 대부분을 얻습니다.")
with st.expander("플러그인 벤치 · 등록된 커널을 이 셀에서 직접 재기 (약 30초)"):
    names = plugin_names(demo.WORST_KEY)
    if not names:
        st.caption("이 장비에서는 플러그인 레지스트리를 불러올 수 없습니다(torch/flash-attn 필요).")
    default = [n for n in ("flashdecoding_paged", "fd_s16_paged", "flashinfer_paged_cudacore") if n in names]
    chosen_plugins = st.multiselect("플러그인", names, default=default, key="bench_plugins")
    bench_dir = root / "hw_4090" / f"demo_bench_{STAMP}"
    bench_argv = lab.compose_command("bench", python=PY, workload=demo.WORST_KEY, plugins=chosen_plugins, results=bench_dir,
                                     cache_state="cold", warmup=5, iters=20)

    def _prepare_bench():
        st.session_state["demo_bench_dir"] = str(bench_dir)
        bench_dir.mkdir(parents=True, exist_ok=True)
        return bench_dir / "bench.log"

    def _bench_done():
        done = st.session_state.get("demo_bench_dir")
        return bool(done and (Path(done) / "summaries.jsonl").exists())

    if chosen_plugins and job_panel("demo_bench", bench_argv, "GPU 호스트에서 벤치", _prepare_bench, _bench_done):
        rows = []
        for line in (Path(st.session_state["demo_bench_dir"]) / "summaries.jsonl").read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        ok = pd.Series({row["plugin"]: float(row["kernel_time_us"]) for row in rows
                        if row.get("status") == "ok" and row.get("kernel_time_us") is not None}).sort_values()
        if not ok.empty:
            chart(figures.variants_bar(pd.DataFrame(), ok, family["heuristic"], theme, "방금 측정 · 플러그인 벤치 (µs)"), "demo_bench_chart")
        for row in rows:
            if row.get("status") != "ok":
                st.caption(f"실패 · {row.get('plugin')}: {str(row.get('error', ''))[:160]}")

# ---------------------------------------------------------------- scene 4 · verify
st.divider()
st.subheader("4 · 검증 — 반복, 토큰 일치, 대조군, 문서 수치 재계산")
campaign = demo.featured_campaign(root) or Path(run_dir).parent
table4 = demo.campaign_table(campaign)
if table4.empty:
    st.info("이 캠페인의 조건별 summary.csv가 없습니다.")
else:
    show = table4.assign(family=table4.family.map(lambda f: demo.FAMILY_LABELS.get(f, f)))
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"scenario": "조건", "family": "배치 종류", "policy": "정책",
                                "tpot_ms": st.column_config.NumberColumn("TPOT (ms)", format="%.2f"),
                                "speedup": st.column_config.NumberColumn("배율", format="%.3f×"),
                                "repeats": "반복", "tokens_equivalent": st.column_config.CheckboxColumn("토큰 일치")})
    uniform = table4[(table4.family == "uniform") & (table4.policy != "heuristic")].dropna(subset=["speedup"])
    if not uniform.empty:
        st.caption(f"대조군 · 균일 배치의 배율 {uniform.speedup.min():.3f}~{uniform.speedup.max():.3f}×. 개선 대상은 길이가 섞인 배치이며, 모든 입력에서 빨라진다고 주장하지 않습니다.")
    else:
        st.caption("대조군 · 이 캠페인에는 균일 배치 조건이 없습니다. Lab 04 Serving에서 graduation_20260922/uniform을 보세요.")
divergence = lab.divergence_overview([campaign])
if not divergence.empty:
    keep = [c for c in ("scenario", "policy", "repeats", "distinct_positions", "tie_1ulp", "tie_2ulp", "clear", "unclassified") if c in divergence]
    st.dataframe(divergence[keep], hide_index=True, width="stretch",
                 column_config={"scenario": "조건", "policy": "정책", "repeats": "반복", "distinct_positions": "분기 사건", "unclassified": "미분류"})
    st.caption("분기 사건 = 요청별 첫 불일치 위치의 수. tie_1ulp/2ulp는 기준 1위 로짓과 후보가 고른 토큰의 차이가 bf16 간격 1~2개인 동점, clear는 조사 대상.")
if st.button("문서 수치 재계산 · kernelscope verify (GPU 불필요)", key="demo_verify"):
    with st.spinner("커밋된 기록에서 다시 계산하는 중…"):
        try:
            result = subprocess.run([PY, "-m", "kernelscope.cli", "verify"], cwd=ROOT, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            result = None
    if result is None:
        st.error("verify가 10분 안에 끝나지 않았습니다.")
    else:
        counts = demo.verify_tail(result.stdout)
        if counts.get("FAIL"):
            st.error(f"FAIL {counts['FAIL']}개 · PASS {counts.get('PASS', 0)}개")
        elif counts.get("PASS"):
            st.success(f"PASS {counts['PASS']}개 · 문서의 수치는 기록에서 다시 계산됩니다.")
        else:
            st.warning("verify 출력을 해석할 수 없습니다.")
        st.code((result.stdout or result.stderr)[-2500:], language="text")

st.divider()
st.markdown('<div class="footnote">KERNELSCOPE · Graduation research demo<br>'
            '실측은 재현 가능한 기록으로, 예측은 검증할 가설로, 서비스 성능은 전체 실행으로 평가합니다. 자세한 탐색은 Lab 페이지.</div>',
            unsafe_allow_html=True)
