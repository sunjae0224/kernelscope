"""Lab page: the five exploration tabs (Diagnose / Kernel map / What-if / Serving / Policy lab)."""
from pathlib import Path
import json
import sys

# Streamlit executes this file by path; a source checkout need not be installed.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from kernelscope.analysis.dispatch import FAMILIES, dispatch_table
from kernelscope.dashboard import data, figures, research, style
from kernelscope.workload import Workload, format_lens


theme = style.current_theme()
st.markdown(style.PAGE_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner="실측 결과를 불러오는 중…")
def get_index(directories, signature):
    return data.load_index(directories)


@st.cache_data(ttl=300, show_spinner=False)
def get_cell(directory, kernel, key, state):
    return data.load_cell(directory, kernel, key, state)


@st.cache_data(show_spinner=False)
def get_predictions(key, family, state, scales=None):
    return data.prediction_table(key, family, state, ROOT / "machines/rtx4090.json",
                                 ROOT / "models/rtx4090.json", scales)


@st.cache_data(ttl=300, show_spinner="각 실험의 독립된 반복 측정을 비교하는 중…")
def get_campaign_overview(directories, signature):
    return research.campaign_overview(directories)


def campaign_explorer(directories):
    explorer = st.expander("일반화 검증 · 모델·seed·요청 분포별 캠페인 비교", key="campaign_explorer", on_change="rerun")
    if not explorer.open:
        return
    with explorer:
        catalog = research.campaign_catalog(directories)
        st.write("각 점은 같은 모델·입력·seed·구현에서 heuristic과 비교한 독립 실험입니다. "
                 "서로 다른 모델이나 seed의 실행 시간을 합쳐 가속 배수를 계산하지 않습니다.")
        campaign_order = catalog.groupby("campaign_group").completed_at.max().sort_values(ascending=False).index.tolist()
        selected_campaign = st.selectbox("캠페인 · 파일 묶음", campaign_order)
        catalog = catalog[catalog.campaign_group == selected_campaign]
        st.caption(f"선택한 캠페인: {selected_campaign}. 예비 실험과 전체 warm-up 확인 실험은 각각의 파일 묶음으로 선택합니다.")
        choices = sorted(catalog.evaluation_split.unique())
        preferred = "heldout_text" if "heldout_text" in choices else choices[0]
        split = st.selectbox("평가 입력의 종류", choices, index=choices.index(preferred))
        subset = catalog[catalog.evaluation_split == split]
        models = sorted(subset.model.unique())
        chosen_models = st.multiselect("비교할 모델", models, default=models)
        subset = subset[subset.model.isin(chosen_models)]
        if split == "heldout_text":
            st.caption("heldout_text · 학습·보정 입력과 구분한 로컬 자연어 구성 실험입니다. 실제 운영 트래픽을 수집한 자료는 아닙니다.")
        elif split == "legacy_synthetic":
            st.caption("legacy_synthetic · 기존 합성 토큰 ID 실험입니다. 새 자연어 입력 실험과 구분해 표시합니다.")
        if subset.empty:
            st.info("선택한 모델의 실험이 없습니다.")
            return
        metrics = st.columns(4)
        metrics[0].metric("모델", subset.model.nunique())
        metrics[1].metric("기록된 seed", subset.loc[subset.seed != "unrecorded", "seed"].nunique())
        metrics[2].metric("요청 분포", subset.scenario_family.nunique())
        metrics[3].metric("완료 / 기록된 실험", f"{subset.status.eq('complete').sum()} / {len(subset)}")
        warmup = ", ".join(sorted(subset.warmup_steps.unique()))
        st.caption(f"Warm-up steps: {warmup} · full_scenario는 시나리오 전체, unrecorded는 설정 미기록입니다. "
                   "실험 수는 현재 폴더에서 발견된 기록을 기준으로 합니다.")
        paths = tuple(subset.experiment)
        signature = tuple((p, (Path(p) / "manifest.json").stat().st_mtime_ns)
                          for p in paths if (Path(p) / "manifest.json").exists())
        overview = get_campaign_overview(paths, signature)
        if overview.empty:
            st.info("비교 가능한 정책별 기록이 아직 없습니다.")
            st.dataframe(subset[["model", "scenario_family", "seed", "status", "declared_repeats"]], hide_index=True)
            return
        candidates = overview[overview.policy != "heuristic"]
        st.caption(f"정책 비교 {len(candidates)}건 중 {int(candidates.claim_eligible.sum())}건이 가속 비교 조건을 통과했습니다. "
                   "나머지 비교의 토큰 일치 비율과 제외 사유도 아래 표에 함께 남깁니다.")
        chart(figures.campaign_comparison(overview, theme), "campaign_comparison")
        st.caption("가속 배수 = 같은 실험의 heuristic TPOT ÷ 정책 TPOT. 오차 막대는 해당 실험의 반복 측정에 대한 "
                   "paired bootstrap 95% 구간입니다. 검증에 실패하거나 반복 짝이 맞지 않는 정책은 가속 그래프에서 제외됩니다.")
        display_columns = ["comparison_id", "model", "scenario_family", "seed", "warmup_runs", "warmup_steps", "policy", "repeats", "declared_repeats",
                           "tpot_ms", "speedup", "token_agreement", "tokens_equivalent", "output_validation_passed",
                           "claim_eligible", "ineligibility", "first_decision_us", "cache_miss_us", "cache_hit_us",
                           "unrecorded_us", "implementation", "experiment"]
        st.dataframe(overview[display_columns], hide_index=True, width="stretch",
                     column_config={"model": "모델", "scenario_family": "요청 분포", "policy": "정책", "repeats": "실측 반복",
                                    "declared_repeats": "계획 반복", "tpot_ms": st.column_config.NumberColumn("TPOT (ms)", format="%.3f"),
                                    "speedup": st.column_config.NumberColumn("검증된 가속 (×)", format="%.3f"),
                                    "token_agreement": st.column_config.NumberColumn("토큰 일치 비율", format="%.6f"),
                                    "claim_eligible": "가속 비교 가능", "first_decision_us": "첫 선택 (µs)",
                                    "cache_miss_us": "이후 miss (µs)", "cache_hit_us": "이후 hit (µs)",
                                    "unrecorded_us": "이후 상태 미기록 (µs)", "ineligibility": "검증 제외 사유"})
        st.caption("캐시 상태가 없는 과거 로그는 '미기록'으로 남깁니다. 빈 값은 측정되지 않은 항목이며 0으로 해석하지 않습니다. "
                   "구현 지문과 입력 해시를 포함한 전체 출처는 CSV에 저장됩니다.")
        st.download_button("독립 실험 비교 CSV 다운로드", overview.to_csv(index=False), "kernelscope-campaign-comparison.csv", mime="text/csv")


def chart(fig, key):
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displaylogo": False})


def fmt(value, suffix="", digits=1):
    return f"{value:,.{digits}f}{suffix}" if pd.notna(value) else "—"


def workload_label(key):
    w = Workload.from_key(key)
    return f"B={w.B} · {format_lens(w.lens())} tokens · {'mixed' if w.is_ragged else 'uniform'}"


def model_note():
    st.caption("MODEL · 보정된 CPU 성능 예측입니다. 검증에서 평균적인 오차는 작아도 최악의 커널 선택 손실 목표는 "
               "통과하지 못했습니다. 하드웨어 변경 결과와 추천 커널은 실험 가설로 사용하세요.")


with st.sidebar:
    st.markdown("### ◈ KernelScope")
    st.caption("GPU ATTENTION OBSERVATORY")
    st.divider()
    root = Path(st.text_input("결과 폴더 · Results", value=str(data.results_root()))).expanduser()
    state = st.radio("캐시 상태 · Cache", ["cold", "warm"], horizontal=True,
                     help="cold는 KV 재사용을 제한한 조건, warm은 같은 데이터를 반복 사용하는 조건입니다.")
    family = st.radio("KV 저장 방식 · Layout", ["dense", "paged"], horizontal=True)
    st.caption("Dense는 연속 텐서, paged는 블록별 KV 캐시입니다. 같은 레이아웃 안에서 비교합니다.")
    geometry_placeholder = st.container()
    if st.button("↻ 결과 새로고침", width="stretch"):
        st.cache_data.clear()
    st.divider()
    st.markdown("**데모 순서**")
    st.caption("01 병목을 관찰합니다.\n\n02 어떤 입력에서 차이가 나는지 찾습니다.\n\n03 하드웨어 가설을 바꿉니다.\n\n04 실제 생성 단계에서 검증합니다.")

directories = data.find_dirs("hw", root)
signature = tuple((str(p), (p / "summaries.jsonl").stat().st_mtime_ns) for p in directories)
index = get_index(tuple(str(p) for p in directories), signature)
selected = index[(index.cache_state == state) & index.kernel.str.match(FAMILIES[family]["members"])].copy()
if not selected.empty:
    selected["geometry"] = selected.workload_key.map(
        lambda k: (lambda w: f"Hq {w.H_q} / Hkv {w.H_kv} / d {w.d} / {w.dtype}")(Workload.from_key(k)))
    with geometry_placeholder:
        geometries = sorted(selected.geometry.unique())
        preferred_geometry = "Hq 32 / Hkv 8 / d 128 / float16"
        geometry = st.selectbox("Attention 형상", geometries,
                                index=geometries.index(preferred_geometry) if preferred_geometry in geometries else 0)
    selected = selected[selected.geometry == geometry]
table = dispatch_table(data.index_rows(selected), family, state)
valid = table.dropna(subset=["heuristic_regret"])
worst = valid.iloc[0] if not valid.empty else None
speedup = float(worst.heuristic_us / worst.best_us) if worst is not None else None

st.markdown('<div class="eyebrow">SYSTEMS RESEARCH / FROM KERNELS TO TOKENS</div>', unsafe_allow_html=True)
st.title("작은 커널 선택, 큰 실행 시간 차이.")
st.markdown('<div class="hero-note">KernelScope는 GPU attention의 병목을 관찰하고, 요청 길이에 맞는 커널을 '
            '선택해 실제 토큰 생성까지 효과를 검증하는 실험실입니다.</div>', unsafe_allow_html=True)
st.markdown('<span class="evidence">● GPU 기록 · 실측</span>'
            '<span class="evidence">◇ What-if · 모델 예측</span>'
            '<span class="evidence">○ Serving · 결과별 증거 확인</span>', unsafe_allow_html=True)
top = st.columns(4)
top[0].metric("GPU 기록 · 전체 측정", f"{len(index):,}", help="선택한 결과 폴더에 기록된 성공한 커널 측정 수입니다.")
top[1].metric("비교 가능한 입력", f"{len(valid):,}", help="현재 레이아웃·캐시·형상에서 heuristic을 포함해 측정한 입력 수입니다.")
top[2].metric("최대 커널 개선 기회", fmt(speedup, "×", 2) if speedup is not None else "—",
              help="현재 선택 범위에서 library heuristic 시간 / 가장 빠른 실측 커널 시간. 전체 LLM 가속 비율이 아닙니다.")
uniform = valid[~valid.ragged]
top[3].metric("균일 길이 · 선택 손실 중앙값", fmt(uniform.heuristic_regret.median() * 100, "%") if len(uniform) else "—")
st.caption(f"SOURCE · {root} · {state.upper()} / {family.upper()} · 실측 최적값은 같은 입력에서 비교한 후보들의 최솟값입니다.")

diagnose_tab, map_tab, whatif_tab, serving_tab, lab_tab = st.tabs(
    ["01  Diagnose", "02  Kernel map", "03  What-if", "04  Serving", "05  Policy lab"])
model_available = (ROOT / "machines/rtx4090.json").exists() and (ROOT / "models/rtx4090.json").exists()
chosen_key = worst.workload_key if worst is not None else (selected.workload_key.iloc[0] if len(selected) else None)

with diagnose_tab:
    st.subheader("어떤 입력에서, 왜 느려질까요?")
    st.write("같은 attention 연산도 KV 길이 분포와 split 수에 따라 실행 시간이 달라집니다. 실측 후보를 직접 비교하세요.")
    if selected.empty:
        st.info("이 조건의 GPU 측정 기록이 없습니다. 결과 폴더, 캐시 상태 또는 KV 저장 방식을 바꾸면 기록을 확인할 수 있습니다.")
    else:
        keys = list(table.workload_key) if not table.empty else sorted(selected.workload_key.unique())
        input_col, variant_col = st.columns([2, 1])
        chosen_key = input_col.selectbox("분석할 입력 · 느린 heuristic 사례부터 정렬", keys, format_func=workload_label)
        cell = selected[selected.workload_key == chosen_key]
        measured = cell.groupby("kernel").kernel_time_us.median().sort_values()
        heuristic = FAMILIES[family]["heuristic"]
        plugins = list(measured.index)
        variant = variant_col.selectbox("자세히 볼 커널", plugins, index=plugins.index(heuristic) if heuristic in plugins else 0)
        row = cell[cell.kernel == variant].iloc[0]
        detail = get_cell(row.source_dir, variant, chosen_key, state)
        detail_table = data.kernel_table(detail, state)
        detail_row = detail_table.iloc[0] if len(detail_table) else pd.Series(dtype=float)
        cols = st.columns(4)
        cols[0].metric("선택 커널 · GPU 시간", fmt(measured[variant], " µs"))
        cols[1].metric("최적 실측 후보", str(measured.idxmin()))
        cols[2].metric("최적 후보 · GPU 시간", fmt(measured.min(), " µs"))
        ratio = measured[variant] / measured.min()
        cols[3].metric("선택 커널 / 최적 후보", fmt(ratio, "×", 2))
        st.caption("같은 입력·커널의 기록이 여러 개면 중앙값으로 비교합니다. 원본 상세는 아래 표시된 결과 폴더의 기록입니다.")
        pred = pd.DataFrame()
        include_model = st.checkbox("실측과 성능 모델을 함께 비교", value=False)
        if include_model and model_available:
            model_note()
            try:
                pred = get_predictions(chosen_key, family, state)
            except (ValueError, KeyError, OSError) as exc:
                st.warning(f"이 입력의 성능 모델을 불러올 수 없습니다: {exc}")
        chart(figures.variants_bar(pred, measured, heuristic, theme, "같은 입력, 다른 커널 · GPU measured"), "variants")
        w = Workload.from_key(chosen_key)
        left, right = st.columns([1.45, 1])
        with left:
            chart(figures.workload_lengths(w, theme), "diagnose_lengths")
        with right:
            st.markdown("**이 배치의 특징**")
            st.write(f"{w.B}개 요청 · 최대 {w.L_kv:,} tokens · 총 {sum(w.lens()):,} KV tokens")
            if w.is_ragged:
                st.write("긴 요청 몇 개가 남아 있으면 긴 구간을 여러 블록으로 나누는 split 선택이 중요해집니다. "
                         "최대 길이와 배치 크기가 같아도 길이 분포가 다르면 최적 커널이 바뀔 수 있습니다.")
            else:
                st.write("모든 요청의 길이가 같습니다. 이 조건에서의 선택 이득이 혼합 길이 배치에도 유지되는지 Kernel map에서 비교합니다.")
            st.caption("이 그래프는 attention 커널 실측입니다. 전체 생성 성능은 Serving에서 별도로 확인합니다.")
        with st.expander("실행 블록과 측정 원본 · 재현 근거"):
            if detail.empty:
                st.caption("이 결과 묶음에는 요약 기록만 있습니다. 실행 블록 정보는 원본 parquet가 있는 결과 폴더에서 확인할 수 있습니다.")
            else:
                metrics = st.columns(4)
                for col, label, name, suffix in zip(metrics, ["실행 블록 · CTAs", "상주 블록 상한 / SM", "실효 대역폭 · 추정", "실행 지연 · CUDA event"],
                                                    ["grid_blocks", "blocks_per_sm_limit", "achieved_gbps", "latency_us"], ["", "", " GB/s", " µs"]):
                    col.metric(label, fmt(detail_row.get(name), suffix))
                st.caption("상주 블록 수와 실효 대역폭은 자원 한계·데이터 이동량에서 계산한 지표입니다.")
                st.dataframe(detail[[c for c in ["backend", "metric", "value", "unit", "launch_idx", "note"] if c in detail]],
                             hide_index=True, width="stretch")
            st.code(chosen_key, language=None)
            st.caption(f"기록 위치: {row.source_dir}")

with map_tab:
    st.subheader("하나의 커널 선택 규칙이 모든 배치에 통할까요?")
    st.write("색은 library heuristic의 시간 손실, 셀의 글자는 가장 빠른 실측 후보입니다. 길이가 섞인 경우는 아래에 따로 표시합니다.")
    if table.empty:
        st.info("선택한 조건의 커널 비교 기록이 없습니다.")
    else:
        if len(table[~table.ragged]):
            chart(figures.regret_heatmap(table, theme), "regret_map")
            st.caption("손실 = heuristic 시간 / 최적 실측 시간 − 1 · s8 = 8 splits · heur = library heuristic · 빈 칸 = 측정 없음")
        if len(table[table.ragged]):
            chart(figures.ragged_bars(table, theme), "ragged_map")
        st.caption("표의 최적 후보는 같은 측정 데이터에서 고른 oracle입니다. 새 입력의 성능을 보장하는 dispatch 정책은 아닙니다.")
        with st.expander("모든 입력 비교표 · CSV 내보내기"):
            st.dataframe(table, hide_index=True, width="stretch")
            st.download_button("실측 비교표 다운로드", table.to_csv(index=False),
                               file_name=f"kernelscope-{family}-{state}.csv", mime="text/csv")

with whatif_tab:
    st.subheader("GPU 자원을 바꾸면 최적 커널도 바뀔까요?")
    model_note()
    if not model_available:
        st.info("machines/rtx4090.json 및 models/rtx4090.json이 있어야 What-if 모델을 실행할 수 있습니다.")
    else:
        mode = st.radio("예측할 입력", ["현재 분석 입력", "직접 구성"], horizontal=True)
        whatif_key = chosen_key or Workload("decode", 1, 1, 32768, 32, 8, 128).key()
        if mode == "직접 구성":
            controls = st.columns(4)
            mixed = controls[0].selectbox("길이 분포", ["uniform", "ragged"])
            B = controls[1].selectbox("배치 크기", [1, 4, 8, 16, 32, 64], index=4)
            L = controls[2].selectbox("최대 KV 길이", [1024, 4096, 8192, 16384, 32768, 65536], index=4)
            lens = None
            if mixed == "ragged" and B > 1:
                n_long = controls[3].number_input("긴 요청 수", min_value=1, max_value=B - 1, value=1)
                L_short = st.select_slider("짧은 요청 KV 길이", options=[128, 256, 512, 1024], value=512)
                lens = tuple([L] * n_long + [L_short] * (B - n_long))
            elif mixed == "ragged":
                st.caption("B=1에는 한 요청만 있어 균일 길이로 계산합니다.")
            whatif_key = Workload("decode", B, 1, L, 32, 8, 128, kv_lens=lens).key()
        st.caption(f"입력 · {workload_label(whatif_key)}")
        controls = st.columns(4)
        sm = controls[0].slider("SM 수 ×", .25, 2.0, 1.0, .25)
        dram = controls[1].slider("DRAM 대역폭 ×", .5, 2.0, 1.0, .25)
        l2 = controls[2].slider("L2 용량 ×", .5, 2.0, 1.0, .25)
        smem = controls[3].selectbox("SM당 공유 메모리", [100, 164], format_func=lambda x: f"{x} KiB")
        try:
            base = get_predictions(whatif_key, family, state)
            scaled = get_predictions(whatif_key, family, state, {"sm": sm, "dram": dram, "l2": l2, "smem": smem / 100})
            a, b, c = st.columns(3)
            a.metric("기준 GPU · 예측 최적", str(base.loc[base.time_us.idxmin(), "plugin"]))
            b.metric("변경 GPU · 예측 최적", str(scaled.loc[scaled.time_us.idxmin(), "plugin"]))
            c.metric("최적 시간 비율 · 예측", f"{base.time_us.min() / scaled.time_us.min():.2f}×")
            chart(figures.whatif_compare(base, scaled, theme), "whatif")
            st.caption("두 그래프 모두 성능 모델의 예측입니다. 기준 GPU의 보정값에 자원 배율을 적용하며, "
                       "다른 실제 GPU의 측정값이나 보정 완료된 Accel-Sim 결과를 의미하지 않습니다.")
            with st.expander("예측 병목 상세"):
                st.dataframe(scaled, hide_index=True, width="stretch")
            validation = ROOT / "docs/plan/2026-09-22-model-validation.md"
            if validation.exists():
                with st.expander("모델 검증 성적표 · 예측 오차와 선택 손실"):
                    validation_text = validation.read_text()
                    st.caption("2026-09-22 검증 · APE는 실행 시간의 절대 비율 오차, regret은 최적 실측 후보 대비 선택 손실입니다.")
                    st.markdown(validation_text.split("Worst cells,")[0])
                    st.download_button("전체 모델 검증 보고서", validation_text, "model-validation.md", mime="text/markdown")
        except (ValueError, KeyError, OSError) as exc:
            st.warning(f"예측을 계산할 수 없습니다: {exc}")

with serving_tab:
    st.subheader("커널의 이득이 실제 토큰 생성까지 이어질까요?")
    st.write("동일 요청과 seed에서 선택 정책만 바꿔 attention 시간, 전체 decode 시간, 토큰 간격을 함께 비교합니다.")
    st.caption("Serving은 저장된 실험의 모델·KV 설정을 사용합니다. 왼쪽 필터는 커널 기록과 What-if에 적용됩니다.")
    text_demo = root / "text_demo.json"
    if not text_demo.exists() and root.resolve() in (data.DEFAULT_ROOT.resolve(), (ROOT / "demo_data").resolve()):
        text_demo = ROOT / "demo_data/text_demo.json"
    if text_demo.exists():
        try:
            sample = json.loads(text_demo.read_text())
            if sample.get("evidence_kind") == "single_request_text_demo":
                with st.expander("실제 LLM 텍스트 생성 예시 · 저장된 단일 요청"):
                    st.caption(f"{sample.get('model', 'unknown')} · prompt {sample.get('prompt_tokens', '—')} tokens · "
                               f"generated {sample.get('generated_tokens', '—')} tokens")
                    st.write("실제 모델의 저장된 응답입니다. 단일 요청의 기능 데모이므로 정책 간 속도 비교의 근거는 아닙니다.")
                    if sample.get("stopped_on_eos") is False:
                        st.caption(f"{sample.get('generated_tokens', '지정된')}토큰 생성 한도에서 중단된 원문입니다. "
                                   "문장이나 단어가 끝나기 전에 잘릴 수 있으며 완결된 답변을 뜻하지 않습니다.")
                    st.text_area("입력 프롬프트", str(sample.get("prompt", "")), disabled=True, height=100)
                    st.text_area("생성된 원문", str(sample.get("generated_text", "")), disabled=True, height=200)
                    if sample.get("command"):
                        st.code(sample["command"], language="bash")
                    st.caption(f"기록 위치: {text_demo}")
        except (OSError, ValueError):
            st.caption("텍스트 데모 기록을 읽을 수 없습니다.")
    scenario_dirs = data.find_dirs("serve", root)
    if not scenario_dirs:
        st.info("서빙 측정 결과가 아직 없습니다. 현재 GPU 커널 기록만으로 전체 LLM 생성의 성능 향상을 주장하지 않습니다.")
        st.caption("serve 실행 결과의 manifest.json과 정책별 steps/tokens/prefill.parquet를 읽습니다. CPU 기능 데모는 GPU 성능 근거와 구분됩니다.")
    else:
        campaign_explorer(scenario_dirs)
        # A partially written experiment remains inspectable, but the default
        # presentation is the most recent completed CUDA campaign.
        scenario_manifests = {p: data.load_manifest(p) for p in scenario_dirs}
        scenario_info = {p: research.experiment_identity(p, scenario_manifests[p]) for p in scenario_dirs}
        scenario_dirs.sort(key=lambda p: (
            scenario_manifests[p].get("status") == "complete",
            scenario_manifests[p].get("evidence_kind") == "cuda_serving",
            scenario_manifests[p].get("completed_at", "")), reverse=True)
        latest_campaign = scenario_info[scenario_dirs[0]]["campaign_group"]
        featured = next((p for p in scenario_dirs if scenario_info[p]["campaign_group"] == latest_campaign
                         and (scenario_info[p]["scenario_family"] == "ragged"
                              or scenario_info[p]["scenario_family"].endswith("_ragged"))
                         and scenario_manifests[p].get("status") == "complete"
                         and scenario_manifests[p].get("evidence_kind") == "cuda_serving"), scenario_dirs[0])
        def scenario_label(path):
            info = scenario_info[path]
            relative = path.relative_to(root / "serve_4090") if path.is_relative_to(root / "serve_4090") else path.name
            return f"{info['model'].rsplit('/', 1)[-1]} · {info['scenario_family']} · seed {info['seed']} · {relative}"
        scenario = st.selectbox("서빙 실험", scenario_dirs, index=scenario_dirs.index(featured), format_func=scenario_label)
        manifest = scenario_manifests[scenario]
        serving = data.load_serving(scenario)
        evidence = manifest.get("evidence_kind", "unclassified")
        performance = (evidence == "cuda_serving" and manifest.get("performance_claim") is True
                       and manifest.get("status") == "complete" and manifest.get("tokens_equivalent") is True)
        selected_warmup_steps = research.experiment_identity(scenario, manifest)["warmup_steps"]
        st.caption(f"MODEL · {manifest.get('model', 'unknown')} · {manifest.get('repeats', '—')} repeats · "
                   f"{manifest.get('warmup_runs', '—')} warm-up runs / {selected_warmup_steps} steps · "
                   f"{manifest.get('model_backend', 'backend unrecorded')}")
        if str(manifest.get("model", "")).startswith("tiny-random"):
            st.caption("무작위 가중치의 소형 decoder와 합성 토큰 입력을 사용한 재현 실험입니다. "
                       "학습된 LLM의 응답 품질이나 상용 서비스 성능을 평가하는 결과는 아닙니다.")
        elif manifest.get("prompt_kind") == "seeded_synthetic_token_ids":
            st.caption("재현 가능한 비교를 위해 합성 토큰 ID 프롬프트와 고정 길이 greedy 생성을 사용합니다. "
                       "웹 요청과 운영용 스케줄러의 오버헤드는 포함하지 않습니다.")
        if performance:
            st.success("GPU SERVING · 실제 CUDA 실행 기록입니다. 아래의 토큰 일치 여부와 반복 측정을 함께 확인하세요.")
        elif evidence == "cpu_functional":
            st.info("CPU FUNCTIONAL · 실제 CPU 디코더의 기능 검증입니다. 아래 시간은 CPU 실행 기록이며 GPU 가속의 근거가 아닙니다.")
        elif evidence == "cuda_serving":
            st.warning("GPU 실행 기록이지만 실험 완료 또는 모든 정책의 토큰 일치가 확인되지 않았습니다. "
                       "정책별 정확도 결과를 확인하세요. 토큰이 달라진 정책의 속도는 유효한 가속 성과로 표시하지 않습니다.")
        else:
            st.warning("이 결과는 GPU 성능 근거로 확인되지 않았습니다. manifest의 측정 장치와 검증 상태를 확인하세요.")
        try:
            from kernelscope.serve.report import summarize
            summary = summarize(scenario)
            if not summary.empty:
                policies = list(summary.policy)
                preferred_policies = (("model", "table", "fixed8", "fa2") if scenario_info[scenario]["evaluation_split"] == "heldout_text"
                                      else ("table", "model", "fixed8", "fa2"))
                default_policy = next((p for p in preferred_policies if p in policies), policies[0])
                compare_policy = st.selectbox("비교할 정책", policies, index=policies.index(default_policy))
                compared = summary[summary.policy == compare_policy].iloc[0]
                baseline = summary[summary.policy == "heuristic"]
                metrics = st.columns(4)
                metrics[0].metric("Heuristic · 평균 TPOT", fmt(baseline.iloc[0].tpot_ms_mean, " ms", 2) if len(baseline) else "—")
                metrics[1].metric(f"{compare_policy} · 평균 TPOT", fmt(compared.tpot_ms_mean, " ms", 2))
                verified_policy = (evidence == "cuda_serving" and manifest.get("status") == "complete"
                                   and bool(compared.get("performance_claim", False))
                                   and pd.notna(compared.get("tokens_equivalent"))
                                   and bool(compared.get("tokens_equivalent")))
                metrics[2].metric("관측된 TPOT 가속", fmt(compared.speedup_vs_heuristic, "×", 2) if verified_policy else "검증 대상 아님")
                equivalent = compared.get("tokens_equivalent")
                metrics[3].metric("선택 정책 · 토큰 일치", "미확인" if pd.isna(equivalent) else "일치" if bool(equivalent) else "불일치")
                low, high = compared.get("speedup_ci95_low"), compared.get("speedup_ci95_high")
                if verified_policy and pd.notna(low) and pd.notna(high):
                    st.caption(f"가속 배수 = heuristic TPOT ÷ {compare_policy} TPOT · paired bootstrap 95% CI: {low:.4f}–{high:.4f}×. "
                               "1×를 포함하면 반복 실행 간 변동을 고려할 때 가속 여부가 명확하지 않습니다.")
                columns = ["policy", "repeats", "attn_ms_per_step", "decode_wall_ms_per_step", "policy_us_per_step",
                           "tpot_ms_mean", "tpot_ms_p90", "throughput_tokens_s", "tokens_equivalent"]
                st.caption("TPOT 평균·p90은 요청별 평균 토큰 간격의 통계입니다. 처리량은 prefill과 전체 실험 시간을 포함합니다.")
                st.dataframe(summary[[c for c in columns if c in summary]], hide_index=True, width="stretch",
                             column_config={"policy": "정책", "repeats": "반복 수", "attn_ms_per_step": st.column_config.NumberColumn("Attention / step (ms)", format="%.3f"),
                                            "decode_wall_ms_per_step": st.column_config.NumberColumn("Decode / step (ms)", format="%.3f"),
                                            "policy_us_per_step": st.column_config.NumberColumn("정책 비용 (µs)", format="%.2f"),
                                            "tpot_ms_mean": st.column_config.NumberColumn("평균 TPOT (ms)", format="%.3f"),
                                            "tpot_ms_p90": st.column_config.NumberColumn("p90 TPOT (ms)", format="%.3f"),
                                            "throughput_tokens_s": st.column_config.NumberColumn("처리량 (tokens/s)", format="%.1f")})
        except (ImportError, KeyError, ValueError, OSError) as exc:
            st.caption(f"요약을 불러오지 못했습니다. 개별 측정 기록을 표시합니다. {exc}")
        if serving:
            with st.expander("정책 선택의 CPU 비용 · 첫 선택·cache miss·cache hit"):
                overhead_table = research.policy_overhead(serving)
                if overhead_table.empty:
                    st.info("이 실험에는 정책 선택 시간 기록이 없습니다.")
                else:
                    chart(figures.policy_overhead(overhead_table, theme), "policy_overhead")
                    st.caption("이 패널의 cache는 CPU 정책 선택 결과를 저장하는 캐시입니다. 첫 선택은 각 반복의 첫 step입니다. "
                               "이후 호출만 명시적인 hit/miss 플래그로 나눕니다. "
                               "null 또는 플래그가 없는 로그는 미기록이며 cache hit으로 추정하지 않습니다. "
                               "y축은 로그 눈금이고 0µs 호출의 평균은 그래프에서 제외됩니다.")
                    st.dataframe(overhead_table.assign(phase=overhead_table.phase.map(research.OVERHEAD_LABELS)), hide_index=True,
                                 width="stretch", column_config={"policy": "정책", "phase": "호출 구분", "calls": "호출 수",
                                                                  "repeats": "관측 반복 수", "mean_us": "평균 (µs)",
                                                                  "p50_us": "p50 (µs)", "p90_us": "p90 (µs)", "total_us": "합계 (µs)"})
                    st.download_button("정책 CPU 비용 CSV 다운로드", overhead_table.to_csv(index=False),
                                       "kernelscope-policy-overhead.csv", mime="text/csv")
                benchmark_path = ROOT / "docs/experiments/policy-latency.json"
                benchmark, benchmark_table = research.offline_policy_benchmark(benchmark_path)
                if not benchmark_table.empty:
                    st.markdown("**별도 CPU 벤치마크 · Python과 C 시뮬레이터 비교**")
                    st.caption("저장된 과거 캠페인 입력으로 측정한 정책 선택 비용입니다. 위 서빙 실험의 모델·자연어 입력과 "
                               "다를 수 있으며, 여기의 CPU 가속 배수는 GPU 서빙 가속 배수가 아닙니다.")
                    setup = st.columns(3)
                    setup[0].metric("CPU 벤치마크 반복", benchmark.get("repeats", "—"))
                    setup[1].metric("최초 빌드 · 1회 비용", fmt(benchmark.get("backend_setup_fresh_build", {}).get("setup_us", float("nan")) / 1000, " ms", 2))
                    setup[2].metric("캐시된 모듈 로드", fmt(benchmark.get("backend_setup_existing_cache", {}).get("setup_us", float("nan")) / 1000, " ms", 2))
                    st.caption("빌드·모듈 로드·정책 생성 비용은 아래 cold/hit 시간에서 제외되어 따로 표시됩니다. "
                               "cold는 새 정책의 첫 선택, hit는 이미 저장된 선택 결과의 재사용입니다.")
                    st.dataframe(benchmark_table, hide_index=True, width="stretch")
                    st.download_button("별도 CPU 벤치마크 CSV", benchmark_table.to_csv(index=False),
                                       "kernelscope-cpu-policy-benchmark.csv", mime="text/csv")
            a, b = st.columns(2)
            with a:
                chart(figures.serving_steps(serving, theme, "Attention · 실제 실행 시간"), "serving_attn")
            with b:
                chart(figures.serving_steps(serving, theme, "Decode step · 실제 실행 시간", "decode_wall_us"), "serving_wall")
            st.caption("반복 실행이 있으면 각 step의 중앙값을 표시합니다. Decode wall time은 정책 선택·동기화·샘플링 비용을 포함합니다.")
            chart(figures.tpot_box(serving, theme), "serving_tpot")
            st.caption("상단 요약은 요청별 평균 TPOT의 통계입니다. 이 분포는 개별 토큰 사이의 간격을 보여 주므로 일시적인 지연도 드러납니다.")
            first = serving.get("heuristic", next(iter(serving.values())))
            chart(figures.serving_batch(first["steps"], theme), "serving_batch")
        equivalence = Path(scenario) / "equivalence.csv"
        if equivalence.exists():
            st.markdown("**정확도 · 정책 간 생성 결과 비교**")
            st.dataframe(pd.read_csv(equivalence), hide_index=True, width="stretch")
        else:
            st.warning("토큰·logit 동등성 보고서가 없습니다. 생성 결과의 일치 여부는 아직 확인되지 않았습니다.")
        with st.expander("실험 설정과 출처"):
            st.json(manifest)
            st.caption(str(scenario))
    st.divider()
    st.markdown("### 전체 생성 속도의 상한을 계산해 보기")
    st.caption("MODELED · Amdahl 법칙으로 계산하는 가정 실험입니다. 서빙 실측값과 독립적입니다.")
    a, b, c = st.columns(3)
    share = a.slider("기존 시간 중 attention 비중 (%)", 0, 100, 40, 5) / 100
    kernel_gain = b.number_input("attention 커널 가속 배율 (×)", min_value=.1, max_value=100.,
                                 value=round(speedup, 2) if speedup is not None else 2., step=.1)
    overhead = c.slider("추가 정책 선택 비용 (% of baseline)", 0, 20, 0, 1) / 100
    estimated = data.amdahl_speedup(share, kernel_gain, overhead)
    st.metric("전체 시간의 예상 가속 · 가정값", f"{estimated:.2f}×")
    chart(figures.amdahl_curve(kernel_gain, share, overhead, theme), "amdahl")
    st.caption("예상 가속 = 1 / (1 − attention 비중 + attention 비중 / 커널 가속 + 추가 비용). "
               "MLP·메모리 관리·스케줄링 등 다른 비용이 유지된다는 가정입니다.")

with lab_tab:
    from kernelscope.dashboard import lab
    st.subheader("정책·규약·백엔드를 같은 자리에서 비교하고, 다음 측정을 조립합니다.")
    st.write("기록된 실험을 읽어 세 가지를 나란히 놓습니다: 정책 캐시 규약(fresh/keep), FlashInfer 대 FlashAttention-2 변형, "
             "생성 토큰의 분기 사건. 아래에서 조립한 명령은 GPU 호스트에서 바로 실행하거나 셸에 붙여 넣을 수 있습니다.")
    serve_dirs = data.find_dirs("serve", root)

    st.markdown("### 1. 측정 규약 · 정책 캐시를 비우는가, 유지하는가")
    st.caption("같은 시나리오·모델·seed의 실험만 짝지어 보여 줍니다. `fresh`는 측정 실행마다 정책 캐시를 비워 첫 결정 비용을 매번 내고, "
               "`keep`은 warm-up에서 본 페이지 구성을 캐시 적중으로 처리합니다(서빙 정상 상태의 낙관적 경계).")
    protocols = lab.protocol_comparison(serve_dirs) if serve_dirs else pd.DataFrame(columns=lab.PROTOCOL_COLUMNS)
    if protocols.empty:
        st.info("비교할 서빙 실험이 없습니다.")
    else:
        groups = protocols.groupby(["model", "scenario_family", "scenario_sha256", "seed"], dropna=False)
        labels = {key: f"{str(key[0]).rsplit('/', 1)[-1]} · {key[1]} · seed {key[3]} · 규약 {g.policy_cache.nunique()}종, 실험 {g.experiment.nunique()}개"
                  for key, g in groups}
        ordered = sorted(labels, key=lambda key: (-groups.get_group(key).policy_cache.nunique(), labels[key]))
        chosen = st.selectbox("시나리오 묶음", ordered, format_func=labels.get)
        block = groups.get_group(chosen)
        shown = block.sort_values(["policy", "policy_cache"])[["campaign_group", "policy_cache", "policy", "repeats", "tpot_ms_mean",
                                                               "speedup_vs_heuristic", "attn_ms_per_step", "step_ms_per_step",
                                                               "decode_wall_ms_per_step", "policy_us_per_step", "cache_misses_per_run",
                                                               "tokens_equivalent"]]
        st.dataframe(shown, hide_index=True, width="stretch",
                     column_config={"campaign_group": "캠페인", "policy_cache": "정책 캐시 규약", "policy": "정책", "repeats": "반복",
                                    "tpot_ms_mean": st.column_config.NumberColumn("평균 TPOT (ms)", format="%.2f"),
                                    "speedup_vs_heuristic": st.column_config.NumberColumn("배율", format="%.3f"),
                                    "attn_ms_per_step": st.column_config.NumberColumn("Attention / step (ms)", format="%.3f"),
                                    "step_ms_per_step": st.column_config.NumberColumn("Step GPU (ms)", format="%.3f"),
                                    "decode_wall_ms_per_step": st.column_config.NumberColumn("Decode wall (ms)", format="%.3f"),
                                    "policy_us_per_step": st.column_config.NumberColumn("선택 비용 (µs/step)", format="%.1f"),
                                    "cache_misses_per_run": st.column_config.NumberColumn("캐시 미스 / 실행", format="%.0f"),
                                    "tokens_equivalent": "토큰 일치"})
        if block.policy_cache.nunique() > 1:
            pivot = block.pivot_table(index="policy", columns="policy_cache", values="tpot_ms_mean", aggfunc="mean")
            st.caption("규약별 평균 TPOT(ms). 두 규약의 차이가 곧 첫 결정 비용이 TPOT에 미치는 몫입니다.")
            st.dataframe(pivot.round(3), width="stretch")
        else:
            st.caption("이 묶음에는 규약이 하나뿐입니다. 아래 4번에서 `--policy-cache keep`으로 같은 시나리오를 다시 재면 짝이 생깁니다.")

    st.markdown("### 2. 커널 백엔드 · FlashInfer 대 FlashAttention-2 변형")
    st.caption("왼쪽 필터의 캐시 상태에서, 셀마다 라이브러리 휴리스틱 · 가장 빠른 FA2 변형(외부 선택의 상한) · FlashInfer(plan 기반 내부 부하 분산, "
               "분할 수를 고를 필요가 없음)를 나란히 둡니다. FlashInfer 열이 비어 있으면 그 격자는 아직 측정 전입니다.")
    backend = lab.backend_comparison(index, state) if not index.empty else pd.DataFrame(columns=lab.BACKEND_COLUMNS)
    measured = backend.dropna(subset=["flashinfer_us"])
    if measured.empty:
        st.info("FlashInfer 측정이 없습니다. `kernelscope bench --plugins flashinfer_paged`로 격자를 재면 여기 나타납니다 (4번 참고).")
    else:
        cols = st.columns(4)
        cols[0].metric("FlashInfer 측정 셀", f"{len(measured):,}")
        cols[1].metric("FlashInfer / 최선 FA2 · 중앙값", fmt(measured.flashinfer_over_best_fa2.median(), "×", 3),
                       help="1 미만이면 FlashInfer가 외부 선택의 상한(가장 빠른 FA2 변형)보다 빠릅니다.")
        ragged_rows = measured[measured.ragged]
        cols[2].metric("혼합 길이 · 휴리스틱 / FlashInfer 최대", fmt(ragged_rows.heuristic_over_flashinfer.max(), "×", 2) if len(ragged_rows) else "—")
        cols[3].metric("혼합 길이 · 휴리스틱 / 최선 FA2 최대", fmt(ragged_rows.heuristic_over_best_fa2.max(), "×", 2) if len(ragged_rows) else "—")
        st.dataframe(measured[["lens", "B", "L_kv", "ragged", "heuristic_us", "best_fa2_kernel", "best_fa2_us", "flashinfer_tensorcore_us",
                               "flashinfer_cudacore_us", "flashinfer_us", "flashinfer_over_best_fa2", "heuristic_over_flashinfer"]],
                     hide_index=True, width="stretch",
                     column_config={"lens": "KV 길이", "ragged": "혼합", "heuristic_us": st.column_config.NumberColumn("휴리스틱 (µs)", format="%.1f"),
                                    "best_fa2_kernel": "최선 FA2 변형", "best_fa2_us": st.column_config.NumberColumn("최선 FA2 (µs)", format="%.1f"),
                                    "flashinfer_tensorcore_us": st.column_config.NumberColumn("FlashInfer tensor-core (µs)", format="%.1f"),
                                    "flashinfer_cudacore_us": st.column_config.NumberColumn("FlashInfer CUDA-core (µs)", format="%.1f"),
                                    "flashinfer_us": st.column_config.NumberColumn("FlashInfer 최선 (µs)", format="%.1f"),
                                    "flashinfer_over_best_fa2": st.column_config.NumberColumn("FlashInfer / 최선 FA2", format="%.3f"),
                                    "heuristic_over_flashinfer": st.column_config.NumberColumn("휴리스틱 / FlashInfer", format="%.2f")})

    st.markdown("### 3. 출력 보존 · 분기 사건과 동점 분류")
    st.caption("토큰 일치율 대신 요청별 첫 분기 위치를 세고, 교사 강제 진단으로 기준 1위 로짓과 후보가 고른 토큰의 차이를 bf16 간격 단위로 분류합니다. "
               "`clear`는 조사 대상이며 구현 오류 여부는 같은 KV 상태의 커널 탐침으로 판정합니다.")
    campaign_dirs = sorted({p.parent for p in serve_dirs if (p.parent / "numerics").is_dir() or (p.parent / "divergence.csv").exists()})
    divergence = lab.divergence_overview(campaign_dirs) if campaign_dirs else pd.DataFrame()
    if divergence.empty:
        st.info("교사 강제 진단이 기록된 캠페인이 없습니다. `scripts/check_policy_numerics.py`와 `scripts/classify_divergence.py`를 참고하세요.")
    else:
        st.dataframe(divergence[["campaign", "scenario", "policy", "repeats", "token_mismatches_max", "distinct_positions", "tie_1ulp",
                                 "tie_2ulp", "clear", "unclassified", "not_reproduced", "same_positions_every_repeat"]],
                     hide_index=True, width="stretch",
                     column_config={"campaign": "캠페인", "scenario": "조건", "policy": "정책", "repeats": "반복",
                                    "token_mismatches_max": "불일치 토큰(최대)", "distinct_positions": "분기 사건",
                                    "unclassified": "미분류", "not_reproduced": "교사 강제 미재현", "same_positions_every_repeat": "반복마다 같은 위치"})

    st.markdown("### 4. 다음 측정 조립 · 실행")
    st.caption("화면에서 고른 설정으로 명령을 만듭니다. GPU 호스트에서는 `serve doctor`가 통과할 때만 실행하고, 로그는 아래에서 봅니다. "
               "GPU가 없는 장비에서는 명령을 복사해 연구실 호스트에서 돌립니다.")
    kind = st.radio("종류", ["serve_run", "bench"], horizontal=True, format_func={"serve_run": "서빙 실험 (serve run)", "bench": "커널 격자 (bench)"}.get)
    stamp = pd.Timestamp.now("UTC").strftime("%Y%m%d")
    if kind == "serve_run":
        a, b = st.columns(2)
        scenario_files = sorted(p.name for p in (ROOT / "scenarios").glob("*.yaml"))
        scenario_file = a.selectbox("시나리오", scenario_files, index=scenario_files.index("heldout_text_arrivals.yaml") if "heldout_text_arrivals.yaml" in scenario_files else 0)
        policies = b.multiselect("정책", ["heuristic", "table:demo_data/dispatch_paged_cold.csv", "hybrid:demo_data/dispatch_paged_cold.csv:0.2", "model", "fa2", "fixed:8", "fixed:16"],
                                 default=["heuristic", "table:demo_data/dispatch_paged_cold.csv", "hybrid:demo_data/dispatch_paged_cold.csv:0.2"])
        c, d, e, f = st.columns(4)
        policy_cache = c.radio("정책 캐시 규약", ["fresh", "keep"], horizontal=True)
        kv_gib = d.number_input("KV 예산 (GiB)", min_value=1, max_value=20, value=4 if scenario_file.startswith("heldout") else 10)
        repeats = e.number_input("반복", min_value=1, max_value=10, value=3)
        warmup_cap = f.number_input("warm-up step 상한 (0 = 전체)", min_value=0, max_value=64, value=0 if scenario_file.startswith("heldout") else 2)
        out_dir = st.text_input("결과 폴더", value=f"../kernelscope/results/serve_4090/lab_{stamp}/{Path(scenario_file).stem}_{policy_cache}")
        argv = lab.compose_command("serve_run", scenario=f"scenarios/{scenario_file}", policies=policies, out=out_dir, kv_gib=kv_gib,
                                   warmup_runs=1, warmup_steps=None if warmup_cap == 0 else int(warmup_cap), repeats=int(repeats),
                                   seed=0, policy_cache=policy_cache)
    else:
        a, b, c = st.columns(3)
        grid_files = sorted(p.name for p in (ROOT / "grids").glob("*.yaml"))
        grid_file = a.selectbox("격자", grid_files, index=grid_files.index("ragged_s1.yaml") if "ragged_s1.yaml" in grid_files else 0)
        plugins = b.multiselect("플러그인", ["flashinfer_paged", "flashdecoding_paged", "fa2_paged", "fd_s8_paged", "fd_s16_paged", "fd_s32_paged"],
                                default=["flashinfer_paged"])
        cache_state = c.radio("캐시 상태", ["cold", "warm"], horizontal=True)
        results_dir = st.text_input("결과 폴더", value=f"../kernelscope/results/hw_4090/{Path(grid_file).stem}_{'_'.join(p.split('_')[0] for p in plugins) or 'bench'}")
        argv = lab.compose_command("bench", grid=f"grids/{grid_file}", plugins=plugins, results=results_dir, cache_state=cache_state)
    st.code(lab.shell_line(argv), language="bash")
    gpu_host = Path("/usr/bin/nvidia-smi").exists() or Path("/usr/local/bin/nvidia-smi").exists()
    job = st.session_state.get("lab_job")
    log_path = st.session_state.get("lab_log")
    run_col, log_col = st.columns([1, 3])
    if run_col.button("GPU 호스트에서 실행", disabled=not gpu_host or (job is not None and job.poll() is None), width="stretch"):
        import subprocess
        doctor = subprocess.run([argv[0], "-m", "kernelscope.cli", "serve", "doctor"], cwd=ROOT, capture_output=True, text=True)
        if doctor.returncode != 0:
            st.error("serve doctor가 실패했습니다. 다른 GPU 프로세스가 있거나 모델·의존성이 준비되지 않았습니다.")
            st.code(doctor.stdout[-1500:] or doctor.stderr[-1500:])
        else:
            log_path = ROOT / "results" / "lab_logs" / f"{kind}_{pd.Timestamp.now('UTC').strftime('%Y%m%dT%H%M%SZ')}.log"
            st.session_state["lab_job"] = lab.launch(argv, log_path, cwd=ROOT)
            st.session_state["lab_log"] = log_path
            st.rerun()
    if log_path:
        running = job is not None and job.poll() is None
        log_col.caption(f"{'실행 중' if running else f'종료 (코드 {job.returncode})' if job is not None else '기록'} · {log_path}")
        st.code(lab.tail(log_path, 40) or "(아직 출력이 없습니다)", language="text")
        if st.button("로그 새로고침"):
            st.rerun()
    if not gpu_host:
        st.caption("이 장비에는 nvidia-smi가 없어 실행 버튼이 비활성입니다. 명령을 복사해 연구실 호스트에서 돌리세요.")

st.divider()
st.markdown('<div class="footnote">KERNELSCOPE · Graduation research demo<br>'
            '실측은 재현 가능한 기록으로, 예측은 검증할 가설로, 서비스 성능은 전체 실행으로 평가합니다.</div>', unsafe_allow_html=True)
