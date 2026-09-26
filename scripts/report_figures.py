"""Figures for the written report, drawn only from the committed evidence (no GPU needed).

    python -m scripts.report_figures --out docs/report/fig [--font /path/to/malgun.ttf]

Every figure is built from ``demo_data/`` with the project's own analysis code, so the report never
reuses a screenshot or someone else's drawing. Sizes stay within half an A4 text page, and the text
font can be set to the report's body font (Malgun Gothic is found automatically when installed).
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from kernelscope.analysis.dispatch import dispatch_table
from kernelscope.model.geometry import build_launch, parse_variant
from kernelscope.results.store import load_dirs
from kernelscope.verify import CAMPAIGN, P0_DIRS, WORST_RAGGED
from kernelscope.workload import Workload

MAX_WIDTH_IN, MAX_HEIGHT_IN = 6.3, 4.6          # A4 text block 16 cm wide; half of a ~24.7 cm text height
FONT_CANDIDATES = [Path.home() / ".local/share/fonts/malgun.ttf", Path("/mnt/c/Windows/Fonts/malgun.ttf"),
                   Path("C:/Windows/Fonts/malgun.ttf")]

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
POLICIES = ["heuristic", "fixed8", "table", "model"]
POLICY_LABEL = {"heuristic": "기본 휴리스틱", "fixed8": "고정 분할 수 8", "table": "조회 테이블", "model": "성능 모델"}
POLICY_COLOR = {"heuristic": "#8a8a86", "fixed8": "#1baf7a", "table": "#2a78d6", "model": "#eb6834"}
SERIES = ["#2a78d6", "#eb6834"]
SCENARIO_LABEL = {"uniform": "균일 길이\n(512 × 32)", "ragged": "가변 길이\n(32K × 1 + 512 × 31)",
                  "arrivals": "요청 도착\n(16K × 1 + 512 × 47, 3회 도착)"}


def setup_style(font=None):
    path = Path(font) if font else next((p for p in FONT_CANDIDATES if p.exists()), None)
    if path is not None:
        font_manager.fontManager.addfont(str(path))
        bold = path.with_name(path.stem + "bd" + path.suffix)       # malgun.ttf -> malgunbd.ttf
        if bold.exists():
            font_manager.fontManager.addfont(str(bold))
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
    matplotlib.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.unicode_minus": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 220, "figure.dpi": 100,
    })
    return path


# ---- Figure 1: system overview ---------------------------------------------------------------------

def fig_system(repo):
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.set_axis_off()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    boxes = [
        (0.5, 1.45, "① 커널 실측", "RTX 4090 · 6,801개 조건\n실행 시간 · 스레드 블록 수"),
        (1.5, 1.45, "② 원인 설명과 예측", "점유율 분석 · Accel-Sim\n실측 기반 성능 모델"),
        (2.5, 1.45, "③ 커널 선택 정책", "기본 휴리스틱 · 고정 분할 수\n조회 테이블 · 성능 모델"),
        (2.5, 0.45, "④ 실제 LLM 생성", "Qwen3-4B · Llama-8B\n연속 배치 · paged KV 캐시"),
        (1.5, 0.45, "⑤ 결과 검증", "TPOT · 생성 토큰 일치\n반복 측정"),
        (0.5, 0.45, "⑥ 재현 · 시각화", "GPU 없이 수치 재계산\n대시보드"),
    ]
    for x, y, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.33), 0.84, 0.66, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor="#f3f6fb", edgecolor="#2a78d6", linewidth=1.0))
        ax.text(x, y + 0.15, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x, y - 0.11, body, ha="center", va="center", fontsize=7.5, color=MUTED, linespacing=1.4)
    arrows = [((0.92, 1.45), (1.08, 1.45)), ((1.92, 1.45), (2.08, 1.45)), ((2.5, 1.12), (2.5, 0.78)),
              ((2.08, 0.45), (1.92, 0.45)), ((1.08, 0.45), (0.92, 0.45))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=INK, linewidth=1.0))
    ax.text(0.5, 1.94, "입력: 요청 길이 분포(워크로드)", ha="center", va="center", fontsize=8, color=MUTED)
    ax.add_patch(FancyArrowPatch((0.5, 1.89), (0.5, 1.79), arrowstyle="-|>", mutation_scale=8, color=MUTED))
    fig.tight_layout(pad=0.2)
    return fig


# ---- Figure 2: work per thread block in the worst mixed-length cell --------------------------------

def fig_imbalance(repo):
    w = Workload.from_key(WORST_RAGGED)
    t = _table(repo, "ragged_s1_dense").loc[WORST_RAGGED]
    runs = [("flashdecoding", "기본 휴리스틱 (split 1)", t["heuristic_us"], SERIES[1]),
            ("fd_s16", "split 16 (측정 최적)", t["best_us"], SERIES[0])]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)
    for ax, (name, label, us, color) in zip(axes, runs):
        keys = np.sort(build_launch(w, parse_variant(name), 128).keys)[::-1]
        ax.fill_between(np.arange(len(keys)), keys, step="post", color=color, alpha=0.85, linewidth=0)
        ax.set_xlim(0, len(keys))
        ax.set_title(f"{label}: 커널 {us:,.0f} µs", loc="left")
        ax.set_xlabel("스레드 블록(CTA), 작업량 순 정렬")
        busy = int((keys == keys.max()).sum())
        ax.text(0.97, 0.93, f"CTA {len(keys):,}개\n최대 {keys.max():,} key를 가진 CTA {busy}개",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK)
    axes[0].set_ylabel("CTA 하나가 처리하는 key 수")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout(pad=0.4)
    return fig


# ---- Figure 3: how much the library heuristic loses, uniform vs mixed-length ---------------------

def _table(repo, group, family="dense"):
    return dispatch_table(load_dirs([repo / "demo_data" / "hw_4090" / group]), family=family,
                          cache_state="cold").set_index("workload_key")


def fig_loss_cdf(repo):
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for (group, label), color in zip([("uniform_s1_dense", "균일 길이 배치"), ("ragged_s1_dense", "가변 길이 배치")],
                                     SERIES):
        t = _table(repo, group)
        s = np.sort((t.heuristic_us / t.best_us).dropna().to_numpy())
        y = np.arange(1, len(s) + 1) / len(s)
        ax.step(s, y, where="post", color=color, linewidth=2)
        ax.text(s[-1] * 1.08, 1.0, f"{label} ({len(s)}개, 최대 {s[-1]:.2f}배)", color=INK, fontsize=8, va="center")
        med = float(np.median(s))
        ax.plot([med], [0.5], "o", color=color, markersize=5, markeredgecolor="white")
        ax.annotate(f"중앙값 {med:.2f}배", (med, 0.5), xytext=(6, -12), textcoords="offset points", fontsize=7.5,
                    color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(0.95, 30)
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}배"))
    ax.set_xlabel("기본 휴리스틱의 커널 시간 / 같은 조건의 최적 커널 시간 (로그 축)")
    ax.set_ylabel("누적 비율")
    fig.tight_layout(pad=0.4)
    return fig


# ---- Figure 4: kernel time against the number of KV splits in the worst cell --------------------

def fig_split_sweep(repo):
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    splits = [1, 2, 4, 8, 16, 32, 64, 128]
    for (family, group, label, heur), color in zip(
            [("dense", "ragged_s1_dense", "연속 KV 캐시", "flashdecoding"),
             ("paged", "ragged_s1_paged", "paged KV 캐시", "flashdecoding_paged")], SERIES):
        rows = load_dirs([repo / "demo_data" / "hw_4090" / group])
        rows = rows[(rows.workload_key == WORST_RAGGED) & (rows.cache_state == "cold")]
        us = rows.groupby("kernel").value.median()
        suffix = "_paged" if family == "paged" else ""
        ys = [us[f"fa2{suffix}"]] + [us[f"fd_s{s}{suffix}"] for s in splits[1:]]
        ax.plot(splits, ys, "-o", color=color, linewidth=2, markersize=5, markeredgecolor="white", label=label)
        ax.axhline(us[heur], color=color, linewidth=1, linestyle="--")
        ax.text(1.05, us[heur] * 1.07, f"{label}: 기본 휴리스틱 {us[heur]:,.0f} µs", color=INK, fontsize=7.5)
        best = int(np.argmin(ys))
        ax.annotate(f"최적 {ys[best]:,.0f} µs", (splits[best], ys[best]), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=7.5, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(splits)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    ax.set_ylim(200, 6000)
    ax.set_yticks([200, 300, 500, 1000, 2000, 3000, 5000])
    ax.get_yaxis().set_major_formatter(plain)
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("KV 분할 수 (num_splits)")
    ax.set_ylabel("attention 커널 시간 (µs, 로그 축)")
    ax.legend(loc="center right")
    fig.tight_layout(pad=0.4)
    return fig


# ---- Figure 5: time per output token on the whole model ------------------------------------------

def fig_tpot(repo):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    scenarios = ["uniform", "ragged", "arrivals"]
    width = 0.2
    for i, sc in enumerate(scenarios):
        s = pd.read_csv(repo / "demo_data" / "serve_4090" / CAMPAIGN / sc / "summary.csv").set_index("policy")
        for j, p in enumerate(POLICIES):
            x = i + (j - 1.5) * (width + 0.02)
            ok = bool(s.loc[p, "tokens_equivalent"])
            ax.bar(x, s.loc[p, "tpot_ms_mean"], width, color=POLICY_COLOR[p], hatch=None if ok else "////",
                   edgecolor="white", linewidth=0.8, label=POLICY_LABEL[p] if i == 0 else None)
            ax.errorbar(x, s.loc[p, "tpot_ms_mean"], yerr=s.loc[p, "tpot_ms_std"], color=INK, linewidth=0.8,
                        capsize=2)
            ax.text(x, s.loc[p, "tpot_ms_mean"] + 2, f"{s.loc[p, 'tpot_ms_mean']:.1f}" + ("" if ok else "*"),
                    ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([SCENARIO_LABEL[s] for s in scenarios])
    ax.set_ylabel("요청별 토큰 간 지연 TPOT (ms)")
    ax.set_ylim(0, 125)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.13))
    ax.text(1.0, -0.3, "* 생성 토큰이 기본 휴리스틱과 달라 생성 결과가 동일한 개선으로 인정하지 않음", transform=ax.transAxes,
            ha="right", fontsize=7, color=MUTED)
    fig.tight_layout(pad=0.4)
    return fig


# ---- Figure 6: surrogate model prediction against measurement (mixed-length set V6) -------------

def fig_model_scatter(repo):
    from kernelscope.model.fit import prepare_rows, row_time_us
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    from kernelscope.model.validate import in_set
    rows = prepare_rows(load_dirs([repo / "demo_data" / "hw_4090" / d for d in P0_DIRS]),
                        MachineSpec.from_json(repo / "machines" / "rtx4090.json"))
    params = ModelParams.from_json(repo / "models" / "rtx4090_uniform.json").states["cold"]
    rs = [r for r in rows if r.cache_state == "cold" and in_set(r, "V6")]
    measured = np.array([r.measured_us for r in rs])
    predicted = np.array([row_time_us(r, params) for r in rs])
    ape = np.abs(predicted / measured - 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.scatter(measured, predicted, s=8, color=SERIES[0], alpha=0.35, linewidths=0)
    lo, hi = 5, 1e4
    ax.plot([lo, hi], [lo, hi], color=INK, linewidth=1)
    for f in (0.775, 1.225):
        ax.plot([lo, hi], [lo * f, hi * f], color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("실측 커널 시간 (µs)")
    ax.set_ylabel("성능 모델 예측 (µs)")
    ax.text(0.04, 0.96, f"가변 길이 검증 집합 V6, {len(rs):,}개\n오차 중앙값 {np.median(ape):.1%}, "
            f"90백분위 {np.percentile(ape, 90):.1%}\n점선: ±22.5%", transform=ax.transAxes, va="top", fontsize=7.5)
    fig.tight_layout(pad=0.4)
    return fig


# ---- Figure 7: GPGPU-Sim cycles against the split count, ragged vs uniform batch ----------------

def fig_sim_splits(repo):
    d = pd.read_csv(repo / "docs" / "experiments" / "gpgpusim-splitkv-sweep.csv")
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    for (w, label), color in zip([("ragged", "가변 길이 (2048 + 128×15)"), ("uniform", "균일 길이 (256×16)")], SERIES):
        g = d[d.workload == w].sort_values("S")
        ax.plot(g.S, g.gpu_sim_cycle, "-o", color=color, linewidth=2, markersize=5, markeredgecolor="white", label=label)
        best = g.loc[g.gpu_sim_cycle.idxmin()]
        ax.annotate(f"S={int(best.S)}: {g.gpu_sim_cycle.iloc[0] / best.gpu_sim_cycle:.2f}배 빠름", (best.S, best.gpu_sim_cycle),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([20000, 50000, 100000, 200000, 500000])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("KV 분할 수 (num_splits)")
    ax.set_ylabel("시뮬레이션 사이클 (GPGPU-Sim, 로그 축)")
    ax.legend(loc="upper right")
    fig.tight_layout(pad=0.4)
    return fig


FIGURES = {
    "fig1_system": fig_system,
    "fig2_imbalance": fig_imbalance,
    "fig3_loss_cdf": fig_loss_cdf,
    "fig4_split_sweep": fig_split_sweep,
    "fig5_tpot": fig_tpot,
    "fig6_model_scatter": fig_model_scatter,
    "fig7_sim_splits": fig_sim_splits,
}


def render_all(repo, out, font=None) -> list[Path]:
    repo, out = Path(repo), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    setup_style(font)
    written = []
    for name, build in FIGURES.items():
        fig = build(repo)
        path = out / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/report/fig")
    ap.add_argument("--font", help="TTF of the report body font (default: Malgun Gothic if installed)")
    args = ap.parse_args(argv)
    matplotlib.use("Agg")
    for p in render_all(Path(__file__).resolve().parents[1], args.out, args.font):
        print(p)


if __name__ == "__main__":
    main()
