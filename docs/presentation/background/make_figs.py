"""Figures for the talk's background part (problem definition). Reads only the bundle `demo_data/`.
Run from the repo root:  .venv/bin/python docs/presentation/background/make_figs.py
Colors/fonts follow docs/report/midterm/src/make_figs_v5.py; slide-size text (1600x900 px)."""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "demo_data"
OUT = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA, GRAY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#c0392b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "font.size": 16, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": False, "figure.facecolor": "white", "axes.titlesize": 20, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.titlecolor": INK})


def base(figsize=(16, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.15)
    return fig, ax


def style(ax, ygrid=True):
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0, labelsize=15)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def save(fig, name, note):
    import textwrap
    fig.text(0.01, 0.012, "\n".join(textwrap.wrap(note, 150)), fontsize=12.5, color=INK2, ha="left", va="bottom")
    fig.savefig(OUT / name, dpi=100)
    plt.close(fig)
    print("wrote", name)


# ---- b1: prompt-length distribution (complementary CDF) ------------------------------------------------------
def fig_b1():
    d = json.loads((DATA / "background/prompt_length_hist.json").read_text())["traces"]
    colors = {"Azure conv": BLUE, "Azure code": ORANGE, "BurstGPT (7일)": AQUA}
    fig, ax = base()
    for t in d:
        e, c, n = np.array(t["hist_edges"]), np.array(t["hist_counts"]), t["n"]
        ccdf = np.append(np.cumsum(c[::-1])[::-1], 0) / n           # P(prompt >= edge) for edges inside [16, 128K]
        keep = (ccdf > 0) & (e <= t["max"])
        xs = np.append(e[keep], t["max"]); ys = np.append(ccdf[keep], 1 / n)
        ax.plot(xs, ys, color=colors[t["name"]], linewidth=3.2, label=f'{t["name"]} (n={n:,})')
        ax.plot([t["max"]], [1 / n], "o", color=colors[t["name"]], markersize=10, markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f'최대 {t["max"]:,}', (t["max"], 1 / n), xytext=(8, 12), textcoords="offset points", color=INK, fontsize=15)
    for x, lab in ((32768, "32K (커널 격자)"), (65536, "64K")):
        ax.axvline(x, color=GRAY, linewidth=2, linestyle=(0, (5, 4)))
        ax.text(x * 1.04, 0.25, lab, rotation=90, va="center", ha="left", color=INK2, fontsize=15)
    ax.axvline(4096, color=GRAY, linewidth=1.2, linestyle=":")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlim(32, 160000); ax.set_ylim(3e-5, 1.3)
    ax.set_xticks([64, 256, 1024, 4096, 16384, 65536]); ax.set_xticklabels(["64", "256", "1K", "4K", "16K", "64K"])
    ax.set_xlabel("프롬프트 길이 (토큰)", fontsize=17); ax.set_ylabel("길이가 그 값 이상인 요청의 비율 (로그)", fontsize=17)
    ax.set_title("공개 트래픽의 프롬프트는 대부분 짧고, 몇 천 토큰짜리가 소수 섞인다")
    style(ax); ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.legend(frameon=False, loc="lower left", fontsize=16)
    save(fig, "fig_b1_length_dist.png", "출처: Azure LLM Inference Trace 2023(conv·code), BurstGPT v2 첫 7일 — demo_data/background/prompt_length_hist.json. 세 트레이스 모두 32K 이상 요청 0건.")


# ---- b2: load sweep (model prediction) + measured check ----------------------------------------------------
def fig_b2():
    t = pd.read_csv(DATA / "traffic_sweep/sweep_table.csv")
    t = t.assign(trace=t.trace.map({"azure_conv": "Azure conv", "azure_code": "Azure code", "burstgpt_7d": "BurstGPT (7일)"}))
    colors = {"Azure conv": BLUE, "Azure code": ORANGE, "BurstGPT (7일)": AQUA}
    fig, ax = base()
    ax.axvspan(0, 26, color="#f1f0ec", zorder=0)
    ax.axvline(26, color=GRAY, linewidth=2, linestyle=(0, (5, 4)))
    ax.text(25.3, 0.171, "B < 26: 휴리스틱이 스스로 분할", ha="right", va="top", color=INK2, fontsize=14.5)
    ax.text(26.7, 0.171, "B ≥ 26: 분할 없음", ha="left", va="top", color=INK2, fontsize=14.5)
    for name, g in t.groupby("trace"):
        g = g.sort_values("batch_mean")
        c = colors[name]
        ax.plot(g.batch_mean, g["loss_ge_1.25_frac"], color=c, linewidth=3, zorder=3)
        over = g["load.wait_gt_1s_frac"] > 0.5            # queue wait > 1 s for most requests = overloaded replay
        ax.scatter(g.batch_mean[~over], g["loss_ge_1.25_frac"][~over], s=110, color=c, edgecolor="white", linewidth=2, zorder=4)
        ax.scatter(g.batch_mean[over], g["loss_ge_1.25_frac"][over], s=110, facecolor="white", edgecolor=c, linewidth=2.5, zorder=4)
        last = g.iloc[-1] if name != "Azure conv" else g[g.batch_mean < 40].iloc[-1]
    lab = {"Azure conv": (43.5, 0.10), "Azure code": (8.2, 0.066), "BurstGPT (7일)": (29, 0.011)}
    for name, (x, y) in lab.items():
        ax.text(x, y, name, color=INK, fontsize=17, fontweight="bold", ha="left")
    pk = t[(t.trace == "Azure conv") & (t.rate_scale == 1)].iloc[0]
    ax.annotate(f'conv ×1: {pk["loss_ge_1.25_frac"]*100:.1f}% (모델 예측)', (pk.batch_mean, pk["loss_ge_1.25_frac"]),
                xytext=(14, 20), textcoords="offset points", fontsize=15.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2))
    # measured check of the same quantity (item 02, stratified sample, 95% interval)
    from kernelscope.analysis import validation as V
    m = V.prepare(pd.read_csv(DATA / "hw_4090/trace_validation_20261024/measured_pass1.csv"))
    est = V.source_stats(m, "azure_conv_x1")["w_meas_frac"]; lo, hi = V.bootstrap(m, "azure_conv_x1", "w_meas_frac")
    ax.errorbar([pk.batch_mean + 0.9], [est], yerr=[[est - lo], [hi - est]], fmt="D", color=INK, markersize=11, capsize=6,
                elinewidth=2.5, zorder=5)
    ax.annotate(f"실측 표본 추정 {est*100:.1f}%\n(95% 구간 {lo*100:.1f}–{hi*100:.1f}%)", (pk.batch_mean + 0.9, est),
                xytext=(-16, -10), textcoords="offset points", fontsize=14.5, color=INK, ha="right")
    ax.scatter([], [], s=110, color=GRAY, label="서버가 감당하는 부하(큐 대기 > 1 s 요청이 절반 미만)")
    ax.scatter([], [], s=110, facecolor="white", edgecolor=GRAY, linewidth=2.5, label="과부하 재생(절반 이상이 1 s 넘게 대기)")
    ax.legend(frameon=False, loc="upper left", fontsize=14.5, bbox_to_anchor=(0.0, 0.93))
    ax.set_xlim(5, 52); ax.set_ylim(0, 0.18)
    ax.set_yticks([0, .05, .10, .15]); ax.set_yticklabels(["0%", "5%", "10%", "15%"])
    ax.set_xlabel("평균 batch 크기 (도착률 배율로 조절)", fontsize=17); ax.set_ylabel("regret ≥ 1.25배인 decode step 비율", fontsize=17)
    ax.set_title("regret가 큰 step은 batch가 26을 넘는 부하에서 최대 약 14%, 짧은 프롬프트 트래픽에서는 0.3% 이하")
    style(ax)
    save(fig, "fig_b2_load_sweep.png", "점·선은 성능 모델 예측(21개 재생, 오차 2.3~12.3%), ◆는 표본 100셀의 커널 실측(conv ×1). regret = 휴리스틱 시간 / 후보 {휴리스틱, split 2·4·8·16} 중 최선 시간. 출처: demo_data/traffic_sweep, hw_4090/trace_validation_20261024.")


# ---- b3: kernel regret, ragged vs uniform and 8K→64K trend ------------------------------------------------
def fig_b3():
    from kernelscope.verify import _defaults_table, _matched_long_cells
    t32 = _defaults_table(DATA, "32k")
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.15, 1]})
    for sel, name, c in ((t32.ragged.astype(bool), "혼합 길이 (162셀)", ORANGE), (~t32.ragged.astype(bool), "균일 길이 (49셀)", BLUE)):
        v = np.sort(t32[sel].fa2_heuristic_loss.values)
        a.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, linewidth=3.4)
        med = float(np.median(v)); mx = float(v.max())
        a.plot([med], [0.5], "o", color=c, markersize=11, markeredgecolor="white", markeredgewidth=2)
        a.text(2.45 if c == ORANGE else 1.4, 0.5 if c == ORANGE else 0.99, f"{name}\n중앙값 {med:.2f}배 · 최악 {mx:.2f}배",
               va="top", color=INK, fontsize=16, fontweight="bold" if c == ORANGE else "normal")
    a.axvline(1.25, color=GRAY, linewidth=1.8, linestyle=(0, (5, 4))); a.text(1.265, 0.04, "1.25배", color=INK2, fontsize=14)
    a.set_xlim(0.98, 4.4); a.set_ylim(0, 1.04)
    a.set_xlabel("regret = FA2 휴리스틱 시간 / 전체 최선 시간", fontsize=16); a.set_ylabel("누적 셀 비율", fontsize=16)
    a.set_title("32K 격자 (batch 26–64, 4090)", fontsize=18); style(a)
    m = _matched_long_cells(DATA)
    lens = [8192, 16384, 32768, 65536]; names = ["8K", "16K", "32K", "64K"]
    med = [m[m.long == L].fa2_loss.median() for L in lens]; mx = [m[m.long == L].fa2_loss.max() for L in lens]
    b.plot(names, mx, color="#86b6ef", linewidth=3.4, marker="o", markersize=11, markeredgecolor="white", markeredgewidth=2)
    b.plot(names, med, color=BLUE, linewidth=3.4, marker="o", markersize=11, markeredgecolor="white", markeredgewidth=2)
    for ys, dy in ((mx, 0), (med, 0)):
        b.annotate(f"{ys[-1]:.2f}배", (3, ys[-1]), xytext=(10, dy), textcoords="offset points", va="center", color=INK, fontsize=15)
    b.text(0.05, mx[0] + 0.28, "최악", color=INK2, fontsize=15); b.text(0.05, med[0] - 0.42, "중앙값", color=INK2, fontsize=15)
    b.set_ylim(1.0, 5.8); b.set_xlim(-0.2, 3.6)
    b.set_xlabel("batch 안 긴 요청 1개의 KV 길이", fontsize=16); b.set_ylabel("regret = FA2 휴리스틱 / 최선 FA2 split", fontsize=16)
    b.set_title("긴 요청이 길수록 커진다", fontsize=18); style(b)
    fig.suptitle("한 batch에 긴 요청이 섞이면 휴리스틱 커널은 중앙값 1.46배, 최악 4.1배 느리다 (균일 길이는 1.17배 안)", x=0.01, ha="left", fontsize=20, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, "fig_b3_kernel_regret.png", "실측(RTX 4090, cold L2, 페이지 KV, Qwen3-4B 헤드 32/8). 왼쪽: FA2 휴리스틱 / FlashInfer 포함 전체 최선. 오른쪽: 길이마다 같은 18셀(B 26·32·64), FA2 split만 비교. 출처: demo_data/hw_4090/*_paged·*_64k_*.")


# ---- b5: per-CTA work of the mixed batch ---------------------------------------------------------------------
def fig_b5():
    from kernelscope.model.geometry import build_launch, parse_variant
    from kernelscope.verify import BG_CELL
    from kernelscope.workload import Workload
    w = Workload.from_key(BG_CELL)
    share = None
    fig, ax = base()
    for name, label, c in (("flashdecoding_paged", "FA2 휴리스틱: split 1 (CTA 256개)", ORANGE), ("fd_s16_paged", "split 16 (작업이 있는 CTA 376개)", BLUE)):
        keys = build_launch(w, parse_variant(name), 128).keys
        share = keys.sum() / 128
        v = np.sort(keys[keys > 0])[::-1]
        ax.step(np.arange(1, len(v) + 1), v, where="mid", color=c, linewidth=3.6, label=label)
    ax.axhline(share, color=INK2, linewidth=2, linestyle=(0, (5, 4)))
    ax.text(375, share * 1.08, f"SM 1개 몫(총 작업 ÷ 128) = {share:,.0f}", ha="right", color=INK2, fontsize=15)
    ax.annotate("가장 큰 CTA = KV 32,768개\n(SM 몫의 10.8배) — 나머지 SM이 끝나도\n이 CTA가 끝날 때까지 kernel이 안 끝난다",
                (8, 32768), xytext=(30, -95), textcoords="offset points", fontsize=15.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2))
    ax.annotate("짧은 요청 CTA: KV 512개", (250, 512), xytext=(0, 14), textcoords="offset points", fontsize=15, color=INK, ha="center")
    ax.annotate("최대 2,048 (SM 몫 이하)", (128, 2048), xytext=(14, -28), textcoords="offset points", fontsize=15, color=INK)
    ax.set_yscale("log"); ax.set_xlim(0, 380); ax.set_ylim(300, 80000)
    ax.set_yticks([512, 2048, 8192, 32768]); ax.set_yticklabels(["512", "2,048", "8,192", "32,768"])
    ax.set_xlabel("CTA (작업량이 큰 순서)", fontsize=17); ax.set_ylabel("CTA 하나가 처리하는 KV 토큰 수 (로그)", fontsize=17)
    ax.set_title("batch 32 = 32K 요청 1개 + 512 요청 31개: split 1이면 CTA 8개가 전체 시간을 정한다")
    ax.legend(frameon=False, loc="upper right", fontsize=16)
    style(ax)
    save(fig, "fig_b5_cta_imbalance.png", "kernelscope.model.geometry.build_launch로 만든 launch 구성(RTX 4090, SM 128, KV head 8, 페이지 KV). 작업량은 CTA가 처리할 KV 토큰 수(예측이 아닌 기하학적 계산).")


if __name__ == "__main__":
    for f in (fig_b1, fig_b2, fig_b3, fig_b5):
        f()
