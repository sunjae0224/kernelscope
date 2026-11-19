"""Report figures for the updated midterm report (v5). Numbers come from docs/experiments/2026-10-06-problem-scope.md
(verified by `kernelscope verify`) and from demo_data/serve_4090/diagnose_20260927/ragged/ops.csv."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path("/home/skkai/AI_Accelerator/kernelscope-design")
OUT = ROOT / "docs/report/fig"
BLUE, ORANGE, AQUA, GRAY, BLUE_L = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#86b6ef"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": False, "savefig.dpi": 220, "figure.facecolor": "white"})


def clean(ax, x=True):
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    if x:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


# ---- fig8: long-request length trend -------------------------------------------------------------
lengths = ["8K", "16K", "32K", "64K"]
fa2_med = [1.545, 1.933, 2.558, 2.839]
fa2_max = [2.295, 2.991, 3.981, 5.183]
fi_tc_med = [1.126, 1.269, 1.443, 1.634]
fig, ax = plt.subplots(figsize=(6.2, 3.1))
for ys, c, lab, ls in [(fa2_max, BLUE_L, "FlashAttention-2 휴리스틱, 최악", "-"),
                       (fa2_med, BLUE, "FlashAttention-2 휴리스틱, 중앙값", "-"),
                       (fi_tc_med, ORANGE, "FlashInfer tensor-core 기본값, 중앙값", "-")]:
    ax.plot(lengths, ys, color=c, linewidth=2, marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.5, label=lab)
    ax.annotate(f"{ys[-1]:.2f}배", (3, ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=8.5)
ax.set_ylim(1.0, 5.6)
ax.set_ylabel("최선 후보 대비 실행 시간 배율")
ax.set_xlabel("긴 요청의 KV 길이 (조건마다 18셀, 페이지 KV, cold)")
ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
ax.spines["left"].set_visible(False); ax.tick_params(length=0)
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.set_xlim(-0.2, 3.55)
fig.tight_layout()
fig.savefig(OUT / "fig8_longctx.png"); plt.close(fig)

# ---- fig9: engine-level comparison (TPOT) ----------------------------------------------------------
panels = [("가변 길이 (32K×1 + 512×31)", [("기본 휴리스틱", 60.99, 1.0), ("FA2 조회 테이블", 33.70, 1.810),
                                        ("FlashInfer CUDA-core", 33.28, 1.833), ("FlashInfer tensor-core", 48.51, 1.257)]),
          ("요청 도착 (16K×1 + 512×47)", [("기본 휴리스틱", 59.18, 1.0), ("FA2 조회 테이블", 50.46, 1.173),
                                        ("FlashInfer CUDA-core", 50.12, 1.181), ("FlashInfer tensor-core", 54.18, 1.092)]),
          ("균일 길이 (512×32)", [("기본 휴리스틱", 27.45, 1.0), ("FA2 조회 테이블", 27.43, 1.001),
                                 ("FlashInfer CUDA-core", 27.66, 0.992), ("FlashInfer tensor-core", 27.57, 0.996)])]
fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.6), sharey=False)
for ax, (title, rows) in zip(axes, panels):
    names = [r[0] for r in rows][::-1]; vals = [r[1] for r in rows][::-1]; sp = [r[2] for r in rows][::-1]
    colors = [GRAY if n == "기본 휴리스틱" else BLUE if n == "FA2 조회 테이블" else ORANGE for n in names]
    ax.barh(names, vals, color=colors, height=0.55)
    for i, (v, s) in enumerate(zip(vals, sp)):
        ax.text(v + 1.0, i, f"{v:.1f} ms" + ("" if s == 1.0 else f" ({s:.2f}배)"), va="center", fontsize=7.5, color=INK)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.set_xlim(0, max(vals) * 1.65)
    clean(ax)
    ax.tick_params(axis="y", labelsize=7.5)
    if ax is not axes[0]:
        ax.set_yticklabels([])
axes[1].set_xlabel("토큰 간 지연 TPOT (ms, 3회 평균) — 회색: 기본값, 파랑: 분할 수 선택(FA2), 주황: FlashInfer 백엔드", fontsize=7.5)
fig.tight_layout(w_pad=0.6)
fig.savefig(OUT / "fig9_engine.png"); plt.close(fig)

# ---- fig10: trace replay --------------------------------------------------------------------------
rows = [("Azure conv ×0.5 (평균 배치 17)", 2.4, False), ("Azure conv ×1 (평균 배치 35)", 13.5, False),
        ("Azure conv ×2 (평균 배치 47)", 4.1, False), ("Azure code ×16 (평균 배치 27)", 14.1, False),
        ("BurstGPT 7일 ×100 (평균 배치 27)", 0.3, False), ("BurstGPT GPT-4 14일 ×300 (평균 배치 30)", 1.5, False),
        ("what-if: conv ×1, 프롬프트 ×2, KV 10 GiB", 25.7, True), ("what-if: conv ×1, 프롬프트 ×4, KV 10 GiB", 2.9, True),
        ("what-if: conv ×1, 프롬프트 ×4, KV 436K 토큰", 39.5, True)]
fig, ax = plt.subplots(figsize=(6.2, 3.3))
names = [r[0] for r in rows][::-1]; vals = [r[1] for r in rows][::-1]; wi = [r[2] for r in rows][::-1]
ax.barh(names, vals, color=[BLUE_L if w else BLUE for w in wi], height=0.55)
for i, v in enumerate(vals):
    ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=8, color=INK)
ax.set_xlim(0, 46)
ax.set_xlabel("attention 시간을 1.25배 이상 잃는 decode step의 비율 (%)")
clean(ax)
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig10_traffic.png"); plt.close(fig)

# ---- fig11: op-class breakdown of the decode step (ragged, heuristic vs table) --------------------
ops = list(csv.DictReader(open(ROOT / "demo_data/serve_4090/diagnose_20260927/ragged/ops.csv")))
groups = {"attention": ["attention"], "GEMM (qkv_proj·o_proj·mlp·lm_head)": ["qkv_proj", "o_proj", "mlp", "lm_head"],
          "norm·rope·embed": ["norm", "rope", "embed"]}
data = {}
for pol in ("heuristic", "table"):
    rows_ = [r for r in ops if r["policy"] == pol and r["phase"] == "decode"]
    step_ms = sum(float(r["gpu_us"]) / float(r["share"]) for r in rows_) / len(rows_) / 1000
    per = {r["op_class"]: float(r["gpu_us"]) / 1000 for r in rows_}
    g = {k: sum(per[c] for c in v) for k, v in groups.items()}
    g["나머지(미측정)"] = step_ms - sum(per.values())
    data[pol] = (step_ms, g)
fig, ax = plt.subplots(figsize=(6.2, 2.3))
labels = {"heuristic": "기본 휴리스틱\n(분할 없음)", "table": "조회 테이블\n(분할 수 16)"}
cols = {"attention": BLUE, "GEMM (qkv_proj·o_proj·mlp·lm_head)": ORANGE, "norm·rope·embed": AQUA, "나머지(미측정)": GRAY}
for i, pol in enumerate(["table", "heuristic"]):
    left = 0
    step_ms, g = data[pol]
    for k, c in cols.items():
        ax.barh(i, g[k], left=left, color=c, height=0.5, edgecolor="white", linewidth=1.5, label=k if i == 1 else None)
        if g[k] > 3.5:
            ax.text(left + g[k] / 2, i, f"{g[k]:.1f}", ha="center", va="center", fontsize=8, color="white")
        left += g[k]
    share = g["attention"] / step_ms * 100
    ax.text(step_ms + 0.6, i, f"step {step_ms:.1f} ms · attention {share:.0f}%", va="center", fontsize=8, color=INK)
ax.set_yticks([0, 1]); ax.set_yticklabels([labels["table"], labels["heuristic"]], fontsize=8)
ax.set_xlim(0, 72); ax.set_xlabel("decode step 시간 (ms, CUDA event로 측정한 연산 클래스별 합)")
clean(ax)
ax.legend(frameon=False, loc="lower right", fontsize=7.5, ncol=4, bbox_to_anchor=(1.0, -0.62))
fig.tight_layout()
fig.savefig(OUT / "fig11_opbreakdown.png", bbox_inches="tight"); plt.close(fig)
print({k: (round(v[0], 2), {kk: round(vv, 2) for kk, vv in v[1].items()}) for k, v in data.items()})
