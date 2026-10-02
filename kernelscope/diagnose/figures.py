"""Stacked op-class bars from a diagnosis summary. matplotlib/Agg so it renders on any host."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kernelscope.dashboard.style import CATEGORICAL  # noqa: E402
from kernelscope.diagnose.opmodel import OP_CLASSES  # noqa: E402

UNATTRIBUTED = "#9a9a9a"


def op_breakdown(summary: dict, out_png, out_svg=None, title="Decode step time by operation class") -> Path:
    policies = list(summary["policies"])
    colors = CATEGORICAL["light"]
    fig, ax = plt.subplots(figsize=(9, 1.4 + 0.75 * len(policies)))
    for i, name in enumerate(policies):
        p = summary["policies"][name]
        by = {row["op_class"]: row["gpu_us"] / 1e3 for row in p["ops"]}
        left = 0.0
        for j, op_class in enumerate(OP_CLASSES):
            width = by.get(op_class, 0.0)
            ax.barh(i, width, left=left, color=colors[j % len(colors)], label=op_class if i == 0 else None)
            left += width
        rest = max(p["step_waterfall"]["step_us_mean"] / 1e3 - left, 0.0)
        ax.barh(i, rest, left=left, color=UNATTRIBUTED, label="unattributed" if i == 0 else None)
        attention = next((row for row in p["ops"] if row["op_class"] == "attention"), None)
        if attention is not None:
            ax.text(left + rest, i, f"  attention {100 * attention['share']:.0f}% · {attention['verdict']}", va="center", fontsize=8)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels(policies)
    ax.invert_yaxis()
    ax.set_xlabel("ms per decode step (CUDA events; unattributed = step − Σ classes)")
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=150)
    if out_svg is not None:
        fig.savefig(out_svg)
    plt.close(fig)
    return out_png
