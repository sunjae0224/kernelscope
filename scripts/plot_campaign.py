"""Export a presentation figure from recorded serving summaries (no new timing)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from kernelscope.serve.report import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scenarios = [name for name in ("uniform", "ragged", "arrivals") if (args.campaign / name / "summary.csv").exists()]
    if not scenarios:
        raise SystemExit("No recorded summary.csv files found")
    policies = ["heuristic", "fixed8", "table", "model"]
    colors = ["#2878d6", "#eb6834", "#1baf7a", "#8a65c5"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    tables = {name: summarize(args.campaign / name).set_index("policy") for name in scenarios}
    x, width = np.arange(len(scenarios)), .18
    for axis, metric, title in zip(axes, ("decode_wall_ms_per_step", "tpot_ms_mean"),
                                   ("Full decode + policy decision", "Mean per-request time per output token")):
        for index, (policy, color) in enumerate(zip(policies, colors)):
            values = [tables[name].loc[policy, metric] if policy in tables[name].index else np.nan for name in scenarios]
            bars = axis.bar(x + (index - 1.5) * width, values, width, label=policy, color=color)
            for name, bar in zip(scenarios, bars):
                if policy in tables[name].index and not bool(tables[name].loc[policy, "tokens_equivalent"]):
                    bar.set_hatch("///")
                    bar.set_edgecolor("#313743")
            axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
        axis.set_xticks(x, scenarios)
        axis.set_ylabel("Milliseconds (lower is faster)")
        axis.set_title(title, fontsize=12, pad=16)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.16)
        axis.set_axisbelow(True)
        axis.set_ylim(0, axis.get_ylim()[1] * 1.15)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("KernelScope | Qwen3-4B on RTX 4090 | Recorded whole-model replay", fontsize=14)
    fig.supxlabel("Hatched bars: generated tokens differ from heuristic; these are not output-preserving gains.\n"
                  "Five repeated runs; fixed-length synthetic prompts. TPOT includes serial prompt-admission stalls.", fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
