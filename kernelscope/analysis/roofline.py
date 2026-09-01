"""Roofline from the analytic track: x = FLOPs/byte, y = achieved TFLOPS, roofs from
measured ceilings (memory roof = AI x HBM GB/s, compute roof = fp16 GEMM peak)."""
from pathlib import Path

import pandas as pd

from kernelscope.workload import Workload


def roofline_points(df: pd.DataFrame) -> pd.DataFrame:
    an = df[df.backend == "analytic"]
    piv = an.pivot_table(index=["kernel", "workload_key"], columns="metric", values="value", aggfunc="median")
    need = ["arithmetic_intensity", "achieved_tflops", "achieved_gbps"]
    piv = piv.reindex(columns=need).dropna()
    out = piv.rename(columns={"arithmetic_intensity": "ai", "achieved_tflops": "tflops", "achieved_gbps": "gbps"})
    out = out.reset_index()
    out["label"] = [_label(k) for k in out["workload_key"]]
    return out


def _label(key: str) -> str:
    w = Workload.from_key(key)
    return f"B{w.B} L{w.L_kv}" if w.phase == "decode" else f"B{w.B} L{w.L_q}/{w.L_kv}"


def plot_roofline(points: pd.DataFrame, ceilings: dict | None, out_path, title: str | None = None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    roof = {}
    hbm = tc = None
    if ceilings:
        hbm = ceilings.get("hbm_gbps") or ceilings.get("hbm_copy_gbps")
        tc = ceilings.get("fp16_matmul_tflops")
    if hbm and tc:
        ridge = tc / (hbm / 1000)
        xs = [1e-2, ridge, ridge * 100]
        ax.plot(xs[:2], [xs[0] * hbm / 1000, tc], color="0.3", lw=1.5, label=f"HBM {hbm:.0f} GB/s")
        ax.plot(xs[1:], [tc, tc], color="0.3", lw=1.5, ls="--", label=f"fp16 GEMM {tc:.0f} TFLOPS")
        for _, r in points.iterrows():
            roof[(r["kernel"], r["workload_key"])] = min(r["ai"] * hbm / 1000, tc)

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, (kernel, grp) in enumerate(points.groupby("kernel")):
        ax.scatter(grp["ai"], grp["tflops"], marker=markers[i % len(markers)], s=55, label=kernel, zorder=3)
        for _, r in grp.iterrows():
            ax.annotate(r["label"], (r["ai"], r["tflops"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte, compulsory traffic)")
    ax.set_ylabel("achieved TFLOPS (analytic FLOPs / kernel time)")
    ax.set_title(title or "Roofline (Nsight-free: analytic traffic vs measured ceilings)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return roof
