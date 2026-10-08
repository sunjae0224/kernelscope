"""Fastest interchangeable kernel variant per workload, and what the library heuristic loses.

Variants in one family compute the same attention on the same cache layout, so a dispatcher may
swap them freely: dense (fa2 = num_splits 1, flashdecoding = the library heuristic, fd_s{N},
sdpa_flash) and paged (the same on a block_table cache). Input: real-HW rows with a
cache_state column (rows without one were measured warm).
"""
import math

import pandas as pd

from kernelscope.workload import Workload, format_lens

FAMILIES = {
    "dense": {"heuristic": "flashdecoding", "fa2": "fa2", "members": r"^(fa2|flashdecoding|fd_s\d+|sdpa_flash)$"},
    "paged": {"heuristic": "flashdecoding_paged", "fa2": "fa2_paged",
              "members": r"^(fa2_paged|flashdecoding_paged|fd_s\d+_paged)$"},
}
COLUMNS = ["workload_key", "B", "L_kv", "H_q", "H_kv", "ragged", "lens", "n_variants", "best_kernel", "best_us",
           "heuristic_us", "heuristic_regret", "fa2_us", "fa2_regret"]


def dispatch_table(df: pd.DataFrame, family: str = "dense", cache_state: str = "cold") -> pd.DataFrame:
    fam = FAMILIES[family]
    states = df["cache_state"].fillna("warm") if "cache_state" in df.columns else pd.Series("warm", index=df.index)
    sel = df[(df.backend == "profile") & (df.metric == "kernel_time_us") & (states == cache_state)
             & df.kernel.str.match(fam["members"])]
    if sel.empty:
        return pd.DataFrame(columns=COLUMNS)
    t = sel.groupby(["workload_key", "kernel"])["value"].median().unstack("kernel")
    rows = []
    for key, r in t.iterrows():
        r = r.dropna()
        w = Workload.from_key(key)
        best_kernel, best = r.idxmin(), r.min()
        h = r.get(fam["heuristic"], math.nan)
        f = r.get(fam["fa2"], math.nan)
        rows.append({"workload_key": key, "B": w.B, "L_kv": w.L_kv, "H_q": w.H_q, "H_kv": w.H_kv,
                     "ragged": w.is_ragged, "lens": format_lens(w.lens()), "n_variants": len(r),
                     "best_kernel": best_kernel, "best_us": best, "heuristic_us": h,
                     "heuristic_regret": h / best - 1, "fa2_us": f, "fa2_regret": f / best - 1})
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("heuristic_regret", ascending=False, na_position="last",
                                                            ignore_index=True)


def regret_summary(table: pd.DataFrame) -> dict:
    r = table["heuristic_regret"].dropna()
    if r.empty:
        return {"cells": len(table), "median_regret": math.nan, "max_regret": math.nan,
                "cells_over_10pct": 0, "worst_key": None}
    return {"cells": len(table), "median_regret": float(r.median()), "max_regret": float(r.max()),
            "cells_over_10pct": int((r > 0.10).sum()),
            "worst_key": table.loc[r.idxmax(), "workload_key"]}


# Two libraries' static defaults on the paged grid: flash-attn's num_splits heuristic and FlashInfer's
# tensor-core / CUDA-core decode variants (vLLM picks the tensor-core variant for GQA group sizes >= 4).
FLASHINFER = {"flashinfer_paged": "flashinfer_tc", "flashinfer_paged_cudacore": "flashinfer_cc"}
DEFAULTS = ("fa2_heuristic", "flashinfer_tc", "flashinfer_cc", "fa2_best")
DEFAULT_COLUMNS = ["workload_key", "B", "L_kv", "H_q", "H_kv", "ragged", "lens", "n_variants", "best_kernel", "best_us",
                   "fa2_heuristic_us", "fa2_heuristic_loss", "flashinfer_tc_us", "flashinfer_tc_loss",
                   "flashinfer_cc_us", "flashinfer_cc_loss", "fa2_best_kernel", "fa2_best_us", "fa2_best_loss", "complete"]
SUMMARY_COLUMNS = ["cells", "default", "n", "incomplete", "loss_median", "loss_p90", "loss_max", "over_1.1", "over_1.25",
                   "over_2"]


def static_default_losses(df: pd.DataFrame, cache_state: str = "cold") -> pd.DataFrame:
    """Per paged cell: the best of every measured variant (FA2 split counts and both FlashInfer variants) and
    the loss of each static default against it. ``fa2_best`` is a dispatcher limited to FA2 split counts.
    ``complete`` says the cell measured all three defaults; a default missing from a cell keeps NaN there."""
    fam = FAMILIES["paged"]
    states = df["cache_state"].fillna("warm") if "cache_state" in df.columns else pd.Series("warm", index=df.index)
    sel = df[(df.backend == "profile") & (df.metric == "kernel_time_us") & (states == cache_state)
             & (df.kernel.str.match(fam["members"]) | df.kernel.isin(FLASHINFER))]
    if sel.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)
    t = sel.groupby(["workload_key", "kernel"])["value"].median().unstack("kernel")
    rows = []
    for key, r in t.iterrows():
        r = r.dropna()
        fa2 = r[[k for k in r.index if k not in FLASHINFER]]
        if fa2.empty:
            continue
        w = Workload.from_key(key)
        best_kernel, best = r.idxmin(), float(r.min())
        h, tc, cc = (float(r.get(k, math.nan)) for k in (fam["heuristic"], "flashinfer_paged", "flashinfer_paged_cudacore"))
        fa2_best_kernel, fa2_best = fa2.idxmin(), float(fa2.min())
        rows.append({"workload_key": key, "B": w.B, "L_kv": w.L_kv, "H_q": w.H_q, "H_kv": w.H_kv, "ragged": w.is_ragged,
                     "lens": format_lens(w.lens()), "n_variants": len(r), "best_kernel": best_kernel, "best_us": best,
                     "fa2_heuristic_us": h, "fa2_heuristic_loss": h / best, "flashinfer_tc_us": tc,
                     "flashinfer_tc_loss": tc / best, "flashinfer_cc_us": cc, "flashinfer_cc_loss": cc / best,
                     "fa2_best_kernel": fa2_best_kernel, "fa2_best_us": fa2_best, "fa2_best_loss": fa2_best / best,
                     "complete": all(math.isfinite(x) for x in (h, tc, cc))})
    return pd.DataFrame(rows, columns=DEFAULT_COLUMNS).sort_values("fa2_heuristic_loss", ascending=False,
                                                                   na_position="last", ignore_index=True)


def static_default_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Loss quantiles and strict threshold counts (loss > 1.1 / 1.25 / 2) of every default over ragged, uniform and
    all cells. Only complete cells are scored, so every default is compared over the same cells and against the
    same best; ``incomplete`` is the number of cells left out of that group."""
    rows = []
    groups = (("ragged", table[table.ragged.astype(bool)]), ("uniform", table[~table.ragged.astype(bool)]), ("all", table))
    for cells, group in groups:
        scored = group[group.complete.astype(bool)] if "complete" in group else group
        for default in DEFAULTS:
            loss = scored[f"{default}_loss"].dropna()
            rows.append({"cells": cells, "default": default, "n": len(loss), "incomplete": int(len(group) - len(scored)),
                         "loss_median": float(loss.median()) if len(loss) else math.nan,
                         "loss_p90": float(loss.quantile(.9)) if len(loss) else math.nan,
                         "loss_max": float(loss.max()) if len(loss) else math.nan,
                         "over_1.1": int((loss > 1.1).sum()), "over_1.25": int((loss > 1.25).sum()),
                         "over_2": int((loss > 2.0).sum())})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
