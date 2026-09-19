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
