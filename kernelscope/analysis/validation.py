"""Measured check of the trace-replay scoring (torch-free; reads the 131-cell measured table of 2026-10-24).

``scripts/sample_replay_cells.py`` sampled decode batches of a trace replay, stratified by the surrogate model's predicted regret;
``gpu_queue/02_trace_validation.sh`` timed them on the RTX 4090 (paged KV, cold L2) for the heuristic and the 2/4/8/16 splits.
The measured regret of a cell is the measured heuristic time over the best measured time of those five variants.
A sampled cell stands for ``stratum_steps / n_sampled`` replay steps of its stratum, so the weighted statistics estimate
step-population values of the replay; the stratum boundaries are model predictions, which is exactly what is being checked.
"""
import numpy as np
import pandas as pd

VARIANTS = ("flashdecoding_paged", "fd_s2_paged", "fd_s4_paged", "fd_s8_paged", "fd_s16_paged")
THRESHOLD = 1.25


def prepare(table: pd.DataFrame, step_weight: str = "stratum") -> pd.DataFrame:
    """Add predicted / measured regret and the per-cell step weight.

    ``step_weight="stratum"``: every sampled configuration of a stratum counts the same (the sample is uniform over distinct
    configurations). ``"key_steps"``: weighted within a stratum by the number of replay steps that share the cell's workload key.
    """
    m = table.copy()
    for kind in ("pred", "meas"):
        columns = [f"{kind}_{v}_us" for v in VARIANTS]
        m[f"{kind}_best_us"] = m[columns].min(axis=1)
        m[f"{kind}_heur_us"] = m[f"{kind}_flashdecoding_paged_us"]
    m["pred_r"] = m.pred_heur_us / m.pred_best_us
    m["meas_r"] = m.meas_heur_us / m.meas_best_us
    n = m.groupby(["source", "stratum"]).id.transform("size")
    if step_weight == "stratum":
        m["w"] = m.stratum_steps / n
    elif step_weight == "key_steps":
        total = m.groupby(["source", "stratum"]).n_steps_with_key.transform("sum")
        m["w"] = m.stratum_steps * m.n_steps_with_key / total
    else:
        raise ValueError(step_weight)
    return m


def _stats(g: pd.DataFrame) -> dict:
    p, q, w = g.pred_r >= THRESHOLD, g.meas_r >= THRESHOLD, g.w
    both = p & q
    return {
        "cells": len(g), "pred_ge": int(p.sum()), "meas_ge": int(q.sum()), "both": int(both.sum()),
        "precision": both.sum() / p.sum() if p.sum() else np.nan,
        "recall": both.sum() / q.sum() if q.sum() else np.nan,
        "w_meas_frac": float((w * q).sum() / w.sum()),
        "w_pred_frac": float((w * p).sum() / w.sum()),
        "w_precision": float((w * both).sum() / (w * p).sum()) if (w * p).sum() else np.nan,
        "w_recall": float((w * both).sum() / (w * q).sum()) if (w * q).sum() else np.nan,
        "w_meas_attention_ratio": float((w * g.meas_heur_us).sum() / (w * g.meas_best_us).sum()),
        "w_pred_attention_ratio": float((w * g.pred_heur_us).sum() / (w * g.pred_best_us).sum()),
    }


def source_stats(m: pd.DataFrame, source: str) -> dict:
    return _stats(m[m.source == source])


def error_by_stratum(m: pd.DataFrame, source: str) -> pd.DataFrame:
    g = m[m.source == source].copy()
    g["abs_err"] = (g.pred_r - g.meas_r).abs()
    g["rel_err"] = (g.pred_r / g.meas_r - 1).abs()
    return g.groupby("stratum").agg(cells=("id", "size"), pred_median=("pred_r", "median"), meas_median=("meas_r", "median"),
                                    abs_err_median=("abs_err", "median"), abs_err_max=("abs_err", "max"),
                                    rel_err_median=("rel_err", "median"), rel_err_max=("rel_err", "max"),
                                    meas_ge=("meas_r", lambda x: int((x >= THRESHOLD).sum())))


def overall_error(m: pd.DataFrame) -> dict:
    err = (m.pred_r - m.meas_r).abs()
    rel = (m.pred_r / m.meas_r - 1).abs()
    return {"abs_err_median": float(err.median()), "abs_err_max": float(err.max()), "rel_err_median": float(rel.median()),
            "rel_err_max": float(rel.max()), "spearman": float(m[["pred_r", "meas_r"]].corr("spearman").iloc[0, 1]),
            "time_pred_over_meas_median": float((m.pred_heur_us / m.meas_heur_us).median()),
            "best_variant_agreement": float((m.pred_best_variant == m.meas_best_variant).mean())}


def bootstrap(m: pd.DataFrame, source: str, key: str, draws: int = 2000, seed: int = 0) -> tuple:
    """95% interval of a weighted statistic, resampling cells within each stratum."""
    g = m[m.source == source].reset_index(drop=True)
    groups = [idx.to_numpy() for _, idx in g.groupby("stratum").groups.items()]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        pick = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in groups])
        out.append(_stats(g.iloc[pick])[key])
    return tuple(float(x) for x in np.nanquantile(out, [0.025, 0.975]))


def run_to_run(first: pd.DataFrame, second: pd.DataFrame) -> dict:
    """Noise between two independent measurement passes of the same cells."""
    a, b = prepare(first), prepare(second)
    regret = (a.meas_r / b.meas_r - 1).abs()
    heur = (a.meas_heur_us / b.meas_heur_us - 1).abs()
    return {"regret_rel_median": float(regret.median()), "regret_rel_max": float(regret.max()),
            "heur_time_rel_median": float(heur.median()), "heur_time_rel_max": float(heur.max())}
