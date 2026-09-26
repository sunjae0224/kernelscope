"""Model/table hybrid: take the measured table's variant when the model predicts it close to its own best.

The surrogate's worst selections put the non-split variant a few percent ahead of a split variant that
measured 20-60 % faster (docs/plan/2026-09-22-model-validation.md). A nearby measured cell knows which
of two close predictions actually won, so within ``delta`` of the predicted best the table decides.
``evaluate`` replays this on the recorded cells with a leave-one-out table, GPU-free.
"""
from collections import defaultdict

import numpy as np

from kernelscope.model.fit import row_time_us
from kernelscope.model.validate import HEURISTIC, in_set


def variant_for_splits(n: int) -> str:
    """Serving convention: 0 = library heuristic, 1 = non-split, N = fixed split, all on the paged cache."""
    return {0: "flashdecoding_paged", 1: "fa2_paged"}.get(n, f"fd_s{n}_paged")


def features(w) -> np.ndarray:
    lens = w.lens()
    return np.log2([len(lens), max(lens), float(np.mean(lens))])       # same features as TablePolicy


def table_neighbor(query, table):
    """(best variant, distance) of the nearest measured workload with the query's head shape."""
    same = [(w, v) for w, v in table if (w.H_q, w.H_kv) == (query.H_q, query.H_kv)]
    if not same:
        return None, None
    q = features(query)
    d = np.sqrt([np.square(features(w) - q).sum() for w, _ in same])
    i = int(np.argmin(d))
    return same[i][1], float(d[i])


def hybrid_pick(predicted: dict, table_variant, delta: float) -> str:
    best = min(predicted, key=predicted.get)
    if table_variant in predicted and predicted[table_variant] <= (1 + delta) * predicted[best]:
        return table_variant
    return best


def _groups(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r.workload.key(), r.cache_state, "paged" if r.plugin.endswith("_paged") else "dense")].append(r)
    return {k: v for k, v in g.items() if len(v) >= 3}


def evaluate(rows, params, set_name: str, cache_state: str, deltas=(0.05, 0.10, 0.20)) -> dict:
    """Regret of model, leave-one-out table and hybrid picks over one validation set.

    The table for a cell is every other measured cell in the same cache state and family (uniform and
    ragged, any set), so a cell never looks itself up. Regret = picked measured time / best - 1.
    """
    groups = _groups(rows)
    best_variant = {k: min(v, key=lambda r: r.measured_us).plugin for k, v in groups.items()}
    regret = defaultdict(list)
    for key, cell in groups.items():
        wkey, state, family = key
        if state != cache_state or not in_set(cell[0], set_name):
            continue
        measured = {r.plugin: r.measured_us for r in cell}
        predicted = {r.plugin: row_time_us(r, params.states[state]) for r in cell}
        best = min(measured.values())
        table = [(groups[k][0].workload, best_variant[k]) for k in groups
                 if k[1] == state and k[2] == family and k[0] != wkey]
        table_variant, _ = table_neighbor(cell[0].workload, table)
        picks = {"model": min(predicted, key=predicted.get), "table": table_variant if table_variant in measured else None,
                 "heuristic": HEURISTIC[family] if HEURISTIC[family] in measured else None}
        for d in deltas:
            picks[f"hybrid_{d:g}"] = hybrid_pick(predicted, table_variant, d)
        for name, pick in picks.items():
            if pick is not None:
                regret[name].append(measured[pick] / best - 1)
    stats = lambda xs: {"n": len(xs), "median": float(np.median(xs)) if xs else float("nan"),  # noqa: E731
                        "max": float(np.max(xs)) if xs else float("nan"),
                        "over_15pct": int(sum(x > 0.15 for x in xs))}
    return {"set": set_name, "cache_state": cache_state, "policies": {k: stats(v) for k, v in regret.items()}}


def markdown(results: list) -> str:
    names = ["heuristic", "model", "table"] + sorted({n for r in results for n in r["policies"] if n.startswith("hybrid")},
                                                    key=lambda s: float(s.split("_")[1]))
    out = ["| set | cache | groups | " + " | ".join(f"{n} median / max (>15 %)" for n in names) + " |",
           "|---|---|---|" + "---|" * len(names)]
    for r in results:
        p = r["policies"]
        cells = [f"{p[n]['median']:.1%} / {p[n]['max']:.1%} ({p[n]['over_15pct']})" if n in p and p[n]["n"] else "-"
                 for n in names]
        out.append(f"| {r['set']} | {r['cache_state']} | {p['model']['n']} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"
