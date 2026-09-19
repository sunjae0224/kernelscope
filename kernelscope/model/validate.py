"""Validation of the surrogate model against held-out measurements (spec §3.3)."""
from collections import defaultdict

import numpy as np

from kernelscope.model.fit import row_time_us

TRAIN_B = (1, 4, 16, 64)
TRAIN_L = (512, 2048, 8192, 32768)
THRESHOLDS = {"V1": (0.15, 0.30), "V5": (0.225, 0.45), "V6": (0.225, 0.45)}
POLICY = (0.05, 0.15)
HEURISTIC = {"dense": "flashdecoding", "paged": "flashdecoding_paged"}


def in_set(row, name: str) -> bool:
    w = row.workload
    if name == "V1":
        return not w.is_ragged and w.H_kv == 8 and not (w.B in TRAIN_B and w.L_kv in TRAIN_L)
    if name == "V5":
        return not w.is_ragged and w.H_kv == 4
    if name == "V6":
        return w.is_ragged and sum(map(ord, w.key())) % 2 == 1
    raise ValueError(name)


def error_stats(rows, params) -> dict:
    if not rows:
        return {"n": 0, "median_ape": float("nan"), "p90_ape": float("nan"), "worst": []}
    preds = [row_time_us(r, params.states[r.cache_state]) for r in rows]
    ape = np.array([abs(p / r.measured_us - 1) for p, r in zip(preds, rows)])
    worst = [(rows[i].workload.key(), rows[i].plugin, rows[i].measured_us, preds[i]) for i in np.argsort(-ape)[:5]]
    return {"n": len(rows), "median_ape": float(np.median(ape)), "p90_ape": float(np.percentile(ape, 90)), "worst": worst}


def policy_stats(rows, params) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r.workload.key(), r.cache_state, "paged" if r.plugin.endswith("_paged") else "dense")].append(r)
    model, heur = [], []
    for (_, state, family), rs in groups.items():
        if len(rs) < 3:
            continue
        best = min(r.measured_us for r in rs)
        pick = min(rs, key=lambda r: row_time_us(r, params.states[state]))
        model.append(pick.measured_us / best - 1)
        h = [r for r in rs if r.plugin == HEURISTIC[family]]
        if h:
            heur.append(h[0].measured_us / best - 1)
    f = lambda xs, fn: float(fn(xs)) if xs else float("nan")  # noqa: E731
    return {"n": len(model), "model_median_regret": f(model, np.median), "model_max_regret": f(model, np.max),
            "heuristic_median_regret": f(heur, np.median), "heuristic_max_regret": f(heur, np.max)}


def report(rows, params, set_name: str, cache_state: str) -> dict:
    rs = [r for r in rows if in_set(r, set_name) and r.cache_state == cache_state]
    return {"set": set_name, "cache_state": cache_state, "threshold": THRESHOLDS[set_name],
            **error_stats(rs, params), "policy": policy_stats(rs, params)}


def markdown(results: list) -> str:
    out = ["| set | cache | n | median APE | p90 APE | threshold (median / p90) | verdict |", "|---|---|---|---|---|---|---|"]
    for r in results:
        med, p90 = r["threshold"]
        ok = r["n"] > 0 and r["median_ape"] <= med and r["p90_ape"] <= p90
        out.append(f"| {r['set']} | {r['cache_state']} | {r['n']} | {r['median_ape']:.1%} | {r['p90_ape']:.1%} | "
                   f"{med:.0%} / {p90:.0%} | {'PASS' if ok else 'FAIL'} |")
    out += ["", "| set | cache | groups | model median / max regret | heuristic median / max regret | verdict (≤ 5 % / 15 %) |",
            "|---|---|---|---|---|---|"]
    for r in results:
        p = r.get("policy")
        if not p or not p["n"]:
            continue
        ok = p["model_median_regret"] <= POLICY[0] and p["model_max_regret"] <= POLICY[1]
        out.append(f"| {r['set']} | {r['cache_state']} | {p['n']} | {p['model_median_regret']:.1%} / {p['model_max_regret']:.1%} | "
                   f"{p['heuristic_median_regret']:.1%} / {p['heuristic_max_regret']:.1%} | {'PASS' if ok else 'FAIL'} |")
    for r in results:
        if r.get("worst"):
            out += ["", f"Worst cells, {r['set']} {r['cache_state']}:", ""]
            out += [f"- `{k}` {p}: measured {m:.1f} µs, predicted {q:.1f} µs" for k, p, m, q in r["worst"]]
    return "\n".join(out) + "\n"
