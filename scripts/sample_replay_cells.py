"""Stratified sample of a trace replay's distinct decode batches, to check the surrogate's scoring on the GPU.

    python -m scripts.sample_replay_cells sample --source azure_conv_x1=../kernelscope/results/traffic/azure_conv_x1:25 \\
        --source azure_code_x16=../kernelscope/results/traffic/azure_code_x16:10 --out DIR
    kernelscope bench --grid DIR/cells_grid.yaml --cache-state cold --results DIR/bench --plugins <the candidates>
    python -m scripts.sample_replay_cells collect --cells DIR/cells.json --results DIR/bench --out DIR/measured.csv

``sample`` takes each replay's decode steps (``steps.parquet`` + ``losses.parquet`` from scripts/traffic_replay.py,
scored with every=1), collapses them to distinct page-quantized length configurations (the key the surrogate's
prediction cache uses, so one configuration has one predicted loss), buckets those by predicted loss (< 1.1,
1.1-1.25, 1.25-1.5, >= 1.5) and draws ``n`` configurations per bucket with a fixed seed (all of them when fewer).
Each cell keeps the exact length vector of the configuration's first step, the surrogate's time for every
candidate and the population counts, so a later analysis can weight the strata back to the trace.
``collect`` joins the measured kernel times (cold cache, median over the bench iterations) to the cells.
GPU-free except for the bench run in between.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from kernelscope.analysis.traffic import CANDIDATES, HEURISTIC, _workload
from kernelscope.model.geometry import parse_variant, resolve_splits
from kernelscope.model.predict import predict
from kernelscope.workload import format_lens

PAGE = 256
BINS = ((0.0, 1.1, "<1.1"), (1.1, 1.25, "1.1-1.25"), (1.25, 1.5, "1.25-1.5"), (1.5, math.inf, ">=1.5"))
HEADS = (32, 8)                      # S1 geometry of the replay and of the grids (grids/ragged_s1.yaml)


def stratum(loss: float) -> str:
    for lo, hi, name in BINS:
        if lo <= loss < hi:
            return name
    raise ValueError(f"loss {loss!r} is not a finite number >= 0")


def distinct_configs(steps: pd.DataFrame, losses: pd.DataFrame) -> pd.DataFrame:
    """One row per page-quantized length configuration: first step, its exact lens, the stored predicted loss,
    and how many decode steps of the replay share the configuration."""
    if len(losses) != len(steps) or not (losses.step.to_numpy() == steps.step.to_numpy()).all():
        raise ValueError("losses must score every step (replay with --every 1)")
    first, count, lens_of = {}, {}, {}
    for i, lens_json in enumerate(steps.lens):
        lens = json.loads(lens_json)
        key = tuple(math.ceil(n / PAGE) for n in lens)
        count[key] = count.get(key, 0) + 1
        if key not in first:
            first[key], lens_of[key] = i, lens
    idx = np.array(list(first.values()))
    out = pd.DataFrame({"first_step": steps.step.to_numpy()[idx], "n_steps_with_key": [count[k] for k in first],
                        "lens": [lens_of[k] for k in first], "pred_regret": losses.loss.to_numpy()[idx],
                        "pred_heuristic_splits": losses.heuristic_splits.to_numpy()[idx]})
    out["stratum"] = out.pred_regret.map(stratum)
    return out


def sample(configs: pd.DataFrame, per_stratum: int, seed: int) -> pd.DataFrame:
    """``per_stratum`` configurations from each loss bucket, drawn without replacement (all when fewer)."""
    parts = []
    for _, _, name in BINS:
        pool = configs[configs.stratum == name]
        take = min(per_stratum, len(pool))
        parts.append(pool.sample(n=take, random_state=seed) if take else pool)
    return pd.concat(parts).sort_values(["pred_regret", "first_step"], ignore_index=True)


def _cell(source, row, machine, params, cache_state, population):
    lens = [int(n) for n in row.lens]
    w = _workload(lens, *HEADS)
    pred = {c: float(predict(c, w, machine, params, cache_state).time_us) for c in CANDIDATES}
    best = min(pred, key=pred.get)
    regret = pred[HEURISTIC] / pred[best]
    if not math.isclose(regret, row.pred_regret, rel_tol=1e-9):
        raise ValueError(f"{source} step {row.first_step}: stored loss {row.pred_regret} != recomputed {regret}")
    return {"id": f"{source}:{int(row.first_step)}", "source": source, "stratum": row.stratum, "B": len(lens),
            "L_kv": max(lens), "lens": lens, "lens_spec": format_lens(lens), "first_step": int(row.first_step),
            "n_steps_with_key": int(row.n_steps_with_key),
            "pred_heuristic_splits": int(row.pred_heuristic_splits), "pred_us": pred, "pred_heuristic_us": pred[HEURISTIC],
            "pred_best_variant": best, "pred_best_us": pred[best], "pred_regret": regret,
            "stratum_distinct": population[row.stratum][0], "stratum_steps": population[row.stratum][1]}


def build_cells(sources, seed, machine, params, cache_state="cold") -> dict:
    """``sources``: (name, steps, losses, per_stratum). Cells are unique by (source, configuration)."""
    cells, meta = [], {}
    for name, steps, losses, per in sources:
        configs = distinct_configs(steps, losses)
        population = {s: (int(g.shape[0]), int(g.n_steps_with_key.sum())) for s, g in configs.groupby("stratum")}
        population = {n: population.get(n, (0, 0)) for _, _, n in BINS}
        drawn = sample(configs, per, seed)
        meta[name] = {"steps": int(len(steps)), "distinct_configs": int(len(configs)), "per_stratum": per,
                      "strata": {n: {"distinct": d, "steps": s, "sampled": int((drawn.stratum == n).sum())}
                                 for n, (d, s) in population.items()}}
        cells += [_cell(name, r, machine, params, cache_state, population) for r in drawn.itertuples(index=False)]
    return {"seed": seed, "candidates": list(CANDIDATES), "heads": list(HEADS), "cache_state": cache_state,
            "page": PAGE, "meta": meta, "cells": cells}


def write_grid(cells: list[dict], path: Path) -> int:
    """A bench grid listing every distinct length spec once (same head geometry and dtype as the other grids)."""
    specs = list(dict.fromkeys(c["lens_spec"] for c in cells))
    lines = ["# generated by scripts/sample_replay_cells.py: sampled trace-replay decode batches",
             "phase: decode", f"H_q: {HEADS[0]}", f"H_kv: {HEADS[1]}", "d: 128", "lens:"]
    lines += [f"  - \"{s}\"" for s in specs]
    path.write_text("\n".join(lines) + "\n")
    return len(specs)


def collect(cells_doc: dict, bench: pd.DataFrame) -> pd.DataFrame:
    """One row per cell: surrogate and measured time of every candidate, and the regret each implies."""
    from kernelscope.workload import Workload
    t = bench[(bench.backend == "profile") & (bench.metric == "kernel_time_us") & (bench.cache_state == "cold")]
    t = t.groupby(["workload_key", "kernel"])["value"].median().unstack("kernel")
    rows = []
    for c in cells_doc["cells"]:
        key = Workload(phase="decode", B=c["B"], L_q=1, L_kv=c["L_kv"], H_q=HEADS[0], H_kv=HEADS[1], d=128,
                       dtype="float16", kv_lens=tuple(c["lens"])).key()
        meas = t.loc[key].dropna().to_dict() if key in t.index else {}
        row = {"id": c["id"], "source": c["source"], "stratum": c["stratum"], "B": c["B"], "L_kv": c["L_kv"],
               "lens_spec": c["lens_spec"], "workload_key": key, "n_steps_with_key": c["n_steps_with_key"],
               "stratum_distinct": c["stratum_distinct"], "stratum_steps": c["stratum_steps"],
               "pred_heuristic_splits": c["pred_heuristic_splits"], "pred_best_variant": c["pred_best_variant"],
               "pred_regret": c["pred_regret"]}
        for v in cells_doc["candidates"]:
            row[f"pred_{v}_us"] = c["pred_us"][v]
            row[f"meas_{v}_us"] = meas.get(v, math.nan)
        mv = {v: meas[v] for v in cells_doc["candidates"] if v in meas}
        complete = len(mv) == len(cells_doc["candidates"])
        row["complete"] = complete
        row["meas_best_variant"] = min(mv, key=mv.get) if complete else None
        row["meas_regret"] = mv[HEURISTIC] / min(mv.values()) if complete else math.nan
        row["pred_vs_meas_heuristic"] = c["pred_heuristic_us"] / mv[HEURISTIC] if HEURISTIC in mv else math.nan
        row["meas_regret_at_pred_best"] = mv[HEURISTIC] / mv[c["pred_best_variant"]] if complete else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_source(text):
    try:
        name, rest = text.split("=", 1)
        run, per = rest.rsplit(":", 1)
        return name, Path(run), int(per)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: expected NAME=RUN_DIR:PER_STRATUM")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("sample")
    ps.add_argument("--source", action="append", required=True, type=_parse_source)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--machine", default="machines/rtx4090.json")
    ps.add_argument("--params", default="models/rtx4090.json")
    ps.add_argument("--cache-state", default="cold", choices=("cold", "warm"))
    ps.add_argument("--out", required=True)
    pc = sub.add_parser("collect")
    pc.add_argument("--cells", required=True)
    pc.add_argument("--results", required=True, help="bench results directory (parquet or summaries.jsonl)")
    pc.add_argument("--out", required=True, help="measured.csv")
    args = parser.parse_args(argv)
    if args.cmd == "sample":
        from kernelscope.model.machine import MachineSpec
        from kernelscope.model.params import ModelParams
        out = Path(args.out)
        machine, params = MachineSpec.from_json(args.machine), ModelParams.from_json(args.params)
        sources = [(name, pd.read_parquet(run / "steps.parquet"), pd.read_parquet(run / "losses.parquet"), per)
                   for name, run, per in args.source]
        doc = build_cells(sources, args.seed, machine, params, args.cache_state)
        doc["runs"] = {name: str(run) for name, run, _ in args.source}
        out.mkdir(parents=True, exist_ok=True)
        (out / "cells.json").write_text(json.dumps(doc, indent=1))
        n = write_grid(doc["cells"], out / "cells_grid.yaml")
        print(f"{len(doc['cells'])} cells, {n} distinct length vectors -> {out}")
    else:
        from kernelscope.verify import load_dirs
        doc = json.loads(Path(args.cells).read_text())
        table = collect(doc, load_dirs([args.results]))
        table.to_csv(args.out, index=False)
        print(f"{int(table.complete.sum())}/{len(table)} cells fully measured -> {args.out}")


if __name__ == "__main__":
    main()
