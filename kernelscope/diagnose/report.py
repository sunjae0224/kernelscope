"""Turn a `serve diagnose` run folder into a per-op-class ceiling report. GPU-free, no torch.

Reads the event run (op-class CUDA event rows) and the control run (no op timers) of every policy,
joins the measured GPU time per class with the compulsory bytes/FLOPs model and the machine
ceilings, and states per class whether it sits at the DRAM roof, the tensor-core roof, is launch
dominated, or is below both roofs for a reason this report cannot see. Attention also gets the
policy's chosen split variant against the nearest measured cell of the dispatch table.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kernelscope.diagnose.opmodel import LAYER_CLASSES, OP_CLASSES, OpCost, op_costs
from kernelscope.serve.report import tpot_us

VERDICTS = ("memory_bound", "compute_bound", "launch_bound", "parallelism_candidate", "below_ceiling_unknown")
LAUNCH_US = 5.0
OPS_CSV_COLUMNS = ["policy", "phase", "op_class", "gpu_us", "share", "weight_bytes", "act_bytes", "bytes", "flops", "ai",
                   "achieved_gbps", "achieved_tflops", "pct_dram", "pct_tc", "layer_mean_us", "verdict"]
ATTENTION_COLUMNS = ["step", "B", "num_splits", "chosen_variant", "source", "neighbor_key", "neighbor_distance",
                     "best_alternative", "best_us", "chosen_us", "regret"]


@dataclass
class Diagnosis:
    summary: dict
    ops: pd.DataFrame
    attention_steps: pd.DataFrame


def config_from(model_config: dict):
    """Attribute view of manifest['model_config']; avoids importing torch-backed serve.hf here."""
    return SimpleNamespace(**model_config)


def verdict(op_class, ai, achieved_gbps, achieved_tflops, layer_mean_us, machine, threshold) -> str:
    ridge = machine.ridge_flop_per_byte()
    if ai < ridge and achieved_gbps >= threshold * machine.dram_gbps:
        return "memory_bound"
    if ai >= ridge and achieved_tflops >= threshold * machine.tc_tflops:
        return "compute_bound"
    if layer_mean_us < LAUNCH_US:
        return "launch_bound"
    if op_class == "attention":
        return "parallelism_candidate"
    return "below_ceiling_unknown"


def amdahl_bound(share: float, chosen_over_best: float) -> float:
    """Step speedup if attention alone reached the best variant (same formula as dashboard.data.amdahl_speedup)."""
    return 1.0 / (1.0 - share + share / chosen_over_best)


def _mean_costs(cost_dicts):
    return {k: OpCost(float(np.mean([c[k].weight_bytes for c in cost_dicts])),
                      float(np.mean([c[k].act_bytes for c in cost_dicts])),
                      float(np.mean([c[k].flops for c in cost_dicts]))) for k in cost_dicts[0]}


def _op_table(gpu_us, costs, base_us, n_layers, machine, threshold, policy, phase) -> pd.DataFrame:
    rows = []
    for op_class in OP_CLASSES:
        us, c = float(gpu_us.get(op_class, 0.0)), costs[op_class]
        ai = c.flops / c.bytes if c.bytes else math.inf
        gbps = c.bytes / us * 1e-3 if us > 0 else math.nan
        tflops = c.flops / us * 1e-6 if us > 0 else math.nan
        layer_mean = us / n_layers if op_class in LAYER_CLASSES else us
        rows.append({"policy": policy, "phase": phase, "op_class": op_class, "gpu_us": us,
                     "share": us / base_us if base_us > 0 else math.nan,
                     "weight_bytes": c.weight_bytes, "act_bytes": c.act_bytes, "bytes": c.bytes, "flops": c.flops, "ai": ai,
                     "achieved_gbps": gbps, "achieved_tflops": tflops,
                     "pct_dram": 100 * gbps / machine.dram_gbps if us > 0 else math.nan,
                     "pct_tc": 100 * tflops / machine.tc_tflops if us > 0 else math.nan,
                     "layer_mean_us": layer_mean,
                     "verdict": verdict(op_class, ai, gbps if us > 0 else 0.0, tflops if us > 0 else 0.0,
                                        layer_mean, machine, threshold)})
    return pd.DataFrame(rows, columns=OPS_CSV_COLUMNS)


def decode_table(steps, ops, cfg, dtype, machine, threshold, policy) -> pd.DataFrame:
    decode = ops[ops.phase == "decode"]
    per_step = decode.groupby(["step", "op_class"]).gpu_us.sum().unstack(fill_value=0.0)
    gpu_us = per_step.reindex(columns=OP_CLASSES, fill_value=0.0).mean().to_dict() if len(per_step) else {}
    costs = _mean_costs([op_costs(cfg, dtype, "decode", int(s.B), tuple(json.loads(s.lens))) for s in steps.itertuples()])
    return _op_table(gpu_us, costs, float(steps.step_us.mean()), cfg.n_layers, machine, threshold, policy, "decode")


def prefill_table(prefill, ops, cfg, dtype, machine, threshold, policy, chunk=4096) -> pd.DataFrame:
    rows = ops[ops.phase == "prefill"]
    if prefill.empty or rows.empty:
        return pd.DataFrame(columns=OPS_CSV_COLUMNS)
    per_rid = rows.groupby(["rid", "op_class"]).gpu_us.sum().unstack(fill_value=0.0)
    gpu_us = per_rid.reindex(columns=OP_CLASSES, fill_value=0.0).mean().to_dict()
    per_request = []
    for n in prefill.prompt_len.astype(int):
        chunks = [op_costs(cfg, dtype, "prefill", min(chunk, n - start), (min(chunk, n - start) + start,))
                  for start in range(0, n, chunk)]
        per_request.append({k: OpCost(sum(c[k].weight_bytes for c in chunks), sum(c[k].act_bytes for c in chunks),
                                      sum(c[k].flops for c in chunks)) for k in chunks[0]})
    return _op_table(gpu_us, _mean_costs(per_request), float(prefill.prefill_us.mean()), cfg.n_layers, machine,
                     threshold, policy, "prefill")


def _expand_lens(spec) -> list:
    """'32768+512x25' -> [32768, 512, ...]; '512x8' -> [512] * 8."""
    out = []
    for part in str(spec).split("+"):
        n, _, times = part.partition("x")
        out.extend([int(n)] * (int(times) if times else 1))
    return out


def _features(lens) -> np.ndarray:
    return np.log2([len(lens), max(lens), float(np.mean(lens))])       # TablePolicy's features


def load_table(csv_path) -> pd.DataFrame:
    if csv_path is None or not Path(csv_path).exists():
        return pd.DataFrame(columns=["workload_key", "H_q", "H_kv", "lens", "best_kernel", "best_us", "lens_list"])
    table = pd.read_csv(csv_path)
    table["lens_list"] = table.lens.map(_expand_lens)
    return table


def load_measured(data_root) -> dict:
    """(workload_key, plugin) -> cold kernel time of every recorded paged cell under data_root/hw_4090."""
    measured = {}
    if data_root is None or not Path(data_root).exists():
        return measured
    for path in sorted(Path(data_root).glob("hw_4090/*_paged/summaries.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("cache_state") == "cold":
                measured[(r["workload_key"], r["plugin"])] = float(r["kernel_time_us"])
    return measured


def nearest_cell(table, lens, n_heads, n_kv_heads):
    same = table[(table.H_q == n_heads) & (table.H_kv == n_kv_heads)]
    if same.empty:
        return None, math.nan
    d = np.sqrt([np.square(_features(l) - _features(lens)).sum() for l in same.lens_list])
    i = int(np.argmin(d))
    return same.iloc[i], float(d[i])


def attention_steps(steps, cfg, table, measured, machine=None, params=None, cache_state="cold") -> pd.DataFrame:
    from kernelscope.model.hybrid import variant_for_splits
    rows = []
    for s in steps.itertuples():
        lens, splits = json.loads(s.lens), int(s.num_splits)
        chosen = variant_for_splits(splits)
        cell, dist = nearest_cell(table, lens, cfg.n_heads, cfg.n_kv_heads)
        row = {"step": int(s.step), "B": int(s.B), "num_splits": splits, "chosen_variant": chosen, "source": "unavailable",
               "neighbor_key": None, "neighbor_distance": dist, "best_alternative": None, "best_us": math.nan, "chosen_us": math.nan}
        if cell is not None:
            chosen_us = float(cell.best_us) if chosen == cell.best_kernel else measured.get((cell.workload_key, chosen), math.nan)
            row.update(source="nearest_measured", neighbor_key=cell.workload_key, best_alternative=cell.best_kernel,
                       best_us=float(cell.best_us), chosen_us=float(chosen_us))
        elif params is not None and machine is not None:
            from kernelscope.model.predict import PAGED_VARIANTS, rank_variants
            from kernelscope.workload import Workload
            w = Workload("decode", len(lens), 1, max(lens), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, "float16",
                         kv_lens=tuple(lens) if len(set(lens)) > 1 else None)
            ranked = rank_variants(w, PAGED_VARIANTS, machine, params, cache_state)
            row.update(source="model", best_alternative=ranked[0].plugin, best_us=ranked[0].time_us,
                       chosen_us={p.plugin: p.time_us for p in ranked}.get(chosen, math.nan))
        row["regret"] = row["chosen_us"] / row["best_us"] - 1 if row["best_us"] > 0 else math.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=ATTENTION_COLUMNS)


def _mode(values):
    s = pd.Series([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))])
    return None if s.empty else s.mode().iloc[0]


def attention_summary(att_steps: pd.DataFrame, share_attention: float) -> dict:
    ratio = (att_steps.chosen_us / att_steps.best_us).replace([np.inf, -np.inf], np.nan).dropna()
    over = float(ratio.mean()) if len(ratio) else math.nan
    return {"n_steps": int(len(att_steps)), "num_splits": _mode(att_steps.num_splits), "chosen_variant": _mode(att_steps.chosen_variant),
            "source": _mode(att_steps.source), "neighbor_key": _mode(att_steps.neighbor_key),
            "neighbor_distance_mean": float(att_steps.neighbor_distance.mean()) if len(att_steps) else math.nan,
            "best_alternative": _mode(att_steps.best_alternative), "share": share_attention, "chosen_over_best": over,
            "regret": over - 1 if math.isfinite(over) else math.nan,
            "amdahl_bound": amdahl_bound(share_attention, over) if math.isfinite(over) and over > 0 else math.nan}


def _frames(directory: Path, ops=False):
    if not (directory / "steps.parquet").exists():
        return None
    frames = {name: pd.read_parquet(directory / f"{name}.parquet") for name in ("steps", "tokens", "prefill")}
    if ops:
        frames["ops"] = pd.read_parquet(directory / "ops.parquet")
    return frames


def diagnose(run_dir, machine, table_csv, data_root, params=None, threshold=0.7) -> Diagnosis:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg, dtype = config_from(manifest["model_config"]), manifest["model_dtype"]
    table, measured = load_table(table_csv), load_measured(data_root)
    policies, ops_tables, att_tables = {}, [], []
    for policy_dir in sorted(p for p in run_dir.iterdir() if (p / "event_000" / "ops.parquet").exists()):
        name = policy_dir.name
        event, control = _frames(policy_dir / "event_000", ops=True), _frames(policy_dir / "control_000")
        steps = event["steps"]
        decode = decode_table(steps, event["ops"], cfg, dtype, machine, threshold, name)
        pre = prefill_table(event["prefill"], event["ops"], cfg, dtype, machine, threshold, name)
        step_us = float(steps.step_us.mean())
        share = float(decode.set_index("op_class").loc["attention", "share"])
        att = attention_steps(steps, cfg, table, measured, machine, params)
        control_step = float(control["steps"].step_us.mean()) if control is not None else math.nan
        tpot = tpot_us(event["tokens"]).dropna()
        policies[name] = {
            "n_steps": int(len(steps)),
            "tpot_waterfall": {"tpot_us_mean": float(tpot.mean()) if len(tpot) else math.nan,
                                "decode_wall_us_mean": float(steps.decode_wall_us.mean())},
            "step_waterfall": {"policy_us_mean": float(steps.policy_us.mean()), "step_us_mean": step_us,
                               "host_residual_us": float((steps.decode_wall_us - steps.step_us - steps.policy_us).mean())},
            "ops": decode.to_dict("records"), "prefill_ops": pre.to_dict("records"),
            "attention": attention_summary(att, share),
            "unattributed_pct": 100 * (step_us - float(decode.gpu_us.sum())) / step_us if step_us > 0 else math.nan,
            "timer_overhead_pct": 100 * (step_us / control_step - 1) if control_step > 0 else math.nan,
        }
        ops_tables += [decode, pre]
        att_tables.append(att.assign(policy=name))
    if not policies:
        raise FileNotFoundError(f"no <policy>/event_000/ops.parquet under {run_dir}")
    summary = {"schema_version": 1,
               "identity": {k: manifest.get(k) for k in ("evidence_kind", "performance_claim", "model", "scenario", "created_at")},
               "ceilings": {"machine": machine.name, "dram_gbps": machine.dram_gbps, "tc_tflops": machine.tc_tflops,
                            "ridge_flop_per_byte": machine.ridge_flop_per_byte(), "threshold": threshold, "launch_us": LAUNCH_US},
               "inputs": {"table": str(table_csv), "data_root": str(data_root), "run_dir": str(run_dir)},
               "policies": policies}
    return Diagnosis(summary, pd.concat(ops_tables, ignore_index=True), pd.concat(att_tables, ignore_index=True))


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def write(run_dir, diagnosis: Diagnosis) -> None:
    run_dir = Path(run_dir)
    (run_dir / "diagnosis.json").write_text(json.dumps(_jsonable(diagnosis.summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    diagnosis.ops.to_csv(run_dir / "ops.csv", index=False)
    diagnosis.attention_steps.to_csv(run_dir / "attention_steps.csv", index=False)
