"""Turn a window of a public request trace into an engine scenario and predict its attention-time ratio on CPU.

    python -m scripts.trace_scenario --trace AzureLLMInferenceTrace_conv.csv --out-dir DIR

Slides a window of ``--n-requests`` consecutive requests over the trace, replays each window with the output
length capped at ``--output-cap`` (the engine run has to finish in minutes), keeps windows whose mean replay
batch is at least ``--min-batch``, scores them with the fitted surrogate (heuristic vs best split variant per
step) and picks the one with the most steps whose predicted regret is >= ``--regret`` at batch >= ``--min-batch``.
Writes the scenario YAML (``arrival_step`` = ceil(arrival / step) of the 30 ms replay clock), ``selection.json``
(every window's replay statistics, the choice and the criteria) and ``predicted.json`` (the surrogate's
prediction for the scenario itself, replayed with the engine's logical-step arrivals). GPU-free.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from kernelscope.analysis.traffic import ReplayConfig, load_trace, replay, step_losses, summarize_losses

STEP_S = 0.03
MAX_BATCH = 64


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cap_window(window, prompt_cap=None, output_cap=None):
    """Window with prompt/output lengths clipped; returns (frame, number of capped prompts, number of capped outputs)."""
    window = window.copy()
    n_prompt = n_output = 0
    if prompt_cap is not None:
        n_prompt = int((window.prompt_tokens > prompt_cap).sum())
        window["prompt_tokens"] = window.prompt_tokens.clip(upper=prompt_cap)
    if output_cap is not None:
        n_output = int((window.output_tokens > output_cap).sum())
        window["output_tokens"] = window.output_tokens.clip(upper=output_cap)
    return window, n_prompt, n_output


def arrival_steps(arrival_s, step_s=STEP_S):
    """Logical decode step at which a request is admitted: the first step whose replay clock reaches its arrival.

    Arrivals are relative to the window's first request (step 0). A small epsilon keeps exact multiples of
    ``step_s`` from rounding up because of binary floating point.
    """
    return [max(0, math.ceil(float(a) / step_s - 1e-9)) for a in arrival_s]


def scenario_rows(window, step_s=STEP_S):
    """[(arrival_step, prompt_len, max_new_tokens)] in arrival order; the first request arrives at step 0."""
    window = window.sort_values("arrival_s", kind="stable")
    base = float(window.arrival_s.iloc[0])
    steps = arrival_steps(window.arrival_s - base, step_s)
    return [(s, int(p), int(o)) for s, p, o in zip(steps, window.prompt_tokens, window.output_tokens)]


def rows_trace(rows, step_s=STEP_S):
    """The scenario as a trace the replay understands (arrivals on the engine's logical-step grid)."""
    return pd.DataFrame({"arrival_s": [r[0] * step_s for r in rows], "prompt_tokens": [r[1] for r in rows],
                         "output_tokens": [r[2] for r in rows]})


def scenario_yaml(rows, header):
    """Scenario text; consecutive identical requests collapse into one ``count`` entry (the loader expands it)."""
    lines = [f"# {line}" for line in header.splitlines()] + ["requests:"]
    groups = []
    for row in rows:
        if groups and groups[-1][0] == row:
            groups[-1][1] += 1
        else:
            groups.append([row, 1])
    for (step, prompt, output), count in groups:
        lines.append(f"  - {{count: {count}, prompt_len: {prompt}, max_new_tokens: {output}, arrival_step: {step}}}")
    return "\n".join(lines) + "\n"


def window_stats(steps, regret=1.25, min_batch=26):
    """Cheap replay statistics of one window (no surrogate)."""
    if len(steps) == 0:
        return {"steps": 0, "batch_mean": 0.0, "steps_long": 0, "steps_batch_ge": 0}
    return {"steps": int(len(steps)), "batch_mean": float(steps.B.mean()), "batch_p90": float(steps.B.quantile(.9)),
            "steps_long": int((steps.n_long >= 1).sum()), "steps_batch_ge": int((steps.B >= min_batch).sum())}


def regret_count(losses, regret=1.25, min_batch=26):
    """Selection criterion: steps with predicted regret >= ``regret`` while the batch is at least ``min_batch``."""
    return int(((losses.loss >= regret) & (losses.B >= min_batch)).sum())


def choose(candidates):
    """The scored candidate with the most criterion steps (earliest window on ties)."""
    scored = [c for c in candidates if c.get("criterion_steps") is not None]
    if not scored:
        return None
    return max(scored, key=lambda c: (c["criterion_steps"], -c["start"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", required=True)
    parser.add_argument("--n-requests", type=int, default=300)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--output-cap", type=int, default=256)
    parser.add_argument("--prompt-cap", type=int, help="clip prompts (the 10 GiB KV pool holds one request of any length in the trace)")
    parser.add_argument("--min-batch", type=float, default=26.0)
    parser.add_argument("--regret", type=float, default=1.25)
    parser.add_argument("--kv-tokens", type=int, default=72_817)
    parser.add_argument("--machine", default="machines/rtx4090.json")
    parser.add_argument("--params", default="models/rtx4090.json")
    parser.add_argument("--out-dir", required=True, help="selection.json and predicted.json go here")
    parser.add_argument("--scenario-out", help="default scenarios/trace_azure_conv_i<start>_n<N>.yaml")
    args = parser.parse_args(argv)
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    out = Path(args.out_dir)
    if out.exists():
        parser.error(f"output exists: {out}")
    trace = load_trace(args.trace)
    cfg = ReplayConfig(max_batch=MAX_BATCH, kv_tokens=args.kv_tokens, step_s=STEP_S)
    machine, params = MachineSpec.from_json(args.machine), ModelParams.from_json(args.params)
    candidates = []
    for start in range(0, len(trace) - args.n_requests + 1, args.stride):
        window, _, _ = cap_window(trace.iloc[start:start + args.n_requests], args.prompt_cap, args.output_cap)
        rows = scenario_rows(window)
        steps = replay(rows_trace(rows), cfg)
        stats = window_stats(steps, args.regret, args.min_batch)
        candidates.append({"start": start, "t0_s": float(trace.arrival_s.iloc[start]), **stats})
        if stats["batch_mean"] >= args.min_batch:
            candidates[-1]["criterion_steps"] = regret_count(step_losses(steps, machine, params), args.regret, args.min_batch)
    chosen = choose(candidates)
    relaxed = chosen is None
    if relaxed:                         # no window reaches the batch target: take the highest-batch one and say so
        best = max(candidates, key=lambda c: c["batch_mean"])
        window, _, _ = cap_window(trace.iloc[best["start"]:best["start"] + args.n_requests], args.prompt_cap, args.output_cap)
        best["criterion_steps"] = regret_count(step_losses(replay(rows_trace(scenario_rows(window)), cfg), machine, params),
                                               args.regret, args.min_batch)
        chosen = best
    start = chosen["start"]
    window, n_prompt, n_output = cap_window(trace.iloc[start:start + args.n_requests], args.prompt_cap, args.output_cap)
    rows = scenario_rows(window)
    scenario = Path(args.scenario_out or f"scenarios/trace_azure_conv_i{start}_n{args.n_requests}.yaml")
    if scenario.exists():
        parser.error(f"scenario exists: {scenario}")
    t0 = float(trace.arrival_s.iloc[start])
    header = (f"Azure LLM Inference 2023 conv trace, requests {start}..{start + args.n_requests - 1} (trace time {t0:.1f}-"
              f"{float(trace.arrival_s.iloc[start + args.n_requests - 1]):.1f} s after the first row).\n"
              f"arrival_step = ceil(arrival / {STEP_S} s) from the window's first request; prompt_len as in the trace"
              f" ({n_prompt} capped); max_new_tokens = min(trace output, {args.output_cap}) ({n_output} capped).\n"
              f"Generated by scripts/trace_scenario.py; selection in the overnight 2026-10-24 notes (section 7).")
    scenario.write_text(scenario_yaml(rows, header))
    steps = replay(rows_trace(rows), cfg)
    losses = step_losses(steps, machine, params)
    summary = summarize_losses(losses)
    summary["criterion_steps"] = regret_count(losses, args.regret, args.min_batch)
    out.mkdir(parents=True)
    predicted = {"scenario": str(scenario), "claim": "surrogate prediction (GPU-free replay, 30 ms constant step, instantaneous prefill); not a measurement",
                 "cache_state": "cold", "requests": len(rows), "steps": int(len(steps)), "kv_tokens": args.kv_tokens, "max_batch": MAX_BATCH,
                 "prompt_capped": n_prompt, "output_capped": n_output, "window": {"start": start, "n_requests": args.n_requests, "t0_s": t0},
                 "replay": window_stats(steps, args.regret, args.min_batch), "losses": summary,
                 "attention_time_ratio_heuristic_over_best": summary["attention_time_ratio"],
                 "predicted_attention_us_per_step": {"heuristic": float(losses.heuristic_us.mean()), "best": float(losses.best_us.mean())},
                 "trace_sha256": _sha256(args.trace)}
    (out / "predicted.json").write_text(json.dumps(predicted, indent=2))
    (out / "selection.json").write_text(json.dumps({
        "criteria": {"n_requests": args.n_requests, "stride": args.stride, "output_cap": args.output_cap, "prompt_cap": args.prompt_cap,
                     "min_batch": args.min_batch, "regret": args.regret, "rule": "most steps with predicted regret >= regret at batch >= min_batch, among windows whose capped replay mean batch >= min_batch",
                     "relaxed_no_window_reached_min_batch": relaxed},
        "chosen": chosen, "scenario": str(scenario), "candidates": candidates}, indent=2))
    print(f"window {start}: {len(rows)} requests, {len(steps)} steps, mean batch {steps.B.mean():.1f}, criterion steps "
          f"{summary['criterion_steps']}, attention ratio {summary['attention_time_ratio']:.3f}; wrote {scenario}")


if __name__ == "__main__":
    main()
