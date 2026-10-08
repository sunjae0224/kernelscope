"""Replay a public request trace through continuous batching and score every decode step's split loss.

    python -m scripts.traffic_replay --trace AzureLLMInferenceTrace_conv.csv --rate-scale 2 --out DIR

Writes ``steps.parquet`` (one row per decode step), ``losses.parquet`` (the surrogate's heuristic-vs-best
prediction for every ``--every``-th step) and ``summary.json``. GPU-free; see kernelscope/analysis/traffic.py
for the scheduler's simplifications.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from kernelscope.analysis.traffic import ReplayConfig, load_trace, replay, step_losses, summarize_losses


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", required=True)
    parser.add_argument("--model", help="BurstGPT: keep this model's rows only")
    parser.add_argument("--window-s", type=float, help="keep arrivals within this many seconds of the first (before scaling)")
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--prompt-scale", type=float, default=1.0,
                        help="what-if: multiply every prompt length (longer-context traffic than the trace recorded)")
    parser.add_argument("--step-ms", type=float, default=30.0)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--kv-tokens", type=int, default=72_817)
    parser.add_argument("--every", type=int, default=1, help="score every n-th decode step")
    parser.add_argument("--machine", default="machines/rtx4090.json")
    parser.add_argument("--params", default="models/rtx4090.json")
    parser.add_argument("--cache-state", default="cold", choices=("cold", "warm"))
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    out = Path(args.out)
    if out.exists():
        parser.error(f"output exists: {out}")
    if args.rate_scale <= 0 or args.step_ms <= 0 or args.max_batch < 1 or args.every < 1 or args.kv_tokens < 256:
        parser.error("--rate-scale and --step-ms must be positive, --max-batch and --every at least 1, --kv-tokens at least one page (256)")
    trace = load_trace(args.trace, model=args.model)
    if args.window_s is not None:
        trace = trace[trace.arrival_s <= args.window_s].reset_index(drop=True)
    if args.prompt_scale <= 0:
        parser.error("--prompt-scale must be positive")
    what_if = []
    if args.prompt_scale != 1.0:
        trace = trace.assign(prompt_tokens=(trace.prompt_tokens * args.prompt_scale).round().clip(lower=1).astype("int64"))
        what_if.append(f"prompt lengths scaled by {args.prompt_scale}")
    cfg = ReplayConfig(max_batch=args.max_batch, kv_tokens=args.kv_tokens, step_s=args.step_ms / 1e3,
                       rate_scale=args.rate_scale)
    started = time.perf_counter()
    steps = replay(trace, cfg)
    machine, params = MachineSpec.from_json(args.machine), ModelParams.from_json(args.params)
    losses = step_losses(steps, machine, params, args.cache_state, args.n_heads, args.n_kv_heads, every=args.every)
    summary = {"trace": {"source": str(args.trace), "sha256": _sha256(args.trace), "kind": trace.attrs["kind"],
                         "model": args.model, "window_s": args.window_s, "prompt_scale": args.prompt_scale,
                         "rows": len(trace), "dropped": trace.attrs["dropped"],
                         "requests": steps.attrs["requests"], "too_large": steps.attrs["too_large"],
                         "span_s": float(trace.arrival_s.max()) if len(trace) else 0.0,
                         "prompt_tokens_p50": float(trace.prompt_tokens.median()) if len(trace) else None,
                         "prompt_tokens_max": int(trace.prompt_tokens.max()) if len(trace) else None,
                         "output_tokens_p50": float(trace.output_tokens.median()) if len(trace) else None},
               "config": cfg.__dict__.copy(), "heads": {"n_heads": args.n_heads, "n_kv_heads": args.n_kv_heads},
               "machine": str(args.machine), "params": str(args.params), "cache_state": args.cache_state,
               "inputs_sha256": {str(args.machine): _sha256(args.machine), str(args.params): _sha256(args.params)},
               "steps": len(steps), "steps_with_long_request": int((steps.n_long >= 1).sum()) if len(steps) else 0,
               "batch_mean_all_steps": float(steps.B.mean()) if len(steps) else None,
               "losses": {**summarize_losses(losses), "every": args.every, "predictions": losses.attrs["predictions"],
                          "candidates": losses.attrs["candidates"]},
               "elapsed_s": time.perf_counter() - started, "what_if": what_if,
               "claim": "GPU-free replay with the fitted surrogate; prefill instantaneous, constant step time"}
    out.mkdir(parents=True)
    steps.to_parquet(out / "steps.parquet", index=False)
    losses.to_parquet(out / "losses.parquet", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    s = summary["losses"]
    print(f"{args.trace} x{args.rate_scale}: {len(steps)} steps, batch mean {s.get('batch_mean', 0):.1f}, "
          f"no-split {s.get('no_split_frac', 0) * 100:.1f}%, loss>=1.25 {s.get('loss_ge_1.25_frac', 0) * 100:.1f}% of steps, "
          f"attention time ratio {s.get('attention_time_ratio', 1):.3f} ({summary['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
