"""Does the ragged-batch decode loss appear in vLLM's FlashAttention backend? Measurement only (PLAN §2 item 5).

Run with the separate vLLM environment (not the project's .venv), e.g.
    ~/.venvs/kernelscope-vllm/bin/python scripts/vllm_reproduce.py --out ../kernelscope/results/vllm_4090/<date>
For each (attention backend, execution mode, scenario) the script generates a fixed number of tokens for a batch
of synthetic token-id prompts in one `LLM.generate` call and derives the decode time per step from two generation
lengths (T(n_long) - T(n_short)) / (n_long - n_short), so prefill is cancelled out. Prefix caching is disabled and
one untimed warm-up generation precedes the timed runs (the first attempt, repro_20261006, had neither and its
first repeat came out negative). Backends: FLASH_ATTN (num_splits left to the library heuristic in eager mode,
a fixed maximum under CUDA graphs) and FLASHINFER (plan-based scheduling); modes: eager and cudagraph. Nothing
here is integrated into vLLM; it is an independent reproduction of the symptom, not a policy.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

SCENARIOS = {
    "ragged": [32768] + [512] * 31,
    "uniform": [512] * 32,
    "arrivals_like": [16384] + [512] * 31,
}


def prompts_for(lens, vocab, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    return [{"prompt_token_ids": rng.integers(1000, vocab - 1000, size=n).tolist()} for n in lens]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--backends", default="FLASH_ATTN,FLASHINFER")
    parser.add_argument("--scenarios", default="ragged,uniform")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--short", type=int, default=4, help="generated tokens of the short run")
    parser.add_argument("--long", type=int, default=64, help="generated tokens of the long run")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--modes", default="eager,cudagraph")
    parser.add_argument("--block-size", type=int, default=None, help="vLLM KV block size (default: vLLM's choice, 16 on this backend)")
    parser.add_argument("--mode", choices=("eager", "cudagraph"), help="(child) a single execution mode")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    backend = os.environ.get("VLLM_ATTENTION_BACKEND")
    if backend is None or args.mode is None:
        # one (backend, mode) per process: vLLM reads the env var at import time, so re-exec ourselves
        rows = []
        for name in args.backends.split(","):
            for mode in args.modes.split(","):
                env = {**os.environ, "VLLM_ATTENTION_BACKEND": name, "HF_HUB_OFFLINE": "1"}
                part = out / f"{name.lower()}_{mode}"
                cmd = [sys.executable, __file__, "--out", str(part), "--model", args.model, "--scenarios", args.scenarios,
                       "--repeats", str(args.repeats), "--short", str(args.short), "--long", str(args.long),
                       "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--mode", mode]
                if args.block_size is not None:
                    cmd += ["--block-size", str(args.block_size)]
                import subprocess
                rc = subprocess.run(cmd, env=env).returncode
                print(f"{name} {mode}: rc={rc}")
                if (part / "summary.json").exists():
                    rows.extend(json.loads((part / "summary.json").read_text())["rows"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps({"rows": rows, "model": args.model}, indent=2))
        for r in rows:
            print(f"{r['backend']:10s} {r['mode']:9s} {r['scenario']:14s} decode_ms_per_step={r['decode_ms_per_step_median']:.3f} "
                  f"(min {r['decode_ms_per_step_min']:.3f}, max {max(r['samples_ms']):.3f}) tokens_identical_across_repeats={r['tokens_identical']}")
        return
    from vllm import LLM, SamplingParams
    import torch
    eager = args.mode == "eager"
    extra = {"block_size": args.block_size} if args.block_size is not None else {}
    llm = LLM(model=args.model, enforce_eager=eager, max_model_len=33000, gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=64, max_num_batched_tokens=33000, dtype="bfloat16", seed=0, enable_prefix_caching=False, **extra)
    vocab = llm.get_tokenizer().vocab_size
    rows = []
    for scenario in args.scenarios.split(","):
        lens = SCENARIOS[scenario]
        prompts = prompts_for(lens, vocab)
        llm.generate(prompts, SamplingParams(max_tokens=args.short, min_tokens=args.short, ignore_eos=True, temperature=0.0), use_tqdm=False)
        steps = []
        tokens_by_repeat = []
        for _ in range(args.repeats):
            times = {}
            for n in (args.short, args.long):
                params = SamplingParams(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0.0)
                torch.cuda.synchronize()
                started = time.perf_counter()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                torch.cuda.synchronize()
                times[n] = time.perf_counter() - started
                if n == args.long:
                    tokens_by_repeat.append([tuple(o.outputs[0].token_ids) for o in outputs])
            steps.append(1000 * (times[args.long] - times[args.short]) / (args.long - args.short))
        rows.append({"backend": os.environ["VLLM_ATTENTION_BACKEND"], "mode": args.mode, "scenario": scenario, "lens": lens[:2] + ["..."],
                     "batch": len(lens), "decode_ms_per_step_median": statistics.median(steps), "decode_ms_per_step_min": min(steps),
                     "samples_ms": steps, "tokens_identical": all(t == tokens_by_repeat[0] for t in tokens_by_repeat),
                     "enforce_eager": eager, "prefix_caching": False, "block_size": args.block_size, "short": args.short, "long": args.long})
    out.mkdir(parents=True, exist_ok=True)
    import vllm
    (out / "summary.json").write_text(json.dumps({"rows": rows, "vllm": vllm.__version__, "torch": torch.__version__,
                                                   "model": args.model, "claim": "decode time per step derived from two generation lengths; not a kernel time"}, indent=2))


if __name__ == "__main__":
    main()
