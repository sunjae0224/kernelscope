"""Where does a vLLM decode step spend its time on a mixed-length batch? torch-profiler traces, then a per-step breakdown.

Run with the separate vLLM environment (not the project's .venv):
    ~/.venvs/kernelscope-vllm/bin/python -m scripts.vllm_profile_step run --out ../kernelscope/results/vllm_4090/profile_<date>
    python -m scripts.vllm_profile_step analyze --root ../kernelscope/results/vllm_4090/profile_<date>     # stdlib only

``run`` uses the batch compositions, model, prompts, prefix-cache setting and warm-up of scripts/vllm_reproduce.py, selects the
attention backend with ``attention_config={"backend": ...}`` (the VLLM_ATTENTION_BACKEND environment variable that
vllm_reproduce.py sets is ignored by vLLM 0.31: its logs say "Unknown vLLM environment variable" and "Using FLASH_ATTN"), and
records a trace over one ``generate`` call with the supported ``profiler_config`` (the older VLLM_TORCH_PROFILER_DIR variable
no longer exists) and ``llm.start_profile()`` / ``llm.stop_profile()``. It also times unprofiled generations with the two-length
method of vllm_reproduce.py, so profiler overhead can be read off.
``analyze`` cuts every trace into engine iterations by vLLM's ``execute_context_N(..)_generation_M(..)`` annotation, assigns each
GPU kernel to the iteration whose annotation contains its launch (CUPTI correlation id), and splits an iteration into attention /
KV-cache write / GEMM / other GPU time, GPU busy time (union of kernel intervals) and the host-side remainder
(iteration wall time minus GPU busy time).
"""
import argparse
import glob
import gzip
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

STEP_RE = re.compile(r"execute_(?:\d+_)?context_(\d+)\(.*?\)_generation_(\d+)\(.*?\)$")
CATEGORIES = ("attention", "kv_write", "gemm", "other")
ATTENTION_RE = re.compile(r"flash|splitkv|fwd_kernel|flashinfer|BatchDecode|BatchPrefill|PagedAttention|paged_attention|"
                          r"merge_attn_states|MergeStates|VariableLengthMerge|attention", re.I)
KV_WRITE_RE = re.compile(r"reshape_and_cache|concat_and_cache|cache_kernel|append_paged_kv|AppendPagedKV", re.I)
GEMM_RE = re.compile(r"gemm|gemv|cutlass|cublas|nvjet|xmma|splitKreduce|sm\d+_.*tn|ampere_|turing_|ada_", re.I)


def classify_kernel(name, extra_attention=None):
    """attention / kv_write / gemm / other for a CUDA kernel name (KV write first: 'reshape_and_cache_flash' contains 'flash')."""
    if KV_WRITE_RE.search(name):
        return "kv_write"
    if ATTENTION_RE.search(name) or (extra_attention and re.search(extra_attention, name)):
        return "attention"
    if GEMM_RE.search(name):
        return "gemm"
    return "other"


def parse_step_name(name):
    """(context requests, generation requests) from vLLM's iteration annotation, or None."""
    match = STEP_RE.search(name)
    return (int(match.group(1)), int(match.group(2))) if match else None


def merged_length(intervals):
    """Total length of the union of (start, end) intervals."""
    total, cur_start, cur_end = 0.0, None, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def load_trace(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def decompose(trace, extra_attention=None):
    """Per-iteration rows from one Chrome trace (times in microseconds). Pure function of the trace dict."""
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    steps = []
    for e in events:
        if e.get("ph") == "X" and isinstance(e.get("name"), str):
            parsed = parse_step_name(e["name"])
            if parsed:
                steps.append({"ts": float(e["ts"]), "end": float(e["ts"]) + float(e.get("dur", 0)), "context": parsed[0],
                              "generation": parsed[1], "name": e["name"]})
    steps.sort(key=lambda s: s["ts"])
    launch_ts = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"):
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                launch_ts[corr] = float(e["ts"])
    starts = [s["ts"] for s in steps]
    for s in steps:
        s["kernels"] = []
    unassigned = 0
    for e in events:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        corr = (e.get("args") or {}).get("correlation")
        t = launch_ts.get(corr, float(e["ts"]))          # without a launch event fall back to the kernel's own start
        idx = None
        for i in range(len(steps) - 1, -1, -1):
            if starts[i] <= t <= steps[i]["end"]:
                idx = i
                break
        if idx is None:
            unassigned += 1
            continue
        steps[idx]["kernels"].append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), e["name"]))
    rows = []
    for i, s in enumerate(steps):
        cats = {c: 0.0 for c in CATEGORIES}
        for start, end, name in s["kernels"]:
            cats[classify_kernel(name, extra_attention)] += end - start
        busy = merged_length([(a, b) for a, b, _ in s["kernels"]])
        wall = steps[i + 1]["ts"] - s["ts"] if i + 1 < len(steps) else None
        rows.append({"index": i, "context_requests": s["context"], "generation_requests": s["generation"],
                     "annotation_us": s["end"] - s["ts"], "wall_us": wall, "gpu_busy_us": busy,
                     "host_gap_us": None if wall is None else wall - busy, "kernels": len(s["kernels"]),
                     **{f"{c}_us": cats[c] for c in CATEGORIES}})
    return rows, unassigned


def decode_rows(rows):
    """Pure-decode iterations that have a successor (their wall time is the gap to the next iteration's start)."""
    return [r for r in rows if r["context_requests"] == 0 and r["generation_requests"] > 0 and r["wall_us"] is not None]


def summarize_rows(rows):
    """Median per-iteration decomposition over pure-decode iterations."""
    dec = decode_rows(rows)
    if not dec:
        return {"decode_iterations": 0}
    keys = ["wall_us", "gpu_busy_us", "host_gap_us", "annotation_us", *(f"{c}_us" for c in CATEGORIES), "kernels"]
    out = {"decode_iterations": len(dec), "batch": statistics.median(r["generation_requests"] for r in dec)}
    for k in keys:
        out[f"{k}_median"] = statistics.median(r[k] for r in dec)
    wall = out["wall_us_median"]
    out["gpu_busy_share_of_wall"] = out["gpu_busy_us_median"] / wall if wall else None
    out["host_gap_share_of_wall"] = out["host_gap_us_median"] / wall if wall else None
    return out


def top_kernels(trace, n=8, extra_attention=None):
    """Aggregate GPU time by kernel name: [(name, category, total_us, calls)] sorted by time."""
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    agg = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") == "kernel":
            t, c = agg.get(e["name"], (0.0, 0))
            agg[e["name"]] = (t + float(e.get("dur", 0)), c + 1)
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][0])[:n]
    return [{"name": k, "category": classify_kernel(k, extra_attention), "total_us": t, "calls": c} for k, (t, c) in ranked]


def analyze(root, extra_attention=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
    runs = []
    for run_dir in sorted(p for p in (root / "traces").iterdir() if p.is_dir()):
        files = sorted(glob.glob(str(run_dir / "*.json*")))
        if not files:
            continue
        trace = load_trace(files[0])
        rows, unassigned = decompose(trace, extra_attention)
        meta = manifest.get("runs", {}).get(run_dir.name, {})
        runs.append({"run": run_dir.name, "trace_file": os.path.basename(files[0]), "unassigned_kernels": unassigned,
                     **meta, "summary": summarize_rows(rows), "top_kernels": top_kernels(trace, extra_attention=extra_attention),
                     "iterations": rows})
    ratios = {}
    by = {(r.get("backend"), r.get("mode"), r.get("scenario")): r for r in runs}
    for (backend, mode, scenario), r in by.items():
        if scenario == "ragged" and (backend, mode, "uniform") in by:
            u, m = by[(backend, mode, "uniform")]["summary"], r["summary"]
            if u.get("decode_iterations") and m.get("decode_iterations"):
                ratios[f"{backend}/{mode}"] = {k: m[f"{k}_median"] / u[f"{k}_median"] if u[f"{k}_median"] else None
                                               for k in ("wall_us", "gpu_busy_us", "host_gap_us", "attention_us", "gemm_us", "other_us")}
    result = {"runs": runs, "ragged_over_uniform": ratios, "claim": "torch-profiler traces; profiler overhead inflates host time, "
              "compare with the unprofiled step times in manifest.json", "extra_attention_regex": extra_attention}
    (root / "step_breakdown.json").write_text(json.dumps(result, indent=1))
    for r in runs:
        s = r["summary"]
        if s.get("decode_iterations"):
            print(f"{r['run']:28s} B={s['batch']:.0f} wall {s['wall_us_median']/1e3:7.2f} ms  gpu busy {s['gpu_busy_us_median']/1e3:7.2f}  "
                  f"host gap {s['host_gap_us_median']/1e3:7.2f}  attn {s['attention_us_median']/1e3:7.2f}  gemm {s['gemm_us_median']/1e3:7.2f}  "
                  f"other {s['other_us_median']/1e3:6.2f}  kv_write {s['kv_write_us_median']/1e3:5.2f}")
    for name, r in ratios.items():
        print(f"{name}: ragged/uniform " + "  ".join(f"{k.removesuffix('_us')} {v:.2f}x" for k, v in r.items() if v))
    return result


def prompts_and_scenarios():
    from scripts.vllm_reproduce import SCENARIOS, prompts_for
    return SCENARIOS, prompts_for


def kv_cache_kwargs(kv_cache_gib):
    """Fixed KV cache size for LLM(). vLLM's utilization-based sizing ignored the FlashInfer workspace and lazily allocated
    activations, so the 2026-10-25 FLASHINFER and CUDA-graph children ran out of memory at 32K prefill (22 GiB in use, 0.4 GiB free)."""
    return {} if kv_cache_gib is None else {"kv_cache_memory_bytes": int(kv_cache_gib * 2**30)}


def run_child(args):
    from vllm import LLM, SamplingParams
    import torch
    import vllm
    scenarios, prompts_for = prompts_and_scenarios()
    out = Path(args.out)
    trace_root = out / "traces"
    eager = args.mode == "eager"
    profile_dir = out / f"_raw_{args.backend.lower()}_{args.mode}"
    llm = LLM(model=args.model, enforce_eager=eager, max_model_len=33000, gpu_memory_utilization=args.gpu_memory_utilization,
              **kv_cache_kwargs(args.kv_cache_gib),
              max_num_seqs=64, max_num_batched_tokens=33000, dtype="bfloat16", seed=0, enable_prefix_caching=False,
              attention_config={"backend": args.backend},
              profiler_config={"profiler": "torch", "torch_profiler_dir": str(profile_dir), "torch_profiler_with_stack": False,
                               "torch_profiler_record_shapes": False, "ignore_frontend": True, "detailed_trace_annotation": True})
    vocab = llm.get_tokenizer().vocab_size
    runs = {}
    for scenario in args.scenarios.split(","):
        lens = scenarios[scenario]
        prompts = prompts_for(lens, vocab)
        llm.generate(prompts, SamplingParams(max_tokens=args.short, min_tokens=args.short, ignore_eos=True, temperature=0.0), use_tqdm=False)
        # unprofiled step time, two-length method (identical to vllm_reproduce.py)
        steps = []
        for _ in range(args.repeats):
            times = {}
            for n in (args.short, args.long):
                params = SamplingParams(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0.0)
                # no torch.cuda.synchronize() here: the engine runs in another process, generate() returns only after its sampled
                # tokens reached the host, and a driver-side CUDA context only competes with the engine for the last GiB (OOM, 2026-10-25)
                started = time.perf_counter()
                llm.generate(prompts, params, use_tqdm=False)
                times[n] = time.perf_counter() - started
            steps.append(1000 * (times[args.long] - times[args.short]) / (args.long - args.short))
        before = set(glob.glob(str(profile_dir / "**" / "*.json*"), recursive=True))
        llm.start_profile(f"{args.backend.lower()}_{args.mode}_{scenario}")
        started = time.perf_counter()
        llm.generate(prompts, SamplingParams(max_tokens=args.profile_tokens, min_tokens=args.profile_tokens, ignore_eos=True, temperature=0.0),
                     use_tqdm=False)
        generate_s = time.perf_counter() - started
        llm.stop_profile()
        new = []
        deadline = time.time() + 180
        while time.time() < deadline:                  # the worker writes the trace after stop_profile returns
            new = sorted(set(glob.glob(str(profile_dir / "**" / "*.json*"), recursive=True)) - before)
            if new and all(time.time() - os.path.getmtime(p) > 5 for p in new):
                break
            time.sleep(2)
        dest = trace_root / f"{args.backend.lower()}_{args.mode}_{scenario}"
        dest.mkdir(parents=True, exist_ok=True)
        for p in new:
            shutil.move(p, dest / os.path.basename(p))
        runs[dest.name] = {"backend": args.backend, "mode": args.mode, "scenario": scenario, "lens_head": lens[:2], "batch": len(lens),
                           "unprofiled_decode_ms_per_step_median": statistics.median(steps), "unprofiled_samples_ms": steps,
                           "profile_tokens": args.profile_tokens, "profiled_generate_s": generate_s, "trace_files": [os.path.basename(p) for p in new]}
    part = out / f"manifest_{args.backend.lower()}_{args.mode}.json"
    part.write_text(json.dumps({"runs": runs, "vllm": vllm.__version__, "torch": torch.__version__, "model": args.model,
                                "enforce_eager": eager, "prefix_caching": False, "requested_backend": args.backend}, indent=1))


def run_parent(args):
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    out.mkdir(parents=True)
    logs = out / "logs"
    logs.mkdir()
    runs, meta = {}, {}
    for spec in args.runs.split(","):
        backend, mode = spec.split(":")
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        cmd = [sys.executable, "-m", "scripts.vllm_profile_step", "child", "--out", str(out), "--backend", backend, "--mode", mode,
               "--model", args.model, "--scenarios", args.scenarios, "--repeats", str(args.repeats), "--short", str(args.short),
               "--long", str(args.long), "--profile-tokens", str(args.profile_tokens),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        if args.kv_cache_gib is not None:
            cmd += ["--kv-cache-gib", str(args.kv_cache_gib)]
        log = logs / f"{backend.lower()}_{mode}.log"
        with open(log, "w") as handle:
            rc = subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        text = log.read_text(errors="replace")
        used = re.findall(r"Using (\w+) attention backend", text)
        print(f"{backend} {mode}: rc={rc}, engine reports backend {used}")
        part = out / f"manifest_{backend.lower()}_{mode}.json"
        if part.exists():
            data = json.loads(part.read_text())
            for name, r in data["runs"].items():
                runs[name] = {**r, "engine_reported_backend": used[-1] if used else None, "child_rc": rc}
            meta = {k: data[k] for k in ("vllm", "torch", "model")}
        else:
            runs[f"{backend.lower()}_{mode}_FAILED"] = {"backend": backend, "mode": mode, "child_rc": rc, "engine_reported_backend": used[-1] if used else None}
    (out / "manifest.json").write_text(json.dumps({"runs": runs, **meta}, indent=1))
    return analyze(out, args.extra_attention)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("run", "child"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
        p.add_argument("--scenarios", default="ragged,uniform")
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--short", type=int, default=4)
        p.add_argument("--long", type=int, default=64)
        p.add_argument("--profile-tokens", type=int, default=24, help="generated tokens of the profiled generate call (prefill plus this many minus one decode iterations)")
        p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
        p.add_argument("--kv-cache-gib", type=float, default=None, help="fixed KV cache size in GiB (overrides utilization-based sizing)")
        if name == "run":
            p.add_argument("--runs", default="FLASH_ATTN:eager,FLASHINFER:eager", help="comma list of BACKEND:MODE (eager|cudagraph)")
            p.add_argument("--extra-attention", help="regex of additional attention kernel names")
        else:
            p.add_argument("--backend", required=True)
            p.add_argument("--mode", choices=("eager", "cudagraph"), required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--root", required=True)
    p.add_argument("--extra-attention")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run_parent(args)
    elif args.cmd == "child":
        run_child(args)
    else:
        analyze(args.root, args.extra_attention)


if __name__ == "__main__":
    main()
