"""Reproducible serving experiments and an explicitly CPU-only functional demo."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def preflight(model=None, require_cuda=True):
    import torch

    result = {"python": sys.version.split()[0], "executable": sys.executable,
              "packages": {name: _version(name) for name in ("torch", "flash-attn", "numpy", "pandas", "pyarrow", "safetensors", "tokenizers")},
              "cuda_available": bool(torch.cuda.is_available()), "torch_cuda": torch.version.cuda,
              "errors": [], "gpu_processes": []}
    if require_cuda:
        if not result["cuda_available"]:
            result["errors"].append("CUDA is unavailable to this process. Check GPU device access and NVIDIA driver; use 'serve demo' for a CPU functional demonstration.")
        if not result["packages"]["flash-attn"]:
            result["errors"].append("flash-attn is missing from this interpreter; install the project's compatible GPU environment.")
        try:
            command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
            process = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
            result["nvidia_smi_output"] = process.stdout.strip()
            if process.returncode:
                result["errors"].append("nvidia-smi failed: " + (process.stderr or process.stdout).strip())
            for line in process.stdout.splitlines() if process.returncode == 0 else []:
                parts = [part.strip() for part in line.split(",", 2)]
                if len(parts) != 3 or not parts[0].isdigit():
                    result["errors"].append("unrecognized nvidia-smi process row: " + line)
                    continue
                pid, name, memory = int(parts[0]), parts[1], parts[2]
                result["gpu_processes"].append({"pid": pid, "name": name, "used_memory_mib": memory})
                if pid != os.getpid() and "rerun" not in Path(name).name.lower():
                    result["errors"].append(f"GPU timing is blocked by another process (PID {pid}, {name}); wait for it to finish.")
        except (OSError, subprocess.TimeoutExpired) as error:
            result["errors"].append("cannot verify exclusive GPU access: " + str(error))
        if result["cuda_available"]:
            result.update(gpu_name=torch.cuda.get_device_name(), device_capability=list(torch.cuda.get_device_capability()),
                          free_memory_bytes=torch.cuda.mem_get_info()[0], total_memory_bytes=torch.cuda.mem_get_info()[1])
    if model and model != "tiny-random":
        from kernelscope.serve.hf import ModelConfig, snapshot_dir
        try:
            path = snapshot_dir(model)
            result.update(model_path=str(path.resolve()), model_config=asdict(ModelConfig.from_dir(path)))
            result["model_config_sha256"] = _sha256(path / "config.json")
            result["weight_files"] = [{"name": p.name, "size": p.stat().st_size} for p in sorted(path.glob("*.safetensors"))]
            if not result["weight_files"]:
                result["errors"].append("local model has no safetensors weights")
        except (FileNotFoundError, ValueError, KeyError) as error:
            result["errors"].append(str(error))
        if result["packages"]["safetensors"] is None:
            result["errors"].append("safetensors is missing from this interpreter")
    result["ready"] = not result["errors"]
    return result


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _git():
    values = {}
    for name, command in (("commit", ["git", "rev-parse", "HEAD"]), ("status", ["git", "status", "--short"])):
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        values[name] = process.stdout.strip() if process.returncode == 0 else None
    return values


def _source_hashes():
    package = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(package.parent)): _sha256(path)
            for directory in (package / "serve", package / "model")
            for path in sorted(directory.rglob("*")) if path.is_file() and path.suffix in (".py", ".c", ".h")}


def _resolve_scenario(args, requests, dataset, vocab):
    from kernelscope.serve.hf import load_tokenizer, snapshot_dir
    from kernelscope.serve.scenarios import needs_tokenizer, resolve_requests
    tokenizer, identity = None, {}
    started = time.perf_counter()
    if needs_tokenizer(requests):
        if args.model == "tiny-random":
            raise ValueError("text scenarios need a local trained-model tokenizer; CPU demo accepts synthetic or explicit token IDs")
        path = snapshot_dir(args.model)
        tokenizer = load_tokenizer(path)
        identity = {name: _sha256(path / name) for name in ("tokenizer.json", "tokenizer_config.json") if (path / name).is_file()}
    tokenizer_load_us = (time.perf_counter() - started) * 1e6
    resolved, metadata = resolve_requests(requests, vocab, tokenizer, args.seed, dataset)
    metadata.update(tokenizer_load_us=tokenizer_load_us, tokenizer_sha256=identity)
    return resolved, metadata


def _request_metadata(request):
    return {name: getattr(request, name) for name in ("rid", "prompt_len", "max_new_tokens", "arrival_step",
                                                     "prompt_kind", "prompt_sha256", "expected_answer")}


def _policy_setup(args, stage):
    started = time.perf_counter()
    policies = _policies(args)
    return policies, {"stage": stage, "setup_us": (time.perf_counter() - started) * 1e6,
                      "policies": [{"name": policy.name, "simulator_backend": getattr(policy, "simulator_backend", None)}
                                   for policy in policies]}


def _load_model(args):
    import torch
    from kernelscope.serve.hf import ModelConfig
    from kernelscope.serve.model import DecoderModel
    if args.model == "tiny-random":
        cfg = ModelConfig(arch="qwen3", n_layers=2, hidden=32, intermediate=64, n_heads=4, n_kv_heads=2,
                          head_dim=8, vocab=128, rope_theta=10000., rope_scaling=None, rms_eps=1e-6,
                          tie_embeddings=False, qk_norm=True)
        if args.device == "cuda":
            cfg = replace(cfg, hidden=256, intermediate=512, head_dim=64)
        return DecoderModel.random(cfg, device=args.device, seed=args.seed)
    if args.device != "cuda":
        raise ValueError("CPU demonstration supports --model tiny-random; use CUDA for the local LLM benchmark")
    return DecoderModel.from_pretrained(args.model, device=args.device)


def _policies(args):
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    from kernelscope.serve.dispatch import make_policy
    machine = MachineSpec.from_json(args.machine) if args.machine else None
    params = ModelParams.from_json(args.params) if args.params else None
    policies = [make_policy(spec, machine, params, args.cache_state) for spec in args.policy]
    if len({policy.name for policy in policies}) != len(policies):
        raise ValueError("policy names must be unique (e.g. fa2 and fixed:1 are the same policy)")
    return policies


def _default_out(label):
    return Path("results/serve") / (label + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))


def _validate_args(args):
    if not math.isfinite(args.kv_gib) or args.kv_gib <= 0:
        raise ValueError("--kv-gib must be finite and positive")
    if args.seed < 0 or args.max_batch <= 0:
        raise ValueError("--seed must be nonnegative and --max-batch positive")
    if getattr(args, "repeats", 1) < 1 or getattr(args, "warmup_runs", 0) < 0:
        raise ValueError("--repeats must be positive and --warmup-runs nonnegative")
    if getattr(args, "warmup_steps", None) is not None and args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be nonnegative")


def run_experiment(args):
    import pandas as pd
    import torch
    from kernelscope.serve.engine import Engine
    from kernelscope.serve.equivalence import compare_results
    from kernelscope.serve.kvcache import PagePool
    from kernelscope.serve.report import oracle, summarize
    from kernelscope.serve.scenarios import load_scenario

    _validate_args(args)
    requests, dataset = load_scenario(args.scenario)
    args.policy = args.policy or ["heuristic", "fa2", "fixed:8"]
    # "keep" reuses the policy objects built here for the warm-up and every measured repeat, so a page
    # configuration seen once is a cache hit afterwards (a server's steady state); "fresh" rebuilds them
    # per measured repeat so every first decision is paid and recorded.
    keep_policies = getattr(args, "policy_cache", "fresh") == "keep"
    policies, policy_setup = _policy_setup(args, "initial")
    status = preflight(args.model, require_cuda=args.device == "cuda")
    if not status["ready"]:
        raise RuntimeError("Serving preflight failed:\n- " + "\n- ".join(status["errors"]))
    out = Path(args.out) if args.out else _default_out("cuda" if args.device == "cuda" else "cpu_demo")
    if out.exists():
        raise FileExistsError(f"output already exists: {out}; choose a new directory to preserve prior evidence")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    model_started = time.perf_counter()
    model = _load_model(args)
    model_setup_us = (time.perf_counter() - model_started) * 1e6
    requests, prompt_metadata = _resolve_scenario(args, requests, dataset, model.cfg.vocab)
    quality_tokenizer, quality_setup_us = None, 0.0
    if any(request.expected_answer is not None for request in requests):
        from kernelscope.serve.hf import load_tokenizer, snapshot_dir
        if args.model == "tiny-random":
            raise ValueError("expected-answer QA scoring requires a local trained-model tokenizer")
        quality_started = time.perf_counter()
        quality_tokenizer = load_tokenizer(snapshot_dir(args.model))
        quality_setup_us = (time.perf_counter() - quality_started) * 1e6
    kv_bytes = int(args.kv_gib * 2**30)
    if args.device == "cuda" and torch.cuda.mem_get_info()[0] < kv_bytes + 256 * 2**20:
        raise MemoryError("not enough free GPU memory for --kv-gib plus working tensors; reduce KV budget or batch scenario")
    manifest = {"schema_version": 1, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "evidence_kind": "cuda_serving" if args.device == "cuda" else "cpu_functional",
                "performance_claim": args.device == "cuda", "model": args.model,
                "model_backend": model.backend, "model_config": asdict(model.cfg),
                "model_dtype": str(model.dtype).removeprefix("torch."), "model_setup_us": model_setup_us,
                "scenario": str(Path(args.scenario).resolve()), "scenario_sha256": _sha256(args.scenario),
                "requests": [_request_metadata(request) for request in requests], "policy_specs": args.policy,
                "seed": args.seed, "max_batch": args.max_batch, "kv_bytes": kv_bytes,
                "warmup_runs": args.warmup_runs, "warmup_steps": args.warmup_steps, "repeats": args.repeats,
                "policy_order": "rotate_each_repeat",
                "policy_cache": "kept_across_runs" if keep_policies else "fresh_for_each_measured_run",
                "prefill": "serial_per_request", "arrival_mode": "logical_decode_step",
                **prompt_metadata, "sampling": "greedy_fixed_length_no_eos_stop",
                "quality_setup_us": quality_setup_us,
                "quality_evaluation": ("small_authored_qa_literal_exact_and_nfkc_casefold_whitespace_exact"
                                       if quality_tokenizer is not None else None),
                "policy_setup_runs": [policy_setup],
                "timing_instrumentation": "per-layer attention events, per-decode events, synchronized host token timestamps",
                "limitations": ["No network or production scheduler overhead", "Step-based arrivals are not an open-loop SLA test",
                                "TPOT includes admission/prefill stalls and dispatch overhead",
                                "Constructed prompts are not production traffic or a validated language-quality dataset"],
                "preflight": status, "git": _git(), "source_sha256": _source_hashes(), "order": []}
    if args.device == "cpu":
        manifest["limitations"].append("CPU SDPA ignores num_splits; CPU policy timing differences are not GPU speed evidence")
    manifest["policy_inputs"] = {str(path): _sha256(path) for path in [args.machine, args.params] +
                                 [spec.split(":", 1)[1] for spec in args.policy if spec.startswith("table:")] if path}
    out.mkdir(parents=True, exist_ok=False)
    (out / "scenario.yaml").write_bytes(Path(args.scenario).read_bytes())
    with (out / "prompts.jsonl").open("x") as handle:
        for request in requests:
            handle.write(json.dumps({**_request_metadata(request), "token_ids": list(request.token_ids)}, ensure_ascii=False) + "\n")
    _json(out / "manifest.json", manifest)
    equivalence_rows, quality_rows = [], []

    def run(policy, scenario):
        pool = PagePool.for_budget(model.cfg, kv_bytes, device=model.device, dtype=model.dtype)
        try:
            return Engine(model, pool, policy, args.max_batch).run(scenario, model.cfg.vocab, seed=args.seed)
        finally:
            del pool

    try:
        warm_requests = requests
        if args.warmup_steps is not None:
            warm_requests = [replace(request, max_new_tokens=min(request.max_new_tokens, args.warmup_steps + 1)) for request in requests]
        for policy in policies:
            for _ in range(args.warmup_runs):
                print(f"Warmup {policy.name}", flush=True)
                run(policy, warm_requests)
        for repeat in range(args.repeats):
            if not keep_policies:
                policies, setup = _policy_setup(args, f"repeat_{repeat:03d}")  # retain measured policy cache misses
                manifest["policy_setup_runs"].append(setup)
            order = policies[repeat % len(policies):] + policies[:repeat % len(policies)]
            manifest["order"].append([policy.name for policy in order])
            results = {}
            for policy in order:
                if args.device == "cuda":
                    check = preflight(require_cuda=True)
                    if not check["ready"]:
                        raise RuntimeError("GPU contention/preflight changed: " + "; ".join(check["errors"]))
                print(f"Repeat {repeat + 1}/{args.repeats}: {policy.name}", flush=True)
                result = run(policy, requests)
                directory = out / policy.name / f"repeat_{repeat:03d}"
                directory.mkdir(parents=True, exist_ok=False)
                for name in ("steps", "tokens", "prefill"):
                    getattr(result, name).to_parquet(directory / f"{name}.parquet", index=False)
                metadata = {**result.metadata, "policy": policy.name, "repeat": repeat,
                            "evidence_kind": manifest["evidence_kind"], "performance_claim": manifest["performance_claim"],
                            "model": args.model, "scenario": manifest["scenario"],
                            "dataset_id": manifest["dataset_id"], "dataset_sha256": manifest["dataset_sha256"],
                            "resolved_prompts_sha256": manifest["resolved_prompts_sha256"],
                            "evaluation_split": manifest["evaluation_split"], "scenario_family": manifest["scenario_family"],
                            "created_at": datetime.now(timezone.utc).isoformat()}
                _json(directory / "meta.json", metadata)
                results[policy.name] = result
                if quality_tokenizer is not None:
                    from kernelscope.serve.quality import score_tokens
                    quality = score_tokens(requests, result.tokens, quality_tokenizer)
                    quality["policy"], quality["repeat"] = policy.name, repeat
                    quality_rows.extend(quality.to_dict("records"))
            reference = "heuristic" if "heuristic" in results else policies[0].name
            for policy, result in results.items():
                if policy != reference:
                    equivalence_rows.append({**compare_results(results[reference], result, reference, policy), "repeat": repeat})
        eq = pd.DataFrame(equivalence_rows)
        if quality_rows:
            pd.DataFrame(quality_rows).to_csv(out / "qa_quality.csv", index=False)
        if not eq.empty:
            eq.to_csv(out / "equivalence.csv", index=False)
        summary = summarize(out)
        summary.to_csv(out / "summary.csv", index=False)
        oracle(out).to_csv(out / "oracle.csv", index=False)
        manifest["status"] = "complete"
        manifest["tokens_equivalent"] = bool(eq.passed.all()) if not eq.empty else None
        manifest["independent_reference_validation"] = "not_performed_by_this_command"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        if not eq.empty and not eq.passed.all():
            manifest["performance_claim"] = False
        _json(out / "manifest.json", manifest)
        print(summary.to_string(index=False))
        print(f"Artifacts: {out.resolve()}")
        if not eq.empty and not eq.passed.all():
            raise RuntimeError("Strict token/schedule equivalence failed; inspect equivalence.csv. Speed results are not an output-preserving gain.")
        return out
    except BaseException as error:
        if manifest["status"] != "complete":
            manifest.update(status="failed", error=str(error))
            _json(out / "manifest.json", manifest)
        raise


def _run(args):
    try:
        return run_experiment(args)
    except (ValueError, RuntimeError, OSError, MemoryError) as error:
        raise SystemExit(str(error)) from None


def _demo(args):
    args.device, args.model, args.policy = "cpu", "tiny-random", ["heuristic", "fa2", "fixed:8"]
    print("CPU FUNCTIONAL DEMO: real tiny random decoder, synthetic prompts; num_splits is ignored by CPU SDPA.")
    return _run(args)


def _doctor(args):
    result = preflight(args.model)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["ready"]:
        raise SystemExit(1)


def _diagnose(args):
    from kernelscope.diagnose.run import run_diagnose
    try:
        return run_diagnose(args)
    except (ValueError, RuntimeError, OSError, MemoryError) as error:
        raise SystemExit(str(error)) from None


def _diagnose_report(args):
    from kernelscope.diagnose.run import run_report
    try:
        return run_report(args)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None


def _compare(args):
    from kernelscope.serve.report import oracle, summarize
    summary = summarize(args.results)
    print(summary.to_string(index=False))
    print(oracle(args.results).to_string(index=False))
    if args.out:
        path = Path(args.out)
        if path.exists():
            raise SystemExit(f"output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path, index=False)


def _equivalence(args):
    from kernelscope.serve.equivalence import compare_policies
    from kernelscope.serve.scenarios import load_scenario
    _validate_args(args)
    args.policy = args.policy or ["heuristic", "fa2", "fixed:8"]
    policies = _policies(args)
    out = Path(args.out) if args.out else _default_out("equivalence").with_suffix(".csv")
    if out.exists():
        raise SystemExit(f"output already exists: {out}")
    check = preflight(args.model, require_cuda=args.device == "cuda")
    if not check["ready"]:
        raise SystemExit("Equivalence preflight failed: " + "; ".join(check["errors"]))
    model = _load_model(args)
    requests, dataset = load_scenario(args.scenario)
    requests, prompt_metadata = _resolve_scenario(args, requests, dataset, model.cfg.vocab)
    result = compare_policies(model, policies, requests, model.cfg.vocab, int(args.kv_gib * 2**30),
                              args.record_steps, args.max_batch, args.seed, args.atol)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    _json(out.with_suffix(".meta.json"), {"model": args.model, "scenario": str(args.scenario), "preflight": check,
                                       "model_dtype": str(model.dtype).removeprefix("torch."), **prompt_metadata,
                                       "record_steps": args.record_steps, "atol": args.atol,
                                       "claim": "Cross-policy greedy tokens and sampled logit closeness, not an independent model oracle"})
    print(result.to_string(index=False))
    if not result.passed.all():
        raise SystemExit("equivalence failed; see " + str(out))


def register_parser(subparsers):
    group = subparsers.add_parser("serve", help="continuous batching benchmark, correctness, and CPU demo")
    sub = group.add_subparsers(dest="serve_cmd", required=True)
    doctor = sub.add_parser("doctor", help="check local model, GPU access, dependencies, and GPU contention")
    doctor.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    doctor.set_defaults(func=_doctor)
    parsers = {}
    for name, handler in (("run", _run), ("demo", _demo), ("equivalence", _equivalence), ("diagnose", _diagnose)):
        parser = parsers[name] = sub.add_parser(name)
        parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
        parser.add_argument("--scenario", default="scenarios/demo.yaml" if name == "demo" else "scenarios/tiny.yaml")
        parser.add_argument("--policy", action="append")
        parser.add_argument("--out")
        parser.add_argument("--kv-gib", type=float, default=.002 if name == "demo" else 9)
        parser.add_argument("--max-batch", type=int, default=64)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu" if name == "demo" else "cuda")
        parser.add_argument("--machine")
        parser.add_argument("--params")
        parser.add_argument("--cache-state", choices=("cold", "warm"), default="cold", help="surrogate assumption; does not flush L2 in the serving loop")
        parser.add_argument("--cpu-threads", type=int, default=1)
        if name == "equivalence":
            parser.add_argument("--record-steps", type=int, default=8)
            parser.add_argument("--atol", type=float, default=.05)
        else:
            parser.add_argument("--repeats", type=int, default=3)
            parser.add_argument("--warmup-runs", type=int, default=1)
            parser.add_argument("--warmup-steps", type=int, help="optional cap on generated warmup decode steps per request")
            parser.add_argument("--policy-cache", choices=("fresh", "keep"), default="fresh",
                                help="fresh: rebuild policies for every measured repeat (every first decision is paid); "
                                     "keep: reuse the policies built before warm-up across repeats (steady-state cache hits)")
        parser.set_defaults(func=handler)
    diag = parsers["diagnose"]
    diag.add_argument("--table", default="demo_data/dispatch_paged_cold.csv", help="measured dispatch table for the attention row")
    diag.add_argument("--data", default="demo_data", help="results root holding hw_4090/*_paged/summaries.jsonl")
    diag.add_argument("--threshold", type=float, default=0.7, help="ceiling fraction that counts as bound")
    report = sub.add_parser("diagnose-report", help="recompute diagnosis.json / ops.csv / figure from a recorded diagnose run (no GPU)")
    report.add_argument("results")
    report.add_argument("--machine", default="machines/rtx4090.json")
    report.add_argument("--params")
    report.add_argument("--table", default="demo_data/dispatch_paged_cold.csv")
    report.add_argument("--data", default="demo_data")
    report.add_argument("--threshold", type=float, default=0.7)
    report.set_defaults(func=_diagnose_report)
    compare = sub.add_parser("compare")
    compare.add_argument("--results", required=True)
    compare.add_argument("--out")
    compare.set_defaults(func=_compare)
    from kernelscope.serve.generate import register_parser as register_generate
    register_generate(sub)
    return group
