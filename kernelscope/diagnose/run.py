"""`serve diagnose`: one control run and one op-timed run per policy, then the ceiling report.

Provenance (preflight, prompt resolution, manifest, source hashes) is reused from kernelscope.serve.cli.
Timings recorded here are diagnostic evidence, never a performance claim.
"""
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def project_path(value, default) -> Path:
    """A CLI path as given when it exists, else the same path relative to the repository root."""
    candidate = Path(value) if value else PROJECT / default
    return candidate if candidate.exists() else PROJECT / candidate


def _report(out, args):
    from kernelscope.diagnose.figures import op_breakdown
    from kernelscope.diagnose.report import diagnose, write
    from kernelscope.model.machine import MachineSpec
    machine = MachineSpec.from_json(project_path(getattr(args, "machine", None), "machines/rtx4090.json"))
    params = None
    if getattr(args, "params", None):
        from kernelscope.model.params import ModelParams
        params = ModelParams.from_json(args.params)
    result = diagnose(out, machine, project_path(args.table, "demo_data/dispatch_paged_cold.csv"),
                      project_path(args.data, "demo_data"), params=params, threshold=args.threshold)
    write(out, result)
    op_breakdown(result.summary, out / "op_breakdown.png", out / "op_breakdown.svg")
    return result


def run_report(args) -> Path:
    out = Path(args.results)
    if not (out / "manifest.json").exists():
        raise FileNotFoundError(f"{out} is not a serve diagnose run: manifest.json is missing")
    result = _report(out, args)
    print(result.ops.to_string(index=False))
    return out


def run_diagnose(args) -> Path:
    import pandas as pd
    import torch
    from kernelscope.serve import cli
    from kernelscope.serve.engine import Engine
    from kernelscope.serve.equivalence import compare_results
    from kernelscope.serve.kvcache import PagePool
    from kernelscope.serve.scenarios import load_scenario

    args.repeats = 1
    cli._validate_args(args)
    if not 0 < args.threshold <= 1:
        raise ValueError("--threshold must be in (0, 1]")
    requests, dataset = load_scenario(args.scenario)
    args.policy = args.policy or ["heuristic"]
    policies, policy_setup = cli._policy_setup(args, "initial")
    cuda = args.device == "cuda"
    status = cli.preflight(args.model, require_cuda=cuda)
    if not status["ready"]:
        raise RuntimeError("Diagnose preflight failed:\n- " + "\n- ".join(status["errors"]))
    out = Path(args.out) if args.out else cli._default_out("diagnose")
    if out.exists():
        raise FileExistsError(f"output already exists: {out}; choose a new directory to preserve prior evidence")
    if not cuda:
        torch.set_num_threads(args.cpu_threads)
    model = cli._load_model(args)
    requests, prompt_metadata = cli._resolve_scenario(args, requests, dataset, model.cfg.vocab)
    kv_bytes = int(args.kv_gib * 2**30)
    if cuda and torch.cuda.mem_get_info()[0] < kv_bytes + 256 * 2**20:
        raise MemoryError("not enough free GPU memory for --kv-gib plus working tensors")
    table, data_root = project_path(args.table, "demo_data/dispatch_paged_cold.csv"), project_path(args.data, "demo_data")
    manifest = {"schema_version": 1, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "evidence_kind": "diagnostic_op_breakdown" if cuda else "cpu_functional", "performance_claim": False,
                "model": args.model, "model_backend": model.backend, "model_config": asdict(model.cfg),
                "model_dtype": str(model.dtype).removeprefix("torch."),
                "scenario": str(Path(args.scenario).resolve()), "scenario_sha256": cli._sha256(args.scenario),
                "requests": [cli._request_metadata(r) for r in requests], "policy_specs": args.policy,
                "seed": args.seed, "max_batch": args.max_batch, "kv_bytes": kv_bytes,
                "warmup_runs": args.warmup_runs, "warmup_steps": args.warmup_steps,
                "ops_mode_runs": ["control", "event"], "threshold": args.threshold,
                "table": str(table), "data_root": str(data_root), "policy_setup_runs": [policy_setup], **prompt_metadata,
                "sampling": "greedy_fixed_length_no_eos_stop", "arrival_mode": "logical_decode_step",
                "timing_instrumentation": "per-layer op-class CUDA events (event run); attention-only events (control run)",
                "limitations": ["Op timers add host work; step times of this run are diagnostic, not performance evidence",
                                "The byte model counts compulsory traffic only, so achieved bandwidth is a lower bound",
                                "Kernel-internal behaviour (stalls, bank conflicts) is outside this report; that is Nsight Compute's domain"],
                "preflight": status, "git": cli._git(), "source_sha256": cli._source_hashes()}
    if not cuda:
        manifest["limitations"].append("CPU SDPA ignores num_splits; CPU op timings are functional only")
    out.mkdir(parents=True, exist_ok=False)
    (out / "scenario.yaml").write_bytes(Path(args.scenario).read_bytes())
    with (out / "prompts.jsonl").open("x") as handle:
        for request in requests:
            handle.write(json.dumps({**cli._request_metadata(request), "token_ids": list(request.token_ids)}, ensure_ascii=False) + "\n")
    cli._json(out / "manifest.json", manifest)

    def run(policy, scenario, ops_mode):
        pool = PagePool.for_budget(model.cfg, kv_bytes, device=model.device, dtype=model.dtype)
        try:
            return Engine(model, pool, policy, args.max_batch).run(scenario, model.cfg.vocab, seed=args.seed, ops_mode=ops_mode)
        finally:
            del pool

    consistency = []
    try:
        warm_requests = requests
        if args.warmup_steps is not None:
            warm_requests = [replace(r, max_new_tokens=min(r.max_new_tokens, args.warmup_steps + 1)) for r in requests]
        for policy in policies:
            for _ in range(args.warmup_runs):
                print(f"Warmup {policy.name}", flush=True)
                run(policy, warm_requests, None)
            results = {}
            for label, mode in (("control", None), ("event", "event")):
                if cuda:
                    check = cli.preflight(require_cuda=True)
                    if not check["ready"]:
                        raise RuntimeError("GPU contention/preflight changed: " + "; ".join(check["errors"]))
                print(f"{label}: {policy.name}", flush=True)
                result = run(policy, requests, mode)
                directory = out / policy.name / f"{label}_000"
                directory.mkdir(parents=True, exist_ok=False)
                for name in ("steps", "tokens", "prefill"):
                    getattr(result, name).to_parquet(directory / f"{name}.parquet", index=False)
                if mode == "event":
                    result.ops.to_parquet(directory / "ops.parquet", index=False)
                cli._json(directory / "meta.json", {**result.metadata, "policy": policy.name, "run": label,
                                                    "evidence_kind": manifest["evidence_kind"], "performance_claim": False,
                                                    "created_at": datetime.now(timezone.utc).isoformat()})
                results[label] = result
            consistency.append({**compare_results(results["control"], results["event"], "control", "event"), "policy": policy.name})
        frame = pd.DataFrame(consistency)
        frame.to_csv(out / "consistency.csv", index=False)
        manifest.update(status="complete", tokens_consistent=bool(frame.passed.all()),
                        completed_at=datetime.now(timezone.utc).isoformat())
        cli._json(out / "manifest.json", manifest)
        result = _report(out, args)
        print(result.ops.to_string(index=False))
        print(f"Artifacts: {out.resolve()}")
        if not manifest["tokens_consistent"]:
            raise RuntimeError("control and op-timed runs generated different tokens; see consistency.csv")
        return out
    except BaseException as error:
        if manifest["status"] != "complete":
            manifest.update(status="failed", error=str(error))
            cli._json(out / "manifest.json", manifest)
        raise
