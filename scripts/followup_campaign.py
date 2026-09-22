"""Prespecified multi-model, multi-seed natural-text validation campaign.

Recorded runs are immutable. A completed negative token comparison does not stop
independent cases, and is retained as a negative outcome in campaign.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"qwen4b": "Qwen/Qwen3-4B-Instruct-2507",
          "llama8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_jobs(models, scenarios, seeds, repeats, out, warmup_steps=None):
    jobs = []
    policies = ["heuristic", "table:" + str(ROOT / "demo_data/dispatch_paged_cold.csv"), "model"]
    for model, scenario, seed in itertools.product(models, scenarios, seeds):
        path = ROOT / "scenarios" / f"heldout_text_{scenario}.yaml"
        if not path.is_file():
            raise FileNotFoundError(path)
        destination = out / model / scenario / f"seed_{seed}"
        command = [sys.executable, "-m", "kernelscope.cli", "serve", "run", "--model", MODELS[model],
                   "--scenario", str(path), "--seed", str(seed), "--repeats", str(repeats),
                   "--kv-gib", "4", "--warmup-runs", "1",
                   "--machine", str(ROOT / "machines/rtx4090.json"), "--params", str(ROOT / "models/rtx4090.json"),
                   "--out", str(destination)]
        if warmup_steps is not None:
            command += ["--warmup-steps", str(warmup_steps)]
        for policy in policies:
            command += ["--policy", policy]
        jobs.append({"id": f"{model}/{scenario}/seed_{seed}", "model": MODELS[model],
                     "scenario": str(path), "scenario_sha256": sha(path), "seed": seed,
                     "repeats": repeats, "policy_specs": policies, "out": str(destination),
                     "warmup_runs": 1, "warmup_steps": warmup_steps, "kv_bytes": 4 * 2**30,
                     "command": command, "status": "planned"})
    return jobs


def compatible_manifest(job, manifest):
    return all(manifest.get(key) == job[key] for key in
               ("model", "scenario_sha256", "seed", "repeats", "policy_specs", "warmup_runs", "warmup_steps", "kv_bytes")
               if key in job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--scenarios", nargs="+", choices=("uniform", "ragged", "arrivals"),
                        default=["uniform", "ragged", "arrivals"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, help="optional partial-warmup cap; default replays the complete scenario")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or any(seed < 0 for seed in args.seeds):
        parser.error("repeats must be positive and seeds nonnegative")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        parser.error("warmup-steps must be nonnegative")
    for name in ("models", "scenarios", "seeds"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(f"duplicate {name} are not allowed")
    args.out = args.out.resolve()
    jobs = build_jobs(args.models, args.scenarios, args.seeds, args.repeats, args.out, args.warmup_steps)
    campaign = {"schema_version": 1, "evaluation_split": "heldout_text", "created_at": datetime.now(timezone.utc).isoformat(),
                "claim_scope": "locally authored natural text; unseen exact lengths/batch sizes; same RTX 4090",
                "training_policy_inputs": {str(path.relative_to(ROOT)): sha(path) for path in
                                           (ROOT / "demo_data/dispatch_paged_cold.csv", ROOT / "models/rtx4090.json")},
                "policy_order": "rotate_each_repeat", "driver_sha256": sha(__file__),
                "warmup_protocol": "full_scenario" if args.warmup_steps is None else "partial_scenario",
                "jobs": jobs}
    if args.plan_only:
        print(json.dumps(campaign, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    campaign_path = args.out / "campaign.json"

    def save():
        temporary = campaign_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(campaign, indent=2) + "\n")
        temporary.replace(campaign_path)

    if campaign_path.exists():
        previous = json.loads(campaign_path.read_text())
        identity_keys = ("id", "model", "scenario_sha256", "seed", "repeats", "policy_specs", "warmup_runs", "warmup_steps", "kv_bytes")
        old_identity = [{k: job.get(k) for k in identity_keys}
                        for job in previous["jobs"]]
        identity = [{k: job.get(k) for k in identity_keys}
                    for job in jobs]
        if old_identity != identity or previous["training_policy_inputs"] != campaign["training_policy_inputs"]:
            raise SystemExit("existing campaign has different inputs; choose a new output directory")
        campaign["created_at"] = previous["created_at"]
    save()
    negative = False
    for job in jobs:
        manifest_path = Path(job["out"]) / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") != "complete" or not compatible_manifest(job, manifest):
                raise SystemExit(f"incomplete or incompatible existing run: {manifest_path}")
            print(f"Keeping completed {job['id']}", flush=True)
        else:
            job["status"] = "running"
            save()
            print(f"=== {job['id']} ===", flush=True)
            log_path = args.out / (job["id"].replace("/", "_") + ".log")
            with log_path.open("x") as log:
                process = subprocess.Popen(job["command"], cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            if manifest.get("status") != "complete":
                job.update(status="failed", returncode=returncode)
                save()
                raise SystemExit(f"operational failure in {job['id']}; see {log_path}")
        valid = manifest.get("tokens_equivalent") is True
        job.update(status="complete", tokens_equivalent=valid)
        negative |= not valid
        save()
    print(f"All {len(jobs)} experiments recorded in {args.out}")
    if negative:
        raise SystemExit("Some strict token comparisons failed; results and negative outcomes are preserved")


if __name__ == "__main__":
    main()
