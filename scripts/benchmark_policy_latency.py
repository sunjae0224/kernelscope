"""Paired CPU policy latency, exact ranking parity, and explicit native setup cost.

Old campaign shapes are read only for this benchmark. Each timed cold decision
uses a fresh ModelPolicy, with no workload preloading or reuse across cases.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import cProfile
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from kernelscope.model.machine import MachineSpec
from kernelscope.model.params import ModelParams
from kernelscope.model.simulate import prepare_simulator
from kernelscope.serve.dispatch import ModelPolicy


@contextmanager
def _backend(name):
    prior = os.environ.get("KERNELSCOPE_SIMULATOR")
    os.environ["KERNELSCOPE_SIMULATOR"] = name
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("KERNELSCOPE_SIMULATOR", None)
        else:
            os.environ["KERNELSCOPE_SIMULATOR"] = prior


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cases(root):
    import pandas as pd
    cases = []
    for scenario in ("uniform", "ragged", "arrivals"):
        source = root / scenario / "heuristic/repeat_000/steps.parquet"
        frame = pd.read_parquet(source)
        seen = set()
        for row in frame.itertuples():
            lens = json.loads(row.lens)
            key = tuple((n + 255) // 256 for n in lens)
            if key in seen:
                continue
            seen.add(key)
            cases.append({"name": f"{scenario}_step{row.step}", "scenario": scenario,
                          "step": int(row.step), "lens": lens, "n_heads": 32, "n_kv_heads": 8,
                          "source": str(source), "source_sha256": _hash(source)})
    return cases


def _summary(values):
    return {"median": float(statistics.median(values)), "min": min(values), "max": max(values),
            "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9)),
            "samples": values}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="machines/rtx4090.json")
    parser.add_argument("--params", default="models/rtx4090.json")
    parser.add_argument("--run-root", type=Path, default=Path("demo_data/serve_4090/graduation_20260922"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hot-calls", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/policy-latency.json"))
    parser.add_argument("--setup-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.setup_only:
        print(json.dumps(prepare_simulator("native")))
        return 0
    if args.repeats < 1 or args.hot_calls < 1:
        parser.error("repeats and hot-calls must be positive")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")

    # A fresh child and temporary cache measure one-time compilation without
    # deleting the real cache or including that setup in later choose timings.
    with tempfile.TemporaryDirectory(prefix="kernelscope-build-probe-") as temporary:
        env = {**os.environ, "KERNELSCOPE_SIMULATOR_CACHE": temporary}
        setup = subprocess.run([sys.executable, "-m", "scripts.benchmark_policy_latency", "--setup-only"],
                               env=env, capture_output=True, text=True, check=True)
        build_setup = json.loads(setup.stdout)
        build_setup["cache_is_ephemeral_probe"] = True
    cached_setup = prepare_simulator("native")
    machine = MachineSpec.from_json(args.machine)
    params = ModelParams.from_json(args.params)
    cases = _cases(args.run_root)
    rows = []
    for case in cases:
        timings = {name: {"cold_us": [], "hot_us": [], "predictions": []} for name in ("python", "native")}
        predictions = {}
        for repeat in range(args.repeats):
            order = ("python", "native") if repeat % 2 == 0 else ("native", "python")
            for backend in order:
                with _backend(backend):
                    policy = ModelPolicy(machine, params, cache_state="cold")
                    before = time.perf_counter_ns()
                    chosen = policy.choose(case["lens"], case["n_heads"], case["n_kv_heads"])
                    elapsed = (time.perf_counter_ns() - before) / 1000
                    assert not policy.last_cache_hit
                    timings[backend]["cold_us"].append(elapsed)
                    predictions[backend] = {"splits": chosen, "ranked": list(policy.last)}
                    before = time.perf_counter_ns()
                    for _ in range(args.hot_calls):
                        assert policy.choose(case["lens"], case["n_heads"], case["n_kv_heads"]) == chosen
                    timings[backend]["hot_us"].append((time.perf_counter_ns() - before) / 1000 / args.hot_calls)
                    assert policy.last_cache_hit
                    timings[backend]["predictions"].append(policy.predictions)
        left, right = predictions["python"], predictions["native"]
        names_equal = [p[0] for p in left["ranked"]] == [p[0] for p in right["ranked"]]
        left_by_name, right_by_name = dict(left["ranked"]), dict(right["ranked"])
        errors = [abs(left_by_name[n] - right_by_name[n]) for n in left_by_name]
        times_equal = bool(np.allclose([left_by_name[n] for n in left_by_name],
                                      [right_by_name[n] for n in left_by_name], rtol=1e-12, atol=1e-9))
        row = {**case, "rankings": predictions, "selected_splits_identical": left["splits"] == right["splits"],
               "ranking_order_identical": names_equal, "predictions_within_tolerance": times_equal,
               "max_abs_prediction_diff_us": max(errors),
               "timing": {name: {"cold_us": _summary(data["cold_us"]), "hot_us": _summary(data["hot_us"]),
                                 "prediction_counts": data["predictions"]} for name, data in timings.items()}}
        row["cold_decision_speedup"] = row["timing"]["python"]["cold_us"]["median"] / row["timing"]["native"]["cold_us"]["median"]
        rows.append(row)
        print(f"{case['name']}: splits={right['splits']}, reference={row['timing']['python']['cold_us']['median']/1000:.3f}ms, "
              f"native={row['timing']['native']['cold_us']['median']/1000:.3f}ms, speedup={row['cold_decision_speedup']:.1f}x", flush=True)

    # Retain the profiled bottleneck as a reviewable artifact, separately from
    # the unprofiled paired timings above.
    profile_case = next(c for c in cases if c["scenario"] == "ragged")
    profiler = cProfile.Profile()
    with _backend("python"):
        policy = ModelPolicy(machine, params)
        profiler.runcall(policy.choose, profile_case["lens"], 32, 8)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(25)
    sources = [Path("kernelscope/model/simulate.py"), Path("kernelscope/model/_simulate_native.c"),
               Path("kernelscope/model/_simulate_backend.py"), Path("kernelscope/serve/dispatch.py"), Path(__file__)]
    report = {"evidence_kind": "cpu_policy_latency", "measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "backend_setup_fresh_build": build_setup, "backend_setup_existing_cache": cached_setup,
              "machine_sha256": _hash(args.machine), "model_params_sha256": _hash(args.params),
              "source_sha256": {str(p): _hash(p) for p in sources},
              "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
              "load_average_at_end": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
              "repeats": args.repeats, "hot_calls_per_sample": args.hot_calls,
              "protocol": "Fresh policy per cold choice; backend setup/constructor excluded and disclosed separately; hot latency is mean of repeated hits; backend order alternates each repeat. Inputs are old campaign page configurations, never held-out future scenarios.",
              "tolerance": {"relative": 1e-12, "absolute_us": 1e-9},
              "all_selected_splits_identical": all(r["selected_splits_identical"] for r in rows),
              "all_rankings_identical": all(r["ranking_order_identical"] for r in rows),
              "all_predictions_within_tolerance": all(r["predictions_within_tolerance"] for r in rows),
              "cases": rows, "reference_profile": stream.getvalue(),
              "limitations": "CPU decision latency only, not serving speedup. Optional local C compiler; unavailable compiler uses explicit NumPy fallback. Existing page-key cache semantics are unchanged."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.out}; fresh build {build_setup['setup_us']/1000:.2f}ms; cached load {cached_setup['setup_us']/1000:.2f}ms")
    return 0 if (report["all_selected_splits_identical"] and report["all_rankings_identical"] and report["all_predictions_within_tolerance"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
