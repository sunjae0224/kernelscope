"""Replay existing scoped traces after a config change, without using the GPU.

Run from the repository root with ``python -m docs.setup.replay_4090 ...``.
The trace/postprocess hooks reuse the exact bytes from a prior successful sweep;
normal AccelSimSweep parsing, variant isolation and ResultStore writes still run.
"""
import argparse
import json
import shutil
from pathlib import Path

from kernelscope.backends.accelsim.paths import AccelSimPaths
from kernelscope.backends.accelsim.sweep import AccelSimSweep
from kernelscope.results.store import ResultStore
from kernelscope.workload import Workload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-results", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--work-dir", default="/home/skkai/accelsim/kernelscope_sim")
    ap.add_argument("--accelsim-root", default="/home/skkai/accelsim/accel-sim-framework")
    ap.add_argument("--sim-jobs", type=int, default=4)
    ap.add_argument("--plugins", help="optional comma-separated subset of source kernels")
    ap.add_argument("--variants", default="base,l2_x2,l2_half,bw_x2,bw_half,sm_x2,sm_half")
    args = ap.parse_args()
    source = ResultStore(args.source_results).load()
    cells = source[(source.backend == "trace") & (source.metric == "warp_insts")]
    if args.plugins:
        selected = set(args.plugins.split(","))
        missing = selected - set(cells.kernel)
        if missing:
            raise SystemExit(f"No source traces for: {sorted(missing)}")
        cells = cells[cells.kernel.isin(selected)]
    if cells.empty or cells.duplicated(["kernel", "workload_key"]).any():
        raise SystemExit("Need exactly one source trace per kernel/workload")
    out = Path(args.results)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Use a new result directory to keep config revisions separate")
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(cells.to_dict("records"), key=lambda r: (
        r["kernel"] == "naive_exec", Workload.from_key(r["workload_key"]).B,
        Workload.from_key(r["workload_key"]).L_kv, r["kernel"]))
    for record in rows:
        original = Path(args.work_dir) / record["run_id"] / f"{record['kernel']}__{record['workload_key']}"
        if not (original / "traces/kernelslist.g").is_file():
            raise SystemExit(f"Missing processed source trace: {original}")

        def reuse_trace(argv, env, cell_dir):
            # Copy rather than hardlink: filtering kernelslist.g must not mutate the source.
            shutil.copytree(original / "traces", cell_dir / "traces")
            (cell_dir / "trace-provenance.json").write_text(json.dumps({
                "source_cell": str(original), "source_results": args.source_results,
                "gpu_execution": False,
            }, indent=2))

        sweep = AccelSimSweep(
            ResultStore(out), AccelSimPaths(args.accelsim_root), args.work_dir,
            arch="SM89_RTX4090", sim_jobs=args.sim_jobs,
            variants=("base",) if record["kernel"] == "naive_exec" else args.variants.split(","),
            trace_fn=reuse_trace, postprocess_fn=lambda *a: None,
        )
        summary = sweep.run_cell(record["kernel"], Workload.from_key(record["workload_key"]))
        summary["source_cell"] = str(original)
        with (out / "summaries.jsonl").open("a") as f:
            f.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
        if summary["status"] != "ok" or any(s != "ok" for s in summary["sim"].values()):
            raise SystemExit("Replay failed; see summaries.jsonl and simulator logs")


if __name__ == "__main__":
    main()
