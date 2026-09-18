"""kernelscope command line.

  kernelscope list      [--registry M:A]
  kernelscope ceilings  --out machine_ceilings.json                      (measured roofline peaks)
  kernelscope sweep     --grid g.yaml | --workload KEY --plugins a,b --results DIR [--ceilings f] [--ncu cmd]
  kernelscope simsweep  --grid g.yaml | --workload KEY --plugins a,b --results DIR --variants base,bw_x2,...
  kernelscope report    --results DIR [DIR ...] [--out summary.csv]  (joins tracks, adds what-if verdict)
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

import yaml

from kernelscope.backends.accelsim.paths import DEFAULT_ROOT, AccelSimPaths
from kernelscope.backends.accelsim.sweep import AccelSimSweep
from kernelscope.backends.realhw.sweep import RealHWSweep
from kernelscope.results.store import ResultStore
from kernelscope.run_kernel import DEFAULT_REGISTRY, load_registry
from kernelscope.workload import Workload, expand_grid


def load_grid(path) -> list[Workload]:
    with open(path) as f:
        spec = yaml.safe_load(f)
    return expand_grid(spec)


def _cmd_list(args):
    for name in load_registry(args.registry).names():
        print(name)


def _workloads(args) -> list[Workload]:
    if bool(args.grid) == bool(args.workload):
        raise SystemExit(f"{args.cmd}: give exactly one of --grid or --workload")
    return load_grid(args.grid) if args.grid else [Workload.from_key(args.workload)]


def _run_with_log(sweep, args, workloads):
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    with open(results / "summaries.jsonl", "a") as f:
        def log(line):
            f.write(line + "\n")
            f.flush()
            print(line)
        sweep.run_grid(args.plugins.split(","), workloads, log=log)


def _cmd_sweep(args):
    workloads = _workloads(args)
    ncu_cmd = None if args.ncu.lower() == "none" else shlex.split(args.ncu)
    ceilings = json.loads(Path(args.ceilings).read_text()) if args.ceilings else None
    sweep = RealHWSweep(
        store=ResultStore(args.results), python_exe=args.python, registry=args.registry,
        device=args.device, ncu_cmd=ncu_cmd, warmup=args.warmup, iters=args.iters,
        atol=args.atol, timeout_s=args.timeout, ceilings=ceilings,
    )
    _run_with_log(sweep, args, workloads)


def _cmd_report(args):
    import pandas as pd
    from kernelscope.analysis.report import summarize, with_verdicts
    df = pd.concat([ResultStore(r).load() for r in args.results], ignore_index=True)
    if df.empty:
        raise SystemExit("report: no rows found under " + ", ".join(args.results))
    s = with_verdicts(summarize(df, clock_mhz=args.clock_mhz))
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4g}".format):
        print(s.to_string())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        s.reset_index().to_csv(args.out, index=False)


def _cmd_plot(args):
    import pandas as pd
    from kernelscope.analysis.roofline import plot_roofline, roofline_points
    df = pd.concat([ResultStore(r).load() for r in args.results], ignore_index=True)
    pts = roofline_points(df)
    if pts.empty:
        raise SystemExit("plot: no analytic rows with achieved_tflops under " + ", ".join(args.results))
    ceilings = json.loads(Path(args.ceilings).read_text()) if args.ceilings else None
    roof = plot_roofline(pts, ceilings, args.out, title=args.title)
    print(f"wrote {args.out} ({len(pts)} points" + (f", {len(roof)} with roofs)" if roof else ")"))


def _cmd_ceilings(args):
    from kernelscope.bench.ceilings import measure_ceilings
    out = measure_ceilings(device=args.device, copy_bytes=args.copy_bytes,
                           matmul_ns=tuple(int(n) for n in args.matmul_ns.split(",")), iters=args.iters)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def _cmd_simsweep(args):
    from kernelscope.backends.accelsim.trace import PLANNING_RATE
    workloads = _workloads(args)
    sweep = AccelSimSweep(
        store=ResultStore(args.results), paths=AccelSimPaths(args.accelsim_root),
        work_dir=Path(args.work_dir), python_exe=args.python, registry=args.registry,
        device=args.device, device_index=args.device_index, arch=args.arch,
        variants=args.variants.split(","), max_sim_s=args.max_sim_s,
        sim_rate=PLANNING_RATE if args.sim_rate is None else args.sim_rate,
        sim_jobs=args.sim_jobs,
        sim_timeout_s=args.sim_timeout,
    )
    _run_with_log(sweep, args, workloads)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kernelscope")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="print registered plugin names")
    p_list.add_argument("--registry", default=DEFAULT_REGISTRY)
    p_list.set_defaults(func=_cmd_list)

    p_sweep = sub.add_parser("sweep", help="real-hardware sweep: check + latency + ncu per cell")
    p_sweep.add_argument("--grid", help="YAML workload grid (see docs)")
    p_sweep.add_argument("--workload", help="single workload key instead of --grid")
    p_sweep.add_argument("--plugins", required=True, help="comma-separated plugin names")
    p_sweep.add_argument("--results", required=True, help="output directory (parquet + summaries.jsonl)")
    p_sweep.add_argument("--registry", default=DEFAULT_REGISTRY)
    p_sweep.add_argument("--device", default="cuda")
    p_sweep.add_argument("--python", default=sys.executable, help="interpreter for per-cell subprocesses")
    p_sweep.add_argument("--ncu", default="none",
                         help="'none' (default: Nsight-free) or an ncu command for hosts without a DCGM profiling lock")
    p_sweep.add_argument("--ceilings", help="machine_ceilings.json from `kernelscope ceilings` (enables dram_util / tc_util)")
    p_sweep.add_argument("--warmup", type=int, default=10)
    p_sweep.add_argument("--iters", type=int, default=50)
    p_sweep.add_argument("--atol", type=float, default=1e-2)
    p_sweep.add_argument("--timeout", type=int, default=3600, help="per-subprocess timeout (s)")
    p_sweep.set_defaults(func=_cmd_sweep)

    p_sim = sub.add_parser("simsweep", help="Accel-Sim track: trace once per cell, simulate per what-if variant")
    p_sim.add_argument("--grid")
    p_sim.add_argument("--workload")
    p_sim.add_argument("--plugins", required=True)
    p_sim.add_argument("--results", required=True)
    p_sim.add_argument("--accelsim-root", default=DEFAULT_ROOT, help="built accel-sim-framework checkout")
    p_sim.add_argument("--work-dir", default="/var/tmp/uceeeee/kernelscope_sim", help="traces + sim logs (local disk!)")
    p_sim.add_argument("--registry", default=DEFAULT_REGISTRY)
    p_sim.add_argument("--device", default="cuda")
    p_sim.add_argument("--device-index", type=int, default=0, help="CUDA_VISIBLE_DEVICES for the tracer")
    p_sim.add_argument("--arch", default="SM80_A100")
    p_sim.add_argument("--variants", default="base", help="comma-separated: base,l2_x2,l2_half,bw_x2,bw_half,sm_x2,sm_half")
    p_sim.add_argument("--max-sim-s", type=float, default=None, help="skip cells whose estimated sim time exceeds this")
    p_sim.add_argument("--sim-rate", type=float, default=None, help="measured warp instructions/s for the simulation budget (default: 15000 on this host)")
    p_sim.add_argument("--sim-jobs", type=int, default=1, help="concurrent CPU variant replays; tracing remains serialized")
    p_sim.add_argument("--sim-timeout", type=int, default=24 * 3600)
    p_sim.add_argument("--python", default=sys.executable)
    p_sim.set_defaults(func=_cmd_simsweep)

    p_rep = sub.add_parser("report", help="one wide row per (kernel, workload) across all tracks + what-if verdict")
    p_rep.add_argument("--results", nargs="+", required=True, help="one or more result dirs (real-HW and sim can be joined)")
    p_rep.add_argument("--out", help="write the table as CSV")
    p_rep.add_argument("--clock-mhz", type=float, default=1410.0, help="core clock to convert sim cycles to µs")
    p_rep.set_defaults(func=_cmd_report)

    p_plot = sub.add_parser("plot", help="roofline PNG from the analytic track (+ ceilings)")
    p_plot.add_argument("--results", nargs="+", required=True)
    p_plot.add_argument("--ceilings", help="machine_ceilings.json (draws the roofs)")
    p_plot.add_argument("--out", default="results/roofline.png")
    p_plot.add_argument("--title")
    p_plot.set_defaults(func=_cmd_plot)

    p_ceil = sub.add_parser("ceilings", help="measure HBM / fp16-GEMM peaks with torch (no profiler)")
    p_ceil.add_argument("--out", default="results/machine_ceilings.json")
    p_ceil.add_argument("--device", default="cuda")
    p_ceil.add_argument("--copy-bytes", type=int, default=1 << 30, help="keep <= 1 GiB (torch 32-bit indexing)")
    p_ceil.add_argument("--matmul-ns", default="4096,8192", help="comma-separated GEMM sizes; best is the ceiling")
    p_ceil.add_argument("--iters", type=int, default=10)
    p_ceil.set_defaults(func=_cmd_ceilings)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
