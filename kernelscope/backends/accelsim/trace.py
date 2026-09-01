"""NVBit tracer plumbing: environment, trace-side stats, simulation-time estimate."""
import os
from pathlib import Path

from kernelscope.backends.accelsim.paths import AccelSimPaths

# Accel-Sim 2.0 paper figure (~27.5K warp-instructions/s); the trace-side
# `total_insts` is warp-level, so est = insts / rate. Planning number, not a promise.
PLANNING_RATE = 27_500


def tracer_env(paths: AccelSimPaths, out_dir, kernel_regex: str, device_index: int = 0,
               base_env=None, window: bool = False) -> dict:
    """Environment for the NVBit tracer. The tool appends its own ``traces/`` under ``TRACES_FOLDER``.

    Range mode (default, for binaries): trace every launch whose *mangled* name matches
    ``kernel_regex`` from the first launch on. tracer_tool.cu uses ``std::regex_match``
    (whole-string), so the search-style regex is wrapped as ``.*(?:re).*``.

    Window mode (``window=True``, for Python plugins): instrumentation starts *off*
    (``NVBIT_INSTRUMENTATION_ENABLED=0``) so JIT compilation and autotuning during warm-up
    are never recorded; `run_kernel --mode trace` flips it on for exactly one run through
    the tracer's exported ``enable_nvbit_instrumentation()`` — the same mechanism as
    Accel-Sim's torch_hook. The regex filter stays active in both modes. (cudaProfilerStart
    from torch does not reach the tracer's ACTIVE_FROM_START=0 path; verified 2026-09-01.)"""
    env = dict(os.environ if base_env is None else base_env)
    env.update({
        "CUDA_INJECTION64_PATH": str(paths.tracer_so),
        "DYNAMIC_KERNEL_RANGE": f"1-@.*(?:{kernel_regex}).*",
        "ACTIVE_FROM_START": "1",
        "NVBIT_INSTRUMENTATION_ENABLED": "0" if window else "1",
        "TERMINATE_UPON_LIMIT": "0",
        "USER_DEFINED_FOLDERS": "1",
        "TRACES_FOLDER": str(out_dir),
        "CUDA_VISIBLE_DEVICES": str(device_index),
    })
    return env


def filter_kernelslist(path, keep_trace_files) -> int:
    """Rewrite ``kernelslist.g`` keeping memcpy lines and only the kernels whose raw trace
    name (``kernel-N-ctx_X.trace.xz`` from stats) is in ``keep_trace_files``. Post-processed
    entries are ``kernel-N-ctx_X.traceg.xz`` / ``.tracez``, so matching is on the stem."""
    stems = {Path(f).name.split(".trace")[0] for f in keep_trace_files}
    kept, out = 0, []
    for line in Path(path).read_text().splitlines():
        if not line.startswith("kernel-"):
            out.append(line)
            continue
        if line.split(".trace")[0] in stems:
            out.append(line)
            kept += 1
    Path(path).write_text("\n".join(out) + ("\n" if out else ""))
    return kept


def read_trace_stats(path) -> list[dict]:
    """``stats_ctx_*``: one row per traced kernel; ``total_insts`` is warp-level."""
    rows = []
    lines = Path(path).read_text().splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        t = [x.strip() for x in line.split(",")]
        rows.append({
            "trace_file": t[0], "kernel_name": t[1],
            "grid": (int(t[2]), int(t[3]), int(t[4])),
            "block": (int(t[6]), int(t[7]), int(t[8])),
            "warp_insts": int(t[10]),
        })
    return rows


def find_trace_stats(cell_dir) -> list[Path]:
    return sorted(Path(cell_dir).glob("traces/stats_ctx_*"))


def find_kernelslist(cell_dir) -> Path | None:
    p = Path(cell_dir) / "traces" / "kernelslist.g"
    return p if p.exists() else None


def estimate_sim_seconds(warp_insts: int, rate: float = PLANNING_RATE) -> float:
    return warp_insts / rate
