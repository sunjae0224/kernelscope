"""Nsight Compute command construction and CSV parsing.

ncu is only used for hardware counters. It locks clocks and serializes/replays
kernels, so its ``gpu__time_duration`` is *not* the latency we report — that
comes from CUDA-event timing in a separate, unprofiled process.
"""
import csv
import io

# Fixed, small metric set: keeps ncu replay passes to a few, and every column
# below has a direct role in the analysis (roofline, occupancy, sim budget).
DEFAULT_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__grid_size",
    "smsp__inst_executed.sum",
]


def build_ncu_command(target, kernel_regex, num_launches, launch_skip=0, metrics=None, ncu_exe="ncu"):
    metrics = list(metrics) if metrics else DEFAULT_METRICS
    exe = [ncu_exe] if isinstance(ncu_exe, str) else list(ncu_exe)
    return [
        *exe,
        "--csv",
        "--metrics", ",".join(metrics),
        "-k", f"regex:{kernel_regex}",
        "--launch-skip", str(launch_skip),
        "--launch-count", str(num_launches),
        *target,
    ]


def parse_ncu_csv(text: str) -> list[dict]:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("==")]
    if not lines:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = []
    for rec in reader:
        rows.append({
            "launch_idx": int(rec["ID"]),
            "kernel_name": rec["Kernel Name"],
            "grid_size": rec["Grid Size"],
            "block_size": rec["Block Size"],
            "metric": rec["Metric Name"],
            "unit": rec["Metric Unit"],
            "value": _to_float(rec["Metric Value"]),
        })
    return rows


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))
