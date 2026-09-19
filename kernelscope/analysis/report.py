"""One wide row per (kernel, workload) joining every track that produced data.

Columns (present when the track ran): latency_us, kernel_time_us, launches,
grid_blocks / sm_coverage / occupancy_device (launch 0), achieved_gbps,
achieved_tflops, dram_util, tc_util, sim_cycles_<variant>, sim_us_base,
sim_vs_kernel_time, sens_<variant> = cycles_variant / cycles_base - 1.
"""
import pandas as pd

CLOCK_MHZ_A100 = 1410.0
ARCH_CLOCK_MHZ = {"SM80_A100": CLOCK_MHZ_A100, "SM89_RTX4090": 2520.0}
KEY = ["kernel", "workload_key", "cache_state"]
_SENS_THRESHOLD = 0.05
_RESOURCE = {"bw": "bandwidth-bound", "sm": "parallelism-bound", "l2": "L2-capacity-bound"}


def _with_cache_state(df: pd.DataFrame, assume: str) -> pd.DataFrame:
    """Hardware rows carry the cache state they were measured in (rows written before
    --cache-state existed get ``assume``); a simulator replay always starts cold."""
    df = df.copy()
    if "cache_state" not in df.columns:
        df["cache_state"] = None
    df["cache_state"] = df["cache_state"].astype(object)
    df.loc[df.backend.str.startswith("sim:"), "cache_state"] = "cold"
    df["cache_state"] = df["cache_state"].fillna(assume)
    return df


def _sim_clock(df: pd.DataFrame, clock_mhz) -> float | None:
    if clock_mhz:
        return float(clock_mhz)
    sim = df[df.backend.str.startswith("sim:")]
    if sim.empty:
        return None
    if "clock_mhz" in sim.columns and sim["clock_mhz"].notna().all() and sim["clock_mhz"].nunique() == 1:
        return float(sim["clock_mhz"].iloc[0])
    archs = set(sim["arch"].dropna()) if "arch" in sim.columns else set()
    if len(archs) == 1 and next(iter(archs)) in ARCH_CLOCK_MHZ:
        return ARCH_CLOCK_MHZ[next(iter(archs))]
    raise ValueError(f"cannot infer the simulator core clock (arch={sorted(archs) or 'unknown'}); pass --clock-mhz")


def _pick(df, backend, metric, launch_idx=None):
    sub = df[(df.backend == backend) & (df.metric == metric)]
    if launch_idx is not None:
        sub = sub[sub.launch_idx == launch_idx]
    return sub.groupby(KEY)["value"].median()


def summarize(df: pd.DataFrame, clock_mhz: float = None, assume_cache_state: str = "warm") -> pd.DataFrame:
    df = _with_cache_state(df, assume_cache_state)
    clock = _sim_clock(df, clock_mhz)
    cols = {
        "latency_us": _pick(df, "latency", "median_s") * 1e6,
        "kernel_time_us": _pick(df, "profile", "kernel_time_us"),
        "launches": _pick(df, "profile", "launches_per_iter"),
        "grid_blocks": _pick(df, "profile", "grid_blocks", 0),
        "sm_coverage": _pick(df, "profile", "sm_coverage", 0),
        "occupancy_device": _pick(df, "profile", "occupancy_device", 0),
        "achieved_gbps": _pick(df, "analytic", "achieved_gbps"),
        "achieved_tflops": _pick(df, "analytic", "achieved_tflops"),
        "dram_util": _pick(df, "analytic", "dram_util"),
        "tc_util": _pick(df, "analytic", "tc_util"),
    }
    variants = sorted({b[4:] for b in df.backend.unique() if b.startswith("sim:")})
    for v in variants:
        cols[f"sim_cycles_{v}"] = _pick(df, f"sim:{v}", "gpu_tot_sim_cycle")
    out = pd.DataFrame(cols)
    out = out.dropna(axis=1, how="all")
    out.index.names = KEY
    if "sim_cycles_base" in out.columns:
        out["sim_us_base"] = out["sim_cycles_base"] / clock
        if "kernel_time_us" in out.columns:
            out["sim_vs_kernel_time"] = out["sim_us_base"] / out["kernel_time_us"]
        for v in variants:
            if v != "base" and f"sim_cycles_{v}" in out.columns:
                out[f"sens_{v}"] = out[f"sim_cycles_{v}"] / out["sim_cycles_base"] - 1
    return out


def verdict(row) -> str:
    row = dict(row)
    sens = {k[5:]: v for k, v in row.items() if k.startswith("sens_") and pd.notna(v)}
    if not sens:
        return "no what-if data"
    name, val = max(sens.items(), key=lambda kv: abs(kv[1]))
    label = f"{name}: {round(val * 100):+d}%"
    if abs(val) < _SENS_THRESHOLD:
        cov = row.get("sm_coverage")
        if cov is not None and pd.notna(cov) and cov < 0.5:
            # nothing helps because the launch never occupies the machine in the first place
            return f"starved: grid covers {cov:.0%} of SMs, no resource helps (largest: {label})"
        return f"insensitive (largest: {label})"
    resource = _RESOURCE.get(name.split("_")[0], f"{name.split('_')[0]}-bound")
    return f"{resource} ({label})"


def with_verdicts(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["verdict"] = [verdict(r) for _, r in out.iterrows()]
    return out
