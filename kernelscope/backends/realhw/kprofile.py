"""Kernel-level timing and launch geometry from torch.profiler (CUPTI *activity* API).

This needs no hardware-counter session, so it works next to DCGM where ncu does
not. The chrome trace carries per-kernel ``dur`` (µs), ``grid``, ``block``,
``registers per thread`` and ``shared memory``, which is enough to place a
kernel on the occupancy side of the story without any counter.
"""
import json
import math
import re
import statistics
from pathlib import Path

# Per-compute-capability limits torch does not expose (CUDA C Programming Guide,
# "Technical Specifications per Compute Capability").
MAX_BLOCKS_PER_SM = {(7, 0): 32, (7, 5): 16, (8, 0): 32, (8, 6): 16, (8, 7): 16, (8, 9): 24, (9, 0): 32}
SMEM_PER_SM = {(7, 0): 98304, (7, 5): 65536, (8, 0): 167936, (8, 6): 102400, (8, 7): 167936,
               (8, 9): 102400, (9, 0): 233472}
REG_ALLOC_UNIT = 256          # registers are allocated per warp in units of 256 (CC 7.x-9.x)

A100_PROPS = {
    "name": "NVIDIA A100-SXM4-80GB", "cc": "8.0", "num_sms": 108, "max_threads_per_sm": 2048,
    "regs_per_sm": 65536, "smem_per_sm": 167936, "max_blocks_per_sm": 32,
    "reserved_smem_per_block": 1024, "warp_size": 32, "max_warps_per_sm": 64, "l2_bytes": 41943040,
}
RTX4090_PROPS = {
    "name": "NVIDIA GeForce RTX 4090", "cc": "8.9", "num_sms": 128, "max_threads_per_sm": 1536,
    "regs_per_sm": 65536, "smem_per_sm": 102400, "max_blocks_per_sm": 24,
    "reserved_smem_per_block": 1024, "warp_size": 32, "max_warps_per_sm": 48, "l2_bytes": 75497472,
}


def props_from_torch(device="cuda") -> dict:
    """Occupancy-relevant limits of the current device. Raises if CUDA is unavailable."""
    import torch
    p = torch.cuda.get_device_properties(device)
    cc = (p.major, p.minor)
    return {
        "name": p.name,
        "cc": f"{p.major}.{p.minor}",
        "num_sms": p.multi_processor_count,
        "max_threads_per_sm": p.max_threads_per_multi_processor,
        "regs_per_sm": getattr(p, "regs_per_multiprocessor", 65536),
        "smem_per_sm": getattr(p, "shared_memory_per_multiprocessor", SMEM_PER_SM.get(cc, 65536)),
        "max_blocks_per_sm": MAX_BLOCKS_PER_SM.get(cc, 16),
        "reserved_smem_per_block": 1024 if p.major >= 8 else 0,
        "warp_size": p.warp_size,
        "max_warps_per_sm": p.max_threads_per_multi_processor // p.warp_size,
        "l2_bytes": p.L2_cache_size,
    }


def kernel_events_from_chrome_trace(src) -> list[dict]:
    obj = src if isinstance(src, dict) else json.loads(Path(src).read_text())
    out = []
    for e in obj.get("traceEvents", []):
        if e.get("cat") != "kernel":
            continue
        a = e.get("args", {})
        out.append({
            "name": e["name"], "dur_us": float(e["dur"]), "ts": e["ts"],
            "grid": tuple(a.get("grid", (0, 0, 0))), "block": tuple(a.get("block", (0, 0, 0))),
            "regs": a.get("registers per thread"), "smem_bytes": a.get("shared memory"),
        })
    out.sort(key=lambda e: e["ts"])
    return out


def summarize_launches(events: list[dict], kernel_regex: str | None, iters: int, marker_regex: str | None = None) -> dict:
    """Per-launch median duration and geometry of the kernels matching ``kernel_regex``.

    With ``marker_regex`` and marker kernels in the trace, iterations are the matching launches
    between consecutive markers; iterations whose launch count differs from the most common one
    (the profiler dropped an event) are discarded and the last ``iters`` good ones summarised.
    Marker kernels are never counted. Without markers the trace must hold exactly ``iters``
    iterations (legacy behaviour).
    """
    rx = re.compile(kernel_regex) if kernel_regex else None
    mx = re.compile(marker_regex) if marker_regex else None

    def is_marker(e):
        return bool(mx and mx.search(e["name"]))

    def matches(e):
        return bool(rx and rx.search(e["name"])) and not is_marker(e)

    unmatched = list(dict.fromkeys(e["name"] for e in events if not is_marker(e) and not matches(e)))
    if mx is not None and any(is_marker(e) for e in events):
        segments, cur = [], None
        for e in events:
            if is_marker(e):
                if cur is not None:
                    segments.append(cur)
                cur = []
            elif cur is not None and matches(e):
                cur.append(e)
        if cur is not None:
            segments.append(cur)
        per = statistics.mode(len(s) for s in segments)
        good = [s for s in segments if len(s) == per]
        used = good[-iters:]
        out = _launch_summary(used, per, unmatched)
        out.update(iterations_used=len(used), iterations_dropped=len(segments) - len(good))
        return out
    matching = [e for e in events if matches(e)]
    n = len(matching)
    if n % iters:
        raise ValueError(f"{n} matching launches is not a multiple of iters={iters}; "
                         f"launch count varies between iterations?")
    per = n // iters
    return _launch_summary([matching[k * per:(k + 1) * per] for k in range(iters)], per, unmatched)


def _launch_summary(iterations: list[list[dict]], per: int, unmatched: list[str]) -> dict:
    if per == 0 or not iterations:
        return {"launches_per_iter": 0, "launches": [], "kernel_time_us_median": None, "unmatched": unmatched}
    launches = []
    for idx in range(per):
        evs = [it[idx] for it in iterations]
        launches.append({
            "idx": idx, "name": evs[0]["name"],
            "dur_us_median": statistics.median(e["dur_us"] for e in evs),
            "grid": evs[0]["grid"], "block": evs[0]["block"],
            "regs": evs[0]["regs"], "smem_bytes": evs[0]["smem_bytes"],
        })
    total = statistics.median(sum(e["dur_us"] for e in it) for it in iterations)
    return {"launches_per_iter": per, "launches": launches, "kernel_time_us_median": total, "unmatched": unmatched}


def blocks_per_sm_limit(threads, regs, smem_bytes, props) -> tuple[int, str]:
    """Resident CTAs per SM and the resource that limits it (CUDA occupancy-calculator rules:
    per-warp register allocation in units of 256, 1 KiB shared memory reserved per block)."""
    warp = props["warp_size"]
    warps_per_block = math.ceil(threads / warp)
    by = {"threads": props["max_threads_per_sm"] // threads, "blocks": props["max_blocks_per_sm"]}
    if regs:
        per_warp = math.ceil(regs * warp / REG_ALLOC_UNIT) * REG_ALLOC_UNIT
        by["regs"] = (props["regs_per_sm"] // per_warp) // warps_per_block
    if smem_bytes:
        by["smem"] = props["smem_per_sm"] // (smem_bytes + props.get("reserved_smem_per_block", 0))
    limiter = min(by, key=by.get)
    return int(by[limiter]), limiter


def occupancy_estimate(grid, block, regs, smem_bytes, props=A100_PROPS) -> dict:
    """First-wave occupancy from launch geometry + per-thread resources."""
    blocks = math.prod(grid)
    threads = math.prod(block)
    warps_per_block = math.ceil(threads / props["warp_size"])
    limit, _ = blocks_per_sm_limit(threads, regs, smem_bytes, props)
    num_sms = props["num_sms"]
    sm_coverage = min(1.0, blocks / num_sms)
    blocks_per_active_sm = min(limit, math.ceil(blocks / num_sms))
    warps_per_active_sm = blocks_per_active_sm * warps_per_block
    warps_per_sm_device = min(blocks, limit * num_sms) * warps_per_block / num_sms
    return {
        "blocks": blocks, "threads_per_block": threads, "warps_per_block": warps_per_block,
        "blocks_per_sm_limit": limit, "sm_coverage": sm_coverage,
        "warps_per_active_sm": warps_per_active_sm,
        "occupancy_active_sm": warps_per_active_sm / props["max_warps_per_sm"],
        "warps_per_sm_device": warps_per_sm_device,
        "occupancy_device": warps_per_sm_device / props["max_warps_per_sm"],
    }
