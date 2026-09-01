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

A100_PROPS = {
    "num_sms": 108, "max_threads_per_sm": 2048, "regs_per_sm": 65536,
    "smem_per_sm": 167936, "max_blocks_per_sm": 32, "warp_size": 32, "max_warps_per_sm": 64,
}


def props_from_torch(device="cuda") -> dict:
    try:
        import torch
        p = torch.cuda.get_device_properties(device)
        return {
            "num_sms": p.multi_processor_count,
            "max_threads_per_sm": p.max_threads_per_multi_processor,
            "regs_per_sm": getattr(p, "regs_per_multiprocessor", A100_PROPS["regs_per_sm"]),
            "smem_per_sm": getattr(p, "shared_memory_per_multiprocessor", A100_PROPS["smem_per_sm"]),
            "max_blocks_per_sm": A100_PROPS["max_blocks_per_sm"],
            "warp_size": p.warp_size,
            "max_warps_per_sm": p.max_threads_per_multi_processor // p.warp_size,
            "name": p.name,
        }
    except Exception:
        return dict(A100_PROPS)


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


def summarize_launches(events: list[dict], kernel_regex: str | None, iters: int) -> dict:
    rx = re.compile(kernel_regex) if kernel_regex else None
    matching = [e for e in events if rx and rx.search(e["name"])]
    unmatched = list(dict.fromkeys(e["name"] for e in events if not (rx and rx.search(e["name"]))))
    n = len(matching)
    if n % iters:
        raise ValueError(f"{n} matching launches is not a multiple of iters={iters}; "
                         f"launch count varies between iterations?")
    per = n // iters
    launches = []
    for idx in range(per):
        evs = [matching[k * per + idx] for k in range(iters)]
        launches.append({
            "idx": idx, "name": evs[0]["name"],
            "dur_us_median": statistics.median(e["dur_us"] for e in evs),
            "grid": evs[0]["grid"], "block": evs[0]["block"],
            "regs": evs[0]["regs"], "smem_bytes": evs[0]["smem_bytes"],
        })
    total = None
    if per:
        total = statistics.median(sum(e["dur_us"] for e in matching[k * per:(k + 1) * per]) for k in range(iters))
    return {"launches_per_iter": per, "launches": launches, "kernel_time_us_median": total, "unmatched": unmatched}


def occupancy_estimate(grid, block, regs, smem_bytes, props=A100_PROPS) -> dict:
    """First-wave occupancy from launch geometry + per-thread resources.

    Ignores register-allocation granularity, so ``blocks_per_sm_limit`` can be
    one too high in edge cases; the profiler's own "warps per SM" agrees with
    ``warps_per_sm_device`` for the kernels checked so far.
    """
    blocks = math.prod(grid)
    threads = math.prod(block)
    warps_per_block = math.ceil(threads / props["warp_size"])
    by_threads = props["max_threads_per_sm"] // threads
    by_regs = props["regs_per_sm"] // (regs * threads) if regs else math.inf
    by_smem = props["smem_per_sm"] // smem_bytes if smem_bytes else math.inf
    limit = int(min(by_threads, by_regs, by_smem, props["max_blocks_per_sm"]))
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
