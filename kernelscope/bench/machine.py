"""Measured machine parameters for the surrogate performance model (machines/<gpu>.json).

Everything the model treats as a machine constant is measured here, except the per-compute-
capability occupancy limits that no API reports (kprofile.MAX_BLOCKS_PER_SM).
"""
import subprocess
import time

from kernelscope.backends.realhw.kprofile import props_from_torch
from kernelscope.bench.stream import stream_gbps


def hit_fraction(gbps: float, dram_gbps: float, l2_gbps: float) -> float:
    """Share of bytes served by L2 implied by an achieved bandwidth: 1/bw = h/l2 + (1-h)/dram."""
    h = (1 / gbps - 1 / dram_gbps) / (1 / l2_gbps - 1 / dram_gbps)
    return min(1.0, max(0.0, h))


def _clocks() -> tuple[float, float]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.max.sm,clocks.max.mem", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.splitlines()[0]
    sm, mem = (float(x) for x in out.split(","))
    return sm, mem


def _tc_tflops(device) -> float:
    from kernelscope.bench.ceilings import measure_ceilings
    return measure_ceilings(device=device)["fp16_matmul_tflops"]


def measure_machine(device="cuda", *, l2_curve_mib=(16, 32, 48, 64, 72, 80, 96, 128, 192, 256, 512),
                    stream=stream_gbps, tc_tflops=None, clocks=None, placement=None) -> dict:
    p = props_from_torch(device)
    n = p["num_sms"]
    dram = max(stream(1 << 30, G) for G in (2 * n, 4 * n))
    l2 = max(stream(p["l2_bytes"] // 2, G) for G in (n, 2 * n))
    cta_dram = stream(1 << 30, 1, target_bytes=256 << 20)
    cta_l2 = stream(8 << 20, 1, target_bytes=256 << 20)
    curve = []
    for mib in l2_curve_mib:
        g = stream(mib << 20, 2 * n)
        curve.append({"mib": mib, "gbps": g, "hit": hit_fraction(g, dram, l2)})
    sm_clock, mem_clock = (clocks or _clocks)()
    try:
        import torch
        prov = {"torch": torch.__version__, "cuda": torch.version.cuda}
    except ImportError:
        prov = {}
    return {
        "name": p["name"], "cc": p["cc"], "n_sm": n, "max_threads_sm": p["max_threads_per_sm"],
        "max_ctas_sm": p["max_blocks_per_sm"], "regs_sm": p["regs_per_sm"], "smem_sm": p["smem_per_sm"],
        "reserved_smem_per_block": p["reserved_smem_per_block"], "l2_bytes": p["l2_bytes"],
        "dram_gbps": dram, "l2_gbps": l2, "cta_dram_gbps": cta_dram, "cta_l2_gbps": cta_l2,
        "l2_hit_curve": curve,
        "tc_tflops": (tc_tflops or (lambda: _tc_tflops(device)))(),
        "clock_mhz": sm_clock, "mem_clock_mhz": mem_clock,
        "block_placement": placement() if placement else None,
        "provenance": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), **prov},
    }
