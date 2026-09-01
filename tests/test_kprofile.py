import json

import pytest

from kernelscope.backends.realhw.kprofile import (
    A100_PROPS, kernel_events_from_chrome_trace, occupancy_estimate, summarize_launches,
)


def _kernel(name, dur, ts, grid, block, regs, smem):
    return {"cat": "kernel", "name": name, "dur": dur, "ts": ts,
            "args": {"grid": grid, "block": block, "registers per thread": regs, "shared memory": smem}}


TRACE = {"traceEvents": [
    {"cat": "cpu_op", "name": "aten::empty", "dur": 5, "ts": 0},
    _kernel("void flash::flash_fwd_splitkv_kernel<...>", 10.0, 100, [1, 8, 8], [128, 1, 1], 244, 81920),
    _kernel("void flash::flash_fwd_splitkv_combine_kernel<...>", 2.0, 120, [8, 1, 1], [128, 1, 1], 52, 160),
    _kernel("void at::native::fill<...>", 1.0, 130, [1, 1, 1], [128, 1, 1], 16, 0),
    _kernel("void flash::flash_fwd_splitkv_kernel<...>", 12.0, 200, [1, 8, 8], [128, 1, 1], 244, 81920),
    _kernel("void flash::flash_fwd_splitkv_combine_kernel<...>", 3.0, 220, [8, 1, 1], [128, 1, 1], 52, 160),
    _kernel("void at::native::fill<...>", 1.0, 230, [1, 1, 1], [128, 1, 1], 16, 0),
]}


def test_kernel_events_are_extracted_in_time_order_with_launch_geometry(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(TRACE))
    ev = kernel_events_from_chrome_trace(p)
    assert [e["name"][:20] for e in ev][:3] == ["void flash::flash_fw", "void flash::flash_fw", "void at::native::fil"]
    assert ev[0] == {"name": "void flash::flash_fwd_splitkv_kernel<...>", "dur_us": 10.0, "ts": 100,
                     "grid": (1, 8, 8), "block": (128, 1, 1), "regs": 244, "smem_bytes": 81920}


def test_summarize_groups_matching_launches_per_iteration():
    ev = kernel_events_from_chrome_trace(TRACE)
    s = summarize_launches(ev, kernel_regex="flash_fwd", iters=2)
    assert s["launches_per_iter"] == 2
    assert [l["idx"] for l in s["launches"]] == [0, 1]
    assert s["launches"][0]["dur_us_median"] == 11.0
    assert s["launches"][1]["dur_us_median"] == 2.5
    assert s["launches"][0]["grid"] == (1, 8, 8)
    assert s["kernel_time_us_median"] == 13.5           # per-iteration sum: 12, 15 -> median
    assert s["unmatched"] == ["void at::native::fill<...>"]


def test_summarize_with_no_match_reports_zero_launches():
    ev = kernel_events_from_chrome_trace(TRACE)
    s = summarize_launches(ev, kernel_regex="nothing_here", iters=2)
    assert s["launches_per_iter"] == 0
    assert s["launches"] == []
    assert s["kernel_time_us_median"] is None


def test_summarize_rejects_launch_count_not_divisible_by_iters():
    ev = kernel_events_from_chrome_trace(TRACE)
    with pytest.raises(ValueError, match="iters"):
        summarize_launches(ev, kernel_regex="flash_fwd", iters=3)


def test_occupancy_estimate_matches_profiler_warps_per_sm_for_flashdecoding():
    o = occupancy_estimate(grid=(1, 8, 8), block=(128, 1, 1), regs=244, smem_bytes=81920, props=A100_PROPS)
    assert o["blocks"] == 64
    assert o["warps_per_block"] == 4
    assert o["blocks_per_sm_limit"] == 2                 # regs: 65536/(244*128)=2.09, smem: 167936/81920=2.05
    assert o["sm_coverage"] == pytest.approx(64 / 108)
    assert o["warps_per_active_sm"] == 4
    assert o["occupancy_active_sm"] == pytest.approx(4 / 64)
    assert o["warps_per_sm_device"] == pytest.approx(2.37037, abs=1e-4)   # profiler's "warps per SM"
    assert o["occupancy_device"] == pytest.approx(2.37037 / 64, abs=1e-5)


def test_occupancy_estimate_for_a_grid_larger_than_the_gpu():
    o = occupancy_estimate(grid=(4096, 1, 1), block=(256, 1, 1), regs=32, smem_bytes=0, props=A100_PROPS)
    assert o["sm_coverage"] == 1.0
    assert o["blocks_per_sm_limit"] == 8                 # threads: 2048/256 = 8 (regs 65536/8192 = 8, smem inf)
    assert o["warps_per_active_sm"] == 64                # 8 blocks x 8 warps saturates the SM
    assert o["occupancy_active_sm"] == 1.0
