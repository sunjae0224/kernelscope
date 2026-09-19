import json
from types import SimpleNamespace

import pytest

from kernelscope.backends.realhw.kprofile import (
    A100_PROPS, RTX4090_PROPS, blocks_per_sm_limit, kernel_events_from_chrome_trace,
    occupancy_estimate, props_from_torch, summarize_launches,
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


def test_splitkv_kernel_fits_once_per_sm_on_ada_because_of_shared_memory():
    # flash_fwd_splitkv_kernel, d=128, measured on the 4090: 128 threads, 244 regs, 80 KiB smem
    assert blocks_per_sm_limit(128, 244, 81920, RTX4090_PROPS) == (1, "smem")


def test_splitkv_kernel_fits_twice_per_sm_on_a100():
    assert blocks_per_sm_limit(128, 244, 81920, A100_PROPS)[0] == 2


def test_fa2_decode_kernel_is_register_limited_to_two_per_sm_on_ada():
    # 255 regs * 32 lanes = 8160 -> 8192 per warp -> 8 warps/SM -> 2 blocks of 4 warps
    assert blocks_per_sm_limit(128, 255, 49152, RTX4090_PROPS) == (2, "regs")


def test_combine_kernel_limit_matches_the_profiler_on_ada():
    assert blocks_per_sm_limit(128, 52, 160, RTX4090_PROPS) == (9, "regs")


def test_register_allocation_rounds_up_to_256_per_warp():
    # 44 regs * 32 = 1408 -> 1536 per warp -> 42 warps -> 21 blocks of 2 warps (23 without rounding)
    assert blocks_per_sm_limit(64, 44, 0, RTX4090_PROPS) == (21, "regs")


def test_cta_cap_limits_tiny_blocks():
    assert blocks_per_sm_limit(32, 16, 0, RTX4090_PROPS) == (24, "blocks")


def _fake_props(**drop):
    p = dict(name="NVIDIA GeForce RTX 4090", major=8, minor=9, multi_processor_count=128,
             max_threads_per_multi_processor=1536, regs_per_multiprocessor=65536,
             shared_memory_per_multiprocessor=102400, warp_size=32, L2_cache_size=75497472)
    for k in drop:
        p.pop(k)
    return SimpleNamespace(**p)


def test_props_from_torch_reads_the_device_and_takes_the_cta_cap_from_the_compute_capability(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: _fake_props())
    assert props_from_torch("cuda") == RTX4090_PROPS


def test_props_from_torch_falls_back_to_the_cc_table_for_missing_fields(monkeypatch):
    import torch
    fake = _fake_props(regs_per_multiprocessor=1, shared_memory_per_multiprocessor=1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: fake)
    p = props_from_torch("cuda")
    assert p["smem_per_sm"] == 102400
    assert p["regs_per_sm"] == 65536


def _ev(name, dur, ts):
    return {"name": name, "dur_us": dur, "ts": ts, "grid": (1, 1, 1), "block": (128, 1, 1), "regs": 32, "smem_bytes": 0}


MARK = "kernelscope_iter_marker"
FLUSH = "kernelscope_l2_flush"
MRX = r"kernelscope_(l2_flush|iter_marker)"


def _iterations(durs, marker=MARK, t0=0):
    """One marker then (main, combine) per iteration; durs = [(main, combine), ...]."""
    ev, t = [], t0
    for a, b in durs:
        ev += [_ev(marker, 0.5, t), _ev("flash_fwd_splitkv_kernel", a, t + 1), _ev("flash_fwd_splitkv_combine_kernel", b, t + 2)]
        t += 10
    return ev


def test_segmented_summary_uses_the_last_iters_complete_iterations():
    ev = _iterations([(100, 9), (100, 9), (10, 1), (12, 2), (14, 3)])   # 2 padding + 3 measured
    s = summarize_launches(ev, "flash_fwd", iters=3, marker_regex=MRX)
    assert s["launches_per_iter"] == 2
    assert s["launches"][0]["dur_us_median"] == 12
    assert s["launches"][1]["dur_us_median"] == 2
    assert s["kernel_time_us_median"] == 14
    assert s["iterations_used"] == 3 and s["iterations_dropped"] == 0
    assert s["unmatched"] == []


def test_segmented_summary_drops_iterations_with_missing_events():
    ev = _iterations([(10, 1), (12, 2), (14, 3), (16, 4)])
    del ev[4]                                       # profiler lost iteration 2's main kernel
    s = summarize_launches(ev, "flash_fwd", iters=3, marker_regex=MRX)
    assert s["iterations_dropped"] == 1
    assert s["iterations_used"] == 3
    assert s["kernel_time_us_median"] == 17         # sums 11, 17, 20 -> median 17


def test_flush_kernels_are_never_counted_even_by_a_catch_all_regex():
    ev = _iterations([(10, 1), (12, 2)], marker=FLUSH)
    s = summarize_launches(ev, ".*", iters=2, marker_regex=MRX)
    assert s["launches_per_iter"] == 2
    assert all("kernelscope" not in l["name"] for l in s["launches"])


def test_launches_before_the_first_marker_are_ignored():
    ev = [_ev("flash_fwd_splitkv_kernel", 999, -5)] + _iterations([(10, 1), (12, 2)])
    s = summarize_launches(ev, "flash_fwd", iters=2, marker_regex=MRX)
    assert s["kernel_time_us_median"] == 12.5


def test_marker_regex_without_markers_in_the_trace_falls_back_to_the_legacy_path():
    ev = kernel_events_from_chrome_trace(TRACE)
    s = summarize_launches(ev, "flash_fwd", iters=2, marker_regex=MRX)
    assert s["kernel_time_us_median"] == 13.5
