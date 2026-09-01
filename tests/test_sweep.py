import json
import sys
from pathlib import Path

import pytest

from kernelscope.backends.realhw.sweep import RealHWSweep, rows_from_ncu
from kernelscope.results.store import ResultStore
from kernelscope.workload import Workload

REG = "tests.fake_plugins:REGISTRY"
W = Workload(phase="decode", B=2, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32")
FAKE_NCU = [sys.executable, str(Path(__file__).parent / "fake_ncu.py")]

FAKE_PROFILE = {
    "launches_per_iter": 2,
    "launches": [
        {"idx": 0, "name": "k_a", "dur_us_median": 10.0, "grid": (1, 8, 8), "block": (128, 1, 1),
         "regs": 244, "smem_bytes": 81920,
         "occupancy": {"blocks": 64, "sm_coverage": 64 / 108, "occupancy_device": 0.037, "warps_per_sm_device": 2.37}},
        {"idx": 1, "name": "k_b", "dur_us_median": 2.5, "grid": (8, 1, 1), "block": (128, 1, 1),
         "regs": 52, "smem_bytes": 160,
         "occupancy": {"blocks": 8, "sm_coverage": 8 / 108, "occupancy_device": 0.005, "warps_per_sm_device": 0.3}},
    ],
    "kernel_time_us_median": 12.5,
    "unmatched": ["void at::native::fill<...>"],
    "device_props": None,
}


def _sweep(tmp_path, **kw):
    kw.setdefault("ncu_cmd", None)
    return RealHWSweep(
        python_exe=sys.executable, registry=REG, device="cpu",
        store=ResultStore(tmp_path / "results"), warmup=2, iters=3, **kw,
    )


def test_rows_from_ncu_tags_every_metric_with_cell_identity():
    parsed = [
        {"launch_idx": 0, "kernel_name": "k<..>", "grid_size": "(32, 1, 1)", "block_size": "(128, 1, 1)",
         "metric": "gpu__time_duration.sum", "unit": "usecond", "value": 12.5},
        {"launch_idx": 1, "kernel_name": "combine", "grid_size": "(1, 1, 1)", "block_size": "(128, 1, 1)",
         "metric": "gpu__time_duration.sum", "unit": "usecond", "value": 1.0},
    ]
    rows = rows_from_ncu(parsed, workload_key=W.key(), plugin="fa2")
    assert len(rows) == 2
    assert rows[0] == {"workload_key": W.key(), "kernel": "fa2", "backend": "ncu",
                       "metric": "gpu__time_duration.sum", "unit": "usecond", "value": 12.5,
                       "launch_idx": 0, "kernel_name": "k<..>", "grid_size": "(32, 1, 1)",
                       "block_size": "(128, 1, 1)"}
    assert rows[1]["launch_idx"] == 1


def test_cell_records_check_latency_profile_and_analytic_rows_without_ncu(tmp_path):
    sweep = _sweep(tmp_path)
    summary = sweep.run_cell("faithful_cpu", W)
    df = sweep.store.load()
    assert summary["status"] == "ok"
    assert summary["check_ok"] is True
    assert summary["launches"] == 0                    # CPU plugin launches no CUDA kernels
    assert summary["ncu"] == "disabled"
    assert {"check", "latency", "profile", "analytic"} <= set(df.backend)
    assert "ncu" not in set(df.backend)
    lat = df[(df.backend == "latency") & (df.metric == "median_s")]
    assert len(lat) == 1 and lat["value"].item() >= 0
    an = df[df.backend == "analytic"].set_index("metric")["value"]
    assert an["total_bytes"] == 1024 + 32768 + 1024   # q + kv + o for W in fp32
    assert an["flops"] == 4 * 2 * 8 * 16 * 64
    assert "achieved_gbps" not in an.index            # no kernel time on CPU -> no rate


def test_profile_summary_is_flattened_into_rows_and_drives_analytic_rates(tmp_path, monkeypatch):
    sweep = _sweep(tmp_path, ceilings={"hbm_copy_gbps": 1500.0, "fp16_matmul_tflops": 250.0})
    monkeypatch.setattr(sweep, "profile", lambda plugin, w: FAKE_PROFILE)
    summary = sweep.run_cell("faithful_cpu", W)
    assert summary["launches"] == 2
    assert summary["kernel_time_us"] == 12.5
    df = sweep.store.load()
    prof = df[df.backend == "profile"]
    assert prof[prof.metric == "kernel_time_us"]["value"].item() == 12.5
    assert prof[prof.metric == "launches_per_iter"]["value"].item() == 2
    l0 = prof[prof.launch_idx == 0].set_index("metric")["value"]
    assert l0["dur_us"] == 10.0 and l0["grid_blocks"] == 64 and l0["block_threads"] == 128
    assert l0["regs"] == 244 and l0["smem_bytes"] == 81920
    assert l0["sm_coverage"] == pytest.approx(64 / 108)
    assert l0["occupancy_device"] == pytest.approx(0.037)
    assert prof[(prof.launch_idx == 0) & (prof.metric == "dur_us")]["note"].item() == "k_a"
    an = df[df.backend == "analytic"].set_index("metric")["value"]
    total = 1024 + 32768 + 1024
    assert an["achieved_gbps"] == pytest.approx(total / 12.5e-6 / 1e9)
    assert an["achieved_tflops"] == pytest.approx(65536 / 12.5e-6 / 1e12)
    assert an["dram_util"] == pytest.approx(an["achieved_gbps"] / 1500.0)
    assert an["tc_util"] == pytest.approx(an["achieved_tflops"] / 250.0)


def test_ncu_is_optional_and_sized_from_profiled_launches(tmp_path, monkeypatch):
    args_file = tmp_path / "ncu_args.json"
    monkeypatch.setenv("FAKE_NCU_ARGS_FILE", str(args_file))
    sweep = _sweep(tmp_path, ncu_cmd=FAKE_NCU)
    monkeypatch.setattr(sweep, "profile", lambda plugin, w: FAKE_PROFILE)
    summary = sweep.run_cell("faithful_cpu", W)
    assert summary["ncu"] == "ok"
    argv = json.loads(args_file.read_text())
    assert argv[argv.index("--launch-skip") + 1] == str(2 * 2)   # warmup * launches
    assert argv[argv.index("--launch-count") + 1] == "2"
    assert argv[argv.index("-k") + 1] == "regex:faithful"
    df = sweep.store.load()
    assert set(df[df.backend == "ncu"].metric) == {"gpu__time_duration.sum", "launch__grid_size"}


def test_ncu_is_skipped_when_no_launch_matches(tmp_path):
    sweep = _sweep(tmp_path, ncu_cmd=FAKE_NCU)
    summary = sweep.run_cell("faithful_cpu", W)
    assert summary["ncu"] == "skipped: kernel_regex matched 0 launches"


def test_ncu_failure_message_includes_ncu_stdout(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_NCU_FAIL", "1")
    sweep = _sweep(tmp_path, ncu_cmd=FAKE_NCU)
    monkeypatch.setattr(sweep, "profile", lambda plugin, w: FAKE_PROFILE)
    summary = sweep.run_cell("faithful_cpu", W)
    assert summary["status"] == "error"
    assert "exit 9" in summary["error"]
    assert "driver resource was unavailable" in summary["error"]
    df = sweep.store.load()
    assert {"check", "latency", "profile", "analytic"} <= set(df.backend)   # earlier measurements are kept


def test_gpu_contention_is_recorded_on_every_row_and_in_summaries(tmp_path, monkeypatch):
    import kernelscope.backends.realhw.sweep as sweep_mod
    monkeypatch.setattr(sweep_mod, "visible_device_index", lambda device: 1)
    monkeypatch.setattr(sweep_mod, "gpu_contention", lambda index, **kw: {
        "busy": True, "index": index, "utilization_pct": 100, "other_pids": [436356],
        "note": "GPU 1 busy: 100% util, other pids [436356]"})
    sweep = _sweep(tmp_path)
    assert sweep.contention["busy"] is True
    s = sweep.run_cell("faithful_cpu", W)
    assert s["gpu_contention"] == "GPU 1 busy: 100% util, other pids [436356]"
    df = sweep.store.load()
    assert (df["gpu_util_at_start"] == 100).all()
    assert (df["other_pids_at_start"] == "436356").all()


def test_unsupported_phase_is_recorded_not_raised(tmp_path):
    sweep = _sweep(tmp_path)
    prefill = Workload(phase="prefill", B=1, L_q=8, L_kv=8, H_q=2, H_kv=2, d=8, dtype="float32")
    from tests.fake_plugins import REGISTRY, FaithfulCPU

    class DecodeOnlyCPU(FaithfulCPU):
        name = "decode_only_cpu"
        phases = frozenset({"decode"})

    if "decode_only_cpu" not in REGISTRY.names():
        REGISTRY.register(DecodeOnlyCPU)
    summary = sweep.run_cell("decode_only_cpu", prefill)
    assert summary["status"] == "unsupported"


def test_run_grid_visits_every_supported_cell_and_returns_summaries(tmp_path):
    sweep = _sweep(tmp_path)
    ws = [W, Workload(phase="decode", B=1, L_q=1, L_kv=32, H_q=2, H_kv=2, d=8, dtype="float32")]
    summaries = sweep.run_grid(["faithful_cpu"], ws)
    assert [s["workload_key"] for s in summaries] == [w.key() for w in ws]
    assert all(s["status"] == "ok" for s in summaries)
    assert set(sweep.store.load()["workload_key"]) == {w.key() for w in ws}
