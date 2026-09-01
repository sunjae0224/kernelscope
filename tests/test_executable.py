"""ExecutablePlugin path: a standalone binary (here a python stand-in) driven by the harness."""
import sys

import pytest

from kernelscope.backends.realhw.executable import check_executable, parse_kernelscope_line, run_executable
from kernelscope.backends.realhw.sweep import RealHWSweep
from kernelscope.results.store import ResultStore
from kernelscope.workload import Workload
from tests.fake_plugins import REGISTRY, FakeExec

W = Workload(phase="decode", B=2, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32")
REG = "tests.fake_plugins:REGISTRY"


def test_executable_plugin_contract_defaults():
    p = FakeExec()
    assert p.command(W)[-1] == "1"                       # iters defaults to 1, no out path
    assert p.command(W, iters=5, out_path="/tmp/o.bin")[-2:] == ["5", "/tmp/o.bin"]
    assert p.output_dtype == "float32"


def test_parse_kernelscope_line_ignores_other_stdout():
    text = "noise\nKERNELSCOPE {\"kernel_time_us\": 3.5, \"launches_per_iter\": 2}\nmore\n"
    assert parse_kernelscope_line(text) == {"kernel_time_us": 3.5, "launches_per_iter": 2}


def test_parse_kernelscope_line_missing_is_an_error():
    with pytest.raises(ValueError, match="KERNELSCOPE"):
        parse_kernelscope_line("no marker here\n")


def test_run_executable_returns_the_reported_timing():
    r = run_executable(FakeExec(), W, iters=3)
    assert r["kernel_time_us"] == 12.5
    assert r["iters"] == 3


def test_check_executable_compares_written_output_with_reference(tmp_path):
    r = check_executable(FakeExec(), W, tmp_path / "out.bin")
    assert r["ok"] is True
    assert r["max_abs_diff"] < 1e-5


def test_check_executable_without_reference_inputs_is_reported_not_raised(tmp_path):
    class NoRef(FakeExec):
        name = "noref"
        reference_inputs = None

    r = check_executable(NoRef(), W, tmp_path / "out.bin")
    assert r["ok"] is False
    assert "reference_inputs" in r["error"]


def test_sweep_runs_executable_cells_with_self_reported_kernel_time(tmp_path):
    sweep = RealHWSweep(store=ResultStore(tmp_path / "r"), python_exe=sys.executable, registry=REG,
                        device="cpu", warmup=1, iters=3)
    s = sweep.run_cell("fake_exec", W)
    assert s["status"] == "ok", s
    assert s["check_ok"] is True
    assert s["kernel_time_us"] == 12.5
    assert s["launches"] == 1
    assert s["profile"] == "self-reported"
    df = sweep.store.load()
    assert df[(df.backend == "profile") & (df.metric == "kernel_time_us")]["value"].item() == 12.5
    an = df[df.backend == "analytic"].set_index("metric")["value"]
    assert an["achieved_gbps"] == pytest.approx((1024 + 32768 + 1024) / 12.5e-6 / 1e9)
    assert "latency" not in set(df.backend)               # no CUDA-event latency for a foreign process
