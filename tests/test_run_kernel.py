import json
import subprocess
import sys

import pytest

from kernelscope import run_kernel

KEY = "decode_B2_Lq1_Lkv64_Hq8_Hkv2_d16_float32_causal"
REG = "tests.fake_plugins:REGISTRY"


def test_latency_mode_prints_json_with_median(capsys):
    run_kernel.main(["--plugin", "faithful_cpu", "--workload", KEY, "--mode", "latency",
                     "--warmup", "1", "--iters", "3", "--registry", REG, "--device", "cpu"])
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "latency"
    assert out["plugin"] == "faithful_cpu"
    assert out["workload_key"] == KEY
    assert out["iters"] == 3
    assert out["median_s"] >= 0


def test_check_mode_prints_correctness_result(capsys):
    run_kernel.main(["--plugin", "faithful_cpu", "--workload", KEY, "--mode", "check",
                     "--registry", REG, "--device", "cpu"])
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "check"
    assert out["ok"] is True
    assert out["max_abs_diff"] < 1e-4


def test_ncu_mode_runs_warmup_plus_one_and_keeps_stdout_clean_for_ncu_csv(capsys):
    run_kernel.main(["--plugin", "faithful_cpu", "--workload", KEY, "--mode", "ncu",
                     "--warmup", "3", "--registry", REG, "--device", "cpu"])
    captured = capsys.readouterr()
    assert captured.out == ""
    out = json.loads(captured.err)
    assert out["mode"] == "ncu"
    assert out["runs"] == 4
    assert out["warmup"] == 3


def test_trace_mode_warms_up_then_runs_once_inside_a_profiler_window(capsys):
    run_kernel.main(["--plugin", "faithful_cpu", "--workload", KEY, "--mode", "trace",
                     "--warmup", "2", "--registry", REG, "--device", "cpu"])
    captured = capsys.readouterr()
    assert captured.out == ""                          # stdout stays clean (tracer banner lives there)
    out = json.loads(captured.err)
    assert out["mode"] == "trace"
    assert out["warmup"] == 2 and out["runs"] == 3
    assert out["instrumentation_window"] is False      # no tracer injected -> nothing to toggle


def test_trace_mode_toggles_the_injected_tracer_around_the_measured_run(capsys, monkeypatch, tmp_path):
    calls = []

    class FakeTracer:
        def enable_nvbit_instrumentation(self):
            calls.append("on")

        def disable_nvbit_instrumentation(self):
            calls.append("off")

    monkeypatch.setenv("CUDA_INJECTION64_PATH", str(tmp_path / "tracer_tool.so"))
    monkeypatch.setattr(run_kernel, "_load_tracer", lambda path: FakeTracer())
    run_kernel.main(["--plugin", "faithful_cpu", "--workload", KEY, "--mode", "trace",
                     "--warmup", "1", "--registry", REG, "--device", "cpu"])
    out = json.loads(capsys.readouterr().err)
    assert out["instrumentation_window"] is True
    assert calls == ["on", "off"]


def test_kernels_mode_lists_launched_kernels_and_regex_matches(capsys):
    run_kernel.main(["--plugin", "faithful_cpu", "--workload", KEY, "--mode", "kernels",
                     "--registry", REG, "--device", "cpu"])
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "kernels"
    assert out["kernel_regex"] == "faithful"
    assert isinstance(out["kernels"], list)      # empty on CPU: no CUDA launches
    assert out["matching"] == [k for k in out["kernels"] if "faithful" in k]


def test_profile_mode_reports_launches_and_kernel_time(capsys):
    run_kernel.main(["--plugin", "faithful_cpu", "--workload", KEY, "--mode", "profile",
                     "--warmup", "1", "--iters", "2", "--registry", REG, "--device", "cpu"])
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "profile"
    assert out["iters"] == 2
    assert out["kernel_regex"] == "faithful"
    assert out["launches"] == []                      # CPU: no CUDA kernels
    assert out["launches_per_iter"] == 0
    assert out["kernel_time_us_median"] is None
    assert isinstance(out["unmatched"], list)


def test_unknown_plugin_exits_nonzero_listing_available(capsys):
    with pytest.raises(SystemExit) as ex:
        run_kernel.main(["--plugin", "nope", "--workload", KEY, "--mode", "latency",
                         "--registry", REG, "--device", "cpu"])
    assert ex.value.code != 0
    assert "faithful_cpu" in capsys.readouterr().err


def test_unsupported_phase_exits_nonzero(capsys):
    from tests.fake_plugins import REGISTRY
    from kernelscope.plugins.base import KernelPlugin

    class PrefillOnly(KernelPlugin):
        name = "prefill_only_tmp"
        phases = frozenset({"prefill"})
        kernel_regex = "x"

        def build_inputs(self, w):
            return {}

        def run(self, inputs):
            return None

        def to_dense_output(self, out):
            return out

    if "prefill_only_tmp" not in REGISTRY.names():
        REGISTRY.register(PrefillOnly)
    with pytest.raises(SystemExit) as ex:
        run_kernel.main(["--plugin", "prefill_only_tmp", "--workload", KEY, "--mode", "latency",
                         "--registry", REG, "--device", "cpu"])
    assert ex.value.code != 0
    assert "decode" in capsys.readouterr().err


def test_module_is_runnable_as_subprocess():
    proc = subprocess.run(
        [sys.executable, "-m", "kernelscope.run_kernel", "--plugin", "faithful_cpu",
         "--workload", KEY, "--mode", "check", "--registry", REG, "--device", "cpu"],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(proc.stdout)["ok"] is True
