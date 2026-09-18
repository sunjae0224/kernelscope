from pathlib import Path

import pytest

from kernelscope.backends.accelsim.paths import AccelSimPaths
from kernelscope.backends.accelsim.trace import (
    estimate_sim_seconds, find_kernelslist, read_trace_stats, tracer_env,
)

FIX = Path(__file__).parent / "fixtures"


def test_paths_derive_tool_locations_from_repo_root(tmp_path):
    p = AccelSimPaths(tmp_path)
    assert p.tracer_so == tmp_path / "util/tracer_nvbit/tracer_tool/tracer_tool.so"
    assert p.post_process == tmp_path / "util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"
    assert p.sim_bin == tmp_path / "gpu-simulator/bin/release/accel-sim.out"
    assert p.gpgpusim_config("SM80_A100") == tmp_path / "gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config"
    assert p.trace_config("SM80_A100") == tmp_path / "gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config"


def test_tracer_env_filters_by_regex_and_writes_to_given_folder(tmp_path):
    env = tracer_env(AccelSimPaths(tmp_path), out_dir=tmp_path / "cell", kernel_regex="flash_fwd", device_index=2)
    assert env["CUDA_INJECTION64_PATH"] == str(tmp_path / "util/tracer_nvbit/tracer_tool/tracer_tool.so")
    # tracer_tool.cu uses std::regex_match (full match) on the *mangled* name -> wrap as search
    assert env["DYNAMIC_KERNEL_RANGE"] == "1-@.*(?:flash_fwd).*"
    assert env["ACTIVE_FROM_START"] == "1"
    assert env["USER_DEFINED_FOLDERS"] == "1"
    assert env["TRACES_FOLDER"] == str(tmp_path / "cell")
    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_tracer_env_window_mode_starts_with_instrumentation_off_but_keeps_the_filter(tmp_path):
    # JIT/autotuned kernels: warm up with instrumentation off, then the runner flips it on
    # via the tracer's exported enable_nvbit_instrumentation() for exactly one run
    # (the mechanism Accel-Sim's torch_hook uses). The regex filter stays active.
    env = tracer_env(AccelSimPaths(tmp_path), out_dir=tmp_path / "cell", kernel_regex="_attn_fwd",
                     device_index=0, window=True)
    assert env["NVBIT_INSTRUMENTATION_ENABLED"] == "0"
    assert env["ACTIVE_FROM_START"] == "1"
    assert env["DYNAMIC_KERNEL_RANGE"] == "1-@.*(?:_attn_fwd).*"
    assert env["TRACES_FOLDER"] == str(tmp_path / "cell")


def test_tracer_env_range_mode_starts_instrumenting_immediately(tmp_path):
    env = tracer_env(AccelSimPaths(tmp_path), out_dir=tmp_path / "cell", kernel_regex="k", device_index=0)
    assert env["NVBIT_INSTRUMENTATION_ENABLED"] == "1"


def test_tracer_uses_matching_cuda_disassembler_instead_of_system_tool(tmp_path):
    cuda = tmp_path / "cuda"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "bin/cuobjdump").touch()
    env = tracer_env(AccelSimPaths(tmp_path), tmp_path / "trace", "flash_fwd",
                     base_env={"ACCELSIM_CUDA_ROOT": str(cuda), "PATH": "/usr/bin"})
    assert env["PATH"] == str(cuda / "bin") + ":/usr/bin"


def test_filter_kernelslist_keeps_only_matching_kernels_and_all_memcpys(tmp_path):
    from kernelscope.backends.accelsim.trace import filter_kernelslist
    kl = tmp_path / "kernelslist.g"
    kl.write_text("MemcpyHtoD,0x7f00,4194304\n"
                  "kernel-1-ctx_0x1.traceg.xz\n"
                  "kernel-2-ctx_0x1.traceg.xz\n"
                  "kernel-3-ctx_0x1.traceg.xz\n")
    kept = filter_kernelslist(kl, keep_trace_files=["kernel-2-ctx_0x1.trace.xz"])
    assert kept == 1
    assert kl.read_text().splitlines() == ["MemcpyHtoD,0x7f00,4194304", "kernel-2-ctx_0x1.traceg.xz"]


def test_filter_kernelslist_handles_tracez_names_too(tmp_path):
    from kernelscope.backends.accelsim.trace import filter_kernelslist
    kl = tmp_path / "kernelslist.g"
    kl.write_text("kernel-1-ctx_0x1.tracez\nkernel-2-ctx_0x1.tracez\n")
    assert filter_kernelslist(kl, keep_trace_files=["kernel-1-ctx_0x1.trace.xz"]) == 1
    assert kl.read_text().splitlines() == ["kernel-1-ctx_0x1.tracez"]


def test_read_trace_stats_gives_warp_instruction_count_per_kernel():
    stats = read_trace_stats(FIX / "stats_ctx_vecadd")
    assert len(stats) == 1
    k = stats[0]
    assert k["trace_file"] == "kernel-1-ctx_0x1f13110.trace.xz"
    assert k["kernel_name"] == "_Z6vecAddPKfS0_Pfi"
    assert k["grid"] == (4096, 1, 1)
    assert k["block"] == (256, 1, 1)
    assert k["warp_insts"] == 524288


def test_estimate_uses_host_planning_rate_and_accepts_calibration():
    assert estimate_sim_seconds(524288) == pytest.approx(524288 / 10_000)
    assert estimate_sim_seconds(524288, rate=524288 / 33.07) == pytest.approx(33.07)
    assert estimate_sim_seconds(0) == 0


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_simulation_rates_fail_before_budgeting(rate):
    with pytest.raises(ValueError, match="finite and positive"):
        estimate_sim_seconds(100, rate)


def test_find_kernelslist_locates_post_processed_list(tmp_path):
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "kernelslist_ctx_0x1").write_text("raw")
    assert find_kernelslist(tmp_path) is None
    (tmp_path / "traces" / "kernelslist.g").write_text("processed")
    assert find_kernelslist(tmp_path) == tmp_path / "traces" / "kernelslist.g"


def test_real_rtx4090_trace_has_nonzero_instructions():
    stats = read_trace_stats(FIX / "SM89_RTX4090_stats_ctx_vecadd")
    assert len(stats) == 1
    assert stats[0]["warp_insts"] == 524288
    assert stats[0]["grid"] == (4096, 1, 1)
