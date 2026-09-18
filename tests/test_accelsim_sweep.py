import shutil
import sys
from pathlib import Path

import pytest

from kernelscope.backends.accelsim.config import get_option
from kernelscope.backends.accelsim.paths import AccelSimPaths
from kernelscope.backends.accelsim.sweep import (
    AccelSimSweep, postprocess_command, simulate_command, trace_command,
)
from kernelscope.results.store import ResultStore
from kernelscope.workload import Workload

FIX = Path(__file__).parent / "fixtures"
REG = "tests.fake_plugins:REGISTRY"
W = Workload(phase="decode", B=2, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32")
SIM_LOG = (FIX / "sim_stdout_vecadd.log").read_text()


def _fake_repo(tmp_path):
    repo = tmp_path / "accel-sim"
    cfg = repo / "gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100"
    cfg.mkdir(parents=True)
    shutil.copy(FIX / "SM80_A100_gpgpusim.config", cfg / "gpgpusim.config")
    tcfg = repo / "gpu-simulator/configs/tested-cfgs/SM80_A100"
    tcfg.mkdir(parents=True)
    (tcfg / "trace.config").write_text("-trace_opcode_latency_initiation_int 4,2\n")
    return repo


def _fake_tools(sim_stdout=SIM_LOG):
    calls = {"trace": [], "post": [], "sim": []}

    def trace(argv, env, cell_dir):
        calls["trace"].append((argv, env))
        tr = cell_dir / "traces"
        tr.mkdir(parents=True)
        # same shape as the real stats file, kernel name matching the fake plugins' regexes
        (tr / "stats_ctx_0x1").write_text(
            "kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, #blocks, block_dimX, "
            "block_dimY, block_dimZ, #threads, total_insts, total_reported_insts\n"
            "kernel-1-ctx_0x1.trace.xz, _Z20faithful_fake_kernelv, 4096, 1, 1, 4096, 256, 1, 1, 256, 524288,524288\n")
        (tr / "kernelslist_ctx_0x1").write_text("kernel-1-ctx_0x1.trace.xz\n")

    def post(argv, cell_dir):
        calls["post"].append(argv)
        (cell_dir / "traces" / "kernelslist.g").write_text("kernel-1-ctx_0x1.traceg.xz\n")

    def sim(argv, log_path):
        calls["sim"].append(argv)
        log_path.write_text(sim_stdout)
        return sim_stdout, 0.5

    return calls, trace, post, sim


def _sweep(tmp_path, variants=("base", "l2_x2"), sim_stdout=SIM_LOG, **kw):
    repo = _fake_repo(tmp_path)
    calls, trace, post, sim = _fake_tools(sim_stdout)
    sweep = AccelSimSweep(
        store=ResultStore(tmp_path / "results"), paths=AccelSimPaths(repo),
        work_dir=tmp_path / "work", python_exe=sys.executable, registry=REG,
        variants=variants, trace_fn=trace, postprocess_fn=post, simulate_fn=sim, **kw,
    )
    return sweep, calls


def test_trace_command_warms_up_then_traces_one_run_in_the_profiler_window():
    argv = trace_command(sys.executable, "fa2", W, REG, "cuda", warmup=5)
    assert argv[:3] == [sys.executable, "-m", "kernelscope.run_kernel"]
    assert argv[argv.index("--mode") + 1] == "trace"
    assert argv[argv.index("--warmup") + 1] == "5"
    assert argv[argv.index("--plugin") + 1] == "fa2"


def test_real_subprocess_failure_and_stderr_reach_the_parser(tmp_path):
    from kernelscope.backends.accelsim.stats import parse_sim_stdout
    sweep, _ = _sweep(tmp_path)
    argv = [sys.executable, "-c", "import sys; print('ERROR: undefined instruction : ADA_NEW', file=sys.stderr); sys.exit(2)"]
    stdout, wall = sweep._run_simulate(argv, tmp_path / "failure.log")
    assert wall >= 0
    assert "KERNELSCOPE_SIM_PROCESS_FAILED returncode=2" in stdout
    assert parse_sim_stdout(stdout)["unsupported_opcode"] == "ADA_NEW"


def test_postprocess_and_simulate_commands_follow_the_gate_report(tmp_path):
    p = AccelSimPaths(tmp_path)
    assert postprocess_command(p, tmp_path / "cell" / "traces", jobs=8) == [
        str(p.post_process), str(tmp_path / "cell" / "traces"), "-j", "8"]
    argv = simulate_command(p, tmp_path / "k.g", tmp_path / "gpgpusim.config", tmp_path / "trace.config")
    assert argv == [str(p.sim_bin), "-trace", str(tmp_path / "k.g"),
                    "-config", str(tmp_path / "gpgpusim.config"), "-config", str(tmp_path / "trace.config")]


def test_cell_traces_once_and_simulates_every_variant(tmp_path):
    sweep, calls = _sweep(tmp_path)
    s = sweep.run_cell("faithful_cpu", W)
    assert s["status"] == "ok"
    assert s["warp_insts"] == 524288
    assert s["sim"] == {"base": "ok", "l2_x2": "ok"}
    assert len(calls["trace"]) == 1 and len(calls["post"]) == 1 and len(calls["sim"]) == 2
    df = sweep.store.load()
    assert {"trace", "sim:base", "sim:l2_x2"} <= set(df.backend)
    cyc = df[(df.backend == "sim:base") & (df.metric == "gpu_tot_sim_cycle")]
    assert cyc["value"].item() == 11333
    assert (df[df.backend == "trace"].set_index("metric")["value"]["warp_insts"]) == 524288


def test_variant_config_files_are_materialized_per_cell(tmp_path):
    sweep, calls = _sweep(tmp_path)
    sweep.run_cell("faithful_cpu", W)
    cfg_arg = calls["sim"][1][calls["sim"][1].index("-config") + 1]
    assert cfg_arg.endswith("l2_x2/gpgpusim.config")
    assert get_option(Path(cfg_arg).read_text(), "gpgpu_cache:dl2").startswith("S:256:")


def test_python_plugins_are_traced_in_a_profiler_window_after_warmup(tmp_path):
    sweep, calls = _sweep(tmp_path)
    sweep.run_cell("faithful_cpu", W)
    argv, env = calls["trace"][0]
    assert argv[argv.index("--mode") + 1] == "trace"
    assert int(argv[argv.index("--warmup") + 1]) >= 1
    assert env["NVBIT_INSTRUMENTATION_ENABLED"] == "0"
    assert env["DYNAMIC_KERNEL_RANGE"] == "1-@.*(?:faithful).*"
    assert env["TRACES_FOLDER"].endswith("faithful_cpu__" + W.key())


def test_executables_are_traced_from_start_with_the_regex_filter(tmp_path):
    sweep, calls = _sweep(tmp_path)
    s = sweep.run_cell("fake_exec", W)
    assert s["status"] == "ok", s
    argv, env = calls["trace"][0]
    assert argv[0] == sys.executable and argv[1].endswith("fake_exec.py")
    assert env["NVBIT_INSTRUMENTATION_ENABLED"] == "1"
    assert env["DYNAMIC_KERNEL_RANGE"] == "1-@.*(?:fake_kernel).*"


def test_kernelslist_is_filtered_to_regex_matching_kernels_before_simulating(tmp_path):
    repo = _fake_repo(tmp_path)
    calls = {"sim": []}

    def trace(argv, env, cell_dir):
        tr = cell_dir / "traces"
        tr.mkdir(parents=True)
        (tr / "stats_ctx_0x1").write_text(
            "kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, #blocks, block_dimX, "
            "block_dimY, block_dimZ, #threads, total_insts, total_reported_insts\n"
            "kernel-1-ctx_0x1.trace.xz, _ZN2at6native4fillv, 1, 1, 1, 1, 128, 1, 1, 128, 40,40\n"
            "kernel-2-ctx_0x1.trace.xz, _ZN5faithful_kernelv, 1, 1, 8, 8, 128, 1, 1, 128, 519944,519944\n")

    def post(argv, cell_dir):
        (cell_dir / "traces" / "kernelslist.g").write_text(
            "kernel-1-ctx_0x1.traceg.xz\nkernel-2-ctx_0x1.traceg.xz\n")

    def sim(argv, log_path):
        calls["sim"].append(Path(argv[argv.index("-trace") + 1]).read_text())
        log_path.write_text(SIM_LOG)
        return SIM_LOG, 0.1

    sweep = AccelSimSweep(store=ResultStore(tmp_path / "r"), paths=AccelSimPaths(repo),
                          work_dir=tmp_path / "w", python_exe=sys.executable, registry=REG,
                          variants=("base",), trace_fn=trace, postprocess_fn=post, simulate_fn=sim)
    s = sweep.run_cell("faithful_cpu", W)
    assert s["status"] == "ok", s
    assert s["n_kernels"] == 1 and s["warp_insts"] == 519944        # the fill kernel is dropped everywhere
    assert calls["sim"][0].splitlines() == ["kernel-2-ctx_0x1.traceg.xz"]


def test_budget_skips_simulation_but_keeps_trace_stats(tmp_path):
    sweep, calls = _sweep(tmp_path, max_sim_s=1.0)   # est = 524288/27500 ≈ 19 s > 1
    s = sweep.run_cell("faithful_cpu", W)
    assert s["status"] == "skipped_budget"
    assert s["est_sim_s"] > 1.0
    assert calls["sim"] == []
    df = sweep.store.load()
    assert set(df.backend) == {"trace"}


def test_unsupported_opcode_is_recorded_per_variant(tmp_path):
    bad = "banner\nERROR: undefined instruction : FOO.BAR\n"
    sweep, calls = _sweep(tmp_path, variants=("base",), sim_stdout=bad)
    s = sweep.run_cell("faithful_cpu", W)
    assert s["status"] == "ok"
    assert s["sim"] == {"base": "unsupported_opcode:FOO.BAR"}
    df = sweep.store.load()
    row = df[(df.backend == "sim:base") & (df.metric == "status")]
    assert row["value"].item() == 0.0
    assert row["note"].item() == "unsupported_opcode:FOO.BAR"


def test_unsupported_phase_is_recorded_not_raised(tmp_path):
    sweep, _ = _sweep(tmp_path)
    from tests.fake_plugins import REGISTRY, FaithfulCPU

    class DecodeOnlyCPU(FaithfulCPU):
        name = "decode_only_cpu"
        phases = frozenset({"decode"})

    if "decode_only_cpu" not in REGISTRY.names():
        REGISTRY.register(DecodeOnlyCPU)
    prefill = Workload(phase="prefill", B=1, L_q=8, L_kv=8, H_q=2, H_kv=2, d=8, dtype="float32")
    assert sweep.run_cell("decode_only_cpu", prefill)["status"] == "unsupported"


def test_trace_with_zero_instructions_is_an_error(tmp_path):
    # the tracer lists every launch in stats_ctx even when its filter instrumented nothing
    repo = _fake_repo(tmp_path)

    def uninstrumented(argv, env, cell_dir):
        tr = cell_dir / "traces"
        tr.mkdir(parents=True)
        (tr / "stats_ctx_0x1").write_text(
            "kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, #blocks, block_dimX, "
            "block_dimY, block_dimZ, #threads, total_insts, total_reported_insts\n"
            "kernel-1-ctx_0x1.trace.xz, _Z4fillv, 1, 1, 1, 1, 128, 1, 1, 128, 0,0\n"
            "kernel-2-ctx_0x1.trace.xz, _ZN5flash16flash_fwd_kernelIv, 1, 1, 8, 8, 128, 1, 1, 128, 0,0\n")

    sweep = AccelSimSweep(store=ResultStore(tmp_path / "r"), paths=AccelSimPaths(repo),
                          work_dir=tmp_path / "w", python_exe=sys.executable, registry=REG,
                          trace_fn=uninstrumented, postprocess_fn=lambda *a: None,
                          simulate_fn=lambda *a: ("", 0))
    s = sweep.run_cell("faithful_cpu", W)
    assert s["status"] == "error"
    assert "0 warp instructions" in s["error"]


def test_uninstrumented_launches_are_not_counted_as_kernels(tmp_path):
    # torch.full's fill kernel is listed by the tracer with 0 insts next to the real kernel
    repo = _fake_repo(tmp_path)

    def mixed(argv, env, cell_dir):
        tr = cell_dir / "traces"
        tr.mkdir(parents=True)
        (tr / "stats_ctx_0x1").write_text(
            "kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, #blocks, block_dimX, "
            "block_dimY, block_dimZ, #threads, total_insts, total_reported_insts\n"
            "kernel-1-ctx_0x1.trace.xz, _Z4fillv, 1, 1, 1, 1, 128, 1, 1, 128, 0,0\n"
            "kernel-2-ctx_0x1.trace.xz, _ZN8faithful_kernelIv, 1, 1, 8, 8, 128, 1, 1, 128, 519944,519944\n")
        (tr / "kernelslist_ctx_0x1").write_text("kernel-2-ctx_0x1.trace.xz\n")

    calls, _, post, sim = _fake_tools()
    sweep = AccelSimSweep(store=ResultStore(tmp_path / "r"), paths=AccelSimPaths(repo),
                          work_dir=tmp_path / "w", python_exe=sys.executable, registry=REG,
                          variants=("base",), trace_fn=mixed, postprocess_fn=post, simulate_fn=sim)
    s = sweep.run_cell("faithful_cpu", W)
    assert s["status"] == "ok"
    assert s["n_kernels"] == 1
    assert s["warp_insts"] == 519944
    df = sweep.store.load()
    per_kernel = df[(df.backend == "trace") & (df.metric == "kernel_warp_insts")]
    assert len(per_kernel) == 1
    assert per_kernel["note"].item().startswith("_ZN8faithful")


def test_instruction_mix_rows_are_recorded_when_the_raw_trace_exists(tmp_path):
    import lzma
    from tests.test_trace_mix import HEADER, LINES
    repo = _fake_repo(tmp_path)

    def trace_with_raw(argv, env, cell_dir):
        tr = cell_dir / "traces"
        tr.mkdir(parents=True)
        (tr / "stats_ctx_0x1").write_text(
            "kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, #blocks, block_dimX, "
            "block_dimY, block_dimZ, #threads, total_insts, total_reported_insts\n"
            "kernel-2-ctx_0x1.trace.xz, _ZN8faithful_kernelIfoo, 1, 1, 8, 8, 128, 1, 1, 128, 8,8\n")
        (tr / "kernelslist_ctx_0x1").write_text("kernel-2-ctx_0x1.trace.xz\n")
        with lzma.open(tr / "kernel-2-ctx_0x1.trace.xz", "wt") as f:
            f.write(HEADER.replace("_ZN5flash16flash_fwd_kernelIfoo", "_ZN8faithful_kernelIfoo")
                    + "\n".join(LINES) + "\n")

    _, _, post, sim = _fake_tools()
    sweep = AccelSimSweep(store=ResultStore(tmp_path / "r"), paths=AccelSimPaths(repo),
                          work_dir=tmp_path / "w", python_exe=sys.executable, registry=REG,
                          variants=("base",), trace_fn=trace_with_raw, postprocess_fn=post, simulate_fn=sim)
    s = sweep.run_cell("faithful_cpu", W)
    assert s["status"] == "ok"
    df = sweep.store.load()
    mix = df[df.backend == "tracemix"].set_index("metric")["value"]
    assert mix["n_warp_insts"] == 8
    assert mix["frac_global_mem"] == pytest.approx(3 / 8)
    assert mix["frac_tensor"] == pytest.approx(1 / 8)
    assert mix["global_load_bytes"] == 32 * 16 + 16 * 16
    assert (df[df.backend == "tracemix"]["note"] == "_ZN8faithful_kernelIfoo").all()


def test_simulation_that_ran_no_kernel_is_not_ok(tmp_path):
    empty_ok = "GPGPU-Sim: *** simulation thread exiting ***\nGPGPU-Sim: *** exit detected ***\n"
    sweep, calls = _sweep(tmp_path, variants=("base",), sim_stdout=empty_ok)
    s = sweep.run_cell("faithful_cpu", W)
    assert s["sim"] == {"base": "no_kernels"}


def test_trace_without_output_is_an_error_not_a_crash(tmp_path):
    repo = _fake_repo(tmp_path)

    def no_trace(argv, env, cell_dir):
        (cell_dir / "traces").mkdir(parents=True)   # tracer ran but regex matched nothing

    sweep = AccelSimSweep(store=ResultStore(tmp_path / "r"), paths=AccelSimPaths(repo),
                          work_dir=tmp_path / "w", python_exe=sys.executable, registry=REG,
                          trace_fn=no_trace, postprocess_fn=lambda *a: None, simulate_fn=lambda *a: ("", 0))
    s = sweep.run_cell("faithful_cpu", W)
    assert s["status"] == "error"
    assert "no trace" in s["error"]
