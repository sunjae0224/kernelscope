from pathlib import Path

from kernelscope.backends.accelsim.stats import parse_sim_stdout

LOG = (Path(__file__).parent / "fixtures" / "sim_stdout_vecadd.log").read_text()


def test_parses_totals_from_a_clean_run():
    s = parse_sim_stdout(LOG)
    assert s["status"] == "ok"
    t = s["totals"]
    assert t["gpu_tot_sim_cycle"] == 11333
    assert t["gpu_tot_sim_insn"] == 15728640
    assert t["gpu_tot_ipc"] == 1387.8619
    assert t["gpu_tot_occupancy"] == 91.4766
    assert t["L2_total_cache_accesses"] == 393216
    assert t["L2_total_cache_misses"] == 131072
    assert t["L2_total_cache_miss_rate"] == 0.3333
    assert t["gpgpu_simulation_time_s"] == 29
    assert t["total_dram_reads"] == 0
    assert t["gpu_stall_dramfull"] == 0


def test_per_kernel_blocks_are_collected_in_order():
    two = LOG.replace("GPGPU-Sim: *** simulation thread exiting ***", "") \
             .replace("GPGPU-Sim: *** exit detected ***", "")
    second = ("kernel_name = _Z7combinePf \nkernel_launch_uid = 2 \n"
              "gpu_sim_cycle = 500\ngpu_sim_insn = 1000\n"
              "gpu_tot_sim_cycle = 11833\ngpu_tot_sim_insn = 15729640\n"
              "L2_total_cache_accesses = 400000\nL2_total_cache_misses = 140000\n"
              "L2_total_cache_miss_rate = 0.35\n"
              "GPGPU-Sim: *** simulation thread exiting ***\nGPGPU-Sim: *** exit detected ***\n")
    s = parse_sim_stdout(two + second)
    assert [k["kernel_name"] for k in s["kernels"]] == ["_Z6vecAddPKfS0_Pfi", "_Z7combinePf"]
    assert [k["gpu_sim_cycle"] for k in s["kernels"]] == [11333, 500]
    assert s["totals"]["gpu_tot_sim_cycle"] == 11833          # last occurrence wins
    assert s["totals"]["L2_total_cache_miss_rate"] == 0.35


def test_missing_exit_sentinel_means_incomplete():
    s = parse_sim_stdout(LOG.replace("GPGPU-Sim: *** exit detected ***", ""))
    assert s["status"] == "incomplete"
    assert s["totals"]["gpu_tot_sim_cycle"] == 11333   # partial stats still returned


def test_undefined_instruction_is_reported_with_opcode():
    s = parse_sim_stdout("banner\nERROR: undefined instruction : LDGSTS.E.BYPASS.128\n")
    assert s["status"] == "unsupported_opcode"
    assert s["unsupported_opcode"] == "LDGSTS.E.BYPASS.128"


def test_clean_exit_without_any_kernel_block_is_no_kernels():
    s = parse_sim_stdout("banner\nGPGPU-Sim: *** exit detected ***\n")
    assert s["status"] == "no_kernels"


def test_empty_output_is_incomplete_with_no_totals():
    s = parse_sim_stdout("")
    assert s["status"] == "incomplete"
    assert s["totals"] == {}
    assert s["kernels"] == []
