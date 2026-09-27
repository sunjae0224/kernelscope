"""Parse accel-sim.out stdout: `key = value` lines, one block per kernel, cumulative totals.

The simulator has no stats file; everything is on stdout. A clean run ends with
``GPGPU-Sim: *** exit detected ***``; an opcode missing from the ISA map aborts with
``ERROR: undefined instruction : <op>`` (trace_driven.cc), which we surface as a
status instead of a crash.
"""
import re

EXIT_SENTINEL = "GPGPU-Sim: *** exit detected ***"
_UNDEF = re.compile(r"ERROR: undefined instruction : (\S+)")
_UNSUPPORTED_BINARY = re.compile(r"unsupported binary version:\s*(\d+)")
_KV = re.compile(r"^([A-Za-z_][A-Za-z0-9_ ]*?)\s*=\s*(.+?)\s*$")

KERNEL_KEYS = {"gpu_sim_cycle", "gpu_sim_insn", "gpu_ipc", "gpu_occupancy"}
TOTAL_KEYS = {
    "gpu_tot_sim_cycle", "gpu_tot_sim_insn", "gpu_tot_ipc", "gpu_tot_occupancy", "gpu_tot_issued_cta",
    "L2_total_cache_accesses", "L2_total_cache_misses", "L2_total_cache_miss_rate",
    "L2_total_cache_reservation_fails", "gpu_stall_dramfull", "gpu_stall_icnt2sh",
    "averagemflatency", "total_dram_reads", "total_dram_writes",
    "partiton_level_parallism_total", "partiton_level_parallism_util_total",
}


def _num(raw: str):
    tok = raw.split()[0].rstrip("%x")
    try:
        return int(tok) if re.fullmatch(r"-?\d+", tok) else float(tok)
    except ValueError:
        return None


def parse_sim_stdout(text: str) -> dict:
    totals, kernels, cur, unsupported = {}, [], None, None
    for line in text.splitlines():
        m = _UNDEF.search(line)
        if m:
            unsupported = m.group(1)
            continue
        m = _KV.match(line)
        if not m:
            continue
        key = m.group(1).strip().replace(" ", "_")
        raw = m.group(2)
        if key == "kernel_name":
            cur = {"kernel_name": raw.strip()}
            kernels.append(cur)
            continue
        if key == "gpgpu_simulation_time":
            m2 = re.search(r"\((\d+) sec\)", raw)
            if m2:
                totals["gpgpu_simulation_time_s"] = int(m2.group(1))
            continue
        if key == "gpgpu_simulation_rate":
            v = _num(raw)
            totals["sim_rate_cycle_per_s" if "cycle" in raw else "sim_rate_inst_per_s"] = v
            continue
        v = _num(raw)
        if v is None:
            continue
        if key in KERNEL_KEYS:
            if cur is not None:
                cur[key] = v
        elif key in TOTAL_KEYS:
            totals[key] = v
    binary = _UNSUPPORTED_BINARY.search(text)
    if binary:
        status = "unsupported_binary:" + binary.group(1)
    elif unsupported:
        status = "unsupported_opcode"
    elif "KERNELSCOPE_SIM_PROCESS_FAILED" in text:
        status = "process_error"
    elif "KERNELSCOPE_SIM_TIMEOUT" in text:
        status = "timeout"
    elif EXIT_SENTINEL not in text:
        status = "incomplete"
    elif not kernels:
        status = "no_kernels"   # clean exit on an empty kernelslist — nothing was simulated
    else:
        status = "ok"
    return {"status": status, "totals": totals, "kernels": kernels, "unsupported_opcode": unsupported}
