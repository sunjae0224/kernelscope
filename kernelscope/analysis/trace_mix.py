"""Instruction mix from the raw NVBit trace (``kernel-N-ctx_*.trace.xz``).

The trace we already produce for the simulator is a per-warp-instruction SASS
log, so opcode histograms, memory-instruction shares and bytes requested come
for free — a counter-free, deterministic view of *what the kernel does*.

Line layout (tracer version 6, lineinfo off): a handful of integer prefixes
(cta x/y/z, warp id, ...) then
``PC mask dest_num [dests] opcode src_num [srcs] mem_width [addr fields] ...``.
The PC is located as the first 4-hex token followed by an 8-hex mask, and the
rest is parsed positionally.
"""
import lzma
import re
from collections import Counter
from pathlib import Path

_HEX4 = re.compile(r"[0-9a-f]{4}")
_HEX8 = re.compile(r"[0-9a-f]{8}")

CATEGORIES = {
    "global_mem": {"LDG", "STG", "LDGSTS", "LD", "ST", "RED", "ATOM", "ATOMG"},
    "shared_mem": {"LDS", "STS", "LDSM", "STSM", "ATOMS"},
    "local_mem": {"LDL", "STL"},
    "tensor": {"HMMA", "IMMA", "DMMA", "BMMA"},
    "control": {"BRA", "BRX", "JMP", "JMX", "EXIT", "BAR", "BSSY", "BSYNC", "BPT", "DEPBAR",
                "LDGDEPBAR", "MEMBAR", "WARPSYNC", "NOP", "RET", "CALL", "ERRBAR"},
}
_LOADS = {"LDG", "LDGSTS", "LD"}
_STORES = {"STG", "ST"}


def categorize(base: str) -> str:
    for cat, ops in CATEGORIES.items():
        if base in ops:
            return cat
    return "other"


def parse_trace_line(line: str) -> dict | None:
    t = line.split()
    for i in range(len(t) - 1):
        if _HEX4.fullmatch(t[i]) and _HEX8.fullmatch(t[i + 1]):
            break
    else:
        return None
    mask = t[i + 1]
    dest_num = int(t[i + 2])
    j = i + 3 + dest_num
    opcode = t[j]
    src_num = int(t[j + 1])
    mem_width = int(t[j + 2 + src_num])
    return {"pc": t[i], "mask": mask, "active": bin(int(mask, 16)).count("1"),
            "opcode": opcode, "base": opcode.split(".")[0], "mem_width": mem_width}


def _open(path):
    path = Path(path)
    return lzma.open(path, "rt") if path.suffix == ".xz" else open(path)


def _tuple(s: str):
    return tuple(int(x) for x in s.strip("() ").split(","))


def instruction_mix(path) -> dict:
    header, opcodes, cats = {}, Counter(), Counter()
    n = load_bytes = store_bytes = 0
    with _open(path) as f:
        for line in f:
            if line.startswith("-"):
                k, _, v = line[1:].partition("=")
                header[k.strip()] = v.strip()
                continue
            if not line.strip() or line.startswith("#"):
                continue
            ins = parse_trace_line(line)
            if ins is None:
                continue
            n += 1
            opcodes[ins["base"]] += 1
            cats[categorize(ins["base"])] += 1
            if ins["mem_width"]:
                if ins["base"] in _LOADS:
                    load_bytes += ins["active"] * ins["mem_width"]
                elif ins["base"] in _STORES:
                    store_bytes += ins["active"] * ins["mem_width"]
    return {
        "kernel_name": header.get("kernel name"),
        "grid": _tuple(header["grid dim"]) if "grid dim" in header else None,
        "block": _tuple(header["block dim"]) if "block dim" in header else None,
        "nregs": int(header["nregs"]) if "nregs" in header else None,
        "shmem": int(header["shmem"]) if "shmem" in header else None,
        "n_warp_insts": n,
        "opcodes": dict(opcodes),
        "categories": dict(cats),
        "fractions": {c: v / n for c, v in cats.items()} if n else {},
        "global_load_bytes": load_bytes,
        "global_store_bytes": store_bytes,
        "global_bytes_requested": load_bytes + store_bytes,
    }
