import lzma

import pytest

from kernelscope.analysis.trace_mix import categorize, instruction_mix, parse_trace_line

HEADER = """\
-kernel name = _ZN5flash16flash_fwd_kernelIfoo
-kernel id = 2
-grid dim = (1,1,8)
-block dim = (128,1,1)
-shmem = 65536
-nregs = 255
-binary version = 80
-cuda stream id = 0
-shmem base_addr = 0x00007f8125000000
-local mem base_addr = 0x00007f8123000000
-nvbit version = 1.8
-accelsim tracer version = 6
-enable lineinfo = 0

#traces format = [line_num] PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [adrrescompress?] [mem_addresses] immediate1 immediate2 Val|NoVal [dest_val_n]

"""
# 11 leading ints (cta xyz, warp id, ...) then PC mask dest_num dests opcode src_num srcs mem_width ...
LINES = [
    "0 0 5 1 0 0 0 0 0 0 0 0000 ffffffff 1 R1 MOV 0 0 0 0 NoVal",
    "0 0 5 1 0 0 0 0 0 0 0 0010 ffffffff 1 R4 LDG.E.128 1 R2 16 1 0x7f00 0 0 NoVal",
    "0 0 5 1 0 0 0 0 0 0 0 0020 0000ffff 0 LDGSTS.E.BYPASS.128 2 R6 R8 16 1 0x7f10 0 0 NoVal",
    "0 0 5 1 0 0 0 0 0 0 0 0030 ffffffff 4 R20 R21 R22 R23 HMMA.16816.F32 3 R20 R24 R26 0 0 0 NoVal",
    "0 0 5 1 0 0 0 0 0 0 0 0040 ffffffff 2 R30 R31 LDSM.16.M88.4 1 R10 16 1 0x100 0 0 NoVal",
    "0 0 5 1 0 0 0 0 0 0 0 0050 ffffffff 0 BAR.SYNC.DEFER_BLOCKING 1 R255 0 0 0 NoVal",
    "0 0 5 1 0 0 0 0 0 0 0 0060 ffffffff 0 STG.E.128 2 R2 R4 16 1 0x7f20 0 0 NoVal",
    "0 0 5 1 0 0 0 0 0 0 0 0070 ffffffff 0 EXIT 0 0 0 0 NoVal",
]


def test_parse_line_is_positional_not_regex_based():
    ins = parse_trace_line(LINES[3])
    assert ins == {"pc": "0030", "mask": "ffffffff", "active": 32, "opcode": "HMMA.16816.F32",
                   "base": "HMMA", "mem_width": 0}
    ins = parse_trace_line(LINES[2])
    assert ins["base"] == "LDGSTS" and ins["active"] == 16 and ins["mem_width"] == 16


def test_categorize_maps_base_opcodes_to_pipes():
    assert categorize("LDG") == "global_mem"
    assert categorize("LDGSTS") == "global_mem"
    assert categorize("STG") == "global_mem"
    assert categorize("LDSM") == "shared_mem"
    assert categorize("LDS") == "shared_mem"
    assert categorize("HMMA") == "tensor"
    assert categorize("BAR") == "control"
    assert categorize("EXIT") == "control"
    assert categorize("MOV") == "other"
    assert categorize("FFMA") == "other"


def test_instruction_mix_over_a_compressed_trace(tmp_path):
    p = tmp_path / "kernel-2-ctx_0x1.trace.xz"
    with lzma.open(p, "wt") as f:
        f.write(HEADER + "\n".join(LINES) + "\n")
    m = instruction_mix(p)
    assert m["kernel_name"] == "_ZN5flash16flash_fwd_kernelIfoo"
    assert m["grid"] == (1, 1, 8) and m["block"] == (128, 1, 1)
    assert m["nregs"] == 255 and m["shmem"] == 65536
    assert m["n_warp_insts"] == 8
    assert m["opcodes"]["LDG"] == 1 and m["opcodes"]["HMMA"] == 1
    assert m["categories"] == {"global_mem": 3, "shared_mem": 1, "tensor": 1, "control": 2, "other": 1}
    assert m["fractions"]["global_mem"] == pytest.approx(3 / 8)
    # bytes requested = active lanes x width for global loads/stores: 32*16 + 16*16 + 32*16
    assert m["global_bytes_requested"] == 32 * 16 + 16 * 16 + 32 * 16
    assert m["global_load_bytes"] == 32 * 16 + 16 * 16
    assert m["global_store_bytes"] == 32 * 16


def test_instruction_mix_accepts_plain_text_too(tmp_path):
    p = tmp_path / "k.trace"
    p.write_text(HEADER + LINES[0] + "\n")
    assert instruction_mix(p)["n_warp_insts"] == 1
