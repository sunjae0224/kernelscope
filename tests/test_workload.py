import pytest

from kernelscope.workload import Workload, expand_grid


def test_key_is_stable_and_encodes_every_field():
    w = Workload(phase="decode", B=1, L_q=1, L_kv=4096, H_q=32, H_kv=8, d=128)
    assert w.key() == "decode_B1_Lq1_Lkv4096_Hq32_Hkv8_d128_float16_causal"


def test_key_reflects_dtype_and_non_causal():
    w = Workload(phase="prefill", B=2, L_q=512, L_kv=512, H_q=8, H_kv=8, d=64,
                 dtype="bfloat16", causal=False)
    assert w.key() == "prefill_B2_Lq512_Lkv512_Hq8_Hkv8_d64_bfloat16_noncausal"


def test_rejects_unknown_phase():
    with pytest.raises(ValueError, match="phase"):
        Workload(phase="train", B=1, L_q=1, L_kv=1, H_q=1, H_kv=1, d=64)


def test_rejects_head_count_not_divisible_by_kv_heads():
    with pytest.raises(ValueError, match="H_q"):
        Workload(phase="decode", B=1, L_q=1, L_kv=1, H_q=6, H_kv=4, d=64)


def test_expand_grid_takes_cartesian_product_of_list_fields():
    spec = {"phase": ["decode"], "B": [1, 16], "L": [1024, 8192],
            "H_q": 32, "H_kv": 8, "d": 128}
    ws = expand_grid(spec)
    assert len(ws) == 4
    assert {(w.B, w.L_kv) for w in ws} == {(1, 1024), (1, 8192), (16, 1024), (16, 8192)}


def test_expand_grid_derives_L_q_from_phase():
    spec = {"phase": ["decode", "prefill"], "B": 1, "L": 2048,
            "H_q": 32, "H_kv": 8, "d": 128}
    by_phase = {w.phase: w for w in expand_grid(spec)}
    assert by_phase["decode"].L_q == 1
    assert by_phase["decode"].L_kv == 2048
    assert by_phase["prefill"].L_q == 2048
    assert by_phase["prefill"].L_kv == 2048


def test_from_key_round_trips():
    w = Workload(phase="prefill", B=4, L_q=2048, L_kv=2048, H_q=32, H_kv=8, d=128,
                 dtype="bfloat16", causal=False)
    assert Workload.from_key(w.key()) == w


def test_from_key_rejects_malformed_key():
    with pytest.raises(ValueError, match="key"):
        Workload.from_key("decode_B1_garbage")


def test_expand_grid_preserves_declaration_order():
    spec = {"phase": "decode", "B": [16, 1], "L": [8192, 1024],
            "H_q": 32, "H_kv": 8, "d": 128}
    ws = expand_grid(spec)
    assert [(w.B, w.L_kv) for w in ws] == [(16, 8192), (16, 1024), (1, 8192), (1, 1024)]
