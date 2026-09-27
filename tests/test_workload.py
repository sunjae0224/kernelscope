import pytest

from kernelscope.workload import Workload, expand_grid, format_lens, parse_lens, ragged_lens


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


RAGGED = Workload(phase="decode", B=32, L_q=1, L_kv=32768, H_q=32, H_kv=8, d=128,
                  kv_lens=[32768] * 2 + [1024] * 30)


def test_ragged_key_compresses_runs_of_equal_lengths():
    assert RAGGED.key() == "decode_B32_Lq1_Lkv32768x2+1024x30_Hq32_Hkv8_d128_float16_causal"


def test_ragged_key_round_trips():
    assert Workload.from_key(RAGGED.key()) == RAGGED
    assert Workload.from_key(RAGGED.key()).kv_lens == (32768, 32768) + (1024,) * 30


def test_uniform_kv_lens_canonicalise_to_the_uniform_key():
    w = Workload(phase="decode", B=4, L_q=1, L_kv=512, H_q=8, H_kv=2, d=64, kv_lens=[512] * 4)
    assert w.kv_lens is None
    assert not w.is_ragged
    assert w.key() == "decode_B4_Lq1_Lkv512_Hq8_Hkv2_d64_float16_causal"


def test_lens_gives_per_sequence_lengths():
    assert RAGGED.lens()[:3] == (32768, 32768, 1024)
    u = Workload(phase="decode", B=3, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16)
    assert u.lens() == (64, 64, 64)


@pytest.mark.parametrize("kw, msg", [
    (dict(phase="prefill", L_q=16), "decode"),
    (dict(kv_lens=[100, 50]), "B=3"),
    (dict(kv_lens=[99, 50, 7]), "max"),
    (dict(kv_lens=[100, 0, 100]), "positive"),
])
def test_rejects_invalid_ragged_workloads(kw, msg):
    base = dict(phase="decode", B=3, L_q=1, L_kv=100, H_q=8, H_kv=2, d=16, kv_lens=[100, 50, 7])
    base.update(kw)
    with pytest.raises(ValueError, match=msg):
        Workload(**base)


def test_dtype_with_underscores_round_trips():
    for causal in (True, False):
        w = Workload(phase="decode", B=1, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16,
                     dtype="float8_e4m3fn", causal=causal)
        assert Workload.from_key(w.key()) == w


def test_format_and_parse_lens():
    assert format_lens([32768, 1024, 1024]) == "32768+1024x2"
    assert parse_lens("32768+1024x2") == (32768, 1024, 1024)
    assert parse_lens(format_lens([5, 5, 7, 5])) == (5, 5, 7, 5)
    with pytest.raises(ValueError, match="lens"):
        parse_lens("12x")


def test_ragged_lens_puts_long_sequences_first():
    assert ragged_lens(B=4, n_long=1, L_long=8192, L_short=512) == "8192+512x3"


def test_expand_grid_lens_sets_B_and_L_kv():
    ws = expand_grid({"phase": "decode", "lens": ["8192+512x3", "1024x2"], "H_q": 8, "H_kv": 2, "d": 16})
    assert [(w.B, w.L_kv, w.is_ragged) for w in ws] == [(4, 8192, True), (2, 1024, False)]


def test_expand_grid_ragged_generator_skips_impossible_combinations():
    ws = expand_grid({"phase": "decode", "H_q": 8, "H_kv": 2, "d": 16,
                      "ragged": {"B": [2, 16], "n_long": [1, 2], "L_long": 4096, "L_short": 256}})
    assert sorted(w.key().split("_")[3] for w in ws) == [
        "Lkv4096+256", "Lkv4096+256x15", "Lkv4096x2+256x14"]


def test_expand_grid_rejects_lens_together_with_B():
    with pytest.raises(ValueError, match="lens"):
        expand_grid({"phase": "decode", "lens": "64x2", "B": 2, "H_q": 8, "H_kv": 2, "d": 16})
