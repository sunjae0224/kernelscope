import pytest

from kernelscope.analytic import attended_pairs, attention_flops, attention_traffic, total_attended_pairs
from kernelscope.workload import Workload

DECODE = Workload(phase="decode", B=1, L_q=1, L_kv=1024, H_q=32, H_kv=8, d=128)          # fp16
PREFILL = Workload(phase="prefill", B=1, L_q=1024, L_kv=1024, H_q=32, H_kv=8, d=128)
CHUNK = Workload(phase="prefill", B=2, L_q=8, L_kv=32, H_q=4, H_kv=4, d=16, causal=True)
NONCAUSAL = Workload(phase="prefill", B=2, L_q=8, L_kv=32, H_q=4, H_kv=4, d=16, causal=False)


def test_decode_traffic_is_dominated_by_reading_the_kv_cache_once():
    t = attention_traffic(DECODE)
    assert t["kv_bytes"] == 2 * 1 * 1024 * 8 * 128 * 2       # K and V, fp16
    assert t["q_bytes"] == 1 * 1 * 32 * 128 * 2
    assert t["o_bytes"] == t["q_bytes"]
    assert t["total_bytes"] == t["kv_bytes"] + t["q_bytes"] + t["o_bytes"]
    assert t["dtype_bytes"] == 2


def test_kv_heads_read_can_be_overridden_for_kernels_without_gqa():
    t = attention_traffic(DECODE, kv_heads_read=32)
    assert t["kv_bytes"] == 4 * attention_traffic(DECODE)["kv_bytes"]


def test_bfloat16_and_float32_change_the_element_size():
    assert attention_traffic(Workload(**{**DECODE.__dict__, "dtype": "bfloat16"}))["dtype_bytes"] == 2
    assert attention_traffic(Workload(**{**DECODE.__dict__, "dtype": "float32"}))["dtype_bytes"] == 4


def test_attended_pairs_counts_the_causal_triangle():
    assert attended_pairs(NONCAUSAL) == 8 * 32
    # bottom-right causal: query i sees L_kv - L_q + i + 1 keys -> 25..32 -> sum = 228
    assert attended_pairs(CHUNK) == sum(32 - 8 + i + 1 for i in range(8))
    assert attended_pairs(PREFILL) == 1024 * 1025 // 2
    assert attended_pairs(DECODE) == 1024


def test_flops_are_four_per_attended_pair_per_head_dim():
    assert attention_flops(DECODE) == 4 * 1 * 32 * 128 * 1024
    assert attention_flops(NONCAUSAL) == 4 * 2 * 4 * 16 * (8 * 32)


def test_arithmetic_intensity_is_flops_per_byte():
    from kernelscope.analytic import arithmetic_intensity
    ai = arithmetic_intensity(DECODE)
    assert ai == pytest.approx(attention_flops(DECODE) / attention_traffic(DECODE)["total_bytes"])
    assert ai < 8   # decode attention lives deep in the memory-bound region


RAGGED = Workload(phase="decode", B=3, L_q=1, L_kv=1000, H_q=4, H_kv=2, d=8, kv_lens=[1000, 24, 100])


def test_ragged_pairs_and_flops_sum_over_sequences():
    assert total_attended_pairs(RAGGED) == 1124
    assert attention_flops(RAGGED) == 4 * 4 * 8 * 1124


def test_ragged_kv_bytes_sum_over_sequences():
    t = attention_traffic(RAGGED)
    assert t["kv_bytes"] == 2 * 1124 * 2 * 8 * 2
    assert t["q_bytes"] == 3 * 1 * 4 * 8 * 2


def test_uniform_totals_are_unchanged():
    w = Workload(phase="decode", B=3, L_q=1, L_kv=1000, H_q=4, H_kv=2, d=8)
    assert total_attended_pairs(w) == 3000
    assert attention_traffic(w)["kv_bytes"] == 2 * 3 * 1000 * 2 * 8 * 2
