from types import SimpleNamespace

import pytest

from kernelscope.analytic import attention_flops, attention_traffic
from kernelscope.diagnose.opmodel import LAYER_CLASSES, OP_CLASSES, OpCost, dtype_bytes, op_costs
from kernelscope.workload import Workload

CFG = SimpleNamespace(n_layers=2, hidden=24, intermediate=48, n_heads=4, n_kv_heads=2, head_dim=8, vocab=128, qk_norm=True)


def test_decode_costs_match_hand_calculation():
    c = op_costs(CFG, "bfloat16", "decode", 3, (5, 7, 9))
    assert set(c) == set(OP_CLASSES)
    b, T, H, I, V, L = 2, 3, 24, 48, 128, 2
    assert c["qkv_proj"].flops == L * 2 * T * H * (4 + 2 * 2) * 8
    assert c["qkv_proj"].weight_bytes == L * H * (4 + 4) * 8 * b
    assert c["mlp"] == OpCost(L * 3 * H * I * b, L * T * b * (3 * H + 6 * I), L * 2 * T * 3 * H * I)
    assert c["embed"] == OpCost(T * H * b, T * H * b, 0)
    assert c["lm_head"] == OpCost(H * b + V * H * b, 2 * T * H * b + T * V * 4, 2 * T * V * H)
    assert c["norm"].act_bytes == L * (2 * (2 * T * H * b) + 2 * T * (4 + 2) * 8 * b) and c["norm"].flops == 0
    w = Workload("decode", 3, 1, 9, 4, 2, 8, "bfloat16", kv_lens=(5, 7, 9))
    assert c["attention"] == OpCost(0, L * attention_traffic(w)["total_bytes"], L * attention_flops(w))
    assert c["attention"].bytes == c["attention"].act_bytes


def test_uniform_decode_uses_a_plain_workload_key():
    c = op_costs(CFG, "float16", "decode", 2, (6, 6))
    w = Workload("decode", 2, 1, 6, 4, 2, 8, "float16")
    assert c["attention"].flops == 2 * attention_flops(w)


def test_prefill_chunk_costs_use_visible_keys():
    c = op_costs(CFG, "float32", "prefill", 5, (13,))
    w = Workload("prefill", 1, 5, 13, 4, 2, 8, "float32")
    assert c["attention"].flops == 2 * attention_flops(w)
    assert c["qkv_proj"].flops == 2 * 2 * 5 * 24 * 8 * 8


def test_rejects_bad_inputs():
    assert dtype_bytes("torch.bfloat16") == 2
    with pytest.raises(ValueError, match="dtype"):
        dtype_bytes("int7")
    with pytest.raises(ValueError, match="phase"):
        op_costs(CFG, "float16", "train", 1, (1,))
    with pytest.raises(ValueError, match="one KV length"):
        op_costs(CFG, "float16", "decode", 2, (4,))
    with pytest.raises(ValueError, match="single"):
        op_costs(CFG, "float16", "prefill", 2, (4, 4))


def test_layer_classes_are_the_per_layer_subset():
    assert set(LAYER_CLASSES) < set(OP_CLASSES) and "embed" not in LAYER_CLASSES


def test_op_classes_match_the_serving_timer():
    torch = pytest.importorskip("torch")
    from kernelscope.serve.model import OP_CLASSES as TIMER_CLASSES
    assert TIMER_CLASSES == OP_CLASSES
