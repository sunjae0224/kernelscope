import torch

from kernelscope.check import check_plugin
from kernelscope.plugins.builtin import REGISTRY
from kernelscope.plugins.builtin.sdpa import SDPAMath
from kernelscope.workload import Workload


def test_builtin_registry_exposes_sdpa_family():
    names = REGISTRY.names()
    assert "sdpa_math" in names
    assert "sdpa_efficient" in names
    assert "sdpa_cudnn" in names
    assert "sdpa_flash" in names


def test_inputs_are_bhld_and_dense_view_is_blhd():
    w = Workload(phase="decode", B=2, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32")
    p = SDPAMath(device="cpu")
    inputs = p.build_inputs(w)
    assert tuple(inputs["q"].shape) == (2, 8, 1, 16)
    assert tuple(inputs["k"].shape) == (2, 2, 64, 16)
    q, k, v = p.to_dense_inputs(inputs)
    assert tuple(q.shape) == (2, 1, 8, 16)
    assert tuple(k.shape) == (2, 64, 2, 16)


def test_math_backend_matches_reference_for_decode_gqa():
    w = Workload(phase="decode", B=2, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32")
    assert check_plugin(SDPAMath(device="cpu"), w, atol=1e-4)["ok"]


def test_math_backend_matches_reference_for_causal_prefill():
    w = Workload(phase="prefill", B=1, L_q=32, L_kv=32, H_q=4, H_kv=4, d=16, dtype="float32")
    assert check_plugin(SDPAMath(device="cpu"), w, atol=1e-4)["ok"]


def test_math_backend_matches_reference_for_bottom_right_causal_chunk():
    # chunked prefill: 8 new queries against 32 keys — must not use top-left is_causal
    w = Workload(phase="prefill", B=1, L_q=8, L_kv=32, H_q=4, H_kv=4, d=16, dtype="float32")
    assert check_plugin(SDPAMath(device="cpu"), w, atol=1e-4)["ok"]


DECODE_GQA = Workload(phase="decode", B=1, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32")
PREFILL = Workload(phase="prefill", B=1, L_q=32, L_kv=32, H_q=8, H_kv=2, d=16, dtype="float32")
CHUNK = Workload(phase="prefill", B=1, L_q=8, L_kv=32, H_q=8, H_kv=2, d=16, dtype="float32")


def test_efficient_backend_expands_kv_heads_because_it_has_no_gqa():
    from kernelscope.plugins.builtin.sdpa import SDPAEfficient
    p = SDPAEfficient(device="cpu")
    assert p.supports_gqa is False
    inputs = p.build_inputs(DECODE_GQA)
    assert tuple(inputs["k"].shape) == (1, 8, 64, 16)      # H_kv 2 -> 8
    assert inputs["enable_gqa"] is False
    # expanded heads must be the GQA mapping: q head h reads kv head h // 4
    k2 = SDPAMath(device="cpu").build_inputs(DECODE_GQA)["k"]
    assert torch.equal(inputs["k"][:, 5], k2[:, 1])
    q, k, v = p.to_dense_inputs(inputs)
    assert tuple(k.shape) == (1, 64, 8, 16)                # reference sees the expanded (MHA) KV
    # numerical agreement of the efficient kernel itself is covered on GPU in test_sdpa_gpu.py


def test_kv_heads_read_reports_the_expansion_for_the_traffic_model():
    from kernelscope.plugins.builtin.sdpa import SDPAEfficient, SDPAFlash
    assert SDPAEfficient(device="cpu").kv_heads_read(DECODE_GQA) == 8      # expanded to H_q
    assert SDPAFlash(device="cpu").kv_heads_read(DECODE_GQA) == 2          # native GQA
    mha = Workload(phase="decode", B=1, L_q=1, L_kv=64, H_q=8, H_kv=8, d=16, dtype="float32")
    assert SDPAEfficient(device="cpu").kv_heads_read(mha) == 8


def test_cudnn_backend_is_prefill_only():
    from kernelscope.plugins.builtin.sdpa import SDPACudnn
    p = SDPACudnn(device="cpu")
    assert not p.supports(DECODE_GQA)
    assert p.supports(PREFILL)
    assert p.supports(CHUNK)


def test_torch_flash_backend_cannot_take_the_chunk_mask():
    from kernelscope.plugins.builtin.sdpa import SDPAFlash
    p = SDPAFlash(device="cpu")
    assert p.supports(DECODE_GQA)
    assert p.supports(PREFILL)
    assert not p.supports(CHUNK)


def test_inputs_are_deterministic_per_workload():
    w = Workload(phase="decode", B=1, L_q=1, L_kv=16, H_q=2, H_kv=2, d=8, dtype="float32")
    a = SDPAMath(device="cpu").build_inputs(w)["q"]
    b = SDPAMath(device="cpu").build_inputs(w)["q"]
    assert torch.equal(a, b)
