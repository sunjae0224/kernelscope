"""Triton's own fused-attention tutorial (v3.4.0, unmodified copy under samples/) as an external plugin."""
from pathlib import Path

import pytest
import torch

from kernelscope.plugins.builtin import REGISTRY
from kernelscope.plugins.builtin.triton_tutorial import TUTORIAL_PATH, TritonTutorialAttn
from kernelscope.workload import Workload

PREFILL = Workload(phase="prefill", B=1, L_q=1024, L_kv=1024, H_q=32, H_kv=8, d=128)
NONCAUSAL = Workload(phase="prefill", B=1, L_q=1024, L_kv=1024, H_q=8, H_kv=8, d=64, causal=False)
CHUNK = Workload(phase="prefill", B=1, L_q=256, L_kv=1024, H_q=32, H_kv=8, d=128)
DECODE = Workload(phase="decode", B=1, L_q=1, L_kv=1024, H_q=32, H_kv=8, d=128)


def test_tutorial_file_is_vendored_unmodified():
    assert TUTORIAL_PATH.name == "06-fused-attention.py"
    text = TUTORIAL_PATH.read_text()
    assert "attention = _attention.apply" in text          # the tutorial's public entry point


def test_registered_with_prefill_square_only_support():
    assert "triton_tutorial" in REGISTRY.names()
    p = TritonTutorialAttn(device="cpu")
    assert p.supports(PREFILL) and p.supports(NONCAUSAL)
    assert not p.supports(CHUNK)          # tutorial has no bottom-right mask
    assert not p.supports(DECODE)         # L_q must equal L_kv
    assert not p.supports(Workload(**{**PREFILL.__dict__, "d": 96}))   # HEAD_DIM in {16,32,64,128,256}


def test_inputs_are_bhld_fp16_with_expanded_kv_because_no_gqa():
    p = TritonTutorialAttn(device="cpu")
    inputs = p.build_inputs(PREFILL)
    assert tuple(inputs["q"].shape) == (1, 32, 1024, 128)
    assert tuple(inputs["k"].shape) == (1, 32, 1024, 128)     # H_kv 8 -> 32
    assert inputs["q"].dtype == torch.float16
    assert inputs["causal"] is True and inputs["sm_scale"] == pytest.approx(128 ** -0.5)
    assert p.kv_heads_read(PREFILL) == 32
    q, k, v = p.to_dense_inputs(inputs)
    assert tuple(q.shape) == (1, 1024, 32, 128) and tuple(k.shape) == (1, 1024, 32, 128)


def test_kernel_regex_targets_the_tutorial_forward_kernel():
    assert TritonTutorialAttn.kernel_regex == r"_attn_fwd"
