"""Triton's fused-attention tutorial (python/tutorials/06-fused-attention.py, tag v3.4.0),
vendored unmodified under samples/triton_tutorial/ and driven as an external kernel.

Tutorial constraints → plugin support matrix: prefill with L_q == L_kv only (no
bottom-right mask), HEAD_DIM in {16, 32, 64, 128, 256}, no GQA (K/V expanded to
H_q heads), fp16 ``[B, H, L, d]``. First call per (N_CTX, HEAD_DIM) autotunes for
~30 s; that lands in the warm-up of each measurement subprocess.
"""
import importlib.util
import os
from pathlib import Path

import torch

from kernelscope.plugins.base import KernelPlugin
from kernelscope.workload import Workload

TUTORIAL_PATH = Path(os.environ.get(
    "KERNELSCOPE_TRITON_TUTORIAL",
    Path(__file__).resolve().parents[3] / "samples" / "triton_tutorial" / "06-fused-attention.py"))

_module = None


def _tutorial():
    global _module
    if _module is None:
        spec = importlib.util.spec_from_file_location("triton_fused_attention_tutorial", TUTORIAL_PATH)
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
    return _module


class TritonTutorialAttn(KernelPlugin):
    name = "triton_tutorial"
    phases = frozenset({"prefill"})
    kernel_regex = r"_attn_fwd"

    def supports(self, w: Workload) -> bool:
        return (w.phase == "prefill" and w.L_q == w.L_kv and w.d in (16, 32, 64, 128, 256)
                and w.dtype in ("float16", "bfloat16"))

    def kv_heads_read(self, w: Workload) -> int:
        return w.H_q

    def build_inputs(self, w: Workload) -> dict:
        g = torch.Generator().manual_seed(0)
        dt = getattr(torch, w.dtype)

        def rand(*shape):
            return torch.randn(*shape, generator=g).to(dt).to(self.device)

        q = rand(w.B, w.H_q, w.L_q, w.d)
        k = rand(w.B, w.H_kv, w.L_kv, w.d)
        v = rand(w.B, w.H_kv, w.L_kv, w.d)
        if w.H_q != w.H_kv:
            k = k.repeat_interleave(w.H_q // w.H_kv, dim=1).contiguous()
            v = v.repeat_interleave(w.H_q // w.H_kv, dim=1).contiguous()
        return {"q": q, "k": k, "v": v, "causal": w.causal, "sm_scale": w.d ** -0.5}

    def run(self, inputs: dict):
        return _tutorial().attention(inputs["q"], inputs["k"], inputs["v"], inputs["causal"], inputs["sm_scale"])

    def to_dense_inputs(self, inputs: dict):
        return (inputs["q"].transpose(1, 2), inputs["k"].transpose(1, 2), inputs["v"].transpose(1, 2))

    def to_dense_output(self, out):
        return out.transpose(1, 2)


PLUGINS = [TritonTutorialAttn]
