"""torch.nn.functional.scaled_dot_product_attention, one plugin per forced backend.

SDPA wants ``[B, H, L, d]``; inputs are built in that layout so ``run()`` is the
bare kernel. ``is_causal=True`` in SDPA is top-left aligned, which is wrong for
``L_q < L_kv`` (a decode query would see only key 0), so the bottom-right mask is
built explicitly for that case.
"""
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from kernelscope.plugins.base import KernelPlugin
from kernelscope.workload import Workload


def _needs_mask(w: Workload) -> bool:
    """Chunked prefill (1 < L_q < L_kv) needs an explicit bottom-right mask."""
    return w.causal and w.L_q > 1 and w.L_q != w.L_kv


class _SDPA(KernelPlugin):
    phases = frozenset({"decode", "prefill"})
    backend: SDPBackend | None = None
    supports_gqa = True       # False -> K/V are repeated to H_q heads (the kernel runs as MHA)
    supports_mask = True      # False -> chunked prefill is not expressible for this backend

    def supports(self, w: Workload) -> bool:
        if w.is_ragged:
            return False
        if w.phase not in self.phases:
            return False
        return self.supports_mask or not _needs_mask(w)

    def kv_heads_read(self, w: Workload) -> int:
        return w.H_kv if self.supports_gqa else w.H_q

    def build_inputs(self, w: Workload) -> dict:
        g = torch.Generator().manual_seed(0)
        dt = getattr(torch, w.dtype)

        def rand(*shape):
            return torch.randn(*shape, generator=g).to(dt).to(self.device)

        q = rand(w.B, w.H_q, w.L_q, w.d)
        k = rand(w.B, w.H_kv, w.L_kv, w.d)
        v = rand(w.B, w.H_kv, w.L_kv, w.d)
        gqa = w.H_q != w.H_kv
        if gqa and not self.supports_gqa:
            k = k.repeat_interleave(w.H_q // w.H_kv, dim=1)
            v = v.repeat_interleave(w.H_q // w.H_kv, dim=1)
            gqa = False
        is_causal, mask = False, None
        if _needs_mask(w):
            i = torch.arange(w.L_q).view(-1, 1)
            j = torch.arange(w.L_kv).view(1, -1)
            mask = (j <= (w.L_kv - w.L_q) + i).to(self.device)
        elif w.causal and w.L_q > 1:
            is_causal = True
        return {"q": q, "k": k, "v": v, "attn_mask": mask, "is_causal": is_causal, "enable_gqa": gqa}

    def run(self, inputs: dict):
        ctx = sdpa_kernel(self.backend) if self.backend is not None else nullcontext()
        with ctx:
            return F.scaled_dot_product_attention(
                inputs["q"], inputs["k"], inputs["v"],
                attn_mask=inputs["attn_mask"], is_causal=inputs["is_causal"],
                enable_gqa=inputs["enable_gqa"],
            )

    def to_dense_inputs(self, inputs: dict):
        return (inputs["q"].transpose(1, 2), inputs["k"].transpose(1, 2), inputs["v"].transpose(1, 2))

    def to_dense_output(self, out):
        return out.transpose(1, 2)


class SDPAMath(_SDPA):
    name = "sdpa_math"
    backend = SDPBackend.MATH
    kernel_regex = None  # many small kernels; reference only, not profiled


class SDPAEfficient(_SDPA):
    """xformers-derived mem-efficient kernel. No GQA: K/V are expanded to H_q heads,
    so with a GQA workload it reads H_q/H_kv times more KV than the flash kernels do."""
    name = "sdpa_efficient"
    backend = SDPBackend.EFFICIENT_ATTENTION
    kernel_regex = r"fmha_cutlass"
    supports_gqa = False


class SDPACudnn(_SDPA):
    """cuDNN fused attention (torch 2.8 / A100): prefill only — L_q = 1 has no kernel."""
    name = "sdpa_cudnn"
    backend = SDPBackend.CUDNN_ATTENTION
    kernel_regex = r"cudnn|sdpa|fprop"  # verify with `--mode kernels` on the target torch build
    phases = frozenset({"prefill"})


class SDPAFlash(_SDPA):
    """torch's bundled FA2 (``pytorch_flash::``). Takes no attn_mask, so chunked prefill is out;
    decode dispatches to split-KV on its own (like the `flashdecoding` plugin)."""
    name = "sdpa_flash"
    backend = SDPBackend.FLASH_ATTENTION
    kernel_regex = r"flash_fwd"
    supports_mask = False


PLUGINS = [SDPAMath, SDPAEfficient, SDPACudnn, SDPAFlash]
