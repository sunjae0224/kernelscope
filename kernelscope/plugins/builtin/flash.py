"""flash-attn 2.x plugins.

Decode goes through ``flash_attn_with_kvcache`` (dense ``[B, L, H, d]`` cache):
  fa2            num_splits=1  -> plain ``flash_fwd_kernel`` (grid = B x H_q CTAs)
  flashdecoding  num_splits=0  -> the library's split-KV heuristic: ``flash_fwd_splitkv_kernel``
                                  + ``flash_fwd_splitkv_combine_kernel`` when B*H_q is small
Prefill goes through ``flash_attn_func`` (causal is bottom-right aligned, like the reference).
"""
import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache

from kernelscope.plugins.base import KernelPlugin
from kernelscope.workload import Workload


class _Flash(KernelPlugin):
    kernel_regex = r"flash_fwd"
    num_splits = 1

    def build_inputs(self, w: Workload) -> dict:
        g = torch.Generator().manual_seed(0)
        dt = getattr(torch, w.dtype)

        def rand(*shape):
            return torch.randn(*shape, generator=g).to(dt).to(self.device)

        inputs = {
            "q": rand(w.B, w.L_q, w.H_q, w.d),
            "k": rand(w.B, w.L_kv, w.H_kv, w.d),
            "v": rand(w.B, w.L_kv, w.H_kv, w.d),
            "causal": w.causal,
            "phase": w.phase,
        }
        if w.phase == "decode":
            inputs["cache_seqlens"] = torch.full((w.B,), w.L_kv, dtype=torch.int32, device=self.device)
        return inputs

    def run(self, inputs: dict):
        if inputs["phase"] == "decode":
            return flash_attn_with_kvcache(
                inputs["q"], inputs["k"], inputs["v"], cache_seqlens=inputs["cache_seqlens"],
                causal=inputs["causal"], num_splits=self.num_splits,
            )
        return flash_attn_func(inputs["q"], inputs["k"], inputs["v"], causal=inputs["causal"])

    def to_dense_inputs(self, inputs: dict):
        return inputs["q"], inputs["k"], inputs["v"]

    def to_dense_output(self, out):
        return out


class FA2(_Flash):
    name = "fa2"
    phases = frozenset({"decode", "prefill"})
    num_splits = 1


class FlashDecoding(_Flash):
    name = "flashdecoding"
    phases = frozenset({"decode"})
    num_splits = 0


PLUGINS = [FA2, FlashDecoding]
