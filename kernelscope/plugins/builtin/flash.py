"""flash-attn 2.x plugins.

Decode goes through ``flash_attn_with_kvcache``; each sequence's live length is passed as
``cache_seqlens`` so ragged batches work. Variants (all compute the same attention):
  fa2                  num_splits=1  -> plain ``flash_fwd_kernel`` (grid = B x H_kv CTAs after GQA packing)
  flashdecoding        num_splits=0  -> the library's heuristic (``num_splits_heuristic`` in flash_api.cpp)
  fd_s{N}              num_splits=N  -> ``flash_fwd_splitkv_kernel`` + ``..._combine_kernel``
  *_paged              same, on a paged cache (``block_table``, page 256). flash-attn always
                       uses the split kernel on this path, even for num_splits=1.
Dense caches have capacity L_kv = the longest live length (never an over-allocated capacity:
the heuristic sizes splits from ``k_cache.size(1)``, spec F17).
Prefill (fa2 only) goes through ``flash_attn_func`` (bottom-right causal, like the reference).
Inputs are generated on the device (a CPU generator would take minutes and tens of GB of
host memory for the largest grid cells).
"""
import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache

from kernelscope.plugins.base import KernelPlugin
from kernelscope.plugins.paged import PAGE, build_block_table, gather_from_pool
from kernelscope.workload import Workload

SPLITS = (2, 4, 8, 16, 32, 64, 128)


class _Flash(KernelPlugin):
    kernel_regex = r"flash_fwd"
    num_splits = 1
    paged = False
    supports_ragged = True

    def build_inputs(self, w: Workload) -> dict:
        dt = getattr(torch, w.dtype)
        g = torch.Generator(device=self.device).manual_seed(0)

        def rand(*shape):
            return torch.randn(*shape, generator=g, device=self.device, dtype=dt)

        inputs = {"q": rand(w.B, w.L_q, w.H_q, w.d), "causal": w.causal, "phase": w.phase, "L_kv": w.L_kv}
        if w.phase == "decode":
            lens = w.lens()
            inputs["lens"] = lens
            inputs["cache_seqlens"] = torch.tensor(lens, dtype=torch.int32, device=self.device)
            if self.paged:
                table, pages = build_block_table(lens)
                inputs["k"] = rand(pages, PAGE, w.H_kv, w.d)
                inputs["v"] = rand(pages, PAGE, w.H_kv, w.d)
                inputs["block_table_cpu"] = table
                inputs["block_table"] = table.to(self.device)
                return inputs
        inputs["k"] = rand(w.B, w.L_kv, w.H_kv, w.d)
        inputs["v"] = rand(w.B, w.L_kv, w.H_kv, w.d)
        return inputs

    def run(self, inputs: dict):
        if inputs["phase"] == "decode":
            return flash_attn_with_kvcache(
                inputs["q"], inputs["k"], inputs["v"], cache_seqlens=inputs["cache_seqlens"],
                block_table=inputs.get("block_table"), causal=inputs["causal"], num_splits=self.num_splits,
            )
        return flash_attn_func(inputs["q"], inputs["k"], inputs["v"], causal=inputs["causal"])

    def to_dense_inputs(self, inputs: dict):
        if "block_table" in inputs:
            t, lens, L = inputs["block_table_cpu"], inputs["lens"], inputs["L_kv"]
            return inputs["q"], gather_from_pool(inputs["k"], lens, t, L), gather_from_pool(inputs["v"], lens, t, L)
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


def _variant(name: str, num_splits: int, paged: bool) -> type:
    return type(name, (_Flash,), {"name": name, "phases": frozenset({"decode"}),
                                  "num_splits": num_splits, "paged": paged})


FIXED = [_variant(f"fd_s{n}", n, False) for n in SPLITS]
PAGED = ([_variant("fa2_paged", 1, True), _variant("flashdecoding_paged", 0, True)]
         + [_variant(f"fd_s{n}_paged", n, True) for n in SPLITS])

PLUGINS = [FA2, FlashDecoding, *FIXED, *PAGED]
