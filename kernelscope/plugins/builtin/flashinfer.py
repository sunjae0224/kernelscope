"""FlashInfer decode plugin (paged KV cache, page 256, GQA).

FlashInfer schedules variable-length batches itself: ``plan()`` (CPU) partitions every request's
KV pages into balanced work for the CTAs, ``run()`` executes the fixed-grid kernel. There is no
``num_splits`` to choose, which is exactly what this project's external selection is compared
against (docs/plan/2026-09-26-plan-flashinfer-comparison.md). ``plan()`` runs once per built
input (per cell) and its wall time is kept in ``inputs["plan_us"]``; the harness times ``run()``
only, so a kernel time from this plugin excludes the plan cost.
"""
import time

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

from kernelscope.plugins.base import KernelPlugin
from kernelscope.plugins.paged import PAGE, build_block_table, gather_from_pool
from kernelscope.workload import Workload

WORKSPACE_BYTES = 128 * 2**20


def paged_indices(lens, table_cpu: torch.Tensor, page: int = PAGE):
    """FlashInfer's CSR view of a block table: indptr [B+1], indices [sum pages], last_page_len [B]."""
    indptr, indices, last = [0], [], []
    for b, n in enumerate(lens):
        pages = (n + page - 1) // page
        indices.extend(int(x) for x in table_cpu[b, :pages])
        indptr.append(indptr[-1] + pages)
        last.append(n - (pages - 1) * page)
    return (torch.tensor(indptr, dtype=torch.int32), torch.tensor(indices, dtype=torch.int32),
            torch.tensor(last, dtype=torch.int32))


class FlashInferPaged(KernelPlugin):
    name = "flashinfer_paged"
    phases = frozenset({"decode"})
    kernel_regex = r"BatchDecodeWithPagedKVCache|BatchPrefillWithPagedKVCache|flashinfer"
    num_launches = 1
    supports_ragged = True
    use_tensor_cores = True        # recommended for GQA group sizes >= 4 (Qwen3-4B: 32/8)

    def __init__(self, device="cuda", **kw):
        super().__init__(device, **kw)
        self._workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)

    def build_inputs(self, w: Workload) -> dict:
        if w.d != 128 and w.d not in (64, 256):
            raise ValueError(f"{self.name}: head_dim {w.d} not supported")
        dt = getattr(torch, w.dtype)
        g = torch.Generator(device=self.device).manual_seed(0)

        def rand(*shape):
            return torch.randn(*shape, generator=g, device=self.device, dtype=dt)

        lens = w.lens()
        table_cpu, pages = build_block_table(lens)
        indptr, indices, last = paged_indices(lens, table_cpu)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(self._workspace, "NHD", use_tensor_cores=self.use_tensor_cores)
        started = time.perf_counter()
        wrapper.plan(indptr.to(self.device), indices.to(self.device), last.to(self.device),
                     w.H_q, w.H_kv, w.d, PAGE, q_data_type=dt, kv_data_type=dt, sm_scale=w.d ** -0.5)
        torch.cuda.synchronize(self.device)
        plan_us = (time.perf_counter() - started) * 1e6
        return {"q": rand(w.B, w.H_q, w.d), "k": rand(pages, PAGE, w.H_kv, w.d), "v": rand(pages, PAGE, w.H_kv, w.d),
                "lens": lens, "L_kv": w.L_kv, "block_table_cpu": table_cpu, "wrapper": wrapper, "plan_us": plan_us,
                "phase": w.phase, "causal": w.causal}

    def run(self, inputs: dict):
        return inputs["wrapper"].run(inputs["q"], (inputs["k"], inputs["v"]))

    def to_dense_inputs(self, inputs: dict):
        t, lens, L = inputs["block_table_cpu"], inputs["lens"], inputs["L_kv"]
        return inputs["q"].unsqueeze(1), gather_from_pool(inputs["k"], lens, t, L), gather_from_pool(inputs["v"], lens, t, L)

    def to_dense_output(self, out):
        return out.unsqueeze(1)          # [B, H_q, d] -> [B, L_q=1, H_q, d]


class FlashInferPagedCudaCore(FlashInferPaged):
    """Same wrapper with FlashInfer's CUDA-core decode kernels (use_tensor_cores=False)."""
    name = "flashinfer_paged_cudacore"
    use_tensor_cores = False


PLUGINS = [FlashInferPaged, FlashInferPagedCudaCore]
