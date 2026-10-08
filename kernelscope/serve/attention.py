"""FlashInfer decode attention behind the engine's flash-attn call signature.

``FlashInferDecode`` is a drop-in for ``flash_attn_with_kvcache`` on decode steps (one query token
per sequence): it appends the step's new K/V into the paged pool and runs FlashInfer's
``BatchDecodeWithPagedKVCacheWrapper`` over the same pages. FlashInfer schedules the variable-length
batch itself in ``plan()`` (CPU), so there is no split count to choose; ``DecoderModel.decode`` plans
once per step before the layer loop and the engine books that time as the policy's selection cost.
Prefill calls (several query tokens) go to flash-attn unchanged.
"""
import math
import time

import torch

WORKSPACE_BYTES = 128 * 2**20
BACKENDS = {"flashinfer": True, "flashinfer_cudacore": False}         # policy name -> use_tensor_cores


def csr_from_block_table(table, lens, page=256):
    """FlashInfer's CSR view of a flash-attn block table: indptr [B+1], indices [sum pages], last_page_len [B]."""
    table = table.cpu()
    indptr, indices, last = [0], [], []
    for b, n in enumerate(lens):
        pages = (int(n) + page - 1) // page
        indices.extend(int(x) for x in table[b, :pages])
        indptr.append(indptr[-1] + pages)
        last.append(int(n) - (pages - 1) * page)
    return (torch.tensor(indptr, dtype=torch.int32), torch.tensor(indices, dtype=torch.int32),
            torch.tensor(last, dtype=torch.int32))


class FlashInferDecode:
    def __init__(self, fallback, use_tensor_cores=False, device="cuda", page=256, wrapper=None):
        self.fallback, self.page = fallback, page
        if wrapper is None:
            from flashinfer import BatchDecodeWithPagedKVCacheWrapper
            workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
            wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=use_tensor_cores)
        self.wrapper = wrapper
        self._planned, self._key, self._pages, self._slots, self._scale = None, None, None, None, None

    def plan(self, cache_lens, block_table, n_heads, n_kv_heads, head_dim, dtype) -> float:
        """Plan this step's kernel for the lengths after the append; returns the wall time in microseconds."""
        started = time.perf_counter()
        lens = [int(n) + 1 for n in cache_lens.tolist()]
        indptr, indices, last = csr_from_block_table(block_table, lens, self.page)
        device = block_table.device
        before = torch.tensor([n - 1 for n in lens], dtype=torch.long)       # the new token's position
        rows = block_table.cpu().long()[torch.arange(len(lens)), before // self.page]
        self._pages, self._slots = rows.to(device), (before % self.page).to(device)
        self.wrapper.plan(indptr.to(device), indices.to(device), last.to(device), n_heads, n_kv_heads, head_dim,
                          self.page, q_data_type=dtype, kv_data_type=dtype, sm_scale=head_dim ** -0.5)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._planned, self._key, self._scale = cache_lens, tuple(lens), head_dim ** -0.5
        return (time.perf_counter() - started) * 1e6

    def __call__(self, q, k_cache, v_cache, *, k, v, cache_seqlens, block_table, softmax_scale, causal, num_splits):
        if q.shape[1] != 1:
            return self.fallback(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens, block_table=block_table,
                                 softmax_scale=softmax_scale, causal=causal, num_splits=num_splits)
        if cache_seqlens is not self._planned and self._key != tuple(int(n) + 1 for n in cache_seqlens.tolist()):
            raise RuntimeError("FlashInfer decode needs plan() for this step's lengths before the layer loop")
        if not math.isclose(float(softmax_scale), self._scale):
            raise ValueError(f"softmax_scale {softmax_scale} differs from the planned head_dim ** -0.5 = {self._scale}")
        k_cache[self._pages, self._slots] = k[:, 0]
        v_cache[self._pages, self._slots] = v[:, 0]
        return self.wrapper.run(q[:, 0], (k_cache, v_cache)).unsqueeze(1)
