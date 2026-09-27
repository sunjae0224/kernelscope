"""Paged KV-cache helpers for flash-attn's ``block_table`` layout.

A pool ``[num_pages, PAGE, H_kv, d]`` holds every sequence's K (or V) in fixed-size pages;
``block_table[b, i]`` is the pool page holding tokens ``[i*PAGE, (i+1)*PAGE)`` of sequence
``b``. flash-attn requires the page size to be a multiple of 256. Pages are handed out in a
seeded random order, like a fragmented pool in a serving engine.
"""
import math

import torch

PAGE = 256


def build_block_table(lens, page: int = PAGE, seed: int = 0) -> tuple[torch.Tensor, int]:
    """CPU int32 table ``[B, max_pages]`` and the pool size. Unused slots hold page 0, which
    the kernel never reads because it stops at each sequence's length."""
    need = [math.ceil(n / page) for n in lens]
    total = sum(need)
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).to(torch.int32)
    table = torch.zeros(len(lens), max(need), dtype=torch.int32)
    pos = 0
    for b, n in enumerate(need):
        table[b, :n] = perm[pos:pos + n]
        pos += n
    return table, total


def gather_from_pool(pool: torch.Tensor, lens, table: torch.Tensor, L: int, page: int = PAGE) -> torch.Tensor:
    """Dense ``[B, L, H, d]`` copy of a paged cache (for the reference check); zeros past each length."""
    out = pool.new_zeros(len(lens), L, pool.shape[2], pool.shape[3])
    for b, n in enumerate(lens):
        for i in range(math.ceil(n / page)):
            lo, hi = i * page, min(n, (i + 1) * page)
            out[b, lo:hi] = pool[int(table[b, i]), : hi - lo]
    return out
