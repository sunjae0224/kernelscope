"""Triton streaming-read kernel with an explicit grid, for bandwidth measurements.

Each of G programs reads BLOCK-element chunks at stride BLOCK*G; in repetition r program p reads
column (p + r) mod G, so every load depends on r and a program never re-reads its own column from
L1. ``num_chunks`` still refuses one chunk per program because with G = 1 the column never changes.
"""
import statistics

import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def kernelscope_stream_read(x_ptr, out_ptr, n, chunks, reps, BLOCK: tl.constexpr, G: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), tl.float32)
    for r in range(reps):
        # Read another program's column each repetition: every load depends on r (no hoisting
        # out of the repetition loop) and no program re-reads its own column from L1.
        base = ((pid + r) % G) * BLOCK + tl.arange(0, BLOCK)
        for c in range(chunks):
            offs = c * BLOCK * G + base
            acc += tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


def num_chunks(n_elems: int, G: int, block: int = BLOCK) -> int:
    c = -(-n_elems // (block * G))
    if c < 2:
        raise ValueError(f"{n_elems} elements over G={G} programs is one chunk per program; "
                         "the compiler could hoist the load out of the repetition loop — use a larger buffer or smaller G")
    return c


def stream_gbps(nbytes: int, G: int, *, target_bytes: int = 2 << 30, num_warps: int = 4, iters: int = 10,
                device="cuda") -> float:
    """Achieved read bandwidth (GB/s) of G programs streaming an fp16 buffer of ``nbytes``."""
    n = nbytes // 2
    chunks = num_chunks(n, G)
    reps = max(1, round(target_bytes / nbytes))
    x = torch.randn(n, device=device, dtype=torch.float16)
    out = torch.empty(G, device=device, dtype=torch.float32)

    def launch():
        kernelscope_stream_read[(G,)](x, out, n, chunks, reps, BLOCK=BLOCK, G=G, num_warps=num_warps)

    for _ in range(2):
        launch()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        launch()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) / 1e3)
    return nbytes * reps / statistics.median(times) / 1e9
