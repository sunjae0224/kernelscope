"""Iteration boundaries and L2 flushing for Nsight-free timing loops.

Before every timed call the harness launches one of two tiny Triton kernels whose names it
recognises in the profiler trace:
  kernelscope_l2_flush     cache_state="cold": writes FLUSH_FACTOR x L2 bytes, evicting the
                           previous call's data (the 4090's L2 is not LRU, so 2 x L2 can leave
                           some of it resident — spec F6)
  kernelscope_iter_marker  cache_state="warm": writes one float
Either one marks an iteration boundary, so the trace can be cut into iterations even when
torch.profiler drops events, and neither is ever counted as kernel time.
In serving, attention is cold: the other layers' weights stream through L2 between two
attention calls of the same layer (spec F8).
"""
MARKER_REGEX = r"kernelscope_(l2_flush|iter_marker)"
CACHE_STATES = ("warm", "cold")
FLUSH_FACTOR = 4
_BLOCK = 4096

try:
    import triton
    import triton.language as tl
except ImportError:          # CPU-only environments: hooks stay inactive
    triton = None

if triton is not None:
    @triton.jit
    def kernelscope_l2_flush(buf_ptr, n, val, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(buf_ptr + offs, tl.zeros((BLOCK,), tl.float32) + val, mask=offs < n)

    @triton.jit
    def kernelscope_iter_marker(buf_ptr, val):
        tl.store(buf_ptr, val)


class IterationHooks:
    def __init__(self, device: str, cache_state: str = "warm", l2_bytes: int | None = None):
        if cache_state not in CACHE_STATES:
            raise ValueError(f"cache_state must be one of {CACHE_STATES}, got {cache_state!r}")
        self.cache_state = cache_state
        self.active = str(device).startswith("cuda") and triton is not None
        self._n = 0
        if not self.active:
            return
        import torch
        if cache_state == "cold":
            if l2_bytes is None:
                l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
            self._buf = torch.empty(FLUSH_FACTOR * l2_bytes // 4, dtype=torch.float32, device=device)
        else:
            self._buf = torch.empty(1, dtype=torch.float32, device=device)

    def between(self) -> None:
        """Launch the boundary kernel that precedes the next timed call (no-op when inactive)."""
        if not self.active:
            return
        self._n += 1
        if self.cache_state == "cold":
            n = self._buf.numel()
            kernelscope_l2_flush[(triton.cdiv(n, _BLOCK),)](self._buf, n, float(self._n), BLOCK=_BLOCK)
        else:
            kernelscope_iter_marker[(1,)](self._buf, float(self._n))
