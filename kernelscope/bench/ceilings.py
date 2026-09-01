"""Measured roofline ceilings — no profiler, just torch.

  hbm_gbps            best of a 1 GiB fp16 copy (read+write) and add_ (2 reads + write)
  fp16_matmul_tflops  best fp16 GEMM over a few sizes (short bursts: an A100 at its
                      power cap throttles on multi-ms GEMMs, and our kernels are µs-scale)

Why these sizes: tensors over 2^31 bytes push torch onto 64-bit-index kernels
(~2x slower — measured 1762 vs 774 GB/s), so the copy stays at 1 GiB. No L2
ceiling: a single-launch torch op over an L2-resident tensor is launch-bound
and would report a meaningless number.

Save the dict as machine_ceilings.json and feed it to `kernelscope sweep --ceilings`
so achieved bandwidth / TFLOPS become fractions of what this box actually does.
"""
from time import perf_counter

DEFAULT_COPY_BYTES = 1 << 30
DEFAULT_MATMUL_NS = (4096, 8192)


def gbps(nbytes: float, seconds: float) -> float:
    return nbytes / seconds / 1e9


def tflops(flops: float, seconds: float) -> float:
    return flops / seconds / 1e12


def _timer(device: str):
    import torch
    if str(device).startswith("cuda"):
        def t(fn):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); e.synchronize()
            return s.elapsed_time(e) / 1e3
        return t

    def t(fn):
        t0 = perf_counter(); fn(); return perf_counter() - t0
    return t


def _best(fn, timer, iters):
    for _ in range(2):
        timer(fn)
    return min(timer(fn) for _ in range(iters))


def measure_ceilings(device="cuda", copy_bytes=DEFAULT_COPY_BYTES, matmul_ns=DEFAULT_MATMUL_NS, iters=10) -> dict:
    import torch
    timer = _timer(device)
    cuda = str(device).startswith("cuda")
    dtype = torch.float16 if cuda else torch.float32
    esz = torch.tensor([], dtype=dtype).element_size()

    x = torch.empty(copy_bytes // esz, dtype=dtype, device=device)
    y = torch.empty_like(x)
    copy_gbps = gbps(2 * copy_bytes, _best(lambda: y.copy_(x), timer, iters))
    add_gbps = gbps(3 * copy_bytes, _best(lambda: y.add_(x), timer, iters))
    del x, y

    by_n = {}
    for n in matmul_ns:
        a = torch.randn(n, n, dtype=dtype, device=device)
        b = torch.randn(n, n, dtype=dtype, device=device)
        by_n[str(n)] = tflops(2 * n ** 3, _best(lambda: a @ b, timer, max(3, iters // 2)))
        del a, b
    best_n = max(by_n, key=by_n.get)

    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if cuda else "cpu",
        "dtype": str(dtype).split(".")[-1],
        "copy_bytes": copy_bytes, "matmul_ns": list(matmul_ns), "iters": iters,
        "hbm_copy_gbps": copy_gbps, "hbm_add_gbps": add_gbps, "hbm_gbps": max(copy_gbps, add_gbps),
        "matmul_tflops_by_n": by_n, "matmul_best_n": int(best_n), "fp16_matmul_tflops": by_n[best_n],
    }
