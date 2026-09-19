"""Real-hardware SM-count what-if: occupy n SMs while a target kernel runs.

Each blocker CTA requests the per-block shared-memory maximum, so exactly one fits on an SM and
no other kernel's CTA needing more than ~1 KiB of shared memory can share that SM. Blocker
CTAs spin on clock64() and touch no global memory, so they take SMs away without taking
bandwidth. Caveat: tiny kernels (e.g. split-KV's combine kernel, <= 640 B of shared memory)
may still land on a blocked SM; verify placement when that matters.
"""
import statistics

import torch

from kernelscope.bench.cuda_ext import load_ext


class SMBlocker:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.ext = load_ext()
        p = torch.cuda.get_device_properties(device)
        self.n_sms = p.multi_processor_count
        self.smem_bytes = getattr(p, "shared_memory_per_block_optin", 101376)
        self.stream = torch.cuda.Stream(device)
        self.cycles_per_us = self._calibrate()

    def _calibrate(self, cycles: int = 20_000_000) -> float:
        """clock64() ticks per microsecond at the current clocks (runs the blocker on one SM)."""
        rate = None
        for _ in range(3):                         # the first runs also ramp the clocks up
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self.stream):
                s.record()
                self.ext.launch_blocker(1, self.smem_bytes, cycles, self.stream.cuda_stream)
                e.record()
            e.synchronize()
            rate = cycles / (s.elapsed_time(e) * 1e3)
        return rate

    def block(self, n_sms: int, duration_us: float) -> None:
        """Occupy n_sms SMs for about duration_us, starting now (asynchronous, side stream)."""
        if not 0 <= n_sms < self.n_sms:
            raise ValueError(f"n_sms must be in [0, {self.n_sms}), got {n_sms}")
        self.ext.launch_blocker(n_sms, self.smem_bytes, int(duration_us * self.cycles_per_us), self.stream.cuda_stream)

    def time_blocked(self, fn, n_sms: int, iters: int = 10, warmup: int = 3) -> float:
        """Median CUDA-event time (µs) of fn on the current stream while n_sms SMs are blocked."""
        cur = torch.cuda.current_stream(self.device)
        t0 = []
        for _ in range(2):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(cur); fn(); e.record(cur); e.synchronize()
            t0.append(s.elapsed_time(e) * 1e3)
        spin_us = 4 * min(t0) * self.n_sms / max(1, self.n_sms - n_sms) + 200.0
        times = []
        for i in range(warmup + iters):
            self.block(n_sms, spin_us)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(cur); fn(); e.record(cur)
            torch.cuda.synchronize(self.device)
            if i >= warmup:
                times.append(s.elapsed_time(e) * 1e3)
        return statistics.median(times)
