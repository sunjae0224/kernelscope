"""Wall-clock kernel latency, measured in an unprofiled process with free-running clocks."""
import statistics
from time import perf_counter


def measure_latency(fn, warmup: int = 10, iters: int = 50, timer=None) -> dict:
    if timer is None:
        timer, name = _default_timer()
    else:
        name = "custom"
    for _ in range(warmup):
        timer(fn)
    samples = [timer(fn) for _ in range(iters)]
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "iters": iters,
        "timer": name,
        "samples_s": samples,
    }


def _default_timer():
    try:
        import torch
        if torch.cuda.is_available():
            def cuda_timer(fn):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                return start.elapsed_time(end) / 1e3
            return cuda_timer, "cuda_event"
    except ImportError:
        pass

    def pc_timer(fn):
        t = perf_counter()
        fn()
        return perf_counter() - t
    return pc_timer, "perf_counter"
