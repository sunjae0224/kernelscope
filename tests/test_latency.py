from kernelscope.backends.realhw.latency import measure_latency


def test_reports_median_of_timed_iterations_only():
    calls = []
    samples = iter([9.0, 9.0, 1.0, 3.0, 2.0])  # 2 warmup samples then 3 timed

    def timer(fn):
        fn()
        return next(samples)

    result = measure_latency(lambda: calls.append(1), warmup=2, iters=3, timer=timer)
    assert result["median_s"] == 2.0
    assert result["min_s"] == 1.0
    assert result["iters"] == 3
    assert len(calls) == 5


def test_default_timer_runs_on_cpu_without_cuda():
    result = measure_latency(lambda: sum(range(1000)), warmup=1, iters=3)
    assert result["median_s"] >= 0.0
    assert result["timer"] in ("cuda_event", "perf_counter")


def test_before_hook_runs_outside_the_timed_region_before_every_call():
    order = []
    def timer(fn):
        order.append("timed"); fn(); return 1.0
    measure_latency(lambda: None, warmup=2, iters=3, timer=timer, before=lambda: order.append("before"))
    assert order == ["before", "timed"] * 5
