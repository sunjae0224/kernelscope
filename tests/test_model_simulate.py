import numpy as np
import pytest

from kernelscope.model.simulate import simulate_launch

BASE = dict(n_sm=1, slots_per_sm=1, sm_order=np.array([0]), cost_us_per_key=0.05, t0_us=0.0, t_empty_us=0.0,
            gamma=0.5, bytes_per_key=512, bw_bytes_per_us=1e12)


def run(keys, **kw):
    return simulate_launch(np.array(keys), **{**BASE, **kw})


def test_single_cta_is_startup_plus_keys_times_cost():
    assert run([1000], t0_us=2.0).makespan_us == pytest.approx(52.0)


def test_two_resident_ctas_share_their_sm():
    assert run([1000, 1000], slots_per_sm=2).makespan_us == pytest.approx(100.0)


def test_a_cta_speeds_up_when_its_partner_leaves():
    # both at 10 keys/us until the short one ends at 20 us, then 20 keys/us for the last 800 keys
    assert run([1000, 200], slots_per_sm=2).makespan_us == pytest.approx(60.0)


def test_bandwidth_cap_scales_every_streaming_cta():
    r = run([1000] * 4, n_sm=4, sm_order=np.arange(4), bw_bytes_per_us=20480.0)
    assert r.makespan_us == pytest.approx(100.0)          # 4 x 10240 B/us asked, 20480 available
    assert r.bw_bound_us == pytest.approx(100.0)


def test_waves_when_ctas_outnumber_slots():
    assert run([100] * 4, n_sm=2, sm_order=np.arange(2), cost_us_per_key=0.1).makespan_us == pytest.approx(20.0)


def test_empty_ctas_cost_only_their_fixed_time():
    assert run([0] * 10, t_empty_us=0.5, t0_us=9.0).makespan_us == pytest.approx(5.0)


def test_block_i_and_i_plus_n_sm_share_an_sm():
    # 4 SMs x 2 slots; CTAs 0 and 4 are long and land on the same SM, so they share it
    keys = [1000, 10, 10, 10, 1000, 10, 10, 10]
    r = run(keys, n_sm=4, slots_per_sm=2, sm_order=np.array([0, 2, 1, 3]), cost_us_per_key=0.01)
    assert r.makespan_us == pytest.approx(20.0)            # 1000 keys at gamma/cost = 50 keys/us


def test_many_equal_ctas_retire_in_batches():
    r = run([500] * 4096, n_sm=128, sm_order=np.arange(128))
    assert r.makespan_us == pytest.approx(32 * 25.0)
    assert r.events <= 2 * 32 + 2


def test_large_ragged_split_grid_is_fast_enough():
    import time
    keys = np.zeros(65536); keys[:256] = 128                # fd_s128-like grid, mostly empty CTAs
    t = time.perf_counter()
    run(keys, n_sm=128, sm_order=np.arange(128), t_empty_us=0.5)
    assert time.perf_counter() - t < 2.0
