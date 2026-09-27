import numpy as np
import pandas as pd
import pytest

from kernelscope.model.fit import fit_state, nelder_mead, prepare_rows, row_time_us
from kernelscope.model.params import KindParams, StateParams
from kernelscope.model.predict import predict
from tests.test_model_predict import M, P

KEYS = ["decode_B1_Lq1_Lkv4096_Hq32_Hkv8_d128_float16_causal",
        "decode_B4_Lq1_Lkv16384_Hq32_Hkv8_d128_float16_causal",
        "decode_B32_Lq1_Lkv8192x2+1024x30_Hq32_Hkv8_d128_float16_causal",
        "decode_B64_Lq1_Lkv2048_Hq32_Hkv8_d128_float16_causal"]
PLUGINS = ["fa2", "flashdecoding", "fd_s4", "fd_s16", "fa2_paged", "fd_s8_paged"]


def _synthetic(params, noise=0.0):
    from kernelscope.workload import Workload
    rng = np.random.default_rng(0)
    rows = []
    for key in KEYS:
        for p in PLUGINS:
            t = predict(p, Workload.from_key(key), M, params, "cold").time_us * (1 + noise * rng.standard_normal())
            rows.append({"workload_key": key, "kernel": p, "backend": "profile", "metric": "kernel_time_us",
                         "unit": "us", "value": t, "launch_idx": 0, "note": None, "cache_state": "cold"})
    rows.append({"workload_key": KEYS[0], "kernel": "sdpa_flash", "backend": "profile", "metric": "kernel_time_us",
                 "unit": "us", "value": 1.0, "launch_idx": 0, "note": None, "cache_state": "cold"})
    return pd.DataFrame(rows)


def test_nelder_mead_finds_a_quadratic_minimum():
    x, fx = nelder_mead(lambda v: float(((v - np.array([1.0, -2.0])) ** 2).sum()), np.zeros(2), maxiter=500)
    assert np.allclose(x, [1.0, -2.0], atol=1e-3) and fx < 1e-6


def test_prepare_rows_keeps_flash_variants_and_precomputes_the_launch():
    rows = prepare_rows(_synthetic(P), M)
    assert len(rows) == len(KEYS) * len(PLUGINS)                 # sdpa_flash skipped
    r = [r for r in rows if r.plugin == "fd_s16" and r.workload.B == 1][0]
    assert r.kind == "split" and r.splits == 16 and r.slots == 1 and len(r.keys) == 128


def test_row_time_matches_predict():
    from kernelscope.workload import Workload
    for r in prepare_rows(_synthetic(P), M):
        assert row_time_us(r, P.states["cold"]) == pytest.approx(
            predict(r.plugin, Workload.from_key(r.workload.key()), M, P, "cold").time_us)


def test_fit_recovers_known_parameters_from_noise_free_data():
    rows = prepare_rows(_synthetic(P), M)
    k0 = KindParams(cost_us_per_key=0.03, t0_us=5.0, t_empty_us=0.2, gamma=0.8, t_fixed_us=3.0)
    init = StateParams(kinds={"nonsplit": k0, "split": k0, "split_paged": k0}, comb_a_us=10.0, comb_b_us=0.003)
    got = fit_state(rows, init, maxiter=600, log=None)
    for kind in ("nonsplit", "split", "split_paged"):
        assert got.kinds[kind].cost_us_per_key == pytest.approx(0.05, rel=0.05)
    errs = [abs(row_time_us(r, got) / r.measured_us - 1) for r in rows]
    assert np.median(errs) < 0.02
