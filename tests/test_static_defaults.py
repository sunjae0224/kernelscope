import pandas as pd
import pytest

from kernelscope.analysis.dispatch import static_default_losses, static_default_summary

R = "decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal"
U = "decode_B32_Lq1_Lkv512_Hq32_Hkv8_d128_float16_causal"


def _kt(key, kernel, us, state="cold"):
    return {"workload_key": key, "kernel": kernel, "backend": "profile", "metric": "kernel_time_us",
            "unit": "us", "value": us, "launch_idx": 0, "note": None, "cache_state": state}


DF = pd.DataFrame([
    _kt(R, "flashdecoding_paged", 1000.0), _kt(R, "fa2_paged", 1000.0), _kt(R, "fd_s8_paged", 300.0),
    _kt(R, "fd_s16_paged", 280.0), _kt(R, "flashinfer_paged", 650.0), _kt(R, "flashinfer_paged_cudacore", 270.0),
    _kt(U, "flashdecoding_paged", 50.0), _kt(U, "fd_s8_paged", 60.0), _kt(U, "flashinfer_paged", 49.0),
    _kt(U, "flashinfer_paged_cudacore", 55.0),
    _kt(R, "fd_s8_paged", 100.0, state="warm"),                        # other cache state: ignored
])


def test_each_library_default_is_measured_against_the_best_of_every_variant():
    t = static_default_losses(DF, cache_state="cold").set_index("workload_key")
    assert t.loc[R, "best_kernel"] == "flashinfer_paged_cudacore" and t.loc[R, "best_us"] == 270.0
    assert t.loc[R, "fa2_heuristic_loss"] == pytest.approx(1000 / 270)
    assert t.loc[R, "flashinfer_tc_loss"] == pytest.approx(650 / 270)
    assert t.loc[R, "flashinfer_cc_loss"] == pytest.approx(1.0)
    # The dispatcher restricted to FA2 split counts: its best variant against the overall best.
    assert t.loc[R, "fa2_best_kernel"] == "fd_s16_paged" and t.loc[R, "fa2_best_loss"] == pytest.approx(280 / 270)
    assert t.loc[U, "best_kernel"] == "flashinfer_paged" and t.loc[U, "fa2_heuristic_loss"] == pytest.approx(50 / 49)
    assert bool(t.loc[R, "ragged"]) and not bool(t.loc[U, "ragged"]) and t.loc[R, "lens"] == "32768+512x31"


def test_cells_without_a_flashinfer_measurement_keep_nan_for_that_default():
    t = static_default_losses(DF[DF.kernel != "flashinfer_paged"], cache_state="cold").set_index("workload_key")
    assert pd.isna(t.loc[R, "flashinfer_tc_loss"]) and t.loc[R, "best_kernel"] == "flashinfer_paged_cudacore"
    assert not t.loc[R, "complete"] and t.loc[U, "complete"] is not None and not t.loc[U, "complete"]
    # The summary scores every default over the same complete cells and says how many it left out.
    s = static_default_summary(t.reset_index()).set_index(["cells", "default"])
    assert s.loc[("ragged", "fa2_heuristic"), "n"] == 0 and s.loc[("ragged", "fa2_heuristic"), "incomplete"] == 1
    assert static_default_losses(DF).set_index("workload_key").complete.all()


def test_summary_reports_each_default_over_ragged_and_uniform_cells():
    s = static_default_summary(static_default_losses(DF)).set_index(["cells", "default"])
    assert s.loc[("ragged", "fa2_heuristic"), "n"] == 1 and s.loc[("ragged", "fa2_heuristic"), "loss_max"] == pytest.approx(1000 / 270)
    assert s.loc[("ragged", "flashinfer_tc"), "over_1.25"] == 1 and s.loc[("ragged", "flashinfer_cc"), "over_1.25"] == 0
    assert s.loc[("uniform", "fa2_heuristic"), "over_1.25"] == 0 and s.loc[("uniform", "fa2_best"), "loss_median"] == pytest.approx(50 / 49)
    assert set(s.index.get_level_values("default")) == {"fa2_heuristic", "flashinfer_tc", "flashinfer_cc", "fa2_best"}
    assert (s["incomplete"] == 0).all()


def test_threshold_counts_are_strict():
    rows = [_kt(R, "flashdecoding_paged", 125.0), _kt(R, "fd_s8_paged", 100.0), _kt(R, "flashinfer_paged", 110.0),
            _kt(R, "flashinfer_paged_cudacore", 100.0)]
    s = static_default_summary(static_default_losses(pd.DataFrame(rows))).set_index(["cells", "default"])
    assert s.loc[("ragged", "fa2_heuristic"), "over_1.25"] == 0 and s.loc[("ragged", "fa2_heuristic"), "over_1.1"] == 1
    assert s.loc[("ragged", "flashinfer_tc"), "over_1.1"] == 0
