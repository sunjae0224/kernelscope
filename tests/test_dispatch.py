import math

import pandas as pd
import pytest

from kernelscope.analysis.dispatch import dispatch_table, regret_summary

U = "decode_B1_Lq1_Lkv8192_Hq32_Hkv8_d128_float16_causal"
R = "decode_B32_Lq1_Lkv32768x2+1024x30_Hq32_Hkv8_d128_float16_causal"


def _kt(key, kernel, us, state="cold"):
    return {"workload_key": key, "kernel": kernel, "backend": "profile", "metric": "kernel_time_us",
            "unit": "us", "value": us, "launch_idx": 0, "note": None, "cache_state": state}


DF = pd.DataFrame([
    _kt(U, "fa2", 466.0), _kt(U, "flashdecoding", 63.4), _kt(U, "fd_s8", 60.1), _kt(U, "fd_s16", 61.1),
    _kt(R, "fa2", 3744.0), _kt(R, "flashdecoding", 3744.0), _kt(R, "fd_s4", 500.0), _kt(R, "fd_s8", 500.0),
    _kt(R, "fd_s8_paged", 400.0),                          # other family: ignored for dense
    _kt(U, "fd_s8", 20.0, state="warm"),                   # other cache state: ignored
])


def test_best_variant_and_heuristic_regret_per_workload():
    t = dispatch_table(DF, family="dense", cache_state="cold").set_index("workload_key")
    assert t.loc[U, "best_kernel"] == "fd_s8" and t.loc[U, "best_us"] == 60.1
    assert t.loc[U, "heuristic_regret"] == pytest.approx(63.4 / 60.1 - 1)
    assert t.loc[R, "heuristic_regret"] == pytest.approx(3744 / 500 - 1)
    assert t.loc[R, "fa2_regret"] == pytest.approx(3744 / 500 - 1)
    assert bool(t.loc[R, "ragged"]) and t.loc[R, "B"] == 32 and t.loc[R, "L_kv"] == 32768
    assert t.loc[R, "lens"] == "32768x2+1024x30"
    assert t.loc[U, "n_variants"] == 4


def test_table_is_sorted_by_regret_and_summary_reports_the_worst_cell():
    t = dispatch_table(DF)
    assert t.iloc[0]["workload_key"] == R
    s = regret_summary(t)
    assert s["cells"] == 2 and s["worst_key"] == R and s["cells_over_10pct"] == 1
    assert s["max_regret"] == pytest.approx(3744 / 500 - 1)


def test_missing_heuristic_gives_nan_regret():
    t = dispatch_table(DF[DF.kernel != "flashdecoding"])
    assert t["heuristic_regret"].isna().all()


def test_rows_without_cache_state_count_as_warm():
    legacy = DF.drop(columns=["cache_state"])
    assert dispatch_table(legacy, cache_state="cold").empty
    assert len(dispatch_table(legacy, cache_state="warm")) == 2
