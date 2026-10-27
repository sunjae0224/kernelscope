import pandas as pd
import pytest

from kernelscope.analysis.dispatch import best_fixed_split, fixed_split_summary, fixed_split_table

R = "decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal"
U = "decode_B32_Lq1_Lkv512_Hq32_Hkv8_d128_float16_causal"
SPLITS = (1, 2, 4, 8, 16, 32, 64, 128)


def _kt(key, kernel, us, state="cold"):
    return {"workload_key": key, "kernel": kernel, "backend": "profile", "metric": "kernel_time_us",
            "unit": "us", "value": us, "launch_idx": 0, "note": None, "cache_state": state}


def _cell(key, heuristic, times, extra=()):
    rows = [_kt(key, "flashdecoding_paged", heuristic)]
    rows += [_kt(key, "fa2_paged" if s == 1 else f"fd_s{s}_paged", t) for s, t in zip(SPLITS, times)]
    return rows + list(extra)


# Ragged cell: best FA2 split is 16 (200 us); uniform cell: best is 1 (50 us) and split 128 costs 4x.
DF = pd.DataFrame(
    _cell(R, 1000.0, [1000, 600, 300, 220, 200, 210, 230, 260], [_kt(R, "flashinfer_paged_cudacore", 180.0)])
    + _cell(U, 50.0, [50, 55, 60, 70, 90, 120, 160, 200])
    + [_kt(R, "fd_s16_paged", 1.0, state="warm")])                       # other cache state: ignored


def test_table_holds_every_fixed_split_and_both_bests():
    t = fixed_split_table(DF).set_index("workload_key")
    assert t.loc[R, "s16_us"] == 200 and t.loc[R, "fa2_best_us"] == 200 and t.loc[R, "best_us"] == 180
    assert t.loc[R, "fa2_heuristic_us"] == 1000 and t.loc[U, "best_us"] == 50 and bool(t.loc[R, "ragged"])
    assert t.complete.all()


def test_summary_is_time_over_reference_per_policy():
    t = fixed_split_table(DF)
    s = fixed_split_summary(t).set_index("policy")
    assert s.loc["fixed:16", "loss_max"] == pytest.approx(90 / 50) and s.loc["fixed:16", "n"] == 2
    assert s.loc["fixed:1", "loss_max"] == pytest.approx(1000 / 200)
    assert s.loc["fa2_heuristic", "over_2"] == 1 and s.loc["fa2_best", "loss_max"] == 1.0
    # Against the overall best (FlashInfer included) fa2_best is not 1 on the ragged cell.
    a = fixed_split_summary(t, against="best").set_index("policy")
    assert a.loc["fa2_best", "loss_max"] == pytest.approx(200 / 180)


def test_best_fixed_split_minimises_the_worst_cell_and_breaks_ties_on_p90():
    t = fixed_split_table(DF)
    assert best_fixed_split(t) == 8          # worst: s8 -> max(220/200, 70/50) = 1.4; s16 -> 1.8; s4 -> 1.5
    assert best_fixed_split(t, criterion="loss_p90") == 8


def test_incomplete_cells_are_not_scored():
    t = fixed_split_table(DF[~((DF.workload_key == U) & (DF.kernel == "fd_s64_paged"))])
    assert not t.set_index("workload_key").loc[U, "complete"]
    assert fixed_split_summary(t).set_index("policy").loc["fixed:8", "n"] == 1
