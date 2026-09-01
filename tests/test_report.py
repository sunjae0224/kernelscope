import pandas as pd
import pytest

from kernelscope.analysis.report import CLOCK_MHZ_A100, summarize, verdict

K = "decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal"


def _rows(kernel, lat_s, kt_us, blocks, cov, gbps, util, base, bw2, sm2=None):
    r = [
        ("latency", "median_s", lat_s, 0, None),
        ("profile", "kernel_time_us", kt_us, 0, None),
        ("profile", "grid_blocks", blocks, 0, "k"),
        ("profile", "sm_coverage", cov, 0, "k"),
        ("profile", "grid_blocks", 8, 1, "combine"),        # second launch must not pollute launch-0 columns
        ("analytic", "achieved_gbps", gbps, 0, None),
        ("analytic", "dram_util", util, 0, None),
        ("sim:base", "gpu_tot_sim_cycle", base, 0, None),
        ("sim:bw_x2", "gpu_tot_sim_cycle", bw2, 0, None),
    ]
    if sm2 is not None:
        r.append(("sim:sm_x2", "gpu_tot_sim_cycle", sm2, 0, None))
    return [{"workload_key": K, "kernel": kernel, "backend": b, "metric": m, "unit": "", "value": float(v),
             "launch_idx": i, "note": n} for b, m, v, i, n in r]


DF = pd.DataFrame(_rows("fa2", 55e-6, 49.2, 8, 8 / 108, 85.0, 0.06, 68482, 68639, 68000)
                  + _rows("flashdecoding", 21.6e-6, 13.6, 64, 64 / 108, 310.0, 0.21, 25116, 23822))


def test_summarize_gives_one_wide_row_per_kernel_and_workload():
    s = summarize(DF)
    assert list(s.index.names) == ["kernel", "workload_key"]
    assert len(s) == 2
    fa2 = s.loc[("fa2", K)]
    assert fa2["latency_us"] == pytest.approx(55.0)
    assert fa2["kernel_time_us"] == 49.2
    assert fa2["grid_blocks"] == 8                     # launch 0 only
    assert fa2["sm_coverage"] == pytest.approx(8 / 108)
    assert fa2["achieved_gbps"] == 85.0
    assert fa2["dram_util"] == 0.06
    assert fa2["sim_cycles_base"] == 68482
    assert fa2["sim_us_base"] == pytest.approx(68482 / CLOCK_MHZ_A100)
    assert fa2["sim_vs_kernel_time"] == pytest.approx(68482 / CLOCK_MHZ_A100 / 49.2)


def test_sensitivity_columns_are_relative_cycle_change_per_variant():
    s = summarize(DF)
    assert s.loc[("fa2", K), "sens_bw_x2"] == pytest.approx(68639 / 68482 - 1)
    assert s.loc[("flashdecoding", K), "sens_bw_x2"] == pytest.approx(23822 / 25116 - 1)
    assert s.loc[("fa2", K), "sens_sm_x2"] == pytest.approx(68000 / 68482 - 1)
    assert pd.isna(s.loc[("flashdecoding", K), "sens_sm_x2"])   # variant not simulated for this cell


def test_summarize_tolerates_missing_tracks():
    only_hw = DF[~DF.backend.str.startswith("sim:")]
    s = summarize(only_hw)
    assert "sim_cycles_base" not in s.columns or s["sim_cycles_base"].isna().all()
    assert s.loc[("fa2", K), "kernel_time_us"] == 49.2


def test_verdict_names_the_resource_that_moves_the_needle():
    assert verdict({"sens_bw_x2": -0.40, "sens_l2_x2": -0.03, "sens_sm_x2": -0.02}) == "bandwidth-bound (bw_x2: -40%)"
    assert verdict({"sens_bw_x2": 0.002, "sens_sm_x2": -0.01}) == "insensitive (largest: sm_x2: -1%)"
    assert verdict({"sens_sm_x2": -0.35, "sens_bw_x2": -0.04}) == "parallelism-bound (sm_x2: -35%)"
    assert verdict({"sens_l2_x2": -0.22}) == "L2-capacity-bound (l2_x2: -22%)"
    assert verdict({}) == "no what-if data"


def test_verdict_calls_out_a_starved_grid_when_nothing_helps():
    # FA2 decode at B=1: 8 CTAs on 108 SMs, and every what-if changes nothing
    row = {"sens_bw_x2": 0.0006, "sens_bw_half": 0.002, "sens_sm_half": -0.0004, "sm_coverage": 8 / 108}
    assert verdict(row) == "starved: grid covers 7% of SMs, no resource helps (largest: bw_half: +0%)"
    # full coverage + insensitive stays plain insensitive
    row2 = {"sens_bw_x2": 0.002, "sens_sm_x2": -0.01, "sm_coverage": 1.0}
    assert verdict(row2) == "insensitive (largest: sm_x2: -1%)"


def test_with_verdicts_uses_the_launch0_coverage_column():
    from kernelscope.analysis.report import summarize, with_verdicts
    s = with_verdicts(summarize(DF))
    # fa2 fixture: every sensitivity < 1 % and launch-0 coverage 8/108 -> starved
    assert s.loc[("fa2", K), "verdict"].startswith("starved: grid covers 7% of SMs")
    # flashdecoding fixture: bw_x2 -5.2 % with coverage 59 % -> bandwidth-bound
    assert s.loc[("flashdecoding", K), "verdict"].startswith("bandwidth-bound")

