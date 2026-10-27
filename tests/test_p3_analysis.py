"""Unit tests for the 2026-10-25 analysis modules (validation of the replay scoring, vLLM step table, GSM8K statistics)."""
import numpy as np
import pandas as pd

from kernelscope.analysis import accuracy, validation, vllm_steps


def _cells():
    rows = []
    for source in ("a",):
        for stratum, n, pred, meas in (("lo", 4, 1.0, 1.1), ("hi", 4, 1.4, 1.3)):
            for i in range(n):
                row = {"id": f"{source}:{stratum}{i}", "source": source, "stratum": stratum, "stratum_steps": 100 if stratum == "lo" else 20,
                       "n_steps_with_key": i + 1, "pred_best_variant": "fd_s4_paged", "meas_best_variant": "fd_s4_paged"}
                for v in validation.VARIANTS:
                    row[f"pred_{v}_us"] = 100.0 * (pred if v == "flashdecoding_paged" else 1.0)
                    row[f"meas_{v}_us"] = 100.0 * (meas if v == "flashdecoding_paged" else 1.0)
                rows.append(row)
    return pd.DataFrame(rows)


def test_validation_weights_restore_stratum_shares():
    m = validation.prepare(_cells())
    assert np.isclose(m.groupby("stratum").w.sum()["lo"], 100) and np.isclose(m.groupby("stratum").w.sum()["hi"], 20)
    s = validation.source_stats(m, "a")
    assert s["pred_ge"] == 4 and s["meas_ge"] == 4 and s["precision"] == 1.0 and s["recall"] == 1.0
    assert np.isclose(s["w_pred_frac"], 20 / 120) and np.isclose(s["w_meas_frac"], 20 / 120)


def test_validation_key_step_weights_sum_to_stratum_steps():
    m = validation.prepare(_cells(), "key_steps")
    assert np.isclose(m.groupby("stratum").w.sum()["lo"], 100)


def test_bootstrap_interval_contains_point_estimate():
    m = validation.prepare(_cells())
    lo, hi = validation.bootstrap(m, "a", "w_meas_frac", draws=50)
    assert lo <= validation.source_stats(m, "a")["w_meas_frac"] <= hi


def test_wilson_and_mcnemar():
    lo, hi = accuracy.wilson(156, 200)
    assert 0.717 < lo < 0.718 and 0.831 < hi < 0.832
    assert accuracy.mcnemar_exact(0, 0) == 1.0
    assert abs(accuracy.mcnemar_exact(13, 15) - 0.8506) < 1e-3
    assert accuracy.mcnemar_exact(0, 6) < 0.05


def test_paired_counts_flips_and_changed_text():
    t = pd.DataFrame({"correct_marker_heuristic": [True, True, False, False], "correct_marker_x": [True, False, True, False],
                      "tokens_identical_x": [True, False, False, True], "pred_marker_heuristic": [1.0, 2.0, 3.0, 4.0],
                      "pred_marker_x": [1.0, 9.0, 3.0, 4.0]})
    p = accuracy.paired(t, "x")
    assert (p["right_to_wrong"], p["wrong_to_right"], p["text_differs"], p["prediction_changed"], p["text_differs_answer_same"]) == (1, 1, 2, 1, 1)


def _run(name, scenario, kernels, busy, attention):
    rows = []
    for i in range(4):
        rows.append({"context_requests": 0, "generation_requests": 32, "kernels": kernels, "gpu_busy_us": busy, "attention_us": attention,
                     "kv_write_us": 0.0, "gemm_us": busy - attention, "other_us": 0.0})
        rows.append({"context_requests": 0, "generation_requests": 32, "kernels": 12, "gpu_busy_us": 1000.0, "attention_us": 0.0,
                     "kv_write_us": 0.0, "gemm_us": 1000.0, "other_us": 0.0})
    return {"run": name, "backend": "B", "mode": "eager", "scenario": scenario, "unprofiled_decode_ms_per_step_median": 10.0, "iterations": rows}


def test_vllm_step_table_ignores_the_alternation_of_forward_and_sampling_rows():
    runs = [_run("r", "ragged", 600, 9000.0, 6000.0), _run("u", "uniform", 600, 5000.0, 2000.0)]
    t = vllm_steps.run_table(runs)
    assert np.isclose(t["r"]["steps"], 4.0)
    assert np.isclose(t["r"]["gpu_busy_ms"], 10.0) and np.isclose(t["r"]["attention_ms"], 6.0)
    assert np.isclose(t["r"]["not_gpu_busy_ms"], 0.0)
