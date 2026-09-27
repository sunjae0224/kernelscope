import pytest

from kernelscope.model.fit import prepare_rows
from kernelscope.model.validate import THRESHOLDS, TRAIN_B, TRAIN_L, error_stats, in_set, markdown, policy_stats
from tests.test_model_fit import _synthetic
from tests.test_model_predict import M, P


def test_perfect_model_has_zero_error_and_zero_regret():
    rows = prepare_rows(_synthetic(P), M)
    e = error_stats(rows, P)
    assert e["n"] == len(rows) and e["median_ape"] == pytest.approx(0.0, abs=1e-9)
    pol = policy_stats(rows, P)
    assert pol["model_max_regret"] == pytest.approx(0.0, abs=1e-9)
    assert pol["heuristic_max_regret"] >= 0.0


def test_sets_partition_the_rows_as_specified():
    rows = prepare_rows(_synthetic(P), M)
    v1 = [r for r in rows if in_set(r, "V1")]
    assert all(not r.workload.is_ragged and r.workload.H_kv == 8 for r in v1)
    assert all((r.workload.B, r.workload.L_kv) not in {(b, l) for b in TRAIN_B for l in TRAIN_L} for r in v1)
    v6 = [r for r in rows if in_set(r, "V6")]
    assert all(r.workload.is_ragged for r in v6)


def test_markdown_marks_pass_and_fail():
    md = markdown([{"set": "V1", "cache_state": "cold", "n": 10, "median_ape": 0.05, "p90_ape": 0.5,
                    "threshold": THRESHOLDS["V1"], "worst": []}])
    assert "FAIL" in md and "V1" in md
