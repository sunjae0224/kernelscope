import pytest

from scripts.evaluate_hybrid import run


@pytest.fixture(scope="module")
def results():
    return run(deltas=(0.0, 0.10))


def test_leave_one_out_replay_covers_every_validation_set(results):
    assert [(r["set"], r["cache_state"]) for r in results] == [("V1", "cold"), ("V1", "warm"), ("V5", "cold"), ("V6", "cold")]
    for r in results:
        assert {"heuristic", "model", "table", "hybrid_0", "hybrid_0.1"} <= set(r["policies"])
        assert r["policies"]["model"]["n"] == r["policies"]["hybrid_0.1"]["n"] > 0


def test_hybrid_with_zero_delta_reproduces_the_model_regret(results):
    for r in results:
        assert r["policies"]["hybrid_0"] == r["policies"]["model"]


def test_model_regret_matches_the_documented_validation_table(results):
    v6 = next(r for r in results if r["set"] == "V6")["policies"]
    assert v6["model"]["max"] == pytest.approx(0.245, abs=0.001)
    assert v6["heuristic"]["max"] == pytest.approx(11.527, abs=0.001)
