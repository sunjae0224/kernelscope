import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kernelscope.serve.engine import RunResult
from kernelscope.serve.equivalence import compare_results


def result(tokens=(4, 5, 6), logits=None):
    return RunResult(pd.DataFrame([dict(step=0, B=1, seq_ids="[0]", lens="[4]")]),
                     pd.DataFrame([dict(rid=0, position=i, token=t) for i, t in enumerate(tokens)]),
                     pd.DataFrame(), [] if logits is None else logits)


def test_first_mismatch_and_missing_tokens_are_strict():
    verdict = compare_results(result(), result((4, 7)))
    assert not verdict["passed"] and verdict["token_agreement"] == pytest.approx(1 / 3)
    assert verdict["missing_tokens"] == 1 and verdict["token_mismatches"] == 2
    assert "position=1" in verdict["first_mismatch"]


def test_logit_tolerance_does_not_excuse_greedy_token_mismatch():
    verdict = compare_results(result(logits=[torch.zeros(1, 3)]), result((4, 5, 7), [torch.ones(1, 3) * .001]))
    assert verdict["max_abs_logit_diff"] < .05 and not verdict["passed"]


def test_nonfinite_logits_and_schedule_mismatch_fail():
    a = result(logits=[torch.zeros(1, 3)])
    b = result(logits=[torch.full((1, 3), float("nan"))])
    assert not compare_results(a, b)["passed"]
    b = result(logits=[torch.zeros(1, 3)])
    b.steps.loc[0, "seq_ids"] = "[9]"
    assert not compare_results(a, b)["schedule_match"]


def test_missing_schedule_identity_or_empty_output_cannot_pass():
    a, b = result(), result()
    b.steps = b.steps.drop(columns="seq_ids")
    assert not compare_results(a, b)["passed"]
    a.tokens = a.tokens.iloc[:0]
    b.tokens = b.tokens.iloc[:0]
    assert not compare_results(a, b)["passed"]
