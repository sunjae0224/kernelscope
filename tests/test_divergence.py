import pandas as pd
import pytest

from kernelscope.serve.divergence import bf16_ulp, divergence_events, tie_class


def _tokens(rows):
    return pd.DataFrame(rows, columns=["rid", "position", "token", "phase"])


def test_divergence_events_reports_the_first_differing_position_and_the_cascade():
    ref = _tokens([(0, p, 100 + p, "decode") for p in range(4)] + [(1, p, 200 + p, "decode") for p in range(4)]
                  + [(0, 0, 5, "prefill")])
    cand = ref.copy()
    cand.loc[(cand.rid == 1) & (cand.position >= 2) & (cand.phase == "decode"), "token"] += 1   # rid 1 diverges at 2
    e = divergence_events(ref, cand)
    assert len(e) == 1
    ev = e.iloc[0]
    assert (ev.rid, ev.first_position, ev.tokens, ev.after_first, ev.differing_after_first, bool(ev.cascade)) == (
        1, 2, 4, 2, 2, True)


def test_divergence_events_distinguishes_a_single_flip_from_a_cascade():
    ref = _tokens([(0, p, 100 + p, "decode") for p in range(6)])
    cand = ref.copy()
    cand.loc[cand.position == 1, "token"] = 999          # flips once, then agrees again
    ev = divergence_events(ref, cand).iloc[0]
    assert (ev.first_position, ev.differing_after_first, ev.after_first, bool(ev.cascade)) == (1, 1, 5, False)


def test_identical_histories_have_no_events():
    ref = _tokens([(0, p, 100 + p, "decode") for p in range(3)])
    assert divergence_events(ref, ref.copy()).empty


def test_bf16_ulp_follows_the_exponent():
    assert bf16_ulp(1.0) == pytest.approx(2 ** -7)
    assert bf16_ulp(16.0) == pytest.approx(0.125)
    assert bf16_ulp(31.9) == pytest.approx(0.125)
    assert bf16_ulp(32.0) == pytest.approx(0.25)


def test_tie_class_measures_the_margin_in_bf16_ulps_of_the_top_logit():
    assert tie_class(margin=0.125, top_logit=20.0) == "tie_1ulp"
    assert tie_class(margin=0.25, top_logit=20.0) == "tie_2ulp"
    assert tie_class(margin=0.5, top_logit=20.0) == "clear"
    assert tie_class(margin=0.0, top_logit=20.0) == "tie_1ulp"
