import pandas as pd
import pytest

from kernelscope.serve.divergence import bf16_ulp, classify_events, divergence_events, tie_class


def _tokens(rows):
    return pd.DataFrame(rows, columns=["rid", "position", "token", "phase"])


def _events(rows):
    return pd.DataFrame(rows, columns=["rid", "first_position", "tokens", "after_first", "differing_after_first", "cascade",
                                       "reference_token", "candidate_token"])


def _teacher(rows):
    return pd.DataFrame(rows, columns=["rid", "position", "argmax_equal", "reference_top1", "candidate_top1",
                                       "reference_top1_margin", "reference_gap_to_candidate_token",
                                       "reference_top1_logit", "max_abs_logit_diff"])


def _event(rid, position, reference_token=7, candidate_token=9):
    return (rid, position, 8, 8 - position, 8 - position, True, reference_token, candidate_token)


def _flip(rid, position, margin, top_logit, gap=None, reference_top1=7, candidate_top1=9, diff=0.25):
    """A teacher-forced row whose argmax changed from token 7 to token 9."""
    return (rid, position, False, reference_top1, candidate_top1, margin, margin if gap is None else gap, top_logit, diff)


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


def test_divergence_events_carry_both_tokens_at_the_first_differing_position():
    ref = _tokens([(1, p, 200 + p, "decode") for p in range(1, 5)])
    cand = ref.copy()
    cand.loc[cand.position >= 3, "token"] += 50
    ev = divergence_events(ref, cand).iloc[0]
    assert (ev.first_position, ev.reference_token, ev.candidate_token) == (3, 203, 253)
    truncated = divergence_events(ref, ref[ref.position < 3]).iloc[0]          # the candidate stopped early
    assert truncated.first_position == 3 and truncated.reference_token == 203 and pd.isna(truncated.candidate_token)


def test_classify_events_measures_the_margin_at_the_first_differing_position_in_ulps_of_the_top_logit():
    events = _events([_event(3, 5), _event(4, 2), _event(5, 1), _event(7, 6)])
    teacher = _teacher([_flip(3, 5, 0.125, 20.0),                               # one bf16 spacing at 16..32
                        (3, 6, True, 7, 7, 4.0, 0.0, 20.0, 0.25),                # a later position of the same request
                        _flip(4, 2, 0.25, 40.0),                                # 0.25 is a single spacing at 32..64
                        _flip(5, 1, 0.25, 20.0),                                # two spacings at 16..32
                        _flip(7, 6, 0.5, 20.0)])                                # four spacings
    out = classify_events(events, teacher).set_index("rid")
    assert out.tie_class.to_dict() == {3: "tie_1ulp", 4: "tie_1ulp", 5: "tie_2ulp", 7: "clear"}
    assert out.margin_ulps.to_dict() == {3: 1.0, 4: 1.0, 5: 2.0, 7: 4.0}
    assert out.reference_top1_logit.to_dict() == {3: 20.0, 4: 40.0, 5: 20.0, 7: 20.0}
    assert out.max_abs_logit_diff.to_dict() == {3: 0.25, 4: 0.25, 5: 0.25, 7: 0.25}
    assert out.teacher_reproduced.to_dict() == {3: True, 4: True, 5: True, 7: True}
    assert out.first_position.to_dict() == {3: 5, 4: 2, 5: 1, 7: 6}


def test_classify_events_uses_the_gap_to_the_token_the_candidate_chose_not_the_runner_up_margin():
    # The reference's top two are one spacing apart, but the candidate jumped to a token 12.0 below the top.
    out = classify_events(_events([_event(3, 5)]), _teacher([_flip(3, 5, margin=0.125, top_logit=20.0, gap=12.0)]))
    assert out.tie_class.tolist() == ["clear"] and out.reference_margin.tolist() == [12.0] and out.margin_ulps.tolist() == [96.0]


def test_classify_events_never_calls_an_unrecorded_position_a_tie():
    events = _events([_event(3, 5)])
    other_position = classify_events(events, _teacher([_flip(3, 4, 0.0, 20.0)]))
    no_diagnostic = classify_events(events, None)
    for out in (other_position, no_diagnostic):
        assert out.tie_class.tolist() == ["unclassified"]
        assert out.teacher_reproduced.isna().all() and out.margin_ulps.isna().all()
        assert "no teacher-forced row" in out.note.iloc[0]


def test_classify_events_rejects_a_diagnostic_recorded_for_another_token_history():
    # Same (rid, position), but the diagnostic's reference produced token 55 where this run's reference produced 7.
    out = classify_events(_events([_event(3, 5)]), _teacher([_flip(3, 5, 0.125, 20.0, reference_top1=55)]))
    assert out.tie_class.tolist() == ["unclassified"] and "reference token" in out.note.iloc[0]


def test_classify_events_flags_an_event_the_teacher_forced_run_did_not_reproduce():
    events = _events([_event(3, 5), _event(4, 2), _event(5, 1, candidate_token=None)])
    teacher = _teacher([(3, 5, True, 7, 7, 0.125, 0.0, 20.0, 0.1),               # no argmax change at all
                        _flip(4, 2, 0.125, 20.0, candidate_top1=11),            # changed, but to another token than 9
                        _flip(5, 1, 0.125, 20.0)])                              # the free run has no token there
    out = classify_events(events, teacher).set_index("rid")
    assert out.teacher_reproduced.to_dict() == {3: False, 4: False, 5: False}
    assert out.tie_class.to_dict() == {3: "tie_1ulp", 4: "tie_1ulp", 5: "tie_1ulp"}   # runner-up margin, kept as a lower bound
    assert out.note.str.contains("did not produce the same token").all()


def test_classify_events_of_no_events_keeps_the_columns():
    out = classify_events(_events([]), _teacher([_flip(3, 5, 0.125, 20.0)]))
    assert out.empty and {"rid", "first_position", "teacher_reproduced", "margin_ulps", "tie_class", "note"} <= set(out.columns)


def test_classify_events_refuses_a_diagnostic_without_the_top_logit():
    old = _teacher([_flip(3, 5, 0.125, 20.0)]).drop(columns="reference_top1_logit")
    with pytest.raises(ValueError, match="reference_top1_logit"):
        classify_events(_events([_event(3, 5)]), old)


def test_classify_events_refuses_rows_of_several_policies():
    mixed = _teacher([_flip(3, 5, 0.125, 20.0), _flip(3, 5, 4.0, 20.0)])
    with pytest.raises(ValueError, match="one policy"):
        classify_events(_events([_event(3, 5)]), mixed)
