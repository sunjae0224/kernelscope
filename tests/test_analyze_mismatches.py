from scripts.analyze_mismatches import collect_events, report, teacher_forced


def test_recorded_follow_up_runs_diverge_in_a_handful_of_events():
    ev = collect_events()
    run = ev[(ev.model == "qwen4b") & (ev.scenario == "arrivals") & (ev.seed == 0) & (ev.policy == "model") & (ev.repeat == 0)]
    assert sorted(zip(run.rid, run.first_position)) == [(21, 14), (23, 25)]
    assert run.events.iloc[0] == 2 and run.token_mismatches.iloc[0] == 25
    clean = ev[(ev.scenario == "uniform")]
    assert (clean.events == 0).all()


def test_report_classifies_every_flipped_position_as_a_bf16_tie():
    teacher, manifest = teacher_forced()
    flips = teacher[~teacher.argmax_equal]
    assert len(flips) == 4 and set(flips.tie_class) <= {"tie_1ulp", "tie_2ulp"}
    md = report(collect_events(), teacher, manifest)
    assert "4/4건이 일치" in md
