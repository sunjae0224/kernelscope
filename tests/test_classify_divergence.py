import pandas as pd
import pytest

from kernelscope.serve.divergence import campaign_events, summarize_campaign_events
from scripts.classify_divergence import main


def _tokens(flips=()):
    """Two requests, one prefill token and three decode tokens each; `flips` are (rid, first differing position)."""
    rows = []
    for rid in (0, 1):
        start = dict(flips).get(rid)
        for position in range(4):
            token = 100 * (rid + 1) + position + (1000 if start is not None and position >= start else 0)
            rows.append(dict(rid=rid, step=position, position=position, token=token, t_us=float(position),
                             phase="prefill" if position == 0 else "decode"))
    return pd.DataFrame(rows)


def _write_tokens(campaign, scenario, policy, repeat, tokens):
    directory = campaign / scenario / policy / f"repeat_{repeat:03d}"
    directory.mkdir(parents=True)
    tokens.to_parquet(directory / "tokens.parquet", index=False)


def _write_run(campaign, scenario, policy, repeat, flips=()):
    _write_tokens(campaign, scenario, policy, repeat, _tokens(flips))


def _write_teacher(campaign, scenario, rows):
    directory = campaign / "numerics" / scenario
    directory.mkdir(parents=True)
    columns = ["policy", "rid", "position", "argmax_equal", "reference_top1", "candidate_top1", "reference_top1_margin",
               "reference_gap_to_candidate_token", "reference_top1_logit", "max_abs_logit_diff"]
    pd.DataFrame(rows, columns=columns).to_csv(directory / "teacher_forced_logits.csv", index=False)


# Request 1 at position 2: the reference produced 202, a flipped candidate produces 1202.
TIE_AT_1_2 = ("table", 1, 2, False, 202, 1202, 0.125, 0.125, 20.0, 0.25)


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "hybrid_test"
    for repeat in (0, 1):
        _write_run(root, "arrivals", "heuristic", repeat)
        _write_run(root, "arrivals", "table", repeat, flips=[(1, 2)])
        _write_run(root, "arrivals", "hybrid", repeat)
        _write_run(root, "uniform", "heuristic", repeat)
        _write_run(root, "uniform", "table", repeat)
    _write_teacher(root, "arrivals", [TIE_AT_1_2, ("table", 1, 3, True, 203, 203, 3.0, 0.0, 20.0, 0.25),
                                      ("hybrid", 1, 2, True, 202, 202, 0.125, 0.0, 20.0, 0.0)])
    (root / "logs").mkdir()
    return root


def test_campaign_events_classifies_each_runs_events_and_keeps_clean_runs_visible(campaign):
    events = campaign_events(campaign)
    assert sorted(set(events.scenario)) == ["arrivals", "uniform"]
    table = events[(events.scenario == "arrivals") & (events.policy == "table")]
    assert table.repeat.tolist() == [0, 1] and table.events.tolist() == [1, 1]
    assert table.rid.tolist() == [1, 1] and table.first_position.tolist() == [2, 2]
    assert table.reference_token.tolist() == [202, 202] and table.candidate_token.tolist() == [1202, 1202]
    assert table.tie_class.tolist() == ["tie_1ulp", "tie_1ulp"] and table.teacher_reproduced.tolist() == [True, True]
    assert table.token_mismatches.tolist() == [2, 2] and table.tokens_compared.tolist() == [8, 8]
    clean = events[events.policy == "hybrid"]
    assert clean.events.tolist() == [0, 0] and clean.token_mismatches.tolist() == [0, 0] and clean.rid.isna().all()
    assert events[events.scenario == "uniform"].events.tolist() == [0, 0]


def test_events_of_a_scenario_without_a_diagnostic_stay_unclassified(tmp_path):
    root = tmp_path / "c"
    _write_run(root, "ragged", "heuristic", 0)
    _write_run(root, "ragged", "table", 0, flips=[(0, 3)])
    events = campaign_events(root)
    assert events.tie_class.tolist() == ["unclassified"] and events.first_position.tolist() == [3]


def test_summary_counts_distinct_positions_per_tie_class(campaign):
    _write_run(campaign, "arrivals", "table", 2, flips=[(0, 3), (1, 2)])   # a third repeat with one more event
    _write_run(campaign, "arrivals", "heuristic", 2)
    _write_run(campaign, "arrivals", "hybrid", 2)
    summary = summarize_campaign_events(campaign_events(campaign)).set_index(["scenario", "policy"])
    table = summary.loc[("arrivals", "table")]
    assert (table.repeats, table.events_repeat_max, table.distinct_positions) == (3, 2, 2)
    assert (table.tie_1ulp, table.tie_2ulp, table.clear, table.unclassified) == (1, 0, 0, 1)   # (0, 3) has no teacher row
    assert (table.not_reproduced, table.missing_tokens, table.mismatches_outside_events) == (0, 0, 0)
    assert not table.same_positions_every_repeat
    hybrid = summary.loc[("arrivals", "hybrid")]
    assert (hybrid.repeats, hybrid.distinct_positions, hybrid.clear) == (3, 0, 0) and hybrid.same_positions_every_repeat


def test_cli_writes_the_event_table_and_passes_when_every_event_is_a_reproduced_tie(campaign, capsys):
    main(["--campaign", str(campaign)])
    written = pd.read_csv(campaign / "divergence.csv", dtype=str)
    assert len(written) == 6
    events = written[written.events != "0"]
    assert events.tie_class.tolist() == ["tie_1ulp", "tie_1ulp"]
    assert events.rid.tolist() == ["1", "1"] and events.first_position.tolist() == ["2", "2"]      # integers, not 1.0 / 2.0
    assert str(campaign / "divergence.csv") in capsys.readouterr().out


def test_cli_fails_on_a_clear_event(campaign):
    _write_teacher(campaign, "uniform", [("table", 0, 1, False, 101, 1101, 1.0, 1.0, 20.0, 0.25)])   # 8 spacings
    for repeat in (0, 1):
        (campaign / "uniform" / "table" / f"repeat_{repeat:03d}" / "tokens.parquet").unlink()
        _tokens(flips=[(0, 1)]).to_parquet(campaign / "uniform" / "table" / f"repeat_{repeat:03d}" / "tokens.parquet", index=False)
    with pytest.raises(SystemExit, match="1 clear"):
        main(["--campaign", str(campaign)])
    assert (campaign / "divergence.csv").exists()                                        # the evidence is still written


def test_cli_fails_on_an_event_without_a_diagnostic(tmp_path):
    root = tmp_path / "c"
    _write_run(root, "ragged", "heuristic", 0)
    _write_run(root, "ragged", "table", 0, flips=[(0, 3)])
    with pytest.raises(SystemExit, match="1 unclassified"):
        main(["--campaign", str(root)])


def test_cli_fails_on_an_event_the_teacher_forced_run_did_not_reproduce(tmp_path):
    root = tmp_path / "c"
    _write_run(root, "ragged", "heuristic", 0)
    _write_run(root, "ragged", "table", 0, flips=[(1, 2)])
    _write_teacher(root, "ragged", [("table", 1, 2, True, 202, 202, 0.125, 0.0, 20.0, 0.1)])    # no argmax change there
    summary = summarize_campaign_events(campaign_events(root))
    assert summary.not_reproduced.tolist() == [1] and summary.tie_1ulp.tolist() == [1]
    with pytest.raises(SystemExit, match="1 not reproduced"):
        main(["--campaign", str(root)])


def test_a_token_difference_outside_every_decode_event_is_not_hidden(tmp_path):
    root = tmp_path / "c"
    _write_run(root, "ragged", "heuristic", 0)
    tokens = _tokens()
    tokens.loc[(tokens.rid == 0) & (tokens.position == 0), "token"] = 7          # only the prefill token differs
    _write_tokens(root, "ragged", "table", 0, tokens)
    summary = summarize_campaign_events(campaign_events(root))
    assert summary.distinct_positions.tolist() == [0] and summary.mismatches_outside_events.tolist() == [1]
    with pytest.raises(SystemExit, match="outside"):
        main(["--campaign", str(root)])


def test_a_truncated_token_history_is_a_structural_failure_not_a_tie(tmp_path):
    root = tmp_path / "c"
    _write_run(root, "ragged", "heuristic", 0)
    tokens = _tokens()
    _write_tokens(root, "ragged", "table", 0, tokens[~((tokens.rid == 1) & (tokens.position >= 2))])   # request 1 stops early
    _write_teacher(root, "ragged", [TIE_AT_1_2])
    summary = summarize_campaign_events(campaign_events(root))
    assert summary.missing_tokens.tolist() == [2] and summary.not_reproduced.tolist() == [1]
    with pytest.raises(SystemExit, match="2 missing tokens"):
        main(["--campaign", str(root)])


def test_runs_that_cannot_be_paired_with_the_reference_are_an_error(tmp_path):
    root = tmp_path / "c"
    _write_run(root, "ragged", "heuristic", 0)
    _write_run(root, "ragged", "table", 0)
    _write_run(root, "ragged", "table", 1)                                        # no reference repeat 1
    with pytest.raises(ValueError, match="ragged/table.*repeat_001"):
        campaign_events(root)
    with pytest.raises(SystemExit, match="ragged/table"):
        main(["--campaign", str(root)])
    (root / "ragged" / "hybrid").mkdir()                                          # a policy without any recorded run
    _write_run(root, "ragged", "heuristic", 1)
    with pytest.raises(ValueError, match="ragged/hybrid"):
        campaign_events(root)


def test_cli_refuses_a_folder_without_runs(tmp_path):
    with pytest.raises(SystemExit, match="no runs"):
        main(["--campaign", str(tmp_path)])
    assert not (tmp_path / "divergence.csv").exists()
