import pandas as pd
import pytest

from kernelscope.serve.scenarios import load_scenario
from scripts.trace_scenario import (arrival_steps, cap_window, choose, regret_count, rows_trace, scenario_rows,
                                    scenario_yaml, window_stats)
from kernelscope.analysis.traffic import ReplayConfig, replay


def _window():
    return pd.DataFrame({"arrival_s": [10.0, 10.0, 10.031, 10.06, 11.0], "prompt_tokens": [300, 9000, 500, 500, 700],
                         "output_tokens": [50, 400, 256, 10, 300]})


def test_cap_window_counts_only_clipped_values():
    capped, n_prompt, n_output = cap_window(_window(), prompt_cap=4096, output_cap=256)
    assert (n_prompt, n_output) == (1, 2)
    assert capped.prompt_tokens.max() == 4096 and capped.output_tokens.max() == 256
    assert _window().prompt_tokens.max() == 9000          # the input frame is untouched
    _, n_prompt, n_output = cap_window(_window())
    assert (n_prompt, n_output) == (0, 0)


def test_arrival_steps_round_up_and_ignore_float_noise():
    assert arrival_steps([0.0, 0.001, 0.03, 0.031, 0.09, 0.6]) == [0, 1, 1, 2, 3, 20]
    assert arrival_steps([0.1 + 0.2]) == [10]                  # 0.30000000000000004 / 0.03 must not become 11


def test_scenario_rows_start_at_step_zero_and_keep_trace_lengths():
    rows = scenario_rows(_window())
    assert rows[0][0] == 0 and rows[1][0] == 0 and rows[2][0] == 2
    assert [r[1] for r in rows] == [300, 9000, 500, 500, 700]


def test_yaml_round_trips_through_the_scenario_loader(tmp_path):
    rows = [(0, 512, 64), (0, 512, 64), (3, 4096, 128), (3, 512, 64)]
    path = tmp_path / "s.yaml"
    path.write_text(scenario_yaml(rows, "line one\nline two"))
    requests, _ = load_scenario(path)
    assert [(r.arrival_step, r.prompt_len, r.max_new_tokens) for r in requests] == rows
    assert path.read_text().count("count: 2") == 1              # identical neighbours collapse


def test_scenario_trace_replays_on_the_engine_step_grid():
    rows = [(0, 512, 5), (0, 512, 5), (7, 512, 5)]
    steps = replay(rows_trace(rows), ReplayConfig(max_batch=8))
    assert steps.B.tolist() == [2, 2, 2, 2, 1, 1, 1, 1]       # 4 decode steps each, third request joins after the gap
    assert window_stats(steps)["steps"] == 8


def test_choose_prefers_more_criterion_steps_then_earlier_window():
    cands = [{"start": 0, "criterion_steps": 5}, {"start": 50, "criterion_steps": 9}, {"start": 100, "criterion_steps": 9},
             {"start": 150}]
    assert choose(cands)["start"] == 50
    assert choose([{"start": 0}]) is None


def test_regret_count_needs_both_regret_and_batch():
    losses = pd.DataFrame({"loss": [1.3, 1.3, 1.1, 2.0], "B": [30, 10, 40, 26]})
    assert regret_count(losses, regret=1.25, min_batch=26) == 2
