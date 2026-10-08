import json

import pandas as pd
import pytest

from kernelscope.analysis.traffic import (ReplayConfig, heuristic_splits, load_trace, replay, step_losses,
                                          summarize_losses)


def _trace(rows):
    return pd.DataFrame(rows, columns=["arrival_s", "prompt_tokens", "output_tokens"])


def test_load_trace_normalizes_azure_and_burstgpt_columns(tmp_path):
    azure = tmp_path / "azure.csv"
    azure.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\n"
                     "2023-11-16 18:15:46.6805900,374,44\n"
                     "2023-11-16 18:15:47.6805900,0,5\n"          # zero prompt tokens: dropped
                     "2023-11-16 18:15:48.1805900,4808,10\n")
    burst = tmp_path / "burst.csv"
    burst.write_text("Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
                     "5,ChatGPT,472,18,490,Conversation log\n"
                     "45,GPT-4,1087,0,1087,API log\n"               # zero response tokens: dropped
                     "50,ChatGPT,30,2,32,Conversation log\n")
    a = load_trace(azure)
    assert a.arrival_s.tolist() == pytest.approx([0.0, 1.5]) and a.prompt_tokens.tolist() == [374, 4808]
    assert a.output_tokens.tolist() == [44, 10] and a.attrs["dropped"] == 1 and a.attrs["kind"] == "azure"
    b = load_trace(burst)
    assert b.arrival_s.tolist() == [0.0, 45.0] and b.prompt_tokens.tolist() == [472, 30]
    assert b.attrs["dropped"] == 1 and b.attrs["kind"] == "burstgpt"
    assert load_trace(burst, model="GPT-4").empty


def test_replay_batches_by_arrival_and_retires_after_the_output_tokens():
    # r0 arrives first and decodes twice (3 output tokens, the first comes from prefill);
    # r1 arrives 10 ms later, so it joins at the second step, and decodes once.
    trace = _trace([(0.0, 10, 3), (0.01, 20, 2)])
    steps = replay(trace, ReplayConfig(step_s=0.03, kv_tokens=4096, max_batch=8))
    assert steps.step.tolist() == [0, 1]
    assert [json.loads(x) for x in steps.lens] == [[11], [12, 21]]
    assert steps.B.tolist() == [1, 2] and steps.len_max.tolist() == [11, 21]
    assert steps.t_s.tolist() == pytest.approx([0.0, 0.03])


def test_replay_admits_fifo_within_the_batch_limit_and_kv_capacity():
    # Each request reserves ceil((prompt + output) / page) pages for its whole life, like the engine.
    trace = _trace([(0.0, 300, 4), (0.0, 300, 4), (0.0, 300, 4)])
    cfg = ReplayConfig(step_s=0.03, kv_tokens=4 * 256, max_batch=8, page=256)   # room for two 2-page requests
    steps = replay(trace, cfg)
    assert steps.B.tolist() == [2, 2, 2, 1, 1, 1]
    limited = replay(trace, ReplayConfig(step_s=0.03, kv_tokens=4096, max_batch=1))
    assert limited.B.tolist() == [1] * 9


def test_replay_skips_requests_the_pool_can_never_hold_and_jumps_idle_gaps():
    trace = _trace([(0.0, 10, 2), (10.0, 9000, 2), (20.0, 10, 2)])
    steps = replay(trace, ReplayConfig(step_s=0.03, kv_tokens=4096, max_batch=8))
    assert steps.attrs["too_large"] == 1 and steps.attrs["requests"] == 2
    assert steps.t_s.tolist() == pytest.approx([0.0, 20.0]) and steps.step.tolist() == [0, 1]


def test_replay_scales_arrival_times():
    # The second request arrives 120 ms after the first at the trace's own rate, 30 ms at four times the rate.
    trace = _trace([(0.0, 10, 2), (0.12, 10, 2)])
    assert replay(trace, ReplayConfig(step_s=0.03, kv_tokens=4096)).t_s.tolist() == pytest.approx([0.0, 0.12])
    assert replay(trace, ReplayConfig(step_s=0.03, kv_tokens=4096, rate_scale=4.0)).t_s.tolist() == pytest.approx([0.0, 0.03])


def test_heuristic_split_count_follows_the_library_rule():
    # S1 geometry (32/8 heads): B * H_kv >= 0.8 * 2 * 128 SMs, i.e. B >= 26, returns without splitting.
    assert heuristic_splits([32768] + [512] * 25, 32, 8, 128) == 1
    assert heuristic_splits([32768] + [512] * 15, 32, 8, 128) > 1


def test_step_losses_predict_the_heuristic_against_the_best_variant_and_cache_page_configurations():
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    machine, params = MachineSpec.from_json("machines/rtx4090.json"), ModelParams.from_json("models/rtx4090.json")
    lens = [8000] + [500] * 25                                     # +1 token stays inside the same pages
    steps = pd.DataFrame({"step": [0, 1, 2], "t_s": [0.0, 0.03, 0.06], "B": [26, 26, 26],
                          "lens": [json.dumps(lens), json.dumps([n + 1 for n in lens]), json.dumps([600] * 26)]})
    losses = step_losses(steps, machine, params, every=1)
    assert losses.step.tolist() == [0, 1, 2]
    assert (losses.heuristic_splits == 1).all()
    assert losses.loc[0, "loss"] == pytest.approx(losses.loc[0, "heuristic_us"] / losses.loc[0, "best_us"])
    assert losses.loc[0, "loss"] > 1.2 and losses.loc[2, "loss"] < 1.1
    assert losses.loc[0, "best_variant"].startswith("fd_s") and not losses.loc[0, "cache_hit"]
    assert losses.loc[1, "cache_hit"]                                  # same page configuration as step 0
    assert losses.attrs["predictions"] == 2
    assert step_losses(steps, machine, params, every=2).step.tolist() == [0, 2]


def test_summarize_losses_reports_step_fractions_and_the_attention_time_ratio():
    losses = pd.DataFrame({"step": [0, 1, 2, 3], "B": [26, 26, 10, 26], "len_max": [8192, 8192, 600, 600],
                           "heuristic_splits": [1, 1, 4, 1], "heuristic_us": [300.0, 300.0, 50.0, 100.0],
                           "best_us": [200.0, 240.0, 50.0, 100.0], "loss": [1.5, 1.25, 1.0, 1.0]})
    s = summarize_losses(losses, thresholds=(1.1, 1.25, 1.5))
    assert s["steps_sampled"] == 4 and s["no_split_frac"] == 0.75
    assert s["loss_ge_1.1_frac"] == 0.5 and s["loss_ge_1.25_frac"] == 0.5 and s["loss_ge_1.5_frac"] == 0.25
    assert s["attention_time_ratio"] == pytest.approx(750 / 590)
    assert s["loss_median"] == pytest.approx(1.125) and s["loss_max"] == 1.5
    assert s["attention_share_loss_ge_1.25"] == pytest.approx(600 / 750)


def test_collect_summaries_tabulates_every_replay_under_a_root(tmp_path):
    from kernelscope.analysis.traffic import collect_summaries
    for name, scale, frac, ratio in (("azure_conv_x1", 1.0, 0.137, 1.117), ("burstgpt_x50", 50.0, 0.003, 1.03)):
        d = tmp_path / name
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({
            "trace": {"kind": name.split("_")[0], "requests": 10, "prompt_scale": 1.0, "window_s": None, "model": None,
                      "prompt_tokens_p50": 1000.0, "prompt_tokens_max": 14050},
            "config": {"max_batch": 64, "kv_tokens": 72817, "step_s": 0.03, "rate_scale": scale, "page": 256},
            "steps": 100, "what_if": [],
            "losses": {"steps_sampled": 100, "batch_mean": 30.0, "no_split_frac": 0.9, "loss_median": 1.05,
                       "loss_p90": 1.4, "loss_max": 3.0, "attention_time_ratio": ratio, "loss_ge_1.1_frac": 0.3,
                       "loss_ge_1.25_frac": frac, "loss_ge_1.5_frac": 0.05, "loss_ge_2.0_frac": 0.01,
                       "attention_share_loss_ge_1.25": 0.2, "every": 1, "predictions": 7}}))
    table = collect_summaries(tmp_path).set_index("run")
    assert table.index.tolist() == ["azure_conv_x1", "burstgpt_x50"]
    assert table.loc["azure_conv_x1", "loss_ge_1.25_frac"] == 0.137 and table.loc["burstgpt_x50", "rate_scale"] == 50.0
    assert table.loc["azure_conv_x1", "attention_time_ratio"] == 1.117 and table.loc["azure_conv_x1", "kind"] == "azure"
    assert table.loc["azure_conv_x1", "what_if"] == "" and table.loc["azure_conv_x1", "steps"] == 100


def test_single_token_requests_never_reach_a_decode_step():
    # The engine retires a request whose only token came from prefill before any decode step.
    assert replay(_trace([(0.0, 10, 1)]), ReplayConfig(step_s=0.03, kv_tokens=4096)).empty
    steps = replay(_trace([(0.0, 10, 1), (0.0, 10, 3)]), ReplayConfig(step_s=0.03, kv_tokens=4096))
    assert [json.loads(x) for x in steps.lens] == [[11], [12]]


def test_replay_refuses_non_finite_arrivals_instead_of_spinning():
    with pytest.raises(ValueError, match="arrival"):
        replay(_trace([(0.0, 10, 3), (float("nan"), 10, 3)]), ReplayConfig(step_s=0.03, kv_tokens=4096))


def test_load_trace_drops_rows_with_unusable_timestamps_or_token_counts(tmp_path):
    burst = tmp_path / "burst.csv"
    burst.write_text("Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
                     "5,ChatGPT,472,18,490,Conversation log\n"
                     ",ChatGPT,30,2,32,Conversation log\n"                 # no timestamp
                     "9,ChatGPT,abc,2,32,Conversation log\n")              # malformed token count
    frame = load_trace(burst)
    assert frame.prompt_tokens.tolist() == [472] and frame.attrs["dropped"] == 2


def test_page_reservation_matches_the_engine_which_never_writes_the_last_token():
    # prompt 200 + 57 tokens: the engine reserves ceil((200 + 57 - 1) / 256) = 1 page, so one page is enough.
    steps = replay(_trace([(0.0, 200, 57)]), ReplayConfig(step_s=0.03, kv_tokens=256))
    assert steps.attrs["too_large"] == 0 and len(steps) == 56
