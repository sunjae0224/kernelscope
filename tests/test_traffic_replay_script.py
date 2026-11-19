import json

import pandas as pd

from scripts.traffic_replay import main


def test_script_replays_a_trace_and_writes_steps_losses_and_summary(tmp_path):
    trace = tmp_path / "azure.csv"
    rows = ["TIMESTAMP,ContextTokens,GeneratedTokens"]
    rows += [f"2023-11-16 18:15:{46 + i // 10:02d}.{i % 10}00000,{8192 if i == 0 else 512},6" for i in range(12)]
    trace.write_text("\n".join(rows) + "\n")
    out = tmp_path / "out"
    main(["--trace", str(trace), "--rate-scale", "4", "--step-ms", "30", "--every", "2", "--max-batch", "64",
          "--kv-tokens", "72817", "--out", str(out)])
    steps = pd.read_parquet(out / "steps.parquet")
    losses = pd.read_parquet(out / "losses.parquet")
    summary = json.loads((out / "summary.json").read_text())
    assert len(steps) > 0 and len(losses) == len(steps.iloc[::2])
    assert summary["trace"]["kind"] == "azure" and summary["trace"]["requests"] == 12
    assert summary["config"] == {"max_batch": 64, "kv_tokens": 72817, "step_s": 0.03, "rate_scale": 4.0, "page": 256}
    assert summary["losses"]["steps_sampled"] == len(losses) and summary["losses"]["predictions"] >= 1
    assert summary["machine"].endswith("rtx4090.json") and summary["params"].endswith("rtx4090.json")
    assert summary["heads"] == {"n_heads": 32, "n_kv_heads": 8}
    assert 0.0 <= summary["losses"]["loss_ge_1.25_frac"] <= 1.0
    assert summary["what_if"] == [] and summary["trace"]["prompt_scale"] == 1.0


def test_script_rejects_nonpositive_rate_scale_batch_and_every(tmp_path):
    import pytest
    trace = tmp_path / "azure.csv"
    trace.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\n2023-11-16 18:15:46.0000000,10,3\n")
    for flag in (("--rate-scale", "0"), ("--max-batch", "0"), ("--every", "0"), ("--kv-tokens", "10")):
        with pytest.raises(SystemExit):
            main(["--trace", str(trace), *flag, "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_prompt_scale_is_a_labeled_what_if_on_the_trace_lengths(tmp_path):
    trace = tmp_path / "azure.csv"
    trace.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\n2023-11-16 18:15:46.0000000,1000,3\n"
                     "2023-11-16 18:15:46.0100000,100,3\n")
    out = tmp_path / "out"
    main(["--trace", str(trace), "--prompt-scale", "4", "--out", str(out)])
    steps = pd.read_parquet(out / "steps.parquet")
    summary = json.loads((out / "summary.json").read_text())
    assert steps.len_max.iloc[0] == 4001 and summary["trace"]["prompt_scale"] == 4.0   # 1000 * 4 + the first token
    assert summary["trace"]["prompt_tokens_max"] == 4000 and summary["losses"]["steps_sampled"] == len(steps)
    assert "prompt lengths scaled by 4.0" in summary["what_if"]
