import json
import math

import pandas as pd
import pytest

from scripts.sample_replay_cells import build_cells, collect, distinct_configs, main, sample, stratum, write_grid
from kernelscope.analysis.traffic import CANDIDATES, ReplayConfig, replay, step_losses
from kernelscope.model.machine import MachineSpec
from kernelscope.model.params import ModelParams


@pytest.fixture(scope="module")
def run():
    rows = [{"arrival_s": 0.0, "prompt_tokens": 8192, "output_tokens": 40}]
    rows += [{"arrival_s": 0.01 * i, "prompt_tokens": 512 + 64 * (i % 3), "output_tokens": 30} for i in range(1, 30)]
    steps = replay(pd.DataFrame(rows), ReplayConfig(rate_scale=1.0))
    machine, params = MachineSpec.from_json("machines/rtx4090.json"), ModelParams.from_json("models/rtx4090.json")
    return steps, step_losses(steps, machine, params), machine, params


def test_strata_follow_the_four_loss_bins():
    assert [stratum(x) for x in (1.0, 1.0999, 1.1, 1.2499, 1.25, 1.4999, 1.5, 9.0)] == \
        ["<1.1", "<1.1", "1.1-1.25", "1.1-1.25", "1.25-1.5", "1.25-1.5", ">=1.5", ">=1.5"]
    with pytest.raises(ValueError):
        stratum(math.nan)


def test_distinct_configs_collapse_steps_sharing_the_page_quantized_key(run):
    steps, losses, *_ = run
    c = distinct_configs(steps, losses)
    assert c.n_steps_with_key.sum() == len(steps) and len(c) < len(steps)
    assert (c.stratum == c.pred_regret.map(stratum)).all()
    with pytest.raises(ValueError):
        distinct_configs(steps, losses.iloc[::2])           # every=2 replays cannot be sampled


def test_sample_is_seeded_and_takes_all_of_a_small_bucket(run):
    steps, losses, *_ = run
    c = distinct_configs(steps, losses)
    a, b = sample(c, 3, 0), sample(c, 3, 0)
    assert list(a.first_step) == list(b.first_step)
    assert a.groupby("stratum").size().max() <= 3
    assert len(sample(c, 10_000, 0)) == len(c)


def test_cells_carry_every_candidate_prediction_and_the_stratum_population(run):
    steps, losses, machine, params = run
    doc = build_cells([("fake", steps, losses, 2)], 0, machine, params)
    assert doc["candidates"] == list(CANDIDATES) and doc["cells"]
    for cell in doc["cells"]:
        assert set(cell["pred_us"]) == set(CANDIDATES) and cell["B"] == len(cell["lens"])
        assert cell["pred_regret"] == pytest.approx(cell["pred_heuristic_us"] / cell["pred_best_us"])
        assert cell["stratum_distinct"] >= 1 and cell["stratum_steps"] >= cell["n_steps_with_key"]
    assert sum(v["steps"] for v in doc["meta"]["fake"]["strata"].values()) == len(steps)


def test_collect_joins_measured_times_and_flags_incomplete_cells(run, tmp_path):
    steps, losses, machine, params = run
    doc = build_cells([("fake", steps, losses, 1)], 0, machine, params)
    n = write_grid(doc["cells"], tmp_path / "g.yaml")
    assert n == len({c["lens_spec"] for c in doc["cells"]})
    from kernelscope.cli import load_grid
    keys = [w.key() for w in load_grid(str(tmp_path / "g.yaml"))]
    rows = []
    for k in keys[:-1]:                                     # the last workload is left unmeasured
        for i, v in enumerate(CANDIDATES):
            rows.append({"workload_key": k, "kernel": v, "backend": "profile", "metric": "kernel_time_us", "unit": "us",
                         "value": 100.0 + 10 * i, "launch_idx": 0, "cache_state": "cold"})
    t = collect(doc, pd.DataFrame(rows))
    done = t[t.complete]
    assert len(done) >= 1
    assert (done.meas_best_variant == CANDIDATES[0]).all() and done.meas_regret.eq(1.0).all()
    assert t[~t.complete].meas_regret.isna().all()


def test_cli_samples_two_runs_into_cells_json_and_a_grid(run, tmp_path):
    steps, losses, *_ = run
    d = tmp_path / "run"
    d.mkdir()
    steps.to_parquet(d / "steps.parquet", index=False)
    losses.to_parquet(d / "losses.parquet", index=False)
    main(["sample", "--source", f"a={d}:2", "--source", f"b={d}:1", "--out", str(tmp_path / "out")])
    doc = json.loads((tmp_path / "out" / "cells.json").read_text())
    assert {c["source"] for c in doc["cells"]} == {"a", "b"} and (tmp_path / "out" / "cells_grid.yaml").exists()
