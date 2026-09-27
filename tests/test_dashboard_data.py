import json

import pandas as pd
import pytest

from kernelscope.dashboard import data, style
from kernelscope.results.store import ResultStore

KEY = "decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal"


def test_summary_index_preserves_cache_and_ignores_failures(tmp_path):
    directory = tmp_path / "hw_4090" / "campaign"
    directory.mkdir(parents=True)
    records = [
        {"status": "ok", "plugin": "fa2", "workload_key": KEY, "cache_state": "cold", "kernel_time_us": 60.},
        {"status": "ok", "plugin": "fa2", "workload_key": KEY, "kernel_time_us": 30.},
        {"status": "failed", "plugin": "fa2", "workload_key": KEY, "kernel_time_us": 1.},
    ]
    (directory / "summaries.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n{unfinished\n[]")
    assert data.find_dirs("hw", tmp_path) == [directory]
    index = data.load_index([directory])
    assert list(index.cache_state) == ["cold", "warm"]
    assert list(index.kernel_time_us) == [60., 30.]
    assert data.index_rows(index).query("cache_state == 'cold'").value.tolist() == [60.]


def test_parquet_fallback_and_launch_zero_metrics(tmp_path):
    rows = [{"workload_key": KEY, "kernel": "fa2", "backend": b, "metric": m, "unit": "", "value": v,
             "launch_idx": launch}
            for b, m, v, launch in [("profile", "kernel_time_us", 60., 0), ("latency", "median_s", 70e-6, 0),
                                    ("profile", "grid_blocks", 8., 0), ("profile", "grid_blocks", 999., 1),
                                    ("analytic", "achieved_gbps", 70., 0)]]
    ResultStore(tmp_path).write(rows, tag="opaque", extra={"cache_state": "cold"})
    index = data.load_index([tmp_path])
    assert index.iloc[0].kernel_time_us == 60.
    detail = data.load_cell(tmp_path, "fa2", KEY, "cold")
    row = data.kernel_table(detail, "cold").loc[("fa2", KEY)]
    assert row.grid_blocks == 8 and row.latency_us == pytest.approx(70.)
    assert row.achieved_gbps == 70.
    assert data.kernel_table(detail, "warm").empty


def test_empty_root_has_no_fake_results(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    assert data.results_root() == tmp_path
    assert data.find_dirs("serve") == []
    assert data.load_index([]).empty
    assert data.kernel_table(data.load_rows([]), "cold").empty
    assert data.load_manifest(tmp_path)["performance_claim"] is False


def test_serving_repeat_boundaries_and_provenance(tmp_path):
    scenario = tmp_path / "serve" / "model" / "mixed"
    for repeat in range(2):
        directory = scenario / "heuristic" / f"repeat_{repeat:03d}"
        directory.mkdir(parents=True)
        pd.DataFrame({"step": [0, 1], "attn_us": [10., 12.]}).to_parquet(directory / "steps.parquet")
        pd.DataFrame({"rid": [0, 0, 1, 1], "step": [0, 1, 0, 1], "t_us": [0., 100., 20., 70.]}).to_parquet(directory / "tokens.parquet")
    (scenario / "manifest.json").write_text(json.dumps({"evidence_kind": "cpu_functional", "performance_claim": False}))
    assert data.find_dirs("serve", tmp_path) == [scenario]
    serving = data.load_serving(scenario)
    assert len(serving["heuristic"]["steps"]) == 4
    assert sorted(data.tpot_samples(serving["heuristic"]["tokens"])) == [50., 50., 100., 100.]
    assert data.load_manifest(scenario)["evidence_kind"] == "cpu_functional"


def test_amdahl_includes_unaccelerated_time_and_dispatch_overhead():
    assert data.amdahl_speedup(0., 10.) == 1.
    assert data.amdahl_speedup(1., 10.) == 10.
    assert data.amdahl_speedup(.5, 2.) == pytest.approx(4 / 3)
    assert data.amdahl_speedup(.5, 2., .25) == 1.
    for args in [(-.1, 2.), (.5, 0.), (.5, float("nan")), (.5, 2., -.1)]:
        with pytest.raises(ValueError):
            data.amdahl_speedup(*args)


def test_policy_colors_follow_entity():
    assert style.policy_color("heuristic", "light") == "#2a78d6"
    assert style.policy_color("model", "dark") == "#d95926"
    assert style.policy_color("fixed8") == style.policy_color("fixed32")
    assert style.policy_dash("fixed8") != style.policy_dash("fixed32")


def test_in_progress_serving_file_is_not_measurement_evidence(tmp_path):
    incomplete = tmp_path / "model" / "repeat_000"
    incomplete.mkdir(parents=True)
    (incomplete / "steps.parquet").write_bytes(b"PAR1")
    assert data.load_serving(tmp_path) == {}
