"""Policy lab: data functions behind the dashboard's experiment console (no Streamlit, no GPU)."""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from kernelscope.dashboard import lab

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "demo_data/serve_4090/hybrid_20261002/heldout_arrivals"


@pytest.fixture
def two_protocols(tmp_path):
    fresh = tmp_path / "fresh" / "heldout_arrivals"
    kept = tmp_path / "keep" / "heldout_arrivals"
    shutil.copytree(HELDOUT, fresh)
    shutil.copytree(HELDOUT, kept)
    manifest = json.loads((kept / "manifest.json").read_text())
    manifest["policy_cache"] = "kept_across_runs"
    (kept / "manifest.json").write_text(json.dumps(manifest))
    for path in (kept / "hybrid").glob("repeat_*/steps.parquet"):          # a kept cache: every decision is a hit
        steps = pd.read_parquet(path)
        steps["policy_us"], steps["policy_cache_hit"] = 10.0, True
        steps.to_parquet(path, index=False)
    return [fresh, kept]


def test_protocol_comparison_pairs_the_same_scenario_under_both_cache_protocols(two_protocols):
    t = lab.protocol_comparison(two_protocols)
    assert set(t.policy_cache) == {"fresh_for_each_measured_run", "kept_across_runs"}
    assert t.scenario_sha256.nunique() == 1 and set(t.policy) == {"heuristic", "table", "hybrid", "model"}
    hybrid = t[t.policy == "hybrid"].set_index("policy_cache")
    assert hybrid.loc["kept_across_runs", "policy_us_per_step"] == pytest.approx(10.0)
    assert hybrid.loc["fresh_for_each_measured_run", "policy_us_per_step"] > 1000
    assert hybrid.loc["kept_across_runs", "cache_misses_per_run"] == 0
    assert hybrid.loc["fresh_for_each_measured_run", "cache_misses_per_run"] == 13
    assert hybrid.loc["fresh_for_each_measured_run", "speedup_vs_heuristic"] == pytest.approx(1.0198, abs=1e-3)


def _rows(cells):
    return pd.DataFrame([dict(kernel=k, workload_key=key, cache_state="cold", kernel_time_us=us, source_dir="x")
                         for k, key, us in cells])


def test_backend_comparison_puts_flashinfer_next_to_the_best_fa2_variant_per_cell():
    key = "decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal"
    other = "decode_B32_Lq1_Lkv512_Hq32_Hkv8_d128_float16_causal"
    rows = _rows([("flashdecoding_paged", key, 1000.0), ("fd_s16_paged", key, 250.0), ("fa2_paged", key, 1000.0),
                  ("flashinfer_paged", key, 600.0), ("flashinfer_paged_cudacore", key, 200.0),
                  ("flashdecoding_paged", other, 100.0), ("fd_s8_paged", other, 90.0)])
    t = lab.backend_comparison(rows, cache_state="cold").set_index("workload_key")
    r = t.loc[key]
    assert (r.heuristic_us, r.best_fa2_us, r.best_fa2_kernel) == (1000.0, 250.0, "fd_s16_paged")
    assert (r.flashinfer_us, r.flashinfer_kernel) == (200.0, "flashinfer_paged_cudacore")     # the faster FlashInfer variant
    assert (r.flashinfer_tensorcore_us, r.flashinfer_cudacore_us) == (600.0, 200.0)
    assert r.flashinfer_over_best_fa2 == pytest.approx(0.8) and r.heuristic_over_flashinfer == pytest.approx(5.0)
    assert r.ragged and not t.loc[other].ragged and pd.isna(t.loc[other].flashinfer_us)


def test_backend_comparison_of_an_empty_index_has_the_columns():
    t = lab.backend_comparison(_rows([]), cache_state="cold")
    assert t.empty and {"workload_key", "flashinfer_us", "flashinfer_over_best_fa2"} <= set(t.columns)


def test_divergence_overview_reads_a_recorded_campaign():
    t = lab.divergence_overview([ROOT / "demo_data/serve_4090/hybrid_20261002"])
    assert set(t.campaign) == {"hybrid_20261002"}
    control = t[(t.scenario == "control_uniform_fixed8") & (t.policy == "fixed8")].iloc[0]
    assert (control.distinct_positions, control.tie_1ulp, control.clear) == (6, 5, 1)


def test_compose_serve_run_command_lists_every_policy_and_the_cache_protocol():
    argv = lab.compose_command("serve_run", scenario="scenarios/graduation_ragged.yaml",
                               policies=["heuristic", "table:demo_data/dispatch_paged_cold.csv"],
                               out="../kernelscope/results/serve_4090/x/ragged", kv_gib=10, warmup_runs=1, warmup_steps=2,
                               repeats=3, seed=0, policy_cache="keep", python=".venv/bin/python")
    assert argv[:5] == [".venv/bin/python", "-m", "kernelscope.cli", "serve", "run"]
    assert argv.count("--policy") == 2 and argv[argv.index("--policy-cache") + 1] == "keep"
    assert argv[argv.index("--warmup-steps") + 1] == "2" and "--machine" in argv and "--params" in argv
    assert "serve run" in lab.shell_line(argv) and "table:demo_data/dispatch_paged_cold.csv" in lab.shell_line(argv)


def test_compose_serve_run_without_a_warmup_cap_omits_the_flag():
    argv = lab.compose_command("serve_run", scenario="s.yaml", policies=["heuristic"], out="o", kv_gib=4, warmup_runs=1,
                               warmup_steps=None, repeats=3, seed=0, policy_cache="fresh", python="python")
    assert "--warmup-steps" not in argv and argv[argv.index("--policy-cache") + 1] == "fresh"


def test_compose_bench_command():
    argv = lab.compose_command("bench", grid="grids/ragged_s1.yaml", plugins=["flashinfer_paged", "fd_s16_paged"],
                               results="../kernelscope/results/hw_4090/ragged_s1_flashinfer", cache_state="cold", python="python")
    assert argv[:4] == ["python", "-m", "kernelscope.cli", "bench"]
    assert argv[argv.index("--plugins") + 1] == "flashinfer_paged,fd_s16_paged" and argv[argv.index("--cache-state") + 1] == "cold"


def test_compose_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        lab.compose_command("sweep", python="python")


def test_launch_runs_in_the_background_and_tail_reads_the_log(tmp_path):
    log = tmp_path / "run.log"
    job = lab.launch([sys.executable, "-c", "print('hello'); print('done')"], log, cwd=tmp_path)
    assert job.wait(timeout=60) == 0
    assert lab.tail(log, lines=1) == "done" and lab.tail(log, lines=5) == "hello\ndone"
    assert lab.tail(tmp_path / "missing.log") == ""


def test_protocol_comparison_tolerates_steps_recorded_without_a_cache_flag(tmp_path):
    old = tmp_path / "old" / "ragged"
    shutil.copytree(HELDOUT, old)
    for path in old.glob("*/repeat_*/steps.parquet"):                          # a 2026-09 style record
        pd.read_parquet(path).drop(columns=["policy_cache_hit"]).to_parquet(path, index=False)
    t = lab.protocol_comparison([old])
    assert len(t) == 4 and t.cache_misses_per_run.isna().all() and t.tpot_ms_mean.notna().all()


def test_compose_bench_single_workload():
    from kernelscope.dashboard import lab
    argv = lab.compose_command("bench", python="py", workload="decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal",
                               plugins=["fd_s16_paged", "flashinfer_paged_cudacore"], results="out", cache_state="cold",
                               warmup=5, iters=20)
    assert argv[:4] == ["py", "-m", "kernelscope.cli", "bench"]
    assert "--grid" not in argv and argv[argv.index("--workload") + 1].startswith("decode_B1")
    assert argv[argv.index("--warmup") + 1] == "5" and argv[argv.index("--iters") + 1] == "20"
    grid = lab.compose_command("bench", grid="grids/g.yaml", plugins=["fa2_paged"], results="out")
    assert "--grid" in grid and "--warmup" not in grid
