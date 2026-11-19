import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def _summary_row(key, plugin, us=10.0):
    return json.dumps({"status": "ok", "plugin": plugin, "workload_key": key, "cache_state": "cold", "check_ok": None,
                       "kernel_time_us": us, "latency_us": 11.0, "launches": 1, "iterations_dropped": 0, "elapsed_s": 0.1})


def test_bundle_skips_failed_runs_and_carries_traffic_summaries(tmp_path):
    results, out = tmp_path / "results", tmp_path / "demo"
    key = "decode_B1_Lq1_Lkv512_Hq32_Hkv8_d128_float16_causal"
    for group in ("uniform_s1_paged", "ragged_s1_flashinfer_failed_jit_20261006"):
        (results / "hw_4090" / group).mkdir(parents=True)
        (results / "hw_4090" / group / "summaries.jsonl").write_text(_summary_row(key, "flashdecoding_paged") + "\n")
    (results / "traffic" / "azure_conv_x1").mkdir(parents=True)
    (results / "traffic" / "azure_conv_x1" / "summary.json").write_text(json.dumps({"steps": 3}))
    (results / "traffic" / "azure_conv_x1" / "steps.parquet").write_bytes(b"not copied")
    subprocess.run([sys.executable, str(PROJECT / "scripts" / "package_demo.py"), "--results", str(results), "--out", str(out)],
                   check=True, capture_output=True, text=True)
    paths = {entry["path"] for entry in json.loads((out / "provenance.json").read_text())["files"]}
    assert "hw_4090/uniform_s1_paged/summaries.jsonl" in paths and "traffic/azure_conv_x1/summary.json" in paths
    assert not any("failed" in p for p in paths) and not (out / "hw_4090" / "ragged_s1_flashinfer_failed_jit_20261006").exists()
    assert not (out / "traffic" / "azure_conv_x1" / "steps.parquet").exists()
    notes = json.loads((out / "provenance.json").read_text())["notes"]
    assert any("traffic/" in note and "surrogate" in note for note in notes)


def test_bundle_honours_a_custom_exclude_glob(tmp_path):
    results, out = tmp_path / "results", tmp_path / "demo"
    key = "decode_B1_Lq1_Lkv512_Hq32_Hkv8_d128_float16_causal"
    for group in ("uniform_s1_paged", "scratch_probe"):
        (results / "hw_4090" / group).mkdir(parents=True)
        (results / "hw_4090" / group / "summaries.jsonl").write_text(_summary_row(key, "flashdecoding_paged") + "\n")
    subprocess.run([sys.executable, str(PROJECT / "scripts" / "package_demo.py"), "--results", str(results), "--out", str(out),
                    "--exclude", "scratch_*"], check=True, capture_output=True, text=True)
    paths = {entry["path"] for entry in json.loads((out / "provenance.json").read_text())["files"]}
    assert paths == {"hw_4090/uniform_s1_paged/summaries.jsonl"}


def _any_bundle(tmp_path, groups):
    """A results tree whose groups hold (workload key, plugin, kernel_us) cold rows."""
    results = tmp_path / "results"
    for group, rows in groups.items():
        (results / "hw_4090" / group).mkdir(parents=True)
        (results / "hw_4090" / group / "summaries.jsonl").write_text(
            "".join(_summary_row(key, plugin, us) + "\n" for key, plugin, us in rows))
    return results


UNIFORM = "decode_B1_Lq1_Lkv512_Hq32_Hkv8_d128_float16_causal"
RAGGED = "decode_B4_Lq1_Lkv2048+512x3_Hq32_Hkv8_d128_float16_causal"
PARTIAL = "decode_B2_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal"      # no CUDA-core measurement
GROUPS = {
    "uniform_s1_paged": [(UNIFORM, "flashdecoding_paged", 50.0), (UNIFORM, "fd_s4_paged", 40.0),
                         (PARTIAL, "flashdecoding_paged", 33.0), (PARTIAL, "fa2_paged", 30.0)],
    "ragged_s1_paged": [(RAGGED, "flashdecoding_paged", 100.0), (RAGGED, "fd_s8_paged", 60.0)],
    "uniform_s1_flashinfer": [(UNIFORM, "flashinfer_paged", 45.0), (PARTIAL, "flashinfer_paged", 31.0)],
    "ragged_s1_flashinfer": [(RAGGED, "flashinfer_paged", 70.0)],
    "uniform_s1_flashinfer_cudacore": [(UNIFORM, "flashinfer_paged_cudacore", 41.0)],
    "ragged_s1_flashinfer_cudacore": [(RAGGED, "flashinfer_paged_cudacore", 55.0)],
}


def _package(results, out, *extra):
    command = [sys.executable, str(PROJECT / "scripts" / "package_demo.py"), "--results", str(results), "--out", str(out)]
    return subprocess.run([*command, *extra], check=True, capture_output=True, text=True)


def test_any_table_holds_only_cells_measured_on_every_library_default(tmp_path):
    import pandas as pd
    out = tmp_path / "demo"
    _package(_any_bundle(tmp_path, GROUPS), out)
    table = pd.read_csv(out / "dispatch_paged_cold_any.csv").set_index("workload_key")
    assert set(table.index) == {UNIFORM, RAGGED} and table.complete.all()          # PARTIAL lacks CUDA-core
    assert table.loc[UNIFORM, "best_kernel"] == "fd_s4_paged" and table.loc[UNIFORM, "best_us"] == 40.0
    assert table.loc[RAGGED, "best_kernel"] == "flashinfer_paged_cudacore" and table.loc[RAGGED, "flashinfer_cc_us"] == 55.0
    assert table.loc[RAGGED, "fa2_best_kernel"] == "fd_s8_paged" and table.loc[RAGGED, "flashinfer_tc_us"] == 70.0
    legacy = pd.read_csv(out / "dispatch_paged_cold.csv")
    assert set(legacy.best_kernel) == {"fd_s4_paged", "fd_s8_paged", "fa2_paged"} and len(legacy) == 3   # FA2 only


def test_any_table_is_skipped_without_flashinfer_cells(tmp_path):
    out = tmp_path / "demo"
    _package(_any_bundle(tmp_path, {g: r for g, r in GROUPS.items() if g.endswith("_paged")}), out)
    assert (out / "dispatch_paged_cold.csv").exists() and not (out / "dispatch_paged_cold_any.csv").exists()


def test_tables_only_rewrites_the_csvs_from_the_bundle_without_touching_anything_else(tmp_path):
    out = tmp_path / "demo"
    results = _any_bundle(tmp_path, GROUPS)
    _package(results, out)
    provenance = (out / "provenance.json").read_text()
    before = {p.name: p.read_bytes() for p in out.glob("*.csv")}
    for name in before:
        (out / name).unlink()
    (results / "hw_4090" / "uniform_s1_paged" / "summaries.jsonl").write_text("")        # the source is not read
    _package(results, out, "--tables-only")
    assert {p.name: p.read_bytes() for p in out.glob("*.csv")} == before
    assert (out / "provenance.json").read_text() == provenance
