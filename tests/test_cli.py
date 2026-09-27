import json
import sys
from pathlib import Path

import pytest

from kernelscope import cli
from kernelscope.workload import Workload

REG = "tests.fake_plugins:REGISTRY"
FAKE_NCU = f"{sys.executable} {Path(__file__).parent / 'fake_ncu.py'}"

GRID_YAML = """\
phase: [decode]
B: [1, 2]
L: [32, 64]
H_q: 8
H_kv: 2
d: 16
dtype: float32
"""


def test_load_grid_expands_yaml_into_workloads(tmp_path):
    f = tmp_path / "grid.yaml"
    f.write_text(GRID_YAML)
    ws = cli.load_grid(f)
    assert len(ws) == 4
    assert all(isinstance(w, Workload) for w in ws)
    assert ws[0].key() == "decode_B1_Lq1_Lkv32_Hq8_Hkv2_d16_float32_causal"


def test_list_prints_plugin_names(capsys):
    cli.main(["list", "--registry", REG])
    assert "faithful_cpu" in capsys.readouterr().out


def test_sweep_writes_parquet_and_summaries_with_ncu_off_by_default(tmp_path, capsys):
    grid = tmp_path / "grid.yaml"
    grid.write_text(GRID_YAML)
    results = tmp_path / "results"
    cli.main(["sweep", "--grid", str(grid), "--plugins", "faithful_cpu", "--results", str(results),
              "--registry", REG, "--device", "cpu", "--python", sys.executable,
              "--warmup", "1", "--iters", "2"])
    assert len(list(results.glob("*.parquet"))) == 4
    lines = (results / "summaries.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert all(json.loads(ln)["status"] == "ok" for ln in lines)
    assert all(json.loads(ln)["ncu"] == "disabled" for ln in lines)


def test_sweep_can_opt_into_ncu_and_ceilings(tmp_path, monkeypatch):
    captured = {}

    class FakeSweep:
        def __init__(self, **kw):
            captured.update(kw)

        def run_grid(self, plugins, workloads, log=print):
            log(json.dumps({"status": "ok"}))
            return []

    monkeypatch.setattr(cli, "RealHWSweep", FakeSweep)
    ceil = tmp_path / "ceil.json"
    ceil.write_text(json.dumps({"hbm_copy_gbps": 1500.0, "fp16_matmul_tflops": 250.0}))
    key = "decode_B1_Lq1_Lkv32_Hq8_Hkv2_d16_float32_causal"
    cli.main(["sweep", "--workload", key, "--plugins", "faithful_cpu", "--results", str(tmp_path / "r"),
              "--registry", REG, "--ncu", "/usr/local/cuda/bin/ncu", "--ceilings", str(ceil)])
    assert captured["ncu_cmd"] == ["/usr/local/cuda/bin/ncu"]
    assert captured["ceilings"]["hbm_copy_gbps"] == 1500.0


def test_report_subcommand_prints_and_writes_the_summary(tmp_path, capsys):
    from kernelscope.results.store import ResultStore
    from tests.test_report import DF
    store = ResultStore(tmp_path / "results")
    store.write(DF.to_dict("records"), tag="fixture")
    out_csv = tmp_path / "summary.csv"
    cli.main(["report", "--results", str(tmp_path / "results"), "--out", str(out_csv)])
    printed = capsys.readouterr().out
    assert "fa2" in printed and "flashdecoding" in printed
    assert "bandwidth-bound" in printed or "insensitive" in printed      # verdict column
    import pandas as pd
    s = pd.read_csv(out_csv)
    assert set(s["kernel"]) == {"fa2", "flashdecoding"}
    assert "sens_bw_x2" in s.columns and "verdict" in s.columns


def test_ceilings_subcommand_writes_json(tmp_path):
    out = tmp_path / "machine_ceilings.json"
    cli.main(["ceilings", "--out", str(out), "--device", "cpu", "--copy-bytes", str(1 << 20),
              "--matmul-ns", "32,64", "--iters", "2"])
    d = json.loads(out.read_text())
    assert d["device"] == "cpu" and d["hbm_gbps"] > 0 and d["matmul_best_n"] in (32, 64)


def test_sweep_accepts_inline_workload_instead_of_grid(tmp_path):
    results = tmp_path / "results"
    key = "decode_B1_Lq1_Lkv32_Hq8_Hkv2_d16_float32_causal"
    cli.main(["sweep", "--workload", key, "--plugins", "faithful_cpu", "--results", str(results),
              "--registry", REG, "--device", "cpu", "--python", sys.executable,
              "--ncu", FAKE_NCU, "--warmup", "1", "--iters", "2"])
    assert len(list(results.glob("*.parquet"))) == 1
    assert json.loads((results / "summaries.jsonl").read_text())["workload_key"] == key


def test_sweep_requires_grid_or_workload():
    with pytest.raises(SystemExit):
        cli.main(["sweep", "--plugins", "faithful_cpu", "--results", "x"])


def test_sweep_runs_one_pass_per_cache_state(tmp_path, capsys):
    grid = tmp_path / "grid.yaml"
    grid.write_text(GRID_YAML)
    out = tmp_path / "res"
    cli.main(["sweep", "--grid", str(grid), "--plugins", "faithful_cpu", "--results", str(out),
              "--registry", REG, "--device", "cpu", "--warmup", "1", "--iters", "2",
              "--cache-state", "warm,cold"])
    import pandas as pd
    from kernelscope.results.store import ResultStore
    df = ResultStore(out).load()
    assert set(df["cache_state"]) == {"warm", "cold"}


def test_simsweep_wires_variants_budget_and_paths(tmp_path, monkeypatch):
    captured = {}

    class FakeSweep:
        def __init__(self, **kw):
            captured.update(kw)

        def run_grid(self, plugins, workloads, log=print):
            captured["plugins"], captured["workloads"] = plugins, workloads
            log(json.dumps({"status": "ok"}))
            return []

    monkeypatch.setattr(cli, "AccelSimSweep", FakeSweep)
    key = "decode_B1_Lq1_Lkv32_Hq8_Hkv2_d16_float32_causal"
    results = tmp_path / "results"
    cli.main(["simsweep", "--workload", key, "--plugins", "fa2,flashdecoding", "--results", str(results),
              "--accelsim-root", str(tmp_path / "repo"), "--work-dir", str(tmp_path / "work"),
              "--variants", "base,bw_x2", "--max-sim-s", "600", "--device-index", "3",
              "--registry", REG, "--python", sys.executable])
    assert captured["plugins"] == ["fa2", "flashdecoding"]
    assert [w.key() for w in captured["workloads"]] == [key]
    assert captured["variants"] == ["base", "bw_x2"]
    assert captured["max_sim_s"] == 600
    assert captured["device_index"] == 3
    assert captured["paths"].root == tmp_path / "repo"
    assert captured["work_dir"] == tmp_path / "work"
    assert (results / "summaries.jsonl").read_text().strip() == '{"status": "ok"}'


def test_dispatch_table_reads_the_portable_demo_bundle(capsys):
    demo = Path(__file__).resolve().parents[1] / "demo_data" / "hw_4090" / "uniform_s1_dense"
    cli.main(["dispatch-table", "--results", str(demo), "--family", "dense", "--cache-state", "cold"])
    out = capsys.readouterr().out
    summary = json.loads(out[: out.index("}") + 1])
    assert summary["cells"] == 49
    assert summary["median_regret"] == pytest.approx(0.00721, abs=5e-5)
    assert summary["max_regret"] == pytest.approx(0.05712, abs=5e-5)
