import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")

from kernelscope.serve.cli import register_parser


def parse(argv):
    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers(required=True))
    return parser.parse_args(argv)


def test_cpu_demo_is_real_runnable_and_does_not_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[1]
    args = parse(["serve", "demo", "--scenario", str(root / "scenarios/demo.yaml"), "--out", str(tmp_path / "demo"),
                  "--repeats", "1", "--warmup-runs", "0"])
    args.func(args)
    manifest = json.loads((tmp_path / "demo/manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["evidence_kind"] == "cpu_functional"
    assert not manifest["performance_claim"] and manifest["tokens_equivalent"]
    assert manifest["model_dtype"] == "float32" and manifest["tokenization_setup_us"] >= 0
    assert manifest["policy_setup_runs"][0]["setup_us"] >= 0
    assert (tmp_path / "demo/prompts.jsonl").is_file() and (tmp_path / "demo/scenario.yaml").is_file()
    assert pd.read_csv(tmp_path / "demo/equivalence.csv").passed.all()
    assert pd.read_csv(tmp_path / "demo/summary.csv").speedup_vs_heuristic.isna().all()
    with pytest.raises(SystemExit, match="already exists"):
        args.func(args)


def test_run_refuses_unavailable_gpu_before_creating_artifacts(tmp_path, monkeypatch):
    import kernelscope.serve.cli as cli
    monkeypatch.setattr(cli, "preflight", lambda *a, **kw: {"ready": False, "errors": ["CUDA unavailable"]})
    scenario = Path(__file__).resolve().parents[1] / "scenarios/demo.yaml"
    args = parse(["serve", "run", "--scenario", str(scenario), "--out", str(tmp_path / "output")])
    with pytest.raises(SystemExit, match="CUDA unavailable"):
        args.func(args)
    assert not (tmp_path / "output").exists()


def test_tokenizer_and_recipe_are_resolved_before_timing(monkeypatch):
    from types import SimpleNamespace
    import kernelscope.serve.cli as cli
    import kernelscope.serve.hf as hf
    from kernelscope.serve.scenarios import Request
    from tests.test_serve_scenarios import CharacterTokenizer
    monkeypatch.setattr(hf, "snapshot_dir", lambda name: Path("/tmp/no-tokenizer-files-needed"))
    monkeypatch.setattr(hf, "load_tokenizer", lambda path: CharacterTokenizer())
    args = SimpleNamespace(model="local/model", seed=0)
    resolved, meta = cli._resolve_scenario(args, [Request(0, None, 2, prompt_text="A question")], {"id": "qa"}, 256)
    assert resolved[0].token_ids == tuple(map(ord, "A question"))
    assert meta["dataset_id"] == "qa" and meta["tokenizer_load_us"] >= 0


def test_diagnose_cpu_tiny_writes_control_event_and_diagnosis(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "diag"
    args = parse(["serve", "diagnose", "--device", "cpu", "--model", "tiny-random", "--scenario", str(root / "scenarios/tiny.yaml"),
                  "--policy", "heuristic", "--policy", "fa2", "--out", str(out), "--warmup-runs", "0", "--kv-gib", "0.002",
                  "--table", str(root / "demo_data/dispatch_paged_cold.csv"), "--data", str(root / "demo_data")])
    args.func(args)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["tokens_consistent"] is True
    assert manifest["evidence_kind"] == "cpu_functional" and manifest["performance_claim"] is False
    assert manifest["ops_mode_runs"] == ["control", "event"] and manifest["threshold"] == 0.7
    for policy in ("heuristic", "fa2"):
        assert (out / policy / "control_000/steps.parquet").is_file() and (out / policy / "event_000/ops.parquet").is_file()
        assert not (out / policy / "control_000/ops.parquet").exists()
        assert json.loads((out / policy / "event_000/meta.json").read_text())["ops_mode"] == "event"
    diagnosis = json.loads((out / "diagnosis.json").read_text())
    assert set(diagnosis["policies"]) == {"heuristic", "fa2"}
    heuristic = diagnosis["policies"]["heuristic"]
    assert len(heuristic["ops"]) == 8 and len(heuristic["prefill_ops"]) == 8
    assert heuristic["attention"]["source"] == "unavailable"          # tiny head shape (4, 2) is not in the measured table
    assert heuristic["unattributed_pct"] is not None
    ops = pd.read_csv(out / "ops.csv")
    assert len(ops) == 2 * 16 and (out / "op_breakdown.png").stat().st_size > 1000
    assert pd.read_csv(out / "consistency.csv").passed.all()
    with pytest.raises(SystemExit, match="already exists"):
        args.func(args)


def test_diagnose_report_regenerates_from_a_recorded_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "diag"
    run = parse(["serve", "diagnose", "--device", "cpu", "--model", "tiny-random", "--scenario", str(root / "scenarios/tiny.yaml"),
                 "--policy", "heuristic", "--out", str(out), "--warmup-runs", "0", "--kv-gib", "0.002"])
    run.func(run)
    (out / "diagnosis.json").unlink()
    report = parse(["serve", "diagnose-report", str(out), "--threshold", "0.5"])
    report.func(report)
    assert json.loads((out / "diagnosis.json").read_text())["ceilings"]["threshold"] == 0.5


def test_diagnose_report_refuses_a_folder_without_manifest(tmp_path):
    args = parse(["serve", "diagnose-report", str(tmp_path)])
    with pytest.raises(SystemExit, match="manifest.json"):
        args.func(args)
