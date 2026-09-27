import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kernelscope.diagnose import report
from kernelscope.diagnose.opmodel import OP_CLASSES
from kernelscope.model.machine import MachineSpec

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(arch="qwen3", n_layers=2, hidden=24, intermediate=48, n_heads=4, n_kv_heads=2, head_dim=8, vocab=128,
           rope_theta=10000.0, rope_scaling=None, rms_eps=1e-6, tie_embeddings=True, qk_norm=True)
KEY = "decode_B2_Lq1_Lkv8+4x1_Hq4_Hkv2_d8_float16_causal"
# per-layer gpu_us chosen so that the verdict rules fire deterministically (see spec §2.3):
# mlp ≈ 33 KB per step in 0.04 µs → 835 GB/s ≥ 0.7·952.6 → memory_bound; rope 1 µs/layer → launch_bound;
# attention 150 µs/layer → parallelism_candidate; qkv 50 µs/layer → below_ceiling_unknown.
LAYER_US = {"norm": 3.0, "qkv_proj": 50.0, "rope": 1.0, "attention": 150.0, "o_proj": 20.0, "mlp": 0.02}
STEP_US, CONTROL_STEP_US = 1000.0, 950.0


def _steps(step_us, lens=(8, 4), n=2, num_splits=0):
    return pd.DataFrame([dict(step=i, policy="heuristic", B=len(lens), len_max=max(lens), len_sum=sum(lens), n_long=0,
                              num_splits=num_splits, attn_us=300.0, step_us=step_us, policy_us=10.0, policy_cache_hit=True,
                              decode_wall_us=step_us + 100.0, seq_ids=json.dumps([0, 1]), lens=json.dumps(list(lens)))
                         for i in range(n)])


def _tokens(n_steps=2):
    rows = [dict(rid=r, step=0, position=0, token=1, t_us=1000.0, phase="prefill") for r in (0, 1)]
    rows += [dict(rid=r, step=s, position=s + 1, token=2, t_us=1000.0 * (s + 2), phase="decode") for r in (0, 1) for s in range(n_steps)]
    return pd.DataFrame(rows)


def _prefill():
    return pd.DataFrame([dict(rid=0, prompt_len=8, prefill_us=500.0, arrival_us=0.0, admitted_us=0.0, first_token_us=500.0, prompt_sha256="a"),
                         dict(rid=1, prompt_len=4, prefill_us=300.0, arrival_us=0.0, admitted_us=500.0, first_token_us=800.0, prompt_sha256="b")])


def _ops(n_steps=2, drop=()):
    rows = []
    for step in range(n_steps):
        rows.append(dict(phase="decode", step=step, rid="", layer=-1, op_class="embed", gpu_us=2.0))
        for layer in range(CFG["n_layers"]):
            for op_class, us in LAYER_US.items():
                if op_class not in drop:
                    rows.append(dict(phase="decode", step=step, rid="", layer=layer, op_class=op_class, gpu_us=us))
        rows.append(dict(phase="decode", step=step, rid="", layer=-1, op_class="lm_head", gpu_us=5.0))
    for rid in ("0", "1"):
        rows.append(dict(phase="prefill", step=0, rid=rid, layer=-1, op_class="embed", gpu_us=1.0))
        for layer in range(CFG["n_layers"]):
            for op_class in LAYER_US:
                rows.append(dict(phase="prefill", step=0, rid=rid, layer=layer, op_class=op_class, gpu_us=10.0))
        rows.append(dict(phase="prefill", step=0, rid=rid, layer=-1, op_class="lm_head", gpu_us=1.0))
    return pd.DataFrame(rows)


def _run_dir(tmp_path, control=True, drop=()):
    run = tmp_path / "run"
    (run / "heuristic" / "event_000").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"evidence_kind": "diagnostic_op_breakdown", "performance_claim": False,
                                                   "model": "tiny", "scenario": "s.yaml", "created_at": "t",
                                                   "model_config": CFG, "model_dtype": "float32"}))
    event = run / "heuristic" / "event_000"
    _steps(STEP_US).to_parquet(event / "steps.parquet", index=False)
    _tokens().to_parquet(event / "tokens.parquet", index=False)
    _prefill().to_parquet(event / "prefill.parquet", index=False)
    _ops(drop=drop).to_parquet(event / "ops.parquet", index=False)
    if control:
        ctrl = run / "heuristic" / "control_000"
        ctrl.mkdir()
        _steps(CONTROL_STEP_US).to_parquet(ctrl / "steps.parquet", index=False)
        _tokens().to_parquet(ctrl / "tokens.parquet", index=False)
        _prefill().to_parquet(ctrl / "prefill.parquet", index=False)
    return run


def _table(tmp_path, h_q=4, h_kv=2):
    csv = tmp_path / "table.csv"
    pd.DataFrame([dict(workload_key=KEY, B=2, L_kv=8, H_q=h_q, H_kv=h_kv, ragged=True, lens="8+4x1", n_variants=3,
                       best_kernel="fd_s2_paged", best_us=10.0, heuristic_us=30.0, heuristic_regret=2.0, fa2_us=30.0, fa2_regret=2.0)]
                 ).to_csv(csv, index=False)
    return csv


def _data(tmp_path):
    d = tmp_path / "data" / "hw_4090" / "ragged_s1_paged"
    d.mkdir(parents=True)
    rows = [dict(status="ok", plugin=p, workload_key=KEY, cache_state="cold", kernel_time_us=us)
            for p, us in (("flashdecoding_paged", 30.0), ("fd_s2_paged", 10.0), ("fa2_paged", 30.0))]
    rows.append(dict(status="ok", plugin="fd_s2_paged", workload_key=KEY, cache_state="warm", kernel_time_us=1.0))
    (d / "summaries.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path / "data"


@pytest.fixture
def machine():
    return MachineSpec.from_json(ROOT / "machines" / "rtx4090.json")


def test_verdict_rules_in_order(machine):
    ridge = machine.ridge_flop_per_byte()
    assert report.verdict("mlp", ridge / 2, 0.8 * machine.dram_gbps, 0.0, 100.0, machine, 0.7) == "memory_bound"
    assert report.verdict("mlp", 2 * ridge, 0.0, 0.8 * machine.tc_tflops, 100.0, machine, 0.7) == "compute_bound"
    assert report.verdict("rope", 0.0, 1.0, 0.0, 1.0, machine, 0.7) == "launch_bound"
    assert report.verdict("attention", 1.0, 1.0, 0.0, 100.0, machine, 0.7) == "parallelism_candidate"
    assert report.verdict("qkv_proj", 1.0, 1.0, 0.0, 100.0, machine, 0.7) == "below_ceiling_unknown"


def test_amdahl_bound_formula():
    assert report.amdahl_bound(0.72, 3.84) == pytest.approx(1 / (0.28 + 0.72 / 3.84))


def test_decode_table_shares_costs_and_verdicts(tmp_path, machine):
    run = _run_dir(tmp_path)
    steps = pd.read_parquet(run / "heuristic/event_000/steps.parquet")
    ops = pd.read_parquet(run / "heuristic/event_000/ops.parquet")
    t = report.decode_table(steps, ops, report.config_from(CFG), "float32", machine, 0.7, "heuristic").set_index("op_class")
    assert list(t.index) == list(OP_CLASSES)
    assert t.loc["attention", "gpu_us"] == pytest.approx(300.0) and t.loc["attention", "share"] == pytest.approx(0.3)
    assert t.loc["mlp", "verdict"] == "memory_bound" and t.loc["rope", "verdict"] == "launch_bound"
    assert t.loc["attention", "verdict"] == "parallelism_candidate" and t.loc["qkv_proj", "verdict"] == "below_ceiling_unknown"
    assert t.loc["mlp", "layer_mean_us"] == pytest.approx(0.02) and t.loc["embed", "layer_mean_us"] == pytest.approx(2.0)
    assert (t.bytes > 0).all() and t.loc["norm", "flops"] == 0 and t.loc["norm", "ai"] == 0
    assert t.loc["mlp", "pct_dram"] > 70 and 0 < t.loc["qkv_proj", "pct_dram"] < 70


def test_missing_class_rows_are_zero_not_error(tmp_path, machine):
    run = _run_dir(tmp_path, drop=("rope",))
    steps = pd.read_parquet(run / "heuristic/event_000/steps.parquet")
    ops = pd.read_parquet(run / "heuristic/event_000/ops.parquet")
    t = report.decode_table(steps, ops, report.config_from(CFG), "float32", machine, 0.7, "heuristic").set_index("op_class")
    assert t.loc["rope", "gpu_us"] == 0 and t.loc["rope", "verdict"] == "launch_bound"
    assert math.isnan(t.loc["rope", "achieved_gbps"]) and math.isnan(t.loc["rope", "pct_dram"])


def test_prefill_costs_sum_over_chunks(tmp_path, machine):
    run = _run_dir(tmp_path)
    prefill = pd.read_parquet(run / "heuristic/event_000/prefill.parquet")
    ops = pd.read_parquet(run / "heuristic/event_000/ops.parquet")
    cfg = report.config_from(CFG)
    whole = report.prefill_table(prefill, ops, cfg, "float32", machine, 0.7, "heuristic", chunk=4096).set_index("op_class")
    chunked = report.prefill_table(prefill, ops, cfg, "float32", machine, 0.7, "heuristic", chunk=3).set_index("op_class")
    assert whole.loc["attention", "gpu_us"] == pytest.approx(20.0)                       # 2 layers × 10 µs, mean over 2 prefills
    assert whole.loc["mlp", "flops"] == chunked.loc["mlp", "flops"]                       # GEMM flops do not depend on chunking
    assert chunked.loc["attention", "flops"] == whole.loc["attention", "flops"]           # causal pairs do not depend on chunking
    assert chunked.loc["attention", "bytes"] > whole.loc["attention", "bytes"]             # but K/V are re-read per chunk
    assert whole.loc["attention", "share"] == pytest.approx(20.0 / 400.0)                 # base = mean prefill_us


def test_attention_steps_keep_one_row_per_step_and_find_the_measured_cell(tmp_path, machine):
    run = _run_dir(tmp_path)
    steps = pd.concat([_steps(STEP_US, lens=(8, 4), n=1), _steps(STEP_US, lens=(9, 4), n=1).assign(step=1)])
    table = report.load_table(_table(tmp_path))
    measured = report.load_measured(_data(tmp_path))
    assert measured[(KEY, "fd_s2_paged")] == 10.0 and len(measured) == 3                    # the warm row is skipped
    rows = report.attention_steps(steps, report.config_from(CFG), table, measured)
    assert len(rows) == 2 and rows.iloc[0].neighbor_distance == pytest.approx(0.0) and rows.iloc[1].neighbor_distance > 0
    assert (rows.source == "nearest_measured").all() and (rows.chosen_variant == "flashdecoding_paged").all()
    assert (rows.best_alternative == "fd_s2_paged").all() and rows.regret.iloc[0] == pytest.approx(2.0)
    summary = report.attention_summary(rows, 0.3)
    assert summary["chosen_over_best"] == pytest.approx(3.0) and summary["regret"] == pytest.approx(2.0)
    assert summary["amdahl_bound"] == pytest.approx(report.amdahl_bound(0.3, 3.0)) and summary["neighbor_key"] == KEY


def test_attention_falls_back_when_the_table_lacks_the_head_shape(tmp_path):
    steps = _steps(STEP_US)
    table = report.load_table(_table(tmp_path, h_q=32, h_kv=8))
    rows = report.attention_steps(steps, report.config_from(CFG), table, {})
    assert (rows.source == "unavailable").all() and rows.regret.isna().all()
    summary = report.attention_summary(rows, 0.3)
    assert math.isnan(summary["amdahl_bound"]) and summary["chosen_variant"] == "flashdecoding_paged"


def test_diagnose_and_write_round_trip(tmp_path, machine):
    run = _run_dir(tmp_path)
    result = report.diagnose(run, machine, _table(tmp_path), _data(tmp_path))
    p = result.summary["policies"]["heuristic"]
    expected_unattributed = 100 * (STEP_US - (2 + 5 + 2 * sum(LAYER_US.values()))) / STEP_US
    assert p["unattributed_pct"] == pytest.approx(expected_unattributed)
    assert p["timer_overhead_pct"] == pytest.approx(100 * (STEP_US / CONTROL_STEP_US - 1))
    assert p["step_waterfall"]["host_residual_us"] == pytest.approx(90.0) and p["tpot_waterfall"]["tpot_us_mean"] == pytest.approx(1000.0)
    assert p["attention"]["amdahl_bound"] == pytest.approx(report.amdahl_bound(0.3, 3.0))
    assert result.summary["ceilings"]["threshold"] == 0.7 and result.summary["identity"]["performance_claim"] is False
    assert len(result.ops) == 2 * len(OP_CLASSES) and set(result.ops.phase) == {"decode", "prefill"}
    report.write(run, result)
    loaded = json.loads((run / "diagnosis.json").read_text())
    assert loaded["policies"]["heuristic"]["n_steps"] == 2
    assert list(pd.read_csv(run / "ops.csv").columns) == report.OPS_CSV_COLUMNS
    assert len(pd.read_csv(run / "attention_steps.csv")) == 2


def test_missing_control_run_gives_nan_overhead(tmp_path, machine):
    run = _run_dir(tmp_path, control=False)
    result = report.diagnose(run, machine, _table(tmp_path), _data(tmp_path))
    assert math.isnan(result.summary["policies"]["heuristic"]["timer_overhead_pct"])


def test_write_serializes_nan_as_null(tmp_path, machine):
    run = _run_dir(tmp_path, control=False)
    result = report.diagnose(run, machine, _table(tmp_path, h_q=32, h_kv=8), tmp_path / "nowhere")
    report.write(run, result)
    loaded = json.loads((run / "diagnosis.json").read_text())
    assert loaded["policies"]["heuristic"]["timer_overhead_pct"] is None
    assert loaded["policies"]["heuristic"]["attention"]["source"] == "unavailable"


def test_diagnose_refuses_a_folder_without_event_runs(tmp_path, machine):
    (tmp_path / "manifest.json").write_text(json.dumps({"model_config": CFG, "model_dtype": "float32"}))
    with pytest.raises(FileNotFoundError, match="event_000"):
        report.diagnose(tmp_path, machine, None, None)
