import json

import pandas as pd
import pytest

from kernelscope.serve.report import oracle, summarize, tpot_us


def write_run(root, name, attn, gap, repeat=0, evidence="cuda_serving", signature="[0]"):
    path = root / name / f"repeat_{repeat:03d}"
    path.mkdir(parents=True)
    pd.DataFrame([dict(step=i, B=1, seq_ids=signature, lens=str([4+i]), attn_us=t,
                       step_us=gap, policy_us=10, decode_wall_us=gap+10) for i, t in enumerate(attn)]).to_parquet(path / "steps.parquet")
    pd.DataFrame([dict(rid=0, position=i, token=i, t_us=i*gap) for i in range(len(attn)+1)]).to_parquet(path / "tokens.parquet")
    (path / "meta.json").write_text(json.dumps(dict(evidence_kind=evidence, performance_claim=evidence=="cuda_serving", repeat=repeat)))


def test_tpot_retains_admission_stalls_and_omits_one_token_requests():
    frame = pd.DataFrame([dict(rid=0, position=0, t_us=100), dict(rid=0, position=1, t_us=300),
                          dict(rid=0, position=2, t_us=1100), dict(rid=1, position=0, t_us=0)])
    values = tpot_us(frame)
    assert values[0] == 500 and pd.isna(values[1])


def test_summary_repeated_paired_timings_and_oracle(tmp_path):
    for repeat in range(3):
        write_run(tmp_path, "heuristic", [10000] * 4, 20000, repeat)
        write_run(tmp_path, "fixed8", [4000, 6000, 4000, 6000], 14000, repeat)
        write_run(tmp_path, "fa2", [8000, 3000, 8000, 3000], 18000, repeat)
    summary = summarize(tmp_path).set_index("policy")
    assert summary.loc["fixed8", "speedup_vs_heuristic"] == pytest.approx(20/14)
    assert summary.loc["fixed8", "attn_share"] == pytest.approx(5/14)
    assert summary.loc["fixed8", "speedup_ci95_low"] == pytest.approx(20/14)
    assert oracle(tmp_path).set_index("policy").loc["fixed8", "oracle_attn_ms"] == 14


def test_cpu_results_never_claim_a_dispatch_speedup(tmp_path):
    write_run(tmp_path, "heuristic", [10, 10], 100, evidence="cpu_functional")
    write_run(tmp_path, "fixed8", [2, 2], 20, evidence="cpu_functional")
    summary = summarize(tmp_path)
    assert summary.speedup_vs_heuristic.isna().all()
    assert not summary.performance_claim.any()


def test_oracle_rejects_mismatched_batches(tmp_path):
    write_run(tmp_path, "fa2", [10, 10], 100)
    write_run(tmp_path, "fixed8", [2, 2], 20, signature="[8]")
    with pytest.raises(ValueError, match="identical"):
        oracle(tmp_path)


def test_logit_failure_does_not_mislabel_identical_tokens(tmp_path):
    write_run(tmp_path, "heuristic", [10], 100)
    write_run(tmp_path, "fixed8", [2], 20)
    pd.DataFrame([dict(policy="fixed8", reference="heuristic", tokens_identical=True, passed=False)]).to_csv(
        tmp_path / "equivalence.csv", index=False)
    row = summarize(tmp_path).set_index("policy").loc["fixed8"]
    assert row.tokens_equivalent and not row.output_validation_passed and not row.performance_claim
