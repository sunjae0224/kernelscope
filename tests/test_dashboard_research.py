import json

import pandas as pd
import pytest

from kernelscope.dashboard import research


def experiment(path, *, model="model-a", seed=0, candidate_gap=1000., repeats=3, failed_policy=None,
               prompt_kind="constructed_natural_text", split="heldout_text"):
    path.mkdir(parents=True)
    manifest = dict(model=model, seed=seed, scenario="heldout_ragged.yaml", scenario_family="ragged",
                    scenario_sha256="scenario", prompt_kind=prompt_kind, evaluation_split=split,
                    dataset_id="original-local-text", dataset_sha256="dataset", resolved_prompts_sha256="prompts",
                    model_backend="flash_attn_paged", model_dtype="float16", source_sha256={"engine.py": "version-a"},
                    status="complete", repeats=repeats, warmup_runs=1, evidence_kind="cuda_serving", performance_claim=False)
    (path / "manifest.json").write_text(json.dumps(manifest))
    eq = []
    for policy in ("heuristic", "table", "fixed8"):
        for repeat in range(repeats):
            run = path / policy / f"repeat_{repeat:03d}"
            run.mkdir(parents=True)
            (run / "meta.json").write_text(json.dumps({"repeat": repeat, "performance_claim": True}))
            pd.DataFrame({"step": [0, 1, 2, 3], "policy_us": [1000., 10., 100., 3.],
                          "policy_cache_hit": [False, True, False, None], "attn_us": [30.] * 4,
                          "decode_wall_us": [100.] * 4}).to_parquet(run / "steps.parquet")
            gap = 2000. if policy == "heuristic" else candidate_gap
            pd.DataFrame({"rid": [0] * 5, "position": list(range(5)), "step": list(range(5)),
                          "t_us": [gap * i for i in range(5)], "token": [1, 2, 3, 4, 5]}).to_parquet(run / "tokens.parquet")
            if policy != "heuristic":
                passed = policy != failed_policy
                eq.append(dict(policy=policy, reference="heuristic", repeat=repeat, passed=passed,
                               tokens_identical=passed, token_agreement=1. if passed else .8, tokens_compared=5))
    pd.DataFrame(eq).to_csv(path / "equivalence.csv", index=False)
    return path


def test_overhead_categories_are_disjoint_and_legacy_hits_are_unknown():
    steps = pd.DataFrame({"repeat": [0, 0, 0, 0, 1, 1], "step": [0, 1, 2, 3, 0, 1],
                          "policy_us": [1000., 10., 100., 3., 2000., 20.],
                          "policy_cache_hit": [False, True, False, None, False, True]})
    costs = research.policy_overhead({"model": {"steps": steps}}).set_index("phase")
    assert costs.calls.sum() == len(steps)
    assert costs.total_us.sum() == steps.policy_us.sum()
    assert costs.loc["first_decision", "mean_us"] == 1500.
    assert costs.loc["cache_hit", "calls"] == 2
    assert costs.loc["cache_miss", "mean_us"] == 100.
    assert costs.loc["unrecorded", "calls"] == 1
    legacy = research.policy_overhead({"model": {"steps": steps.drop(columns="policy_cache_hit")}})
    assert set(legacy.phase) == {"first_decision", "unrecorded"}
    assert legacy.set_index("phase").loc["unrecorded", "calls"] == 4


def test_models_seeds_and_datasets_remain_separate(tmp_path):
    paths = [experiment(tmp_path / "a", model="model-a", seed=0, candidate_gap=1000.),
             experiment(tmp_path / "b", model="model-a", seed=1, candidate_gap=1500.),
             experiment(tmp_path / "c", model="model-b", seed=0, candidate_gap=2500.)]
    overview = research.campaign_overview(paths)
    candidates = overview[overview.policy == "table"].set_index("experiment")
    assert len(candidates) == 3
    assert candidates.loc[str(paths[0]), "speedup"] == 2.
    assert candidates.loc[str(paths[1]), "speedup"] == pytest.approx(4 / 3)
    assert candidates.loc[str(paths[2]), "speedup"] == .8
    assert candidates.claim_eligible.all()
    assert candidates.comparison_id.nunique() == 3
    assert candidates.implementation.nunique() == 1
    assert candidates.repeats.tolist() == [3, 3, 3]
    assert candidates.cache_hit_calls.tolist() == [3, 3, 3]


def test_failed_policy_does_not_mask_valid_peer(tmp_path):
    path = experiment(tmp_path / "mixed", failed_policy="fixed8")
    rows = research.campaign_overview([path]).set_index("policy")
    assert rows.loc["table", "claim_eligible"]
    assert rows.loc["table", "speedup"] == 2.
    assert not rows.loc["fixed8", "claim_eligible"]
    assert pd.isna(rows.loc["fixed8", "speedup"])
    assert rows.loc["fixed8", "token_agreement"] == pytest.approx(.8)


def test_source_change_inside_experiment_cannot_get_a_pooled_ratio(tmp_path):
    path = experiment(tmp_path / "source-change")
    meta = path / "table/repeat_002/meta.json"
    meta.write_text(json.dumps({"repeat": 2, "performance_claim": True, "source_sha256": {"engine.py": "version-b"}}))
    rows = research.campaign_overview([path])
    table = rows[rows.policy == "table"]
    assert len(table) == 2
    assert not table.claim_eligible.any()
    assert table.speedup.isna().all()
    assert any("no_compatible_reference" in reason for reason in table.ineligibility)


def test_missing_equivalence_and_missing_repetition_are_not_eligible(tmp_path):
    path = experiment(tmp_path / "partial")
    (path / "table/repeat_002/tokens.parquet").unlink()
    rows = research.campaign_overview([path]).set_index("policy")
    assert not rows.loc["table", "claim_eligible"]
    assert "missing_repetitions" in rows.loc["table", "ineligibility"]
    eq = pd.read_csv(path / "equivalence.csv")
    eq = eq[~((eq.policy == "fixed8") & (eq.repeat == 2))]
    eq.to_csv(path / "equivalence.csv", index=False)
    rows = research.campaign_overview([path]).set_index("policy")
    assert not rows.loc["fixed8", "claim_eligible"]
    assert "output_validation_failed_or_missing" in rows.loc["fixed8", "ineligibility"]


def test_missing_evaluation_metadata_never_becomes_heldout_text(tmp_path):
    path = tmp_path / "legacy"
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"model": "old", "prompt_kind": "seeded_synthetic_token_ids"}))
    catalog = research.campaign_catalog([path])
    assert catalog.iloc[0].evaluation_split == "legacy_synthetic"
    assert catalog.iloc[0].implementation == "unrecorded"


def test_campaign_and_warmup_settings_stay_explicit(tmp_path):
    pilot = tmp_path / "serve_4090" / "pilot" / "model" / "arrivals" / "seed_0"
    confirmation = tmp_path / "serve_4090" / "fullwarmup" / "model" / "arrivals" / "seed_0"
    first = research.experiment_identity(pilot, {"warmup_steps": 2})
    second = research.experiment_identity(confirmation, {"warmup_steps": None})
    legacy = research.experiment_identity(tmp_path, {})
    assert first["campaign_group"] == "pilot" and first["warmup_steps"] == "2"
    assert second["campaign_group"] == "fullwarmup" and second["warmup_steps"] == "full_scenario"
    assert legacy["warmup_steps"] == "unrecorded"
