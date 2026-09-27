from scripts.followup_campaign import build_jobs, compatible_manifest


def test_resume_refuses_different_seed_or_scenario_even_with_same_model():
    job = dict(model="qwen", scenario_sha256="abc", seed=0, repeats=3, policy_specs=["heuristic", "model"])
    assert compatible_manifest(job, dict(job, status="complete"))
    for key, value in (("seed", 1), ("scenario_sha256", "changed"), ("repeats", 4),
                       ("policy_specs", ["heuristic", "table"])):
        assert not compatible_manifest(job, dict(job, **{key: value}))


def test_arrivals_default_warmup_replays_full_overlap_and_cannot_resume_partial(tmp_path):
    job = build_jobs(["qwen4b"], ["arrivals"], [0], 3, tmp_path)[0]
    assert "--warmup-steps" not in job["command"]
    assert job["warmup_steps"] is None
    assert compatible_manifest(job, dict(job))
    assert not compatible_manifest(job, dict(job, warmup_steps=2))
