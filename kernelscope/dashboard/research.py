"""Comparable serving evidence, with experiment identities and cache provenance.

No result directories are pooled. A comparison pairs repetitions only inside
one experiment and one identical model/data/implementation identity.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from kernelscope.dashboard.data import load_manifest

IDENTITY = ["experiment", "campaign_group", "model", "model_backend", "model_dtype", "scenario", "scenario_sha256",
            "scenario_family", "prompt_kind", "evaluation_split", "dataset_id", "dataset_sha256",
            "resolved_prompts_sha256", "seed", "implementation", "evidence_kind", "warmup_runs", "warmup_steps"]
OVERHEAD_PHASES = ("first_decision", "cache_miss", "cache_hit", "unrecorded")
OVERHEAD_LABELS = {"first_decision": "첫 선택", "cache_miss": "이후 cache miss",
                   "cache_hit": "이후 cache hit", "unrecorded": "이후 cache 상태 미기록"}


def _value(value, missing="unrecorded"):
    if value is None:
        return missing
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _true(value):
    return (isinstance(value, (bool, np.bool_)) and bool(value)) or (isinstance(value, str) and value.lower() == "true")


def _implementation(meta):
    for field in ("source_fingerprint", "code_fingerprint", "source_sha256", "source_hashes"):
        if meta.get(field):
            value = meta[field]
            return hashlib.sha256(_value(value).encode()).hexdigest() if isinstance(value, (dict, list)) else str(value)
    return "unrecorded"


def experiment_identity(directory, meta):
    dataset = meta.get("dataset") if isinstance(meta.get("dataset"), dict) else {}
    scenario = _value(meta.get("scenario"), Path(directory).name)
    prompt_kind = _value(meta.get("prompt_kind"))
    split = meta.get("evaluation_split", dataset.get("evaluation_split"))
    if split is None:
        split = "legacy_synthetic" if prompt_kind == "seeded_synthetic_token_ids" else "unclassified"
    path = Path(directory).resolve()
    campaign_group = meta.get("campaign_id")
    if not campaign_group:
        parts = path.parts
        anchor = next((i for i, part in enumerate(parts) if part in ("serve_4090", "serve")), None)
        campaign_group = parts[anchor + 1] if anchor is not None and anchor + 1 < len(parts) else path.parent.name
    warmup_steps = ("full_scenario" if meta["warmup_steps"] is None else _value(meta["warmup_steps"])) if "warmup_steps" in meta else "unrecorded"
    return dict(experiment=str(path), campaign_group=_value(campaign_group), model=_value(meta.get("model")),
                model_backend=_value(meta.get("model_backend")), model_dtype=_value(meta.get("model_dtype")),
                scenario=scenario, scenario_family=_value(meta.get("scenario_family"), Path(scenario).stem),
                scenario_sha256=_value(meta.get("scenario_sha256")), prompt_kind=prompt_kind,
                evaluation_split=_value(split), dataset_id=_value(meta.get("dataset_id", dataset.get("id"))),
                dataset_sha256=_value(meta.get("dataset_sha256")),
                resolved_prompts_sha256=_value(meta.get("resolved_prompts_sha256")),
                seed=_value(meta.get("seed")), implementation=_implementation(meta),
                evidence_kind=_value(meta.get("evidence_kind")), warmup_runs=_value(meta.get("warmup_runs")),
                warmup_steps=warmup_steps)


def campaign_catalog(directories):
    records = []
    for directory in directories:
        meta = load_manifest(directory)
        records.append({**experiment_identity(directory, meta), "status": meta.get("status", "unrecorded"),
                        "declared_repeats": meta.get("repeats"), "completed_at": meta.get("completed_at", "")})
    return pd.DataFrame(records, columns=IDENTITY + ["status", "declared_repeats", "completed_at"])


def offline_policy_benchmark(path):
    """Load the independent CPU benchmark without executing either backend."""
    try:
        meta = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}, pd.DataFrame()
    if not isinstance(meta, dict) or meta.get("evidence_kind") != "cpu_policy_latency":
        return {}, pd.DataFrame()
    rows = []
    for case in meta.get("cases", []):
        timing = case.get("timing", {})
        python, native = timing.get("python", {}), timing.get("native", {})
        rows.append(dict(case=case.get("name"), scenario=case.get("scenario"),
                         python_cold_us=python.get("cold_us", {}).get("median"),
                         native_cold_us=native.get("cold_us", {}).get("median"),
                         python_hit_us=python.get("hot_us", {}).get("median"),
                         native_hit_us=native.get("hot_us", {}).get("median"),
                         cold_decision_speedup=case.get("cold_decision_speedup"),
                         selected_splits_identical=case.get("selected_splits_identical"),
                         ranking_order_identical=case.get("ranking_order_identical"),
                         predictions_within_tolerance=case.get("predictions_within_tolerance"),
                         max_abs_prediction_diff_us=case.get("max_abs_prediction_diff_us")))
    return meta, pd.DataFrame(rows)


def policy_overhead(serving):
    """First decision and non-first hit/miss costs; never guess a cache state.

    Categories are disjoint, so their call counts and total CPU times add up.
    First means the first recorded step in each independent repetition.
    """
    rows = []
    for policy, frames in serving.items():
        steps = frames.get("steps", pd.DataFrame())
        if steps.empty or not {"step", "policy_us"} <= set(steps):
            continue
        steps = steps.copy()
        if "repeat" not in steps:
            steps["repeat"] = 0
        steps = steps.sort_values(["repeat", "step"], kind="stable").reset_index(drop=True)
        steps["policy_us"] = pd.to_numeric(steps.policy_us, errors="coerce")
        steps["phase"] = "unrecorded"
        if "policy_cache_hit" in steps:
            # A null flag also occurs for policies that do not implement a cache.
            hits = steps.policy_cache_hit.map(_true)
            misses = steps.policy_cache_hit.map(lambda x: (isinstance(x, (bool, np.bool_)) and not bool(x))
                                                or (isinstance(x, str) and x.lower() == "false"))
            steps.loc[hits, "phase"] = "cache_hit"
            steps.loc[misses, "phase"] = "cache_miss"
        first = steps.groupby("repeat", sort=False).head(1).index
        steps.loc[first, "phase"] = "first_decision"
        steps = steps[np.isfinite(steps.policy_us) & steps.policy_us.ge(0)]
        for phase, group in steps.groupby("phase", sort=False):
            rows.append(dict(policy=policy, phase=phase, calls=len(group), repeats=group.repeat.nunique(),
                             mean_us=float(group.policy_us.mean()), p50_us=float(group.policy_us.median()),
                             p90_us=float(group.policy_us.quantile(.9)), total_us=float(group.policy_us.sum())))
    return pd.DataFrame(rows, columns=["policy", "phase", "calls", "repeats", "mean_us", "p50_us", "p90_us", "total_us"])


def _validation(eq, policy, repeat, total_runs):
    if eq.empty or not {"policy", "reference", "passed"} <= set(eq):
        return None, None, math.nan
    selected = eq[eq.policy == policy]
    if selected.empty and policy in set(eq.reference):
        return True, True, 1.0  # The reference is identical to itself.
    if "repeat" in selected:
        selected = selected[selected.repeat == repeat]
    elif total_runs != 1:
        return None, None, math.nan
    if selected.empty:
        return None, None, math.nan
    valid = bool(selected.passed.map(_true).all())
    identical = bool(selected.tokens_identical.map(_true).all()) if "tokens_identical" in selected else valid
    if "token_agreement" in selected:
        agreement = pd.to_numeric(selected.token_agreement, errors="coerce")
        weights = pd.to_numeric(selected.tokens_compared, errors="coerce") if "tokens_compared" in selected else pd.Series(1., index=selected.index)
        keep = agreement.notna() & weights.gt(0)
        fraction = float(np.average(agreement[keep], weights=weights[keep])) if keep.any() else math.nan
    else:
        fraction = 1.0 if identical else math.nan
    return identical, valid, fraction


def campaign_overview(directories):
    """One row per experiment/compatible identity/policy, with matched-run ratios."""
    from kernelscope.serve.report import iter_runs, tpot_us

    run_records = []
    for directory in directories:
        directory = Path(directory)
        manifest = load_manifest(directory)
        try:
            eq = pd.read_csv(directory / "equivalence.csv") if (directory / "equivalence.csv").exists() else pd.DataFrame()
            runs = list(iter_runs(directory))
        except (OSError, ValueError, KeyError):
            continue
        policy_counts = pd.Series([r[0] for r in runs]).value_counts()
        for policy, repeat, run_dir, meta in runs:
            try:
                steps = pd.read_parquet(run_dir / "steps.parquet")
                tokens = pd.read_parquet(run_dir / "tokens.parquet")
                request_tpot = tpot_us(tokens).dropna() / 1000
            except (OSError, ValueError, KeyError, AttributeError):
                continue
            identical, valid, agreement = _validation(eq, policy, repeat, int(policy_counts[policy]))
            costs = policy_overhead({policy: {"steps": steps.assign(repeat=repeat)}})
            extra = {}
            for phase in OVERHEAD_PHASES:
                group = costs[costs.phase == phase]
                extra[f"{phase}_us"] = float(group.mean_us.iloc[0]) if len(group) else math.nan
                extra[f"{phase}_calls"] = int(group.calls.iloc[0]) if len(group) else 0
            def mean(column):
                return float(steps[column].mean()) if column in steps and len(steps) else math.nan
            run_records.append({**experiment_identity(directory, meta), **extra, "policy": policy, "repeat": repeat,
                                "tpot_ms": float(request_tpot.mean()), "requests": len(request_tpot),
                                "generated_tokens": len(tokens),
                                "policy_us": mean("policy_us"), "decode_wall_ms": mean("decode_wall_us") / 1000,
                                "attention_ms": mean("attn_us") / 1000, "tokens_identical": identical,
                                "validation_passed": valid, "token_agreement": agreement,
                                "performance_claim": _true(meta.get("performance_claim")),
                                "status": manifest.get("status", "unrecorded"),
                                "declared_repeats": meta.get("repeats"), "observed_steps": len(steps)})
    if not run_records:
        return pd.DataFrame(columns=IDENTITY + ["policy", "repeats", "tpot_ms", "claim_eligible", "speedup"])
    runs = pd.DataFrame(run_records)
    rows = []
    for identity, same_experiment in runs.groupby(IDENTITY, dropna=False, sort=False):
        reference = same_experiment[same_experiment.policy == "heuristic"]
        comparison_id = hashlib.sha256(json.dumps(list(identity), ensure_ascii=False).encode()).hexdigest()[:10]
        for policy, group in same_experiment.groupby("policy", sort=False):
            row = {**dict(zip(IDENTITY, identity)), "comparison_id": comparison_id, "policy": policy, "repeats": group.repeat.nunique(),
                   "declared_repeats": group.declared_repeats.iloc[0], "requests_per_repeat": float(group.requests.mean()),
                   "status": group.status.iloc[0], "tpot_ms": float(group.tpot_ms.mean()),
                   "tpot_run_std_ms": float(group.tpot_ms.std(ddof=1)), "policy_us": float(group.policy_us.mean()),
                   "decode_wall_ms": float(group.decode_wall_ms.mean()), "attention_ms": float(group.attention_ms.mean()),
                   "tokens_equivalent": bool(group.tokens_identical.map(_true).all()),
                   "output_validation_passed": bool(group.validation_passed.map(_true).all()),
                   "token_agreement": float(np.average(group.token_agreement, weights=group.generated_tokens))
                       if group.token_agreement.notna().all() and group.generated_tokens.sum() > 0 else math.nan,
                   "baseline_tpot_ms": float(reference.tpot_ms.mean()) if not reference.empty else math.nan}
            for phase in OVERHEAD_PHASES:
                counts = group[f"{phase}_calls"]
                row[f"{phase}_calls"] = int(counts.sum())
                row[f"{phase}_us"] = float((group[f"{phase}_us"].fillna(0) * counts).sum() / counts.sum()) if counts.sum() else math.nan
            reasons = []
            if row["evidence_kind"] != "cuda_serving": reasons.append("not_cuda_serving")
            if not group.status.eq("complete").all(): reasons.append("incomplete_experiment")
            if not group.performance_claim.all(): reasons.append("run_not_eligible")
            if not row["tokens_equivalent"]: reasons.append("tokens_mismatch_or_unchecked")
            if not row["output_validation_passed"]: reasons.append("output_validation_failed_or_missing")
            if reference.empty: reasons.append("no_compatible_reference")
            elif set(group.repeat) != set(reference.repeat): reasons.append("unmatched_repetitions")
            if group.repeat.duplicated().any() or reference.repeat.duplicated().any(): reasons.append("duplicate_repetition")
            declared = pd.to_numeric(group.declared_repeats, errors="coerce")
            if declared.notna().any() and not declared.dropna().eq(row["repeats"]).all(): reasons.append("missing_repetitions")
            if not np.isfinite(group.tpot_ms).all() or not group.tpot_ms.gt(0).all(): reasons.append("no_valid_tpot")
            if not reference.empty and (not reference.performance_claim.all() or not np.isfinite(reference.tpot_ms).all()
                                        or not reference.tpot_ms.gt(0).all()): reasons.append("invalid_reference")
            row.update(claim_eligible=not reasons, ineligibility=";".join(reasons), speedup=math.nan,
                       speedup_ci95_low=math.nan, speedup_ci95_high=math.nan)
            if row["claim_eligible"]:
                paired = reference[["repeat", "tpot_ms"]].merge(group[["repeat", "tpot_ms"]], on="repeat", suffixes=("_baseline", "_candidate"))
                baseline, candidate = paired.tpot_ms_baseline.to_numpy(), paired.tpot_ms_candidate.to_numpy()
                row["speedup"] = float(baseline.mean() / candidate.mean())
                if len(paired) >= 3:
                    picks = np.random.default_rng(0).integers(0, len(paired), (2000, len(paired)))
                    ratios = baseline[picks].mean(axis=1) / candidate[picks].mean(axis=1)
                    row["speedup_ci95_low"], row["speedup_ci95_high"] = map(float, np.quantile(ratios, [.025, .975]))
            rows.append(row)
    return pd.DataFrame(rows)
