"""Serving reports from recorded timings, preserving repetition and evidence labels."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def tpot_us(tokens: pd.DataFrame) -> pd.Series:
    """Per-request mean inter-token gap. Admission/prefill stalls remain included."""
    if tokens.empty:
        return pd.Series(dtype=float, name="tpot_us")
    order = ["rid", "position"] if "position" in tokens else ["rid", "t_us"]
    ordered = tokens.sort_values(order, kind="stable")
    gaps = ordered.groupby("rid", sort=False).t_us.diff()
    if (gaps.dropna() < 0).any():
        raise ValueError("token timestamps must increase with generated position")
    return gaps.groupby(ordered.rid).mean().rename("tpot_us")


def iter_runs(results_dir):
    """Yield (policy, repeat, directory, metadata), accepting original flat artifacts."""
    root = Path(results_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for policy_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        run_dirs = [policy_dir] if (policy_dir / "steps.parquet").exists() else sorted(policy_dir.glob("repeat_*"))
        for index, directory in enumerate(run_dirs):
            if not (directory / "steps.parquet").exists() or not (directory / "tokens.parquet").exists():
                continue
            meta_path = directory / "meta.json"
            metadata = {**manifest, **(json.loads(meta_path.read_text()) if meta_path.exists() else {})}
            yield policy_dir.name, int(metadata.get("repeat", index)), directory, metadata


def _mean(frame, column):
    return float(frame[column].mean()) if column in frame and len(frame) else math.nan


def summarize(results_dir) -> pd.DataFrame:
    records, tpots = [], {}
    for policy, repeat, directory, meta in iter_runs(results_dir):
        steps, tokens = pd.read_parquet(directory / "steps.parquet"), pd.read_parquet(directory / "tokens.parquet")
        tpot = tpot_us(tokens).dropna() / 1000
        tpots.setdefault(policy, []).extend(tpot.tolist())
        duration = float(meta.get("total_wall_us", tokens.t_us.max() if len(tokens) else 0))
        pre_path = directory / "prefill.parquet"
        pre = pd.read_parquet(pre_path) if pre_path.exists() else pd.DataFrame()
        ttft = (pre.first_token_us - pre.arrival_us) / 1000 if {"first_token_us", "arrival_us"} <= set(pre) else pd.Series(dtype=float)
        records.append(dict(policy=policy, repeat=repeat, steps=len(steps), generated_tokens=len(tokens),
                            attn_ms_per_step=_mean(steps, "attn_us") / 1000,
                            step_ms_per_step=_mean(steps, "step_us") / 1000,
                            policy_us_per_step=_mean(steps, "policy_us"),
                            decode_wall_ms_per_step=_mean(steps, "decode_wall_us") / 1000,
                            tpot_ms_mean=float(tpot.mean()), ttft_ms_mean=float(ttft.mean()),
                            throughput_tokens_s=len(tokens) * 1e6 / duration if duration > 0 else math.nan,
                            evidence_kind=meta.get("evidence_kind", "legacy_unverified"),
                            performance_claim=meta.get("performance_claim", False)))
    if not records:
        return pd.DataFrame(columns=["policy", "repeats", "steps", "attn_ms_per_step", "step_ms_per_step",
                                     "attn_share", "tpot_ms_mean", "tpot_ms_p50", "tpot_ms_p90", "speedup_vs_heuristic"])
    runs = pd.DataFrame(records)
    rows = []
    for policy, group in runs.groupby("policy", sort=False):
        kinds = set(group.evidence_kind)
        if len(kinds) != 1:
            raise ValueError(f"cannot aggregate mixed evidence types for policy {policy}")
        row = {column: float(group[column].mean()) for column in
               ("steps", "generated_tokens", "attn_ms_per_step", "step_ms_per_step", "policy_us_per_step",
                "decode_wall_ms_per_step", "tpot_ms_mean", "ttft_ms_mean", "throughput_tokens_s")}
        row.update(policy=policy, repeats=len(group), evidence_kind=next(iter(kinds)),
                   performance_claim=bool(group.performance_claim.all()),
                   tpot_ms_std=float(group.tpot_ms_mean.std(ddof=1)) if len(group) > 1 else math.nan,
                   tpot_ms_p50=float(np.quantile(tpots[policy], .5)) if tpots[policy] else math.nan,
                   tpot_ms_p90=float(np.quantile(tpots[policy], .9)) if tpots[policy] else math.nan)
        row["attn_share"] = row["attn_ms_per_step"] / row["step_ms_per_step"] if row["step_ms_per_step"] > 0 else math.nan
        row.update(speedup_vs_heuristic=math.nan, speedup_ci95_low=math.nan, speedup_ci95_high=math.nan)
        baseline = runs[runs.policy == "heuristic"]
        # Legacy files retain descriptive ratios for compatibility; only explicit
        # CUDA runs can carry a performance claim. CPU split timings are irrelevant.
        if not baseline.empty and row["evidence_kind"] != "cpu_functional":
            paired = baseline[["repeat", "tpot_ms_mean"]].merge(group[["repeat", "tpot_ms_mean"]], on="repeat", suffixes=("_base", "_candidate"))
            paired = paired.dropna()
            paired = paired[(paired.tpot_ms_mean_base > 0) & (paired.tpot_ms_mean_candidate > 0)]
            if not paired.empty:
                row["speedup_vs_heuristic"] = float(paired.tpot_ms_mean_base.mean() / paired.tpot_ms_mean_candidate.mean())
                if len(paired) >= 3:
                    picks = np.random.default_rng(0).integers(0, len(paired), (2000, len(paired)))
                    ratios = (paired.tpot_ms_mean_base.to_numpy()[picks].mean(axis=1) /
                              paired.tpot_ms_mean_candidate.to_numpy()[picks].mean(axis=1))
                    row["speedup_ci95_low"], row["speedup_ci95_high"] = map(float, np.quantile(ratios, [.025, .975]))
        rows.append(row)
    out = pd.DataFrame(rows)
    equivalence_path = Path(results_dir) / "equivalence.csv"
    if equivalence_path.exists():
        eq = pd.read_csv(equivalence_path)
        if not eq.empty and "passed" in eq:
            verdict = eq.groupby("policy").passed.all().to_dict()
            tokens = eq.groupby("policy").tokens_identical.all().to_dict() if "tokens_identical" in eq else verdict
            out["tokens_equivalent"] = out.policy.map(tokens)
            out["output_validation_passed"] = out.policy.map(verdict)
            # The reference compared with itself is defined as exact.
            reference_names = set(eq.reference) - set(eq.policy)
            out.loc[out.policy.isin(reference_names), "tokens_equivalent"] = True
            out.loc[out.policy.isin(reference_names), "output_validation_passed"] = True
            out.loc[out.output_validation_passed == False, "performance_claim"] = False  # noqa: E712
    return out


def oracle(results_dir, fixed=None) -> pd.DataFrame:
    """Per-step best fixed policy, using mean repeats and identical schedules only."""
    policy_runs = {}
    for policy, _, directory, _ in iter_runs(results_dir):
        frame = pd.read_parquet(directory / "steps.parquet")
        if not frame.empty:
            policy_runs.setdefault(policy, []).append(frame)
    if not policy_runs:
        return pd.DataFrame()
    names = list(fixed) if fixed is not None else [n for n in policy_runs if n == "fa2" or n.startswith("fixed")]
    names = [n for n in names if n in policy_runs]
    if not names:
        return pd.DataFrame()
    reference = policy_runs[names[0]][0]
    identity = [c for c in ("step", "B", "seq_ids", "lens") if c in reference]
    averages = {}
    for policy, frames in policy_runs.items():
        for frame in frames:
            if not set(identity) <= set(frame) or not reference[identity].reset_index(drop=True).equals(frame[identity].reset_index(drop=True)):
                raise ValueError("oracle requires identical decode steps and batch composition across all runs")
        averages[policy] = pd.concat(frames).groupby("step").attn_us.mean()
    table = pd.DataFrame(averages)
    best_sum = float(table[names].min(axis=1).sum())
    if best_sum <= 0:
        raise ValueError("oracle timings must be positive")
    return pd.DataFrame([dict(policy=name, steps=len(table), attn_ms=float(table[name].sum() / 1000),
                              oracle_attn_ms=best_sum / 1000, attn_ratio_vs_oracle=float(table[name].sum() / best_sum),
                              regret=float(table[name].sum() / best_sum - 1)) for name in table])
