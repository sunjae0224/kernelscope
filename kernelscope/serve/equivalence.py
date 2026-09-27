"""Strict greedy-token comparison; floating-point closeness is a separate result."""
import math

import pandas as pd
import torch

from kernelscope.serve.engine import Engine
from kernelscope.serve.kvcache import PagePool

COLUMNS = ["policy", "reference", "steps_compared", "max_abs_logit_diff", "token_agreement",
           "tokens_compared", "token_mismatches", "missing_tokens", "schedule_match", "logits_checked",
           "logits_finite", "tokens_identical", "passed", "first_mismatch"]


def _token_map(frame):
    frame = frame.copy()
    if "position" not in frame:
        frame["position"] = frame.groupby("rid", sort=False).cumcount()
    if frame.duplicated(["rid", "position"]).any():
        raise ValueError("duplicate (rid, position) token records")
    return {(int(row.rid), int(row.position)): int(row.token) for row in frame.itertuples()}


def compare_results(reference, candidate, reference_name="heuristic", policy_name="candidate", atol=0.05):
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("logit tolerance must be finite and nonnegative")
    left, right = _token_map(reference.tokens), _token_map(candidate.tokens)
    keys = sorted(left.keys() | right.keys())
    missing = sum(key not in left or key not in right for key in keys)
    mismatches = [key for key in keys if key not in left or key not in right or left[key] != right[key]]
    agreement = (len(keys) - len(mismatches)) / len(keys) if keys else math.nan
    identity = ("step", "B", "seq_ids", "lens")
    left_columns = {c for c in identity if c in reference.steps}
    right_columns = {c for c in identity if c in candidate.steps}
    alignment = [c for c in identity if c in left_columns & right_columns]
    schedule_match = left_columns == right_columns and reference.steps[alignment].reset_index(drop=True).equals(
        candidate.steps[alignment].reset_index(drop=True))
    keys_a = reference.metadata.get("logit_step_keys")
    keys_b = candidate.metadata.get("logit_step_keys")
    alignment_ok = schedule_match and (keys_a == keys_b)
    checked = bool(reference.logits and candidate.logits)
    count = 0
    finite, maximum = True, math.nan
    if checked:
        alignment_ok &= len(reference.logits) == len(candidate.logits)
        maximum = 0.0
        for a, b in zip(reference.logits, candidate.logits):
            if a.shape != b.shape:
                alignment_ok = False
                continue
            finite &= bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            if finite:
                maximum = max(maximum, float((a.float() - b.float()).abs().max().item()))
            count += 1
        if not finite:
            maximum = math.inf
    elif bool(reference.logits) != bool(candidate.logits):
        alignment_ok = False
    passed = bool(keys) and not mismatches and alignment_ok and (not checked or (finite and maximum <= atol))
    first = None
    if mismatches:
        key = mismatches[0]
        first = f"rid={key[0]}, position={key[1]}, reference={left.get(key)}, candidate={right.get(key)}"
    return dict(policy=policy_name, reference=reference_name, steps_compared=count,
                max_abs_logit_diff=maximum, token_agreement=agreement, tokens_compared=len(keys),
                token_mismatches=len(mismatches), missing_tokens=missing, schedule_match=bool(alignment_ok),
                logits_checked=checked, logits_finite=finite if checked else None,
                tokens_identical=not mismatches, passed=bool(passed), first_mismatch=first)


def compare_policies(model, policies, requests, vocab, kv_bytes, record_steps=32, max_batch=64, seed=0,
                     atol=0.05) -> pd.DataFrame:
    policies, requests = list(policies), list(requests)
    if len(policies) < 2:
        raise ValueError("equivalence requires at least two policies")
    reference, rows = None, []
    for policy in policies:
        pool = PagePool.for_budget(model.cfg, kv_bytes, device=model.device, dtype=model.dtype)
        result = Engine(model, pool, policy, max_batch).run(requests, vocab, record_steps, seed)
        del pool
        if reference is None:
            reference = result
        else:
            rows.append(compare_results(reference, result, policies[0].name, policy.name, atol))
    return pd.DataFrame(rows, columns=COLUMNS)
