"""Teacher-forced split-policy diagnostics; never replaces strict free-generation checks.

Run from the project root with ``python -m scripts.check_policy_numerics --help``.
Both runs receive the baseline's token histories; their KV caches evolve independently.
The CPU logit copies invalidate timing measurements, so no performance claim is made.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


def logit_statistics(reference, candidate):
    """One row per sequence; retain ties and nonfinite outputs explicitly."""
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] < 2:
        raise ValueError("expected matching [batch, vocabulary>=2] logits")
    reference, candidate = reference.detach().float().cpu(), candidate.detach().float().cpu()
    rows = []
    for left, right in zip(reference, candidate):
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        if not finite:
            rows.append({"finite": False, "argmax_equal": False})
            continue
        ref_values, ref_ids = torch.topk(left, 2)
        cand_values, cand_ids = torch.topk(right, 2)
        # torch.argmax, not topk's unspecified tie order, matches generation.
        ref_id, cand_id = int(left.argmax()), int(right.argmax())
        maximum = float((left - right).abs().max())
        margin = float(ref_values[0] - ref_values[1])
        rows.append(dict(finite=True, argmax_equal=ref_id == cand_id, reference_top1=ref_id, candidate_top1=cand_id,
                         # The top logit's magnitude sets the bf16 spacing used to classify a margin as a tie.
                         reference_top1_logit=float(ref_values[0]), candidate_top1_logit=float(cand_values[0]),
                         reference_top1_margin=margin, candidate_top1_margin=float(cand_values[0] - cand_values[1]),
                         reference_tied_top1=bool(margin == 0), max_abs_logit_diff=maximum,
                         mean_abs_logit_diff=float((left - right).abs().mean()),
                         cosine_similarity=float(F.cosine_similarity(left[None], right[None]).item()),
                         candidate_token_rank_in_reference=int((left > left[cand_id]).sum()) + 1,
                         reference_gap_to_candidate_token=float(left[ref_id] - left[cand_id]),
                         guaranteed_stable_by_linf_bound=bool(margin > 2 * maximum)))
    return rows


class TeacherForcedModel:
    def __init__(self, model, reference, requests):
        self.model, self.reference = model, reference
        self.prompts = {r.rid: r.prompt_len for r in requests}
        self.tokens = {(int(r.rid), int(r.position)): int(r.token) for r in reference.tokens.itertuples()}
        self.rows, self.call = [], 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def decode(self, seq_ids, token_ids, pool, **kwargs):
        forced = [self.tokens[(rid, pool.length(rid) - self.prompts[rid])] for rid in seq_ids]
        output = self.model.decode(seq_ids, forced, pool, **kwargs)
        if self.call < len(self.reference.logits):
            step, expected_ids = self.reference.metadata["logit_step_keys"][self.call]
            if list(seq_ids) != expected_ids:
                raise RuntimeError("teacher-forced batch alignment changed")
            stats = logit_statistics(self.reference.logits[self.call], output)
            self.rows.extend({**row, "step": step, "rid": rid,
                              "position": pool.length(rid) - self.prompts[rid], "forced_input_token": token,
                              "num_splits": kwargs["num_splits"]} for row, rid, token in zip(stats, seq_ids, forced))
        self.call += 1
        return output


def prepare_requests(args, model):
    """Use the serving runner's tokenizer and provenance path, before any replay."""
    from kernelscope.serve.cli import _resolve_scenario
    from kernelscope.serve.scenarios import load_scenario
    requests, dataset = load_scenario(args.scenario)
    requests = [replace(request, max_new_tokens=min(request.max_new_tokens, args.steps + 1)) for request in requests]
    return _resolve_scenario(args, requests, dataset, model.cfg.vocab)


def main(argv=None):
    from kernelscope.serve.cli import _json, _policies, _sha256, _source_hashes, preflight
    from kernelscope.serve.engine import Engine
    from kernelscope.serve.kvcache import PagePool
    from kernelscope.serve.model import DecoderModel
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--scenario", default="scenarios/graduation_uniform.yaml")
    parser.add_argument("--reference", default="heuristic")
    parser.add_argument("--policy", action="append")
    parser.add_argument("--steps", type=int, default=16,
                        help="per-request decode cap; use 64 to preserve the full heldout arrival scenario")
    parser.add_argument("--logit-steps", type=int,
                        help="number of batch decode calls to compare (default: --steps); arrivals can need 55")
    parser.add_argument("--kv-gib", type=float, default=10)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--machine")
    parser.add_argument("--params")
    parser.add_argument("--cache-state", default="cold", choices=("cold", "warm"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.steps < 1 or not math.isfinite(args.kv_gib) or args.kv_gib <= 0 or args.seed < 0 or args.max_batch < 1:
        parser.error("steps, kv-gib, max-batch must be positive and seed nonnegative")
    if args.logit_steps is not None and args.logit_steps < 1:
        parser.error("--logit-steps must be positive")
    # Small per-request CPU reductions otherwise incur excessive thread-launch
    # overhead; this command makes no timing claim.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    logit_steps = args.steps if args.logit_steps is None else args.logit_steps
    out = Path(args.out)
    if out.exists():
        parser.error(f"output exists: {out}")
    status = preflight(args.model)
    if not status["ready"]:
        raise SystemExit("; ".join(status["errors"]))
    args.policy = [args.reference] + (args.policy or ["fixed:8"])
    policies = _policies(args)
    model = DecoderModel.from_pretrained(args.model)
    requests, prompt_metadata = prepare_requests(args, model)
    pool = PagePool.for_budget(model.cfg, int(args.kv_gib * 2**30), device=model.device, dtype=model.dtype)
    reference = Engine(model, pool, policies[0], args.max_batch).run(requests, model.cfg.vocab, logit_steps, args.seed)
    rows = []
    for policy in policies[1:]:
        wrapped = TeacherForcedModel(model, reference, requests)
        Engine(wrapped, pool, policy, args.max_batch).run(requests, model.cfg.vocab, seed=args.seed)
        rows.extend({**row, "policy": policy.name, "reference": policies[0].name} for row in wrapped.rows)
    out.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "teacher_forced_logits.csv", index=False)
    reference.tokens.to_parquet(out / "reference_tokens.parquet", index=False)
    _json(out / "manifest.json", {"schema_version": 1, "evidence_kind": "teacher_forced_numeric_diagnostic",
                                  "performance_claim": False, "model": args.model, "scenario": args.scenario,
                                  "scenario_sha256": _sha256(args.scenario), "requests": [asdict(r) for r in requests],
                                  **prompt_metadata, "model_dtype": str(model.dtype).removeprefix("torch."),
                                  "seed": args.seed, "steps": args.steps, "logit_steps": logit_steps,
                                  "decode_calls_recorded": len(reference.logits), "policy_specs": args.policy,
                                  "kv_gib": args.kv_gib, "max_batch": args.max_batch, "preflight": status,
                                  "source_sha256": {**_source_hashes(), "scripts/check_policy_numerics.py": _sha256(__file__)},
                                  "protocol": "Baseline greedy token histories fed to every policy; independently evolved KV caches",
                                  "claim": "Diagnostic only. Does not make differing free-generation outputs equivalent."})
    columns = [c for c in ("argmax_equal", "reference_tied_top1", "max_abs_logit_diff", "cosine_similarity") if c in frame]
    print(frame.groupby("policy")[columns].agg(["mean", "min", "max"]).to_string())
    print(f"Diagnostics: {out.resolve()}")
    if not frame.finite.all():
        raise SystemExit("Nonfinite logits observed; inspect diagnostics")


if __name__ == "__main__":
    main()
