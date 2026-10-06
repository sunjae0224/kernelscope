"""Synthetic serving recordings for dashboard tests (same files `serve run` writes, tiny numbers)."""
import json
from pathlib import Path

import pandas as pd


def make_run(root, campaign="camp", scenario="ragged", policies=("heuristic", "table"), repeats=2, B=3,
             new_tokens=4, diverge=None, natural=False, created="2026-10-01T00:00:00+00:00",
             scenario_family=None, flat=False, arrivals=False, model="test/model"):
    """Write <root>/serve_4090/<campaign>/<scenario>/ with per-policy repeat folders and a summary.

    ``diverge=(policy, rid, position)`` makes that policy emit different tokens from ``position`` on.
    ``flat`` puts steps/tokens/prefill directly under the policy folder (no repeat_* level).
    ``arrivals`` admits requests 1.. one decode step apart instead of all at step 0."""
    run = Path(root) / "serve_4090" / campaign / scenario
    lens = [4096] + [128] * (B - 1)
    summary = []
    for policy in policies:
        speed = 1.0 if policy == "heuristic" else 0.5
        for r in range(repeats if not flat else 1):
            d = run / policy if flat else run / policy / f"repeat_{r:03d}"
            d.mkdir(parents=True)
            first = [500.0 + 50.0 * i for i in range(B)]
            arrival = [0.0] * B
            if arrivals:  # request i arrives i steps late and is admitted right away
                arrival = [0.0] + [first[-1] + i * 1e6 * speed for i in range(1, B)]
                first = [first[0]] + [a + 50.0 for a in arrival[1:]]
            prefill = pd.DataFrame({"rid": range(B), "prompt_len": lens, "prefill_us": [500.0] + [50.0] * (B - 1),
                                    "arrival_us": arrival, "admitted_us": arrival, "first_token_us": first,
                                    "prompt_sha256": "x"})
            tokens = [dict(rid=i, step=0 if not arrivals else i, position=0, token=100 + i, t_us=first[i], phase="prefill")
                      for i in range(B)]
            steps = []
            t0 = first[0] if arrivals else first[-1]
            for s in range(new_tokens - 1):
                t = t0 + (s + 1) * 1e6 * speed * (1 + 0.1 * r)
                for i in range(B):
                    if arrivals and s < i:
                        continue
                    token = 200 + 10 * s + i
                    if diverge and policy == diverge[0] and i == diverge[1] and s + 1 >= diverge[2]:
                        token += 1000
                    tokens.append(dict(rid=i, step=s, position=s + 1 - (i if arrivals else 0), token=token, t_us=t, phase="decode"))
                steps.append(dict(step=s, policy=policy, B=B, len_max=lens[0] + s, len_sum=sum(lens) + B * s, n_long=1,
                                  num_splits=0 if policy == "heuristic" else 16, attn_us=800.0 * speed,
                                  step_us=950.0 * speed, policy_us=3.0, policy_cache_hit=True, decode_wall_us=960.0 * speed,
                                  seq_ids=json.dumps(list(range(B))), lens=json.dumps([n + s + 1 for n in lens])))
            prefill.to_parquet(d / "prefill.parquet", index=False)
            pd.DataFrame(tokens).to_parquet(d / "tokens.parquet", index=False)
            pd.DataFrame(steps).to_parquet(d / "steps.parquet", index=False)
        summary.append(dict(policy=policy, repeats=repeats, tpot_ms_mean=1.0 * speed, speedup_vs_heuristic=1.0 / speed,
                            tokens_equivalent=not (diverge and diverge[0] == policy)))
    pd.DataFrame(summary).to_csv(run / "summary.csv", index=False)
    manifest = {"model": model, "scenario": f"/x/scenarios/{scenario}.yaml", "created_at": created, "repeats": repeats,
                "prompt_kind": "local_text" if natural else "seeded_synthetic_token_ids",
                "requests": [{"prompt_len": n, "max_new_tokens": new_tokens} for n in lens]}
    if scenario_family:
        manifest["scenario_family"] = scenario_family
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run
