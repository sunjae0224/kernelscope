"""Kernel-level probe for the `clear` divergence event (control_uniform_fixed8, request 8, position 41).

Replays the heuristic reference history of the uniform scenario (teacher forcing, all 32 requests) and, at the
decode step that produced the event, compares for every layer the flash-attn paged decode kernel with
num_splits=1 (what the heuristic resolves to here) and num_splits=8 against a float32 reference computed
from the same Q and the same KV cache contents. Also records the single-step logit difference between the
two splits on the identical KV state. Diagnostic only; no timing claim. Run from the repository root.
"""
import json, sys
from pathlib import Path

import pandas as pd
import torch

from kernelscope.serve.kvcache import PagePool
from kernelscope.serve.model import DecoderModel
from kernelscope.serve.scenarios import Request

CAMPAIGN = Path(sys.argv[1] if len(sys.argv) > 1 else "../kernelscope/results/serve_4090/hybrid_20261002")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else CAMPAIGN / "numerics" / "control_uniform_fixed8" / "kernel_probe")
TARGET_STEP, TARGET_RID, CANDIDATE_SPLITS = 40, 8, 8
run = CAMPAIGN / "control_uniform_fixed8"
prompts = [json.loads(line) for line in (run / "prompts.jsonl").read_text().splitlines()]
reference = pd.read_parquet(CAMPAIGN / "numerics" / "control_uniform_fixed8" / "reference_tokens.parquet")
tokens = {(int(r.rid), int(r.position)): int(r.token) for r in reference.itertuples()}

model = DecoderModel.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
cfg = model.cfg
pool = PagePool.for_budget(cfg, 10 * 2**30, device=model.device, dtype=model.dtype)
flash = model._flash_attention
records, state = [], {"probe": False, "layer": 0}


def fp32_reference(q, kcache, vcache, k_new, v_new, cache_lens, table, scale):
    """Exact attention in float32 over the cached keys plus the new key, per sequence (GQA)."""
    batch, _, hq, d = q.shape
    hk = kcache.shape[2]
    group = hq // hk
    outs = []
    for b in range(batch):
        n = int(cache_lens[b])
        pages = table[b, : (n + 256 - 1) // 256].long()
        k = kcache[pages].reshape(-1, hk, d)[:n].float()                 # [n, hk, d]
        v = vcache[pages].reshape(-1, hk, d)[:n].float()
        k = torch.cat([k, k_new[b].float()], dim=0)                        # the new token's key/value
        v = torch.cat([v, v_new[b].float()], dim=0)
        qb = q[b, 0].float()                                               # [hq, d]
        kk = k.repeat_interleave(group, dim=1)                             # [n+1, hq, d]
        vv = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,nhd->hn", qb, kk) * scale
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hn,nhd->hd", probs, vv))
    return torch.stack(outs).unsqueeze(1)                                  # [B, 1, hq, d] float32


def probing_attention(q, kcache, vcache, k=None, v=None, cache_seqlens=None, block_table=None, softmax_scale=None,
                      causal=True, num_splits=0):
    out = flash(q, kcache, vcache, k=k, v=v, cache_seqlens=cache_seqlens, block_table=block_table,
                softmax_scale=softmax_scale, causal=causal, num_splits=num_splits)
    if state["probe"]:
        layer = state["layer"]
        # The kernel has already written k/v into the cache at cache_seqlens; re-running with k=None reads them.
        lens = cache_seqlens + 1
        kw = dict(cache_seqlens=lens, block_table=block_table, softmax_scale=softmax_scale, causal=causal)
        one = flash(q, kcache, vcache, num_splits=1, **kw)
        eight = flash(q, kcache, vcache, num_splits=CANDIDATE_SPLITS, **kw)
        exact = fp32_reference(q, kcache, vcache, k, v, cache_seqlens, block_table, softmax_scale)
        for name, got in (("split1", one), ("split8", eight), ("kernel_as_called", out)):
            err = (got.float() - exact).abs()
            per_seq = err.amax(dim=(1, 2, 3))
            records.append(dict(layer=layer, variant=name, num_splits_called=num_splits,
                                max_abs_err=float(err.max()), mean_abs_err=float(err.mean()),
                                max_abs_err_rid8=float(per_seq[TARGET_RID]), max_abs_out=float(exact.abs().max()),
                                max_abs_out_rid8=float(exact[TARGET_RID].abs().max()),
                                split1_vs_split8_max=float((one.float() - eight.float()).abs().max()),
                                split1_vs_split8_max_rid8=float((one[TARGET_RID].float() - eight[TARGET_RID].float()).abs().max()),
                                exact_equal_split1=bool(torch.equal(one, out)) if num_splits in (0, 1) else None))
        state["layer"] = layer + 1
    return out


model._flash_attention = probing_attention
seq_ids = [p["rid"] for p in prompts]
for p in prompts:
    model.prefill(p["rid"], p["token_ids"], pool)
assert all(tokens[(rid, 0)] == tokens[(rid, 0)] for rid in seq_ids)
logit_rows = []
for step in range(TARGET_STEP + 1):
    forced = [tokens[(rid, step)] for rid in seq_ids]       # the token generated at position `step` is this step's input
    if step == TARGET_STEP:
        lengths = [pool.length(rid) for rid in seq_ids]
        # single-step comparison on the identical KV state: candidate split first, then roll back
        cand = model.decode(seq_ids, forced, pool, num_splits=CANDIDATE_SPLITS)
        for rid, n in zip(seq_ids, lengths):
            pool.set_length(rid, n)
        state["probe"], state["layer"] = True, 0
        ref = model.decode(seq_ids, forced, pool, num_splits=0)
        state["probe"] = False
        for i, rid in enumerate(seq_ids):
            a, b = ref[i], cand[i]
            logit_rows.append(dict(rid=rid, position=step + 1, ref_top1=int(a.argmax()), cand_top1=int(b.argmax()),
                                   ref_top1_logit=float(a.max()), cand_top1_logit=float(b.max()),
                                   ref_margin=float(torch.topk(a, 2).values.diff().abs()),
                                   max_abs_logit_diff=float((a - b).abs().max()),
                                   cosine=float(torch.nn.functional.cosine_similarity(a[None], b[None]))))
        break
    model.decode(seq_ids, forced, pool, num_splits=0)

OUT.mkdir(parents=True, exist_ok=True)
layers = pd.DataFrame(records)
logits = pd.DataFrame(logit_rows)
layers.to_csv(OUT / "attention_vs_fp32.csv", index=False)
logits.to_csv(OUT / "single_step_logits.csv", index=False)
json.dump({"campaign": str(CAMPAIGN), "run": "control_uniform_fixed8", "target_step": TARGET_STEP, "target_rid": TARGET_RID,
           "candidate_splits": CANDIDATE_SPLITS, "model": "Qwen/Qwen3-4B-Instruct-2507", "dtype": str(model.dtype),
           "flash_attn": __import__("flash_attn").__version__, "torch": torch.__version__,
           "claim": "Diagnostic only: per-layer kernel outputs vs a float32 reference on identical inputs, and the single-step logit difference of the two splits on the identical KV state."},
          open(OUT / "manifest.json", "w"), indent=2)
pd.set_option("display.width", 220)
print(layers.groupby("variant")[["max_abs_err", "mean_abs_err", "max_abs_err_rid8", "max_abs_out", "split1_vs_split8_max", "split1_vs_split8_max_rid8"]].max().to_string())
print(layers[layers.variant == "split8"][["layer", "max_abs_err", "max_abs_err_rid8", "max_abs_out_rid8", "split1_vs_split8_max_rid8"]].to_string(index=False))
print(logits[logits.rid == TARGET_RID].to_string(index=False))
print("single-step: rows with argmax change:", int((logits.ref_top1 != logits.cand_top1).sum()), "max |dlogit| over batch:", float(logits.max_abs_logit_diff.max()))
print("Artifacts:", OUT.resolve())
