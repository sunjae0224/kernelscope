"""Kernel-level probe for one divergence event of a teacher-forced reference run.

Replays the heuristic reference token history of a scenario (teacher forcing, every request of the run) and, at the
decode step that produced the event, compares for every layer the flash-attn paged decode kernel with
num_splits=1 (what the heuristic resolves to in the original event) and a candidate against a float32
reference computed from the same Q and the same KV cache contents. The candidate is flash-attn with
num_splits=<candidate> (`--candidate-backend fa2`, the default) or a FlashInfer decode backend (`flashinfer`,
`flashinfer_cudacore`), planned once per step for the pre-append lengths exactly as the engine does. Also records the
single-step logit difference between the default (num_splits=0) decode and the candidate on the identical KV state.
Diagnostic only; no timing claim. Run from the repository root.

With no arguments it reproduces the original probe: the `clear` event of control_uniform_fixed8 (request 8,
position 41, decode step 40, candidate split 8). The 64K event is, for example:

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m scripts.probe_split_kernel \\
        --campaign ../kernelscope/results/serve_4090/longctx_20261006 --run ragged_64k \\
        --target-step 26 --target-rid 0 --candidate-splits 8 --kv-gib 13

and a `clear` event of a FlashInfer serving policy:

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m scripts.probe_split_kernel \\
        --campaign ../kernelscope/results/serve_4090/flashinfer_20261006 --run ragged \\
        --target-step 40 --target-rid 8 --candidate-backend flashinfer_cudacore --kv-gib 10

Layout read from the campaign: `<campaign>/<run>/prompts.jsonl` and
`<campaign>/numerics/<run>/reference_tokens.parquet`. The token generated at position `step` is the input of decode
step `step`; at `step == target_step` the logits of position `step + 1` are computed.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from kernelscope.serve.kvcache import PAGE, PagePool
from kernelscope.serve.model import DecoderModel

DEFAULT_CAMPAIGN = "../kernelscope/results/serve_4090/hybrid_20261002"
BACKENDS = ("fa2", "flashinfer", "flashinfer_cudacore")
CLAIM = ("Diagnostic only: per-layer kernel outputs vs a float32 reference on identical inputs, and the "
         "single-step logit difference of the default decode and the candidate on the identical KV state.")


@dataclass
class BackendCandidate:
    """A FlashInfer decode backend as the candidate: `backend` has plan(...) and __call__(...) like FlashInferDecode."""
    name: str
    backend: object


def candidate_label(candidate):
    """Variant label of the candidate in the records: `split<N>` for a split count, the backend name otherwise."""
    return candidate.name if isinstance(candidate, BackendCandidate) else f"split{candidate}"


def fp32_reference(q, kcache, vcache, k_new, v_new, cache_lens, table, scale):
    """Exact attention in float32 over the cached keys plus the new key, per sequence (GQA).

    The query is grouped as [hk, group, d] (query head = kv_head * group + g, the order repeat_interleave
    along the head dim would give) so K/V are never expanded to the query-head count.
    """
    batch, _, hq, d = q.shape
    hk = kcache.shape[2]
    group = hq // hk
    outs = []
    for b in range(batch):
        n = int(cache_lens[b])
        pages = table[b, : (n + PAGE - 1) // PAGE].long()
        k = kcache[pages].reshape(-1, hk, d)[:n].float()                  # [n, hk, d]
        v = vcache[pages].reshape(-1, hk, d)[:n].float()
        k = torch.cat([k, k_new[b].float()], dim=0)                        # the new token's key/value
        v = torch.cat([v, v_new[b].float()], dim=0)
        qb = q[b, 0].float().reshape(hk, group, d)                         # [hk, group, d]
        scores = torch.einsum("hgd,nhd->hgn", qb, k) * scale
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hgn,nhd->hgd", probs, v).reshape(hq, d))
    return torch.stack(outs).unsqueeze(1)                                  # [B, 1, hq, d] float32


def make_probing_attention(flash, candidate, target_index, records, state):
    """Wrap `flash` so that, while state["probe"] is set, every call also appends per-layer error records.

    `candidate` is a flash-attn split count (int) or a BackendCandidate. A backend candidate is planned on the first
    probed layer (state["layer"] == 0) with the pre-append lengths of that call, as DecoderModel.decode does, and then
    called on every probed layer; it writes the same k/v the flash-attn call has just stored.
    """
    label = candidate_label(candidate)

    def probing_attention(q, kcache, vcache, k=None, v=None, cache_seqlens=None, block_table=None,
                          softmax_scale=None, causal=True, num_splits=0):
        out = flash(q, kcache, vcache, k=k, v=v, cache_seqlens=cache_seqlens, block_table=block_table,
                    softmax_scale=softmax_scale, causal=causal, num_splits=num_splits)
        if state["probe"]:
            layer = state["layer"]
            # The kernel has already written k/v into the cache at cache_seqlens; re-running with k=None reads them.
            lens = cache_seqlens + 1
            kw = dict(cache_seqlens=lens, block_table=block_table, softmax_scale=softmax_scale, causal=causal)
            one = flash(q, kcache, vcache, num_splits=1, **kw)
            if isinstance(candidate, BackendCandidate):
                if layer == 0:                                  # once per step, for the lengths before the append
                    candidate.backend.plan(cache_seqlens, block_table, q.shape[2], kcache.shape[2], q.shape[3], q.dtype)
                cand = candidate.backend(q, kcache, vcache, k=k, v=v, cache_seqlens=cache_seqlens,
                                         block_table=block_table, softmax_scale=softmax_scale, causal=causal,
                                         num_splits=0)
            else:
                cand = flash(q, kcache, vcache, num_splits=candidate, **kw)
            exact = fp32_reference(q, kcache, vcache, k, v, cache_seqlens, block_table, softmax_scale)
            diff = (one.float() - cand.float()).abs()
            for name, got in (("split1", one), (label, cand), ("kernel_as_called", out)):
                err = (got.float() - exact).abs()
                per_seq = err.amax(dim=(1, 2, 3))
                records.append(dict(layer=layer, variant=name, num_splits_called=num_splits,
                                    max_abs_err=float(err.max()), mean_abs_err=float(err.mean()),
                                    max_abs_err_target=float(per_seq[target_index]), max_abs_out=float(exact.abs().max()),
                                    max_abs_out_target=float(exact[target_index].abs().max()),
                                    split1_vs_candidate_max=float(diff.max()),
                                    split1_vs_candidate_max_target=float(diff[target_index].max()),
                                    exact_equal_split1=bool(torch.equal(one, out)) if num_splits in (0, 1) else None))
            state["layer"] = layer + 1
        return out
    return probing_attention


def check_reference(tokens, seq_ids, target_step):
    """The teacher-forced replay needs a reference token for every request at every position 0..target_step."""
    missing = [(rid, pos) for rid in seq_ids for pos in range(target_step + 1) if (rid, pos) not in tokens]
    if missing:
        raise ValueError(f"reference_tokens has no token for {len(missing)} (rid, position) pairs up to position "
                         f"{target_step}, e.g. {missing[:5]}; the reference run is too short for --target-step "
                         f"{target_step} or does not cover every request")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--campaign", type=Path, default=Path(DEFAULT_CAMPAIGN),
                   help=f"campaign directory [{DEFAULT_CAMPAIGN}]")
    p.add_argument("--run", default="control_uniform_fixed8",
                   help="run name: <campaign>/<run>/prompts.jsonl and "
                        "<campaign>/numerics/<run>/reference_tokens.parquet [control_uniform_fixed8]")
    p.add_argument("--target-step", type=int, default=40, help="decode step of the event (logits of position step+1) [40]")
    p.add_argument("--target-rid", type=int, default=8, help="request id of the event [8]")
    p.add_argument("--candidate-backend", choices=BACKENDS, default="fa2",
                   help="candidate attention: fa2 = flash-attn with --candidate-splits, flashinfer / flashinfer_cudacore "
                        "= that FlashInfer decode backend as the serving policy of the same name runs it [fa2]")
    p.add_argument("--candidate-splits", type=int, default=8,
                   help="num_splits compared against num_splits=1, >= 2; only for --candidate-backend fa2, ignored "
                        "(recorded as null) with a FlashInfer backend [8]")
    p.add_argument("--kv-gib", type=float, default=10, help="KV page pool budget in GiB [10]")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="model id or path [Qwen/Qwen3-4B-Instruct-2507]")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory [<campaign>/numerics/<run>/kernel_probe, or kernel_probe_<backend> "
                        "with a FlashInfer backend]")
    args = p.parse_args(argv)
    if args.target_step < 0:
        p.error("--target-step must be >= 0")
    if args.candidate_backend == "fa2":
        if args.candidate_splits < 2:
            p.error("--candidate-splits must be >= 2 (split1 is always the other variant)")
    else:
        args.candidate_splits = None
    if args.kv_gib <= 0:
        p.error("--kv-gib must be positive")
    if args.out is None:
        suffix = "kernel_probe" if args.candidate_backend == "fa2" else f"kernel_probe_{args.candidate_backend}"
        args.out = args.campaign / "numerics" / args.run / suffix
    return args


def decode_kwargs(args):
    """Arguments of DecoderModel.decode that select the candidate for the single-step logit comparison."""
    if args.candidate_backend == "fa2":
        return dict(num_splits=args.candidate_splits)
    return dict(num_splits=0, attention=args.candidate_backend)


def main(argv=None):
    args = parse_args(argv)
    run = args.campaign / args.run
    prompts = [json.loads(line) for line in (run / "prompts.jsonl").read_text().splitlines()]
    reference = pd.read_parquet(args.campaign / "numerics" / args.run / "reference_tokens.parquet")
    tokens = {(int(r.rid), int(r.position)): int(r.token) for r in reference.itertuples()}
    seq_ids = [p["rid"] for p in prompts]
    if args.target_rid not in seq_ids:
        raise ValueError(f"--target-rid {args.target_rid} is not a request of {run / 'prompts.jsonl'}")
    target_index = seq_ids.index(args.target_rid)     # batch row of the target; rid is not necessarily the row
    check_reference(tokens, seq_ids, args.target_step)

    model = DecoderModel.from_pretrained(args.model)
    cfg = model.cfg
    pool = PagePool.for_budget(cfg, int(args.kv_gib * 2**30), device=model.device, dtype=model.dtype)
    records, state = [], {"probe": False, "layer": 0}
    if args.candidate_backend == "fa2":
        candidate = args.candidate_splits
    else:   # built before the probing wrapper replaces model._flash_attention, so its fallback is the plain kernel
        candidate = BackendCandidate(args.candidate_backend, model.attention_backend(args.candidate_backend))
    model._flash_attention = make_probing_attention(model._flash_attention, candidate, target_index, records, state)
    for p in prompts:
        model.prefill(p["rid"], p["token_ids"], pool)
    logit_rows = []
    for step in range(args.target_step + 1):
        forced = [tokens[(rid, step)] for rid in seq_ids]   # the token generated at position `step` is this step's input
        if step == args.target_step:
            lengths = [pool.length(rid) for rid in seq_ids]
            # single-step comparison on the identical KV state: the candidate first, then roll back
            cand = model.decode(seq_ids, forced, pool, **decode_kwargs(args))
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

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    layers = pd.DataFrame(records)
    logits = pd.DataFrame(logit_rows)
    layers.to_csv(out / "attention_vs_fp32.csv", index=False)
    logits.to_csv(out / "single_step_logits.csv", index=False)
    manifest = {"campaign": str(args.campaign), "run": args.run, "target_step": args.target_step,
                "target_rid": args.target_rid, "candidate_backend": args.candidate_backend,
                "candidate_splits": args.candidate_splits, "kv_gib": args.kv_gib,
                "model": args.model, "out": str(out), "dtype": str(model.dtype),
                "flash_attn": __import__("flash_attn").__version__, "torch": torch.__version__, "claim": CLAIM}
    if args.candidate_backend != "fa2":
        manifest["flashinfer"] = getattr(__import__("flashinfer"), "__version__", None)
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    cand_name = candidate_label(candidate)
    pd.set_option("display.width", 220)
    print(layers.groupby("variant")[["max_abs_err", "mean_abs_err", "max_abs_err_target", "max_abs_out",
                                     "split1_vs_candidate_max", "split1_vs_candidate_max_target"]].max().to_string())
    print(layers[layers.variant == cand_name][["layer", "max_abs_err", "max_abs_err_target", "max_abs_out_target",
                                               "split1_vs_candidate_max_target"]].to_string(index=False))
    print(logits[logits.rid == args.target_rid].to_string(index=False))
    print("single-step: rows with argmax change:", int((logits.ref_top1 != logits.cand_top1).sum()),
          "max |dlogit| over batch:", float(logits.max_abs_logit_diff.max()))
    print("Artifacts:", out.resolve())


if __name__ == "__main__":
    main()
