#!/usr/bin/env python3
"""Reproduce local BF16 decoder validation against Hugging Face on a CUDA GPU.

Run from the repository root:
  .venv/bin/python scripts/verify_qwen.py --out docs/experiments/qwen3-4b-validation.json

The oracle and KernelScope model are loaded sequentially to bound GPU memory.
This is a correctness check, not a performance benchmark; no model is downloaded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kernelscope.serve.hf import snapshot_dir
from kernelscope.serve.kvcache import PagePool
from kernelscope.serve.model import DecoderModel


def verify(model_id="Qwen/Qwen3-4B-Instruct-2507", chunk=13, splits=8):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; real-model GPU validation cannot be substituted with CPU demo data")
    inventory = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if torch.cuda.mem_get_info()[0] < 18 * 2**30:
        raise RuntimeError("at least 18 GiB free GPU memory is required; leave other workloads running")
    from transformers import AutoModelForCausalLM

    path = snapshot_dir(model_id)
    prompt, continuation = list(range(100, 132)), [800, 801]
    print("Loading the local Hugging Face oracle...", flush=True)
    oracle = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False,
    ).cuda().eval()
    expected = []
    with torch.inference_mode():
        for n in range(len(continuation) + 1):
            ids = torch.tensor([prompt + continuation[:n]], device="cuda")
            expected.append(oracle(ids, use_cache=False).logits[0, -1].float().cpu())
    del oracle
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading the local KernelScope decoder...", flush=True)
    model = DecoderModel.from_pretrained(str(path))
    cfg = model.cfg
    pool = PagePool(cfg.n_layers, 2, cfg.n_kv_heads, cfg.head_dim,
                    dtype=model.dtype, device=model.device)
    actual = [model.prefill("validation", prompt, pool, chunk=chunk).cpu()]
    for token in continuation:
        actual.append(model.decode(["validation"], [token], pool, num_splits=splits)[0].cpu())
    del model, pool
    gc.collect()
    torch.cuda.empty_cache()

    comparisons = []
    for n, (got, want) in enumerate(zip(actual, expected)):
        difference = (got - want).abs()
        comparisons.append({
            "stage": "prefill" if n == 0 else f"decode_{n}",
            "context_tokens": len(prompt) + n,
            "cosine_similarity": torch.nn.functional.cosine_similarity(got, want, dim=0).item(),
            "max_abs_logit_diff": difference.max().item(),
            "mean_abs_logit_diff": difference.mean().item(),
            "reference_top1": want.argmax().item(),
            "kernelscope_top1": got.argmax().item(),
            "top1_matches": bool(got.argmax() == want.argmax()),
            "finite_logits": bool(torch.isfinite(got).all()),
        })
    # Chosen before measurement, following the implementation plan. Max absolute
    # differences are reported, not retrofitted into a passing threshold.
    tolerances = {"cosine_similarity_min": 0.999, "require_top1_match": True,
                  "require_finite_logits": True, "bit_identity_required": False}
    passed = all(row["cosine_similarity"] > tolerances["cosine_similarity_min"]
                 and row["top1_matches"] and row["finite_logits"] for row in comparisons)
    hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
              for name in ("config.json", "model.safetensors.index.json") if (path / name).is_file()}
    return {
        "format_version": 1, "kind": "hf_correctness_validation", "evidence": "measured_gpu",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(), "model": model_id,
        "snapshot_revision": path.name, "local_file_sha256": hashes,
        "dtype": "bfloat16", "gpu": torch.cuda.get_device_name(),
        "versions": {name: version(name) for name in ("torch", "transformers", "flash-attn", "safetensors")},
        "gpu_processes_before": inventory,
        "oracle": {"implementation": "Hugging Face SDPA", "use_cache": False,
                   "local_files_only": True, "trust_remote_code": False},
        "kernelscope": {"implementation": "flash_attn_with_kvcache", "page_tokens": 256,
                        "prefill_chunk": chunk, "decode_num_splits": splits},
        "inputs": {"prompt_token_ids": prompt, "continuation_token_ids": continuation},
        "tolerances": tolerances, "comparisons": comparisons, "passed": passed,
        "scope": "One local checkpoint, one synthetic token prompt, chunked prefill and two teacher-forced decode steps. This is not broad generation equivalence or serving speed evidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/qwen3-4b-validation.json"))
    parser.add_argument("--chunk", type=int, default=13)
    parser.add_argument("--splits", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    report = verify(args.model, chunk=args.chunk, splits=args.splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "comparisons": report["comparisons"], "out": str(args.out)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
