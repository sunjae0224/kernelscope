"""Per-step decomposition of the vLLM 0.31 decode profiles of 2026-10-25 (torch-free; reads ``step_breakdown.json``).

``scripts/vllm_profile_step.py analyze`` reports the median of every column over all pure-decode iterations. In these traces the
iterations alternate between a forward pass (about 500-620 kernels) and a 12-kernel sampling annotation, and in the uniform eager
runs one forward pass is split across two annotations, so those medians mix different kinds of iteration (the logged "ragged wall 39 ms,
attention 1.4 ms" is such a mix) and must not be used. Here a decode step is the sum over the iterations of a batch of 32 and its
kernel total divided by the kernels of one step (forward pass + sampling, from the same backend and mode's ragged run, where the
alternation is clean). Times are GPU kernel times per step; the host share is the unprofiled step time minus the GPU busy time.
"""
import json
import statistics
from pathlib import Path

CATEGORIES = ("attention", "kv_write", "gemm", "other")
BATCH = 32


def _decode(run):
    return [x for x in run["iterations"] if x["context_requests"] == 0 and x["generation_requests"] == BATCH]


def kernels_per_step(ragged_run) -> float:
    rows = _decode(ragged_run)
    forward = [x["kernels"] for x in rows if x["kernels"] > 100]
    sampling = [x["kernels"] for x in rows if 0 < x["kernels"] <= 100]
    return statistics.median(forward) + statistics.median(sampling)


def run_table(runs: list) -> dict:
    """{run name: per-step milliseconds and counts}."""
    ragged = {(r["backend"], r["mode"]): r for r in runs if r["scenario"] == "ragged"}
    out = {}
    for r in runs:
        rows = _decode(r)
        k = kernels_per_step(ragged[(r["backend"], r["mode"])])
        steps = sum(x["kernels"] for x in rows) / k
        busy = sum(x["gpu_busy_us"] for x in rows) / steps / 1000
        entry = {"backend": r["backend"], "mode": r["mode"], "scenario": r["scenario"], "steps": steps,
                 "unprofiled_ms": r["unprofiled_decode_ms_per_step_median"], "gpu_busy_ms": busy}
        for c in CATEGORIES:
            entry[f"{c}_ms"] = sum(x[f"{c}_us"] for x in rows) / steps / 1000
        entry["not_gpu_busy_ms"] = entry["unprofiled_ms"] - busy
        entry["attention_share_of_busy"] = entry["attention_ms"] / busy
        out[r["run"]] = entry
    return out


def load(root) -> list:
    return json.loads((Path(root) / "step_breakdown.json").read_text())["runs"]


def last_row_after_long_request_leaves(run) -> dict:
    """The decode iteration of 31 requests after the 32K request finished (ragged runs): the same engine, no long request."""
    rows = [x for x in run["iterations"] if x["context_requests"] == 0 and x["generation_requests"] == BATCH - 1 and x["kernels"] > 100]
    return rows[0] if rows else {}
