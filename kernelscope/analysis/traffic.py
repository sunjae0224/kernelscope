"""How often real request traffic produces the decode batches the split heuristic mishandles.

The kernel grids show the loss of flash-attn's ``num_splits`` heuristic for a batch *shape*; this
module asks how often a serving engine sees such shapes. A public request trace (arrival time,
prompt tokens, output tokens) is replayed through a continuous-batching scheduler at decode-step
granularity, with the engine's admission rule (FIFO, batch limit, page reservations for a request's
whole life). Every decode step's KV lengths are then scored with the fitted surrogate model: the
heuristic's predicted time against the best split variant, i.e. the loss a per-step dispatcher
would recover. Nothing here needs torch or a GPU.

Simplifications, deliberately kept in the open: prefill is instantaneous (a request joins the next
step after it arrives), every decode step takes ``step_s`` regardless of its batch, and arrivals
can be compressed with ``rate_scale`` to reach a load one GPU would actually see.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from kernelscope.model.geometry import parse_variant, resolve_splits
from kernelscope.model.predict import PAGED_VARIANTS, predict
from kernelscope.workload import Workload

TRACE_COLUMNS = ["arrival_s", "prompt_tokens", "output_tokens"]
STEP_COLUMNS = ["step", "t_s", "B", "len_max", "len_mean", "len_sum", "n_long", "lens"]
LOSS_COLUMNS = ["step", "t_s", "B", "len_max", "heuristic_splits", "heuristic_us", "best_variant", "best_us",
                "loss", "cache_hit"]
LONG = 4096
HEURISTIC = "flashdecoding_paged"
# The ragged grids never chose more than 16 splits; the larger counts only add simulation time.
CANDIDATES = (HEURISTIC, "fd_s2_paged", "fd_s4_paged", "fd_s8_paged", "fd_s16_paged")
_AZURE = {"TIMESTAMP": "arrival_s", "ContextTokens": "prompt_tokens", "GeneratedTokens": "output_tokens"}
_BURST = {"Timestamp": "arrival_s", "Request tokens": "prompt_tokens", "Response tokens": "output_tokens"}


def load_trace(path, model=None) -> pd.DataFrame:
    """Azure LLM inference trace or BurstGPT rows as arrival_s (from the first row), prompt and output tokens.

    Rows without a positive prompt and output length are dropped and counted in ``attrs["dropped"]``;
    ``model`` keeps one BurstGPT model's rows.
    """
    raw = pd.read_csv(path)
    if set(_AZURE) <= set(raw.columns):
        kind, columns = "azure", _AZURE
        stamps = pd.to_datetime(raw["TIMESTAMP"])
        raw["TIMESTAMP"] = (stamps - stamps.min()).dt.total_seconds()
    elif set(_BURST) <= set(raw.columns):
        kind, columns = "burstgpt", _BURST
        if model is not None:
            raw = raw[raw["Model"] == model]
    else:
        raise ValueError(f"{path}: expected Azure (TIMESTAMP, ContextTokens, GeneratedTokens) or BurstGPT columns")
    frame = raw.rename(columns=columns)[TRACE_COLUMNS].copy()
    for column in TRACE_COLUMNS:                    # malformed cells become NaN and are dropped below, not raised
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = (frame.prompt_tokens >= 1) & (frame.output_tokens >= 1) & np.isfinite(frame.arrival_s.astype(float))
    frame = frame[valid].astype({"prompt_tokens": "int64", "output_tokens": "int64"})
    frame = frame.sort_values("arrival_s", kind="stable").reset_index(drop=True)
    frame["arrival_s"] = frame.arrival_s.astype(float) - (float(frame.arrival_s.iloc[0]) if len(frame) else 0.0)
    frame.attrs.update({"kind": kind, "dropped": int((~valid).sum()), "source": str(path)})
    return frame


@dataclass(frozen=True)
class ReplayConfig:
    max_batch: int = 64
    kv_tokens: int = 72_817          # 10 GiB of Qwen3-4B bf16 KV (36 layers, 8 heads, d=128)
    step_s: float = 0.03
    rate_scale: float = 1.0          # arrivals at arrival_s / rate_scale
    page: int = 256

    def __post_init__(self):
        if self.max_batch < 1 or self.kv_tokens < self.page or self.step_s <= 0 or self.rate_scale <= 0 or self.page < 1:
            raise ValueError("replay needs a positive batch limit, at least one page of KV, and positive step and rate scale")


def replay(trace: pd.DataFrame, cfg: ReplayConfig) -> pd.DataFrame:
    """One row per decode step: the live KV lengths of the batch, in admission order.

    A request reserves ceil((prompt + output - 1) / page) pages when admitted (the last token is never
    written to the cache) and keeps them until its last token, as the engine does; its first token
    comes from prefill, so it decodes output - 1 times with KV lengths prompt + 1, prompt + 2, ... A
    single-token request therefore never reaches a decode step. Requests the pool can never hold are
    skipped and counted in ``attrs["too_large"]``. When nothing is active the clock jumps to the next
    arrival.
    """
    if not np.isfinite(trace.arrival_s.astype(float)).all():
        raise ValueError("every arrival_s must be finite (load_trace drops rows without a usable timestamp)")
    capacity = cfg.kv_tokens // cfg.page
    pending = []
    too_large = 0
    for row in trace.sort_values("arrival_s", kind="stable").itertuples(index=False):
        pages = max(1, math.ceil((int(row.prompt_tokens) + int(row.output_tokens) - 1) / cfg.page))
        if pages > capacity:
            too_large += 1
            continue
        pending.append((float(row.arrival_s) / cfg.rate_scale, int(row.prompt_tokens), int(row.output_tokens), pages))
    pending.reverse()                                   # pop() from the end is the earliest arrival
    active, reserved, rows, t, step = [], 0, [], 0.0, 0
    while pending or active:
        if not active and pending and pending[-1][0] > t:
            t = pending[-1][0]
        while pending and pending[-1][0] <= t and len(active) < cfg.max_batch and reserved + pending[-1][3] <= capacity:
            _, prompt, output, pages = pending.pop()
            if output < 2:                                        # prefill only: its pages come and go within the step
                continue
            reserved += pages
            active.append([prompt + 1, output - 1, pages])        # [live KV length at the next step, decode steps left, pages]
        if active:
            lens = [a[0] for a in active]
            rows.append({"step": step, "t_s": t, "B": len(lens), "len_max": max(lens), "len_mean": sum(lens) / len(lens),
                         "len_sum": sum(lens), "n_long": sum(n >= LONG for n in lens), "lens": json.dumps(lens)})
            for a in active:
                a[0] += 1
                a[1] -= 1
            finished = [a for a in active if a[1] <= 0]
            reserved -= sum(a[2] for a in finished)
            active = [a for a in active if a[1] > 0]
            step += 1
            t += cfg.step_s
    frame = pd.DataFrame(rows, columns=STEP_COLUMNS)
    frame.attrs.update({"requests": len(trace) - too_large, "too_large": too_large, "config": cfg.__dict__.copy()})
    return frame


def heuristic_splits(lens, n_heads: int, n_kv_heads: int, n_sm: int) -> int:
    """The split count flash-attn's heuristic picks for this batch (1 = it returned without splitting)."""
    return resolve_splits(parse_variant(HEURISTIC), _workload(lens, n_heads, n_kv_heads), n_sm)


def _workload(lens, n_heads, n_kv_heads):
    return Workload(phase="decode", B=len(lens), L_q=1, L_kv=max(lens), H_q=n_heads, H_kv=n_kv_heads, d=128,
                    dtype="bfloat16", kv_lens=tuple(lens))


def step_losses(steps: pd.DataFrame, machine, params, cache_state="cold", n_heads=32, n_kv_heads=8,
                candidates=CANDIDATES, every=1, page=256) -> pd.DataFrame:
    """Predicted heuristic time against the best candidate for every ``every``-th step.

    Predictions are cached per page-quantized length configuration (the policy cache's key), so
    consecutive steps of the same batch cost one prediction; ``attrs["predictions"]`` counts them.
    """
    if every < 1:
        raise ValueError("every must be a positive integer")
    cache, rows = {}, []
    for row in steps.iloc[::every].itertuples(index=False):
        lens = json.loads(row.lens)
        key = tuple(math.ceil(n / page) for n in lens)
        hit = key in cache
        if not hit:
            w = _workload(lens, n_heads, n_kv_heads)
            times = {c: float(predict(c, w, machine, params, cache_state).time_us) for c in candidates}
            if any(not math.isfinite(t) or t <= 0 for t in times.values()):
                raise ValueError(f"surrogate produced a nonpositive or nonfinite prediction for {row.lens}")
            best = min(times, key=times.get)
            cache[key] = (resolve_splits(parse_variant(HEURISTIC), w, machine.n_sm), times[HEURISTIC], best, times[best])
        splits, h_us, best, best_us = cache[key]
        rows.append({"step": int(row.step), "t_s": float(row.t_s), "B": len(lens), "len_max": max(lens),
                     "heuristic_splits": splits, "heuristic_us": h_us, "best_variant": best, "best_us": best_us,
                     "loss": h_us / best_us, "cache_hit": hit})
    frame = pd.DataFrame(rows, columns=LOSS_COLUMNS)
    frame.attrs.update({"predictions": len(cache), "every": every, "candidates": list(candidates), "cache_state": cache_state})
    return frame


def summarize_losses(losses: pd.DataFrame, thresholds=(1.1, 1.25, 1.5, 2.0)) -> dict:
    """Step fractions by loss threshold, the no-split fraction, and the attention-time ratio heuristic/best."""
    n = len(losses)
    if n == 0:
        return {"steps_sampled": 0}
    total_h, total_best = float(losses.heuristic_us.sum()), float(losses.best_us.sum())
    out = {"steps_sampled": n, "batch_mean": float(losses.B.mean()), "batch_p90": float(losses.B.quantile(.9)),
           "len_max_mean": float(losses.len_max.mean()), "no_split_frac": float((losses.heuristic_splits == 1).mean()),
           "loss_median": float(losses.loss.median()), "loss_p90": float(losses.loss.quantile(.9)),
           "loss_max": float(losses.loss.max()), "attention_time_ratio": total_h / total_best}
    for th in thresholds:
        over = losses.loss >= th
        out[f"loss_ge_{th}_frac"] = float(over.mean())
        out[f"attention_share_loss_ge_{th}"] = float(losses.heuristic_us[over].sum() / total_h)
    return out


SUMMARY_FIELDS = ["kind", "requests", "rate_scale", "prompt_scale", "kv_tokens", "step_s", "max_batch", "what_if", "steps",
                  "batch_mean", "no_split_frac", "loss_median", "loss_p90", "loss_max", "loss_ge_1.1_frac",
                  "loss_ge_1.25_frac", "loss_ge_1.5_frac", "loss_ge_2.0_frac", "attention_share_loss_ge_1.25",
                  "attention_time_ratio", "prompt_tokens_p50", "prompt_tokens_max", "predictions"]


def collect_summaries(root) -> pd.DataFrame:
    """One row per ``<root>/<run>/summary.json`` written by scripts/traffic_replay.py, sorted by run name."""
    rows = []
    for path in sorted(Path(root).glob("*/summary.json")):
        s = json.loads(path.read_text())
        trace, cfg, losses = s["trace"], s["config"], s["losses"]
        rows.append({"run": path.parent.name, "kind": trace["kind"], "requests": trace["requests"],
                     "rate_scale": cfg["rate_scale"], "prompt_scale": trace.get("prompt_scale", 1.0),
                     "kv_tokens": cfg["kv_tokens"], "step_s": cfg["step_s"], "max_batch": cfg["max_batch"],
                     "what_if": "; ".join(s.get("what_if", [])), "steps": s["steps"],
                     "prompt_tokens_p50": trace.get("prompt_tokens_p50"), "prompt_tokens_max": trace.get("prompt_tokens_max"),
                     **{k: losses.get(k) for k in SUMMARY_FIELDS if k in losses}})
    return pd.DataFrame(rows, columns=["run"] + SUMMARY_FIELDS)
