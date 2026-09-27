"""Deterministic continuous batching with separate GPU and host-visible timings.

Arrivals are logical decode steps; every policy sees the same request schedule.
Sequential prompt admission can delay active requests, and these delays remain
in token timestamps and TPOT. This is a research replay engine, not a web server.
"""
import json
import math
import time
from dataclasses import dataclass, field

import pandas as pd
import torch

from kernelscope.serve.kvcache import PAGE
from kernelscope.serve.model import AttentionTimer, OpTimer
from kernelscope.serve.scenarios import prompt_ids

STEP_COLUMNS = ["step", "policy", "B", "len_max", "len_sum", "n_long", "num_splits", "attn_us",
                "step_us", "policy_us", "policy_cache_hit", "decode_wall_us", "seq_ids", "lens"]
TOKEN_COLUMNS = ["rid", "step", "position", "token", "t_us", "phase"]
PREFILL_COLUMNS = ["rid", "prompt_len", "prefill_us", "arrival_us", "admitted_us", "first_token_us", "prompt_sha256"]
OPS_COLUMNS = ["phase", "step", "rid", "layer", "op_class", "gpu_us"]
OPS_MODES = (None, "event")


@dataclass
class RunResult:
    steps: pd.DataFrame
    tokens: pd.DataFrame
    prefill: pd.DataFrame
    logits: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    ops: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=OPS_COLUMNS))


class _CallTimer:
    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda"
        self.device = device

    def start(self):
        if self.cuda:
            with torch.cuda.device(self.device):
                self.begin = torch.cuda.Event(enable_timing=True)
                self.end = torch.cuda.Event(enable_timing=True)
                self.begin.record()
        else:
            self.begin = time.perf_counter()

    def stop(self):
        if self.cuda:
            with torch.cuda.device(self.device):
                self.end.record()
                self.end.synchronize()
            return self.begin.elapsed_time(self.end) * 1000
        return (time.perf_counter() - self.begin) * 1e6


class Engine:
    def __init__(self, model, pool, policy, max_batch=64):
        if not isinstance(max_batch, int) or max_batch < 1:
            raise ValueError("max_batch must be a positive integer")
        self.model, self.pool, self.policy, self.max_batch = model, pool, policy, max_batch

    @torch.inference_mode()
    def run(self, requests, vocab, record_logits_steps=0, seed=0, ops_mode=None) -> RunResult:
        if record_logits_steps < 0:
            raise ValueError("record_logits_steps must be nonnegative")
        if ops_mode not in OPS_MODES:
            raise ValueError(f"ops_mode must be one of {OPS_MODES}, got {ops_mode!r}")
        requests = list(requests)
        if len({r.rid for r in requests}) != len(requests):
            raise ValueError("request IDs must be unique")
        if any(r.prompt_text is not None or r.prompt_recipe is not None or r.prompt_len is None for r in requests):
            raise ValueError("resolve natural-text requests with the local tokenizer before Engine.run")
        cfg, device = self.model.cfg, self.model.device
        if vocab != cfg.vocab:
            raise ValueError("scenario vocabulary must equal the model vocabulary")
        for request in requests:
            if request.token_ids is not None:
                prompt_ids(request, vocab, seed)  # Reject invalid explicit tokens before timing/allocation.
        prompt_kinds = {r.prompt_kind or ("explicit_token_ids" if r.token_ids is not None else "seeded_synthetic_token_ids")
                        for r in requests}
        if getattr(self.policy, "name", None) in {"model", "table"} and (
                cfg.head_dim != 128 or self.model.dtype not in (torch.float16, torch.bfloat16)):
            raise ValueError("model/table policies are calibrated for d=128 fp16/bf16 only")
        pending = sorted(requests, key=lambda r: (r.arrival_step, r.rid))
        # Logical reservations guarantee an admitted request can finish. Actual
        # pages still grow with live lengths, preserving flash-attn capacity.
        capacity = self.pool.free_pages
        required = {r.rid: math.ceil((r.prompt_len + r.max_new_tokens - 1) / PAGE) for r in requests}
        too_large = [r.rid for r in requests if required[r.rid] > capacity]
        if too_large:
            raise MemoryError(f"requests {too_large} exceed this pool's {capacity} pages; increase --kv-gib")
        active, owned, last_tok, produced, arrivals = [], set(), {}, {}, {}
        steps, tokens, prefills, logits, logit_keys, ops_rows = [], [], [], [], [], []
        reserved = 0
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        now_us = lambda: (time.perf_counter() - started) * 1e6
        step = 0

        def retire_finished():
            nonlocal active, reserved
            keep = []
            for request in active:
                if produced[request.rid] >= request.max_new_tokens:
                    self.pool.release(request.rid)
                    owned.discard(request.rid)
                    reserved -= required[request.rid]
                else:
                    keep.append(request)
            active = keep

        try:
            while pending or active:
                if not active and pending and pending[0].arrival_step > step:
                    step = pending[0].arrival_step
                arrival_time = now_us()
                for request in pending:
                    if request.arrival_step > step:
                        break
                    arrivals.setdefault(request.rid, arrival_time)
                while pending and pending[0].arrival_step <= step and len(active) < self.max_batch:
                    request = pending[0]
                    if reserved + required[request.rid] > capacity:
                        break  # FIFO backpressure; active requests release their pages.
                    pending.pop(0)
                    ids = prompt_ids(request, vocab, seed)
                    admitted = now_us()
                    timer = _CallTimer(device)
                    owned.add(request.rid)
                    timer.start()
                    if ops_mode == "event":
                        ops = OpTimer(device=device)
                        output = self.model.prefill(request.rid, ids, self.pool, timer=ops)
                    else:
                        output = self.model.prefill(request.rid, ids, self.pool)
                    elapsed = timer.stop()
                    if ops_mode == "event":
                        ops_rows.extend({"phase": "prefill", "step": step, "rid": str(request.rid), **row} for row in ops.rows())
                    first = int(output.argmax().item())
                    stamp = now_us()
                    last_tok[request.rid], produced[request.rid] = first, 1
                    prefills.append(dict(rid=request.rid, prompt_len=request.prompt_len, prefill_us=elapsed,
                                         arrival_us=arrivals[request.rid], admitted_us=admitted, first_token_us=stamp,
                                         prompt_sha256=request.prompt_sha256))
                    tokens.append(dict(rid=request.rid, step=step, position=0, token=first, t_us=stamp, phase="prefill"))
                    active.append(request)
                    reserved += required[request.rid]
                    retire_finished()
                if not active:
                    step += 1
                    continue
                seq_ids = [r.rid for r in active]
                # Selection predicts the attention call after appending its new KV.
                lens = [self.pool.length(rid) + 1 for rid in seq_ids]
                wall_started = time.perf_counter()
                select_started = time.perf_counter()
                splits = self.policy.choose(lens, cfg.n_heads, cfg.n_kv_heads)
                selection_us = (time.perf_counter() - select_started) * 1e6
                attention = OpTimer(device=device) if ops_mode == "event" else AttentionTimer(device=device)
                timer = _CallTimer(device)
                timer.start()
                output = self.model.decode(seq_ids, [last_tok[rid] for rid in seq_ids], self.pool,
                                           num_splits=splits, timer=attention)
                step_us = timer.stop()
                if ops_mode == "event":
                    ops_rows.extend({"phase": "decode", "step": step, "rid": "", **row} for row in attention.rows())
                next_tokens = output.argmax(-1).tolist()
                stamp = now_us()
                wall_us = (time.perf_counter() - wall_started) * 1e6
                for request, token in zip(active, next_tokens):
                    rid = request.rid
                    tokens.append(dict(rid=rid, step=step, position=produced[rid], token=int(token),
                                       t_us=stamp, phase="decode"))
                    last_tok[rid] = int(token)
                    produced[rid] += 1
                steps.append(dict(step=step, policy=self.policy.name, B=len(seq_ids), len_max=max(lens),
                                  len_sum=sum(lens), n_long=sum(n >= 4096 for n in lens), num_splits=splits,
                                  attn_us=attention.total_us(), step_us=step_us, policy_us=selection_us,
                                  policy_cache_hit=getattr(self.policy, "last_cache_hit", None),
                                  decode_wall_us=wall_us, seq_ids=json.dumps(seq_ids), lens=json.dumps(lens)))
                if len(logits) < record_logits_steps:
                    logits.append(output.detach().float().cpu().clone())
                    logit_keys.append([step, seq_ids])
                retire_finished()
                step += 1
        finally:
            for rid in owned:
                self.pool.release(rid)
        return RunResult(pd.DataFrame(steps, columns=STEP_COLUMNS), pd.DataFrame(tokens, columns=TOKEN_COLUMNS),
                         pd.DataFrame(prefills, columns=PREFILL_COLUMNS), logits,
                         {"total_wall_us": now_us(), "arrival_mode": "logical_decode_step",
                          "sampling": "greedy_fixed_length_no_eos_stop", "seed": seed,
                          "prompt_kind": next(iter(prompt_kinds)) if len(prompt_kinds) == 1 else "mixed",
                          "prompt_resolution": "pre_resolved" if all(r.token_ids is not None for r in requests) else "synthetic_at_admission",
                          "model_dtype": str(self.model.dtype).removeprefix("torch."), "logit_step_keys": logit_keys,
                          "timing_backend": "cuda_events" if torch.device(device).type == "cuda" else "cpu_perf_counter",
                          "model_backend": getattr(self.model, "backend", "unknown"),
                          "measurement_scope": "single_process_continuous_batching_replay",
                          "ops_mode": ops_mode},
                         ops=pd.DataFrame(ops_rows, columns=OPS_COLUMNS))
