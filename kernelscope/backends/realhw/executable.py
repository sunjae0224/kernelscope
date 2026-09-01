"""Driving a standalone binary (ExecutablePlugin) on real hardware.

Contract for the binary (see samples/naive_attn/naive_attn.cu):
  * ``plugin.command(w, iters, out_path)`` runs the kernel ``iters`` times, timing each
    launch with CUDA events, and prints one line ``KERNELSCOPE {json}`` with at least
    ``kernel_time_us`` (median) and optionally ``launches_per_iter``;
  * with ``out_path`` it writes the dense output ``[B, L_q, H_q, d]`` as raw
    ``plugin.output_dtype`` so the harness can compare it with the reference computed
    from ``plugin.reference_inputs(w)``.
The harness cannot put CUDA events or torch.profiler around a foreign process, so
timing is self-reported; launch geometry comes from the NVBit trace (sim track).
"""
import json
import math
import subprocess

import numpy as np
import torch

from kernelscope.reference import reference_attention

KS_PREFIX = "KERNELSCOPE "


def parse_kernelscope_line(text: str) -> dict:
    for line in text.splitlines():
        if line.startswith(KS_PREFIX):
            return json.loads(line[len(KS_PREFIX):])
    raise ValueError("executable printed no 'KERNELSCOPE {json}' line")


def run_executable(plugin, w, iters: int = 1, out_path=None, timeout_s: int = 3600, env=None) -> dict:
    argv = plugin.command(w, iters=iters, out_path=str(out_path) if out_path else None)
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{plugin.name} exit {proc.returncode}: {proc.stderr.strip()[-1500:]}")
    d = parse_kernelscope_line(proc.stdout)
    d.setdefault("launches_per_iter", plugin.num_launches)
    return d


def check_executable(plugin, w, out_path, atol: float = 1e-2) -> dict:
    if getattr(plugin, "reference_inputs", None) is None:
        return {"ok": False, "max_abs_diff": math.nan,
                "error": f"{plugin.name}: no reference_inputs; correctness check unavailable"}
    try:
        run_executable(plugin, w, iters=1, out_path=out_path)
        arr = np.fromfile(str(out_path), dtype=np.dtype(plugin.output_dtype))
        out = torch.from_numpy(arr).reshape(w.B, w.L_q, w.H_q, w.d)
        q, k, v = plugin.reference_inputs(w)
        ref = reference_attention(q, k, v, w.causal)
        diff = (out.float() - ref.float()).abs().max().item()
        return {"ok": diff <= atol, "max_abs_diff": diff, "error": None}
    except Exception as e:
        return {"ok": False, "max_abs_diff": math.nan, "error": f"{type(e).__name__}: {e}"}
