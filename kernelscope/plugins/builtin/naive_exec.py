"""The sample CUDA binary (samples/naive_attn/naive_attn.cu) as an ExecutablePlugin —
the template for registering any standalone kernel binary.

fp32, decode only (L_q = 1). Inputs are rebuilt here with the same integer hash the
binary uses, so the harness can check the binary's written output against the reference.
"""
import os
from pathlib import Path

import torch

from kernelscope.plugins.base import ExecutablePlugin
from kernelscope.workload import Workload

DEFAULT_BIN = Path(__file__).resolve().parents[3] / "samples" / "naive_attn" / "naive_attn"


def hash_init(n: int, seed: int) -> torch.Tensor:
    """Same as init_val() in naive_attn.cu: low 16 bits of a uint32 multiply, mapped to [-0.5, 0.5)."""
    i = torch.arange(n, dtype=torch.int64)
    x = ((i + seed * 1000003) * 2654435761) & 0xFFFF
    return x.to(torch.float32) / 65536.0 - 0.5


class NaiveDecodeExec(ExecutablePlugin):
    name = "naive_exec"
    phases = frozenset({"decode"})
    kernel_regex = r"naive_decode_attn"
    num_launches = 1
    output_dtype = "float32"

    def __init__(self, device="cuda", binary=None, **_):
        super().__init__(device=device)
        self.binary = Path(binary or os.environ.get("KERNELSCOPE_NAIVE_BIN", DEFAULT_BIN))

    def supports(self, w: Workload) -> bool:
        return w.phase == "decode" and w.L_q == 1 and w.dtype == "float32" and w.d <= 256

    def command(self, w: Workload, iters: int = 1, out_path: str | None = None) -> list[str]:
        argv = [str(self.binary), str(w.B), str(w.L_kv), str(w.H_q), str(w.H_kv), str(w.d), str(iters)]
        if out_path:
            argv.append(str(out_path))
        return argv

    def reference_inputs(self, w: Workload):
        q = hash_init(w.B * w.H_q * w.d, seed=1).reshape(w.B, 1, w.H_q, w.d)
        k = hash_init(w.B * w.L_kv * w.H_kv * w.d, seed=2).reshape(w.B, w.L_kv, w.H_kv, w.d)
        v = hash_init(w.B * w.L_kv * w.H_kv * w.d, seed=3).reshape(w.B, w.L_kv, w.H_kv, w.d)
        return q, k, v


PLUGINS = [NaiveDecodeExec]
