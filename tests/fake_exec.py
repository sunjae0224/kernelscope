"""Stand-in for a user's CUDA binary: obeys the ExecutablePlugin contract on CPU.

usage: fake_exec.py <workload_key> <iters> [out_path]
Prints one line `KERNELSCOPE {json}` and (optionally) writes the dense output
[B, L_q, H_q, d] as raw float32 to out_path.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script: make `kernelscope` importable
from kernelscope.reference import reference_attention  # noqa: E402
from kernelscope.workload import Workload  # noqa: E402


def inputs_for(w: Workload):
    g = torch.Generator().manual_seed(1234)
    q = torch.randn(w.B, w.L_q, w.H_q, w.d, generator=g)
    k = torch.randn(w.B, w.L_kv, w.H_kv, w.d, generator=g)
    v = torch.randn(w.B, w.L_kv, w.H_kv, w.d, generator=g)
    return q, k, v


def main():
    w = Workload.from_key(sys.argv[1])
    iters = int(sys.argv[2])
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    q, k, v = inputs_for(w)
    out = None
    for _ in range(iters):
        out = reference_attention(q, k, v, w.causal)
    if out_path:
        out.float().numpy().tofile(out_path)
    print("some unrelated stdout noise")
    print("KERNELSCOPE " + json.dumps({"kernel_time_us": 12.5, "launches_per_iter": 1, "iters": iters}))


if __name__ == "__main__":
    main()
