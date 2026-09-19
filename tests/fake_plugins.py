"""CPU-only plugins used to exercise the runner/sweep machinery without a GPU."""
import sys
from pathlib import Path

import torch

from kernelscope.plugins.base import ExecutablePlugin, KernelPlugin
from kernelscope.plugins.registry import PluginRegistry
from kernelscope.reference import reference_attention


class FaithfulCPU(KernelPlugin):
    name = "faithful_cpu"
    phases = frozenset({"decode", "prefill"})
    kernel_regex = "faithful"
    supports_ragged = True

    def build_inputs(self, w):
        g = torch.Generator().manual_seed(0)
        dt = getattr(torch, w.dtype)
        return {
            "q": torch.randn(w.B, w.L_q, w.H_q, w.d, generator=g).to(dt),
            "k": torch.randn(w.B, w.L_kv, w.H_kv, w.d, generator=g).to(dt),
            "v": torch.randn(w.B, w.L_kv, w.H_kv, w.d, generator=g).to(dt),
            "causal": w.causal,
            "kv_lens": w.kv_lens,
        }

    def run(self, inputs):
        return reference_attention(inputs["q"], inputs["k"], inputs["v"], inputs["causal"],
                                    kv_lens=inputs.get("kv_lens"))

    def to_dense_inputs(self, inputs):
        return inputs["q"], inputs["k"], inputs["v"]

    def to_dense_output(self, out):
        return out


class FakeExec(ExecutablePlugin):
    """Drives tests/fake_exec.py the way a plugin would drive a CUDA binary."""
    name = "fake_exec"
    phases = frozenset({"decode", "prefill"})
    kernel_regex = "fake_kernel"

    def command(self, w, iters=1, out_path=None):
        argv = [sys.executable, str(Path(__file__).parent / "fake_exec.py"), w.key(), str(iters)]
        if out_path:
            argv.append(str(out_path))
        return argv

    def reference_inputs(self, w):
        from tests.fake_exec import inputs_for
        return inputs_for(w)


REGISTRY = PluginRegistry()
REGISTRY.register(FaithfulCPU)
REGISTRY.register(FakeExec)
