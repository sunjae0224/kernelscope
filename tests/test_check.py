import torch

from kernelscope.check import check_plugin
from kernelscope.plugins.base import KernelPlugin
from kernelscope.reference import reference_attention
from kernelscope.workload import Workload

W = Workload(phase="decode", B=2, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16)


class _DenseBase(KernelPlugin):
    phases = frozenset({"decode", "prefill"})
    kernel_regex = "x"

    def build_inputs(self, w):
        g = torch.Generator().manual_seed(0)
        dt = getattr(torch, w.dtype)
        return {
            "q": torch.randn(w.B, w.L_q, w.H_q, w.d, generator=g).to(dt),
            "k": torch.randn(w.B, w.L_kv, w.H_kv, w.d, generator=g).to(dt),
            "v": torch.randn(w.B, w.L_kv, w.H_kv, w.d, generator=g).to(dt),
            "causal": w.causal,
        }

    def to_dense_inputs(self, inputs):
        return inputs["q"], inputs["k"], inputs["v"]

    def to_dense_output(self, out):
        return out


class Faithful(_DenseBase):
    name = "faithful"

    def run(self, inputs):
        return reference_attention(inputs["q"], inputs["k"], inputs["v"], inputs["causal"])


class Broken(_DenseBase):
    name = "broken"

    def run(self, inputs):
        return torch.zeros(W.B, W.L_q, W.H_q, W.d, dtype=inputs["q"].dtype)


def test_faithful_plugin_passes_with_tiny_diff():
    r = check_plugin(Faithful(), W, atol=1e-2)
    assert r["ok"] is True
    assert r["max_abs_diff"] < 1e-4


def test_broken_plugin_fails_and_reports_diff():
    r = check_plugin(Broken(), W, atol=1e-2)
    assert r["ok"] is False
    assert r["max_abs_diff"] > 1e-2


def test_shape_mismatch_is_reported_not_raised():
    class WrongShape(_DenseBase):
        name = "wrong_shape"

        def run(self, inputs):
            return torch.zeros(1, 1, 1, 1)

    r = check_plugin(WrongShape(), W, atol=1e-2)
    assert r["ok"] is False
    assert "shape" in r["error"]
