import torch

from kernelscope.check import check_outputs, check_plugin
from kernelscope.plugins.base import KernelPlugin
from kernelscope.reference import reference_attention
from kernelscope.workload import Workload
from tests.fake_plugins import FaithfulCPU

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


RAGGED_W = Workload(phase="decode", B=3, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32",
                    kv_lens=[64, 5, 30])


class IgnoresLengths(FaithfulCPU):
    """Attends to the whole cache: wrong for a ragged batch."""
    name = "ignores_lengths"

    def run(self, inputs):
        return reference_attention(inputs["q"], inputs["k"], inputs["v"], inputs["causal"])


def test_check_passes_a_plugin_that_honours_ragged_lengths():
    r = check_plugin(FaithfulCPU(device="cpu"), RAGGED_W)
    assert r["ok"] is True, r


def test_check_fails_a_plugin_that_ignores_ragged_lengths():
    r = check_plugin(IgnoresLengths(device="cpu"), RAGGED_W)
    assert r["ok"] is False
    assert r["max_abs_diff"] > 1e-2


def test_check_outputs_uses_the_given_inputs():
    p = FaithfulCPU(device="cpu")
    inputs = p.build_inputs(RAGGED_W)
    assert check_outputs(p, RAGGED_W, inputs)["ok"] is True
