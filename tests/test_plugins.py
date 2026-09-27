import pytest

from kernelscope.plugins.base import ExecutablePlugin, KernelPlugin
from kernelscope.plugins.registry import PluginRegistry
from kernelscope.workload import Workload
from tests.fake_plugins import FakeExec, FaithfulCPU


class DecodeOnly(KernelPlugin):
    name = "decode_only"
    phases = frozenset({"decode"})
    kernel_regex = r"my_decode_kernel"

    def build_inputs(self, w):
        return {"w": w}

    def run(self, inputs):
        return inputs["w"].L_kv

    def to_dense_output(self, out):
        return out


class BinaryAttn(ExecutablePlugin):
    name = "binary_attn"
    phases = frozenset({"decode", "prefill"})
    kernel_regex = r"attn_kernel"
    num_launches = 2

    def command(self, w, iters=1, out_path=None):
        return ["./attn", "--B", str(w.B), "--L", str(w.L_kv)]


DECODE = Workload(phase="decode", B=1, L_q=1, L_kv=4096, H_q=32, H_kv=8, d=128)
PREFILL = Workload(phase="prefill", B=1, L_q=4096, L_kv=4096, H_q=32, H_kv=8, d=128)


def test_kernel_plugin_supports_only_declared_phases():
    p = DecodeOnly()
    assert p.supports(DECODE)
    assert not p.supports(PREFILL)


def test_num_launches_defaults_to_one():
    assert DecodeOnly().num_launches == 1


def test_executable_plugin_builds_command_from_workload():
    p = BinaryAttn()
    assert p.command(DECODE) == ["./attn", "--B", "1", "--L", "4096"]
    assert p.num_launches == 2


def test_registry_round_trips_by_name():
    reg = PluginRegistry()
    reg.register(DecodeOnly)
    reg.register(BinaryAttn)
    assert reg.names() == ["decode_only", "binary_attn"]
    assert isinstance(reg.get("decode_only"), DecodeOnly)


def test_registry_rejects_duplicate_names():
    reg = PluginRegistry()
    reg.register(DecodeOnly)
    with pytest.raises(ValueError, match="decode_only"):
        reg.register(DecodeOnly)


def test_registry_unknown_name_lists_available():
    reg = PluginRegistry()
    reg.register(DecodeOnly)
    with pytest.raises(KeyError, match="decode_only"):
        reg.get("nope")


def test_registry_filters_plugins_supporting_a_workload():
    reg = PluginRegistry()
    reg.register(DecodeOnly)
    reg.register(BinaryAttn)
    assert [p.name for p in reg.supporting(PREFILL)] == ["binary_attn"]
    assert [p.name for p in reg.supporting(DECODE)] == ["decode_only", "binary_attn"]


_R = Workload(phase="decode", B=2, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32", kv_lens=[64, 3])


def test_plugins_do_not_support_ragged_workloads_unless_they_opt_in():
    assert FaithfulCPU(device="cpu").supports(_R) is True
    assert FakeExec(device="cpu").supports(_R) is False


def test_builtin_non_flash_plugins_refuse_ragged_workloads():
    from kernelscope.plugins.builtin.sdpa import SDPAFlash
    assert SDPAFlash(device="cpu").supports(_R) is False
