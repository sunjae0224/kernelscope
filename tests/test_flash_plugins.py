"""GPU-only: flash-attn based plugins. Run with the gradkernel env on a CUDA device."""
import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from kernelscope.check import check_plugin  # noqa: E402
from kernelscope.plugins.builtin import REGISTRY  # noqa: E402
from kernelscope.plugins.builtin.flash import FA2, FlashDecoding  # noqa: E402
from kernelscope.run_kernel import launched_kernels  # noqa: E402
from kernelscope.workload import Workload  # noqa: E402

pytestmark = pytest.mark.gpu

DECODE = Workload(phase="decode", B=1, L_q=1, L_kv=4096, H_q=32, H_kv=8, d=128)
PREFILL = Workload(phase="prefill", B=1, L_q=1024, L_kv=1024, H_q=32, H_kv=8, d=128)
CHUNK = Workload(phase="prefill", B=1, L_q=256, L_kv=1024, H_q=32, H_kv=8, d=128)


def test_builtin_registry_exposes_flash_plugins():
    assert {"fa2", "flashdecoding"} <= set(REGISTRY.names())


@pytest.mark.parametrize("w", [DECODE, PREFILL, CHUNK], ids=lambda w: w.phase + str(w.L_q))
def test_fa2_matches_reference(w):
    r = check_plugin(FA2(device="cuda"), w, atol=2e-2)
    assert r["ok"], r


def test_flashdecoding_matches_reference_on_decode():
    r = check_plugin(FlashDecoding(device="cuda"), DECODE, atol=2e-2)
    assert r["ok"], r


def test_fa2_decode_launches_a_single_non_split_kernel():
    p = FA2(device="cuda")
    names = launched_kernels(p, p.build_inputs(DECODE), "cuda")
    matching = [n for n in names if re.search(p.kernel_regex, n)]
    assert len(matching) == 1
    assert "splitkv_combine" not in matching[0]


def test_flashdecoding_decode_launches_split_and_combine_kernels():
    p = FlashDecoding(device="cuda")
    names = launched_kernels(p, p.build_inputs(DECODE), "cuda")
    matching = [n for n in names if re.search(p.kernel_regex, n)]
    assert len(matching) == 2
    assert any("splitkv_combine" in n for n in matching)


def test_flashdecoding_is_decode_only():
    assert not FlashDecoding(device="cuda").supports(PREFILL)


from kernelscope.plugins.builtin import flash as flash_mod  # noqa: E402
from kernelscope.run_kernel import profile_launches  # noqa: E402

RAGGED = Workload(phase="decode", B=4, L_q=1, L_kv=1024, H_q=32, H_kv=8, d=128, kv_lens=(1024, 300, 700, 5))
UNIFORM = Workload(phase="decode", B=2, L_q=1, L_kv=2048, H_q=32, H_kv=8, d=128)
DECODE_PLUGINS = [c for c in flash_mod.PLUGINS if "decode" in c.phases]


def test_registry_exposes_the_fixed_split_and_paged_families():
    names = set(REGISTRY.names())
    for n in flash_mod.SPLITS:
        assert {f"fd_s{n}", f"fd_s{n}_paged"} <= names
    assert {"fa2_paged", "flashdecoding_paged"} <= names


@pytest.mark.parametrize("cls", DECODE_PLUGINS, ids=lambda c: c.name)
@pytest.mark.parametrize("w", [RAGGED, UNIFORM], ids=["ragged", "uniform"])
def test_every_decode_variant_matches_the_reference(cls, w):
    r = check_plugin(cls(device="cuda"), w, atol=2e-2)
    assert r["ok"], r


def test_fixed_split_plugin_launches_exactly_that_many_splits():
    p = REGISTRY.get("fd_s8", device="cuda")
    inputs = p.build_inputs(Workload(phase="decode", B=1, L_q=1, L_kv=4096, H_q=32, H_kv=8, d=128))
    for _ in range(3):
        p.run(inputs)
    s = profile_launches(p, inputs, "cuda", iters=3)
    main = [l for l in s["launches"] if "combine" not in l["name"]][0]
    assert main["grid"][1] == 8          # grid = (m_blocks, num_splits, B * H_kv)


def test_paged_fa2_runs_the_split_kernel():
    p = REGISTRY.get("fa2_paged", device="cuda")
    names = launched_kernels(p, p.build_inputs(UNIFORM), "cuda")
    assert any("splitkv" in n for n in names), names
