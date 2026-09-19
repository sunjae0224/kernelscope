"""GPU-only: cold measurements really are colder than warm ones for an L2-resident working set."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from kernelscope.backends.realhw.cache import IterationHooks  # noqa: E402
from kernelscope.plugins.builtin import REGISTRY  # noqa: E402
from kernelscope.run_kernel import profile_launches  # noqa: E402
from kernelscope.workload import Workload  # noqa: E402

pytestmark = pytest.mark.gpu

# 64 MiB of K/V: fits in the 72 MiB L2, so warm iterations are served from L2 (spec F7)
W = Workload(phase="decode", B=16, L_q=1, L_kv=1024, H_q=32, H_kv=8, d=128)


def _kernel_time(state):
    p = REGISTRY.get("flashdecoding", device="cuda")
    inputs = p.build_inputs(W)
    hooks = IterationHooks("cuda", state)
    for _ in range(5):
        hooks.between(); p.run(inputs)
    s = profile_launches(p, inputs, "cuda", iters=20, hooks=hooks, pad=5)
    assert all("kernelscope" not in l["name"] for l in s["launches"])
    assert s["iterations_used"] == 20
    return s["kernel_time_us_median"]


def test_cold_is_slower_than_warm_for_an_l2_resident_cell():
    assert _kernel_time("cold") > 1.2 * _kernel_time("warm")
