"""GPU-only: the boundary kernels exist under the names the profiler parser expects."""
import re

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from kernelscope.backends.realhw.cache import FLUSH_FACTOR, MARKER_REGEX, IterationHooks  # noqa: E402
from kernelscope.run_kernel import launched_kernels  # noqa: E402

pytestmark = pytest.mark.gpu


class _Plugin:
    """Adapter so launched_kernels() can profile a bare callable."""
    def __init__(self, fn):
        self.run = lambda inputs: fn()


@pytest.mark.parametrize("state, name", [("cold", "kernelscope_l2_flush"), ("warm", "kernelscope_iter_marker")])
def test_boundary_kernels_carry_their_names_into_the_profiler(state, name):
    h = IterationHooks("cuda", state)
    h.between()                                   # JIT compile outside the profiled region
    torch.cuda.synchronize()
    names = launched_kernels(_Plugin(h.between), None, "cuda")
    assert any(name in n for n in names), names
    assert all(re.search(MARKER_REGEX, n) for n in names if "kernelscope" in n)


def test_cold_flush_buffer_is_four_times_l2():
    h = IterationHooks("cuda", "cold")
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    assert h._buf.numel() * 4 == FLUSH_FACTOR * l2
