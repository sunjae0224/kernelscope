"""GPU-only: every SDPA backend agrees with the reference on the workloads it claims to support,
and its kernel_regex matches what it actually launches."""
import re

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from kernelscope.check import check_plugin  # noqa: E402
from kernelscope.plugins.builtin.sdpa import SDPACudnn, SDPAEfficient, SDPAFlash  # noqa: E402
from kernelscope.run_kernel import launched_kernels  # noqa: E402
from kernelscope.workload import Workload  # noqa: E402

pytestmark = pytest.mark.gpu

DECODE = Workload(phase="decode", B=1, L_q=1, L_kv=4096, H_q=32, H_kv=8, d=128)
PREFILL = Workload(phase="prefill", B=1, L_q=1024, L_kv=1024, H_q=32, H_kv=8, d=128)
CHUNK = Workload(phase="prefill", B=1, L_q=256, L_kv=1024, H_q=32, H_kv=8, d=128)
ALL = [DECODE, PREFILL, CHUNK]


@pytest.mark.parametrize("cls", [SDPAEfficient, SDPACudnn, SDPAFlash], ids=lambda c: c.name)
@pytest.mark.parametrize("w", ALL, ids=lambda w: w.phase + str(w.L_q))
def test_supported_cells_match_reference_and_regex(cls, w):
    p = cls(device="cuda")
    if not p.supports(w):
        pytest.skip(f"{p.name} declares no support for {w.key()}")
    r = check_plugin(p, w, atol=2e-2)
    assert r["ok"], r
    names = launched_kernels(p, p.build_inputs(w), "cuda")
    matching = [n for n in names if re.search(p.kernel_regex, n)]
    assert matching, f"{p.name} regex {p.kernel_regex!r} matched none of {names}"
