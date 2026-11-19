"""GPU-only: the FlashInfer decode plugin computes the same attention as the reference on ragged batches."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flashinfer")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from kernelscope.check import check_plugin  # noqa: E402
from kernelscope.plugins.builtin import REGISTRY  # noqa: E402
from kernelscope.plugins.builtin.flashinfer import FlashInferPaged, FlashInferPagedCudaCore, paged_indices  # noqa: E402
from kernelscope.workload import Workload  # noqa: E402

pytestmark = pytest.mark.gpu

UNIFORM = Workload(phase="decode", B=4, L_q=1, L_kv=1024, H_q=32, H_kv=8, d=128, dtype="bfloat16")
RAGGED = Workload(phase="decode", B=4, L_q=1, L_kv=4096, H_q=32, H_kv=8, d=128, dtype="bfloat16",
                  kv_lens=(4096, 300, 257, 1))


def test_registry_exposes_both_flashinfer_plugins():
    assert {"flashinfer_paged", "flashinfer_paged_cudacore"} <= set(REGISTRY.names())


def test_paged_indices_follow_the_block_table():
    table = torch.tensor([[7, 3, 0], [5, 0, 0]], dtype=torch.int32)
    indptr, indices, last = paged_indices([300, 256], table)
    assert indptr.tolist() == [0, 2, 3] and indices.tolist() == [7, 3, 5] and last.tolist() == [44, 256]


@pytest.mark.parametrize("plugin", [FlashInferPaged, FlashInferPagedCudaCore], ids=["tensorcore", "cudacore"])
@pytest.mark.parametrize("w", [UNIFORM, RAGGED], ids=["uniform", "ragged"])
def test_flashinfer_matches_reference(plugin, w):
    r = check_plugin(plugin(device="cuda"), w, atol=2e-2)
    assert r["ok"], r


def test_plan_cost_is_recorded_per_built_input():
    inputs = FlashInferPaged(device="cuda").build_inputs(RAGGED)
    assert inputs["plan_us"] > 0 and inputs["wrapper"] is not None
