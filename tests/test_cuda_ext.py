import pytest

from kernelscope.bench.placement import smem_for_ctas_per_sm


@pytest.mark.parametrize("smem_per_sm", [102400, 167936])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_smem_request_admits_exactly_k_ctas_per_sm(k, smem_per_sm):
    s = smem_for_ctas_per_sm(k, smem_per_sm, reserved=1024)
    assert smem_per_sm // (s + 1024) == k


@pytest.fixture(scope="module")
def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch


@pytest.mark.gpu
def test_block_placement_fills_every_sm_exactly(cuda):
    from kernelscope.bench.placement import block_placement
    r = block_placement(ctas_per_sm=2)
    n = cuda.cuda.get_device_properties(0).multi_processor_count
    assert r["n_ctas"] == 2 * n
    assert r["distinct_sms"] == n and r["max_ctas_on_one_sm"] == 2
    assert all(0 <= s < n for s in r["smid"])


@pytest.mark.gpu
def test_blocking_half_the_sms_slows_a_gemm_but_not_idle_time(cuda):
    from kernelscope.bench.sm_blocker import SMBlocker
    torch = cuda
    b = SMBlocker()
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    fn = lambda: a @ a  # noqa: E731
    free = b.time_blocked(fn, 0)
    half = b.time_blocked(fn, b.n_sms // 2)
    assert 1.6 < half / free < 2.4, (free, half)       # probe: 1.83x at 64 of 128 SMs
