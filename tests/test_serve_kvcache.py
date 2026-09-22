import subprocess

import pytest
import torch

from kernelscope.serve.kvcache import PAGE, PagePool


def _pool(n_pages=8):
    return PagePool(2, n_pages, 2, 8, dtype=torch.float32, device="cpu")


def test_reserve_growth_preserves_committed_length():
    pool = _pool()
    pool.reserve("a", 300)
    pool.set_length("a", 300)
    assert pool.free_pages == 6
    pool.reserve("a", 512)
    assert pool.free_pages == 6
    pool.reserve("a", 513)
    assert pool.free_pages == 5 and pool.length("a") == 300


def test_released_pages_reuse_without_aliasing_live_sequence():
    pool = _pool()
    pool.reserve("a", 600)
    pool.reserve("b", 10)
    old = pool.block_table(["a"])[0].tolist()
    live = pool.block_table(["b"])[0].tolist()
    pool.release("a")
    pool.release("a")
    pool.reserve("c", 600)
    assert pool.block_table(["c"])[0].tolist() == old
    assert not set(old).intersection(live)
    assert pool.free_pages == 4


def test_block_table_orders_pads_and_handles_empty_batches():
    pool = _pool()
    pool.reserve("a", 600)
    pool.reserve("b", 10)
    table = pool.block_table(["b", "a"])
    assert table.dtype == torch.int32 and table.shape == (2, 3)
    assert table[0, 1:].tolist() == [0, 0]
    assert len(set(table[1].tolist())) == 3
    assert pool.block_table([]).shape == (0, 0)
    assert pool.lengths([]).shape == (0,)


def test_batch_exhaustion_is_atomic():
    pool = _pool(2)
    pool.reserve("a", 1)
    with pytest.raises(MemoryError):
        pool.reserve_many({"a": 300, "b": 300})
    assert pool.free_pages == 1 and pool.capacity("a") == PAGE
    with pytest.raises(KeyError):
        pool.length("b")


@pytest.mark.parametrize("bad", [-1, 1.5, True, 2**31])
def test_invalid_lengths_do_not_mutate_pool(bad):
    pool = _pool()
    with pytest.raises(ValueError):
        pool.reserve("a", bad)
    assert pool.free_pages == 8


def test_length_cannot_exceed_reserved_storage():
    pool = _pool()
    pool.reserve("a", 1)
    with pytest.raises(ValueError):
        pool.set_length("a", PAGE + 1)
    with pytest.raises(KeyError):
        pool.set_length("absent", 1)


def test_budget_counts_all_layers_and_rounds_down():
    from types import SimpleNamespace
    cfg = SimpleNamespace(n_layers=2, n_kv_heads=2, head_dim=8)
    page_bytes = PagePool.bytes_per_page(2, 2, 8, torch.float32)
    assert page_bytes == 2 * 2 * 256 * 2 * 8 * 4
    pool = PagePool.for_budget(cfg, page_bytes * 3 + 7, device="cpu", dtype=torch.float32)
    assert pool.n_pages == 3
    with pytest.raises(ValueError, match="one page"):
        PagePool.for_budget(cfg, page_bytes - 1, device="cpu", dtype=torch.float32)


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    try:
        probe = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                "--format=csv"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("GPU process inventory is unavailable")
    if torch.cuda.mem_get_info()[0] < 18 * 2**30:
        pytest.skip("functional GPU tests require at least 18 GiB free")
    return probe.stdout


@pytest.mark.gpu
def test_flash_attention_appends_to_correct_page():
    _require_gpu()
    flash_attn = pytest.importorskip("flash_attn")
    pool = PagePool(2, 16, 8, 128)
    for name, length in (("a", 300), ("b", 40)):
        pool.reserve(name, length + 1)
        pool.set_length(name, length)
    # Populate the existing context as valid zero K/V, not uninitialized storage.
    pool.k[0].zero_()
    pool.v[0].zero_()
    table = pool.block_table(["a", "b"])
    q = torch.randn(2, 1, 32, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 1, 8, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output = flash_attn.flash_attn_with_kvcache(q, pool.k[0], pool.v[0], k=key, v=value,
        cache_seqlens=pool.lengths(["a", "b"]), block_table=table, causal=True)
    torch.testing.assert_close(pool.k[0][table[0, 300 // PAGE], 300 % PAGE], key[0, 0], rtol=0, atol=0)
    torch.testing.assert_close(pool.v[0][table[1, 0], 40], value[1, 0], rtol=0, atol=0)
    assert torch.isfinite(output).all()
