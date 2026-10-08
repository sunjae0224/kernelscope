import pytest

torch = pytest.importorskip("torch")

from kernelscope.serve.attention import FlashInferDecode, csr_from_block_table


class FakeWrapper:
    """Records plan() arguments and runs a checkable stand-in for the FlashInfer kernel."""

    def __init__(self):
        self.plans, self.runs = [], 0

    def plan(self, indptr, indices, last_page_len, n_heads, n_kv_heads, head_dim, page, **kw):
        self.plans.append((indptr.tolist(), indices.tolist(), last_page_len.tolist(), n_heads, n_kv_heads, head_dim, page))

    def run(self, q, kv):
        self.runs += 1
        k, _ = kv
        return q + k[0, 0, 0, 0]            # [B, H_q, d]


def test_csr_from_block_table_lists_live_pages_and_last_page_lengths():
    table = torch.tensor([[7, 3, 0], [5, 0, 0]], dtype=torch.int32)
    indptr, indices, last = csr_from_block_table(table, [300, 256], page=256)
    assert indptr.tolist() == [0, 2, 3] and indices.tolist() == [7, 3, 5] and last.tolist() == [44, 256]


def test_decode_plans_once_per_step_appends_the_new_token_and_runs_the_kernel():
    wrapper = FakeWrapper()
    attention = FlashInferDecode(fallback=None, wrapper=wrapper, page=256)
    B, H_q, H_kv, d = 2, 4, 2, 8
    k_cache, v_cache = torch.zeros(4, 256, H_kv, d), torch.zeros(4, 256, H_kv, d)
    table = torch.tensor([[2, 1], [3, 0]], dtype=torch.int32)
    cache_lens = torch.tensor([256, 10], dtype=torch.int32)                     # lengths before this token
    plan_us = attention.plan(cache_lens, table, H_q, H_kv, d, torch.float32)
    assert plan_us >= 0.0
    assert wrapper.plans == [([0, 2, 3], [2, 1, 3], [1, 11], H_q, H_kv, d, 256)]  # lengths after the append
    q, k, v = torch.randn(B, 1, H_q, d), torch.randn(B, 1, H_kv, d), torch.randn(B, 1, H_kv, d)
    out = attention(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_lens, block_table=table,
                    softmax_scale=d ** -0.5, causal=True, num_splits=0)
    assert out.shape == (B, 1, H_q, d) and wrapper.runs == 1
    assert torch.equal(k_cache[1, 0], k[0, 0]) and torch.equal(v_cache[1, 0], v[0, 0])   # token 257 of sequence 0: page 1, slot 0
    assert torch.equal(k_cache[3, 10], k[1, 0]) and torch.equal(v_cache[3, 10], v[1, 0])
    assert torch.equal(out, (q[:, 0] + k_cache[0, 0, 0, 0]).unsqueeze(1))
    attention(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_lens, block_table=table,
              softmax_scale=d ** -0.5, causal=True, num_splits=0)                 # the next layer reuses the plan
    assert len(wrapper.plans) == 1 and wrapper.runs == 2


def test_prefill_calls_go_to_the_fallback_kernel_unchanged():
    calls = []

    def fallback(q, k_cache, v_cache, **kw):
        calls.append(kw["num_splits"])
        return q

    attention = FlashInferDecode(fallback=fallback, wrapper=FakeWrapper())
    q = torch.randn(1, 5, 4, 8)
    out = attention(q, None, None, k=None, v=None, cache_seqlens=None, block_table=None, softmax_scale=1.0,
                    causal=True, num_splits=0)
    assert calls == [0] and out is q


def test_decode_without_a_plan_for_these_lengths_is_refused():
    attention = FlashInferDecode(fallback=None, wrapper=FakeWrapper())
    k_cache = torch.zeros(2, 256, 2, 8)
    with pytest.raises(RuntimeError, match="plan"):
        attention(torch.randn(1, 1, 4, 8), k_cache, k_cache.clone(), k=torch.randn(1, 1, 2, 8), v=torch.randn(1, 1, 2, 8),
                  cache_seqlens=torch.tensor([3], dtype=torch.int32), block_table=torch.tensor([[1]], dtype=torch.int32),
                  softmax_scale=1.0, causal=True, num_splits=0)


def test_a_plan_for_the_previous_step_is_refused_for_the_next_lengths():
    attention = FlashInferDecode(fallback=None, wrapper=FakeWrapper())
    k_cache = torch.zeros(2, 256, 2, 8)
    table = torch.tensor([[1]], dtype=torch.int32)
    attention.plan(torch.tensor([3], dtype=torch.int32), table, 4, 2, 8, torch.float32)
    q, k, v = torch.randn(1, 1, 4, 8), torch.randn(1, 1, 2, 8), torch.randn(1, 1, 2, 8)
    attention(q, k_cache, k_cache.clone(), k=k, v=v, cache_seqlens=torch.tensor([3], dtype=torch.int32), block_table=table,
              softmax_scale=8 ** -0.5, causal=True, num_splits=0)                 # a fresh tensor with the planned lengths is fine
    with pytest.raises(RuntimeError, match="plan"):
        attention(q, k_cache, k_cache.clone(), k=k, v=v, cache_seqlens=torch.tensor([4], dtype=torch.int32), block_table=table,
                  softmax_scale=8 ** -0.5, causal=True, num_splits=0)


def test_a_softmax_scale_other_than_the_planned_one_is_refused():
    attention = FlashInferDecode(fallback=None, wrapper=FakeWrapper())
    k_cache = torch.zeros(2, 256, 2, 8)
    table = torch.tensor([[1]], dtype=torch.int32)
    lens = torch.tensor([3], dtype=torch.int32)
    attention.plan(lens, table, 4, 2, 8, torch.float32)
    with pytest.raises(ValueError, match="softmax_scale"):
        attention(torch.randn(1, 1, 4, 8), k_cache, k_cache.clone(), k=torch.randn(1, 1, 2, 8), v=torch.randn(1, 1, 2, 8),
                  cache_seqlens=lens, block_table=table, softmax_scale=1.0, causal=True, num_splits=0)
