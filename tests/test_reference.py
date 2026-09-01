import torch

from kernelscope.reference import reference_attention


def _manual(q, k, v, mask=None):
    # q [B,Lq,H,d], k/v [B,Lkv,H,d] with H already matched; plain float32 softmax attention
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / q.shape[-1] ** 0.5
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", p, v.float())


def test_matches_manual_softmax_attention_non_causal():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 5, 4, 8)
    v = torch.randn(2, 5, 4, 8)
    out = reference_attention(q, k, v, causal=False)
    torch.testing.assert_close(out, _manual(q, k, v))


def test_output_keeps_query_dtype_and_shape():
    q = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 6, 4, 8, dtype=torch.bfloat16)
    v = torch.randn(1, 6, 4, 8, dtype=torch.bfloat16)
    out = reference_attention(q, k, v, causal=True)
    assert out.dtype == torch.bfloat16
    assert out.shape == (1, 2, 4, 8)


def test_gqa_query_head_h_uses_kv_head_h_div_group():
    torch.manual_seed(1)
    q = torch.randn(1, 2, 4, 8)          # H_q = 4
    k = torch.randn(1, 6, 2, 8)          # H_kv = 2 -> group size 2
    v = torch.randn(1, 6, 2, 8)
    out = reference_attention(q, k, v, causal=False)
    k_rep = k.repeat_interleave(2, dim=2)  # kv head 0 -> q heads 0,1 ; kv head 1 -> q heads 2,3
    v_rep = v.repeat_interleave(2, dim=2)
    torch.testing.assert_close(out, _manual(q, k_rep, v_rep))


def test_causal_square_masks_future_keys():
    torch.manual_seed(2)
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 4, 1, 8)
    v = torch.randn(1, 4, 1, 8)
    out = reference_attention(q, k, v, causal=True)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    torch.testing.assert_close(out, _manual(q, k, v, mask))


def test_causal_with_shorter_query_is_bottom_right_aligned():
    # flash-attn convention: query i (of Lq) may see keys j <= Lkv - Lq + i
    torch.manual_seed(3)
    q = torch.randn(1, 2, 1, 8)
    k = torch.randn(1, 4, 1, 8)
    v = torch.randn(1, 4, 1, 8)
    out = reference_attention(q, k, v, causal=True)
    mask = torch.tensor([[1, 1, 1, 0],
                         [1, 1, 1, 1]], dtype=torch.bool)
    torch.testing.assert_close(out, _manual(q, k, v, mask))


def test_decode_single_query_causal_equals_non_causal():
    torch.manual_seed(4)
    q = torch.randn(3, 1, 8, 16)
    k = torch.randn(3, 32, 8, 16)
    v = torch.randn(3, 32, 8, 16)
    torch.testing.assert_close(
        reference_attention(q, k, v, causal=True),
        reference_attention(q, k, v, causal=False),
    )
