"""Pure-torch reference attention in dense ``[B, L, H, d]`` layout.

Conventions follow flash-attn: GQA query head ``h`` reads kv head ``h // (H_q // H_kv)``;
causal masks with ``L_q < L_kv`` are bottom-right aligned (query ``i`` sees keys
``j <= L_kv - L_q + i``). Computed in float32, returned in the query dtype.
``kv_lens`` (a ragged batch) masks out keys at positions ``>= kv_lens[b]`` for each
batch ``b``, and the bottom-right causal alignment is then relative to that sequence's
own length rather than to ``L_kv``.
"""
import torch


def reference_attention(q, k, v, causal: bool, kv_lens=None):
    B, L_q, H_q, d = q.shape
    L_kv, H_kv = k.shape[1], k.shape[2]
    group = H_q // H_kv
    kf = k.float().repeat_interleave(group, dim=2)
    vf = v.float().repeat_interleave(group, dim=2)
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kf) * d ** -0.5
    lens = torch.as_tensor(list(kv_lens) if kv_lens is not None else [L_kv] * B, device=q.device).view(-1, 1, 1, 1)
    j = torch.arange(L_kv, device=q.device).view(1, 1, 1, -1)
    s = s.masked_fill(j >= lens, float("-inf"))
    if causal:
        i = torch.arange(L_q, device=q.device).view(1, 1, -1, 1)
        s = s.masked_fill(j > (lens - L_q) + i, float("-inf"))
    p = torch.softmax(s, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", p, vf)
    return out.to(q.dtype)
