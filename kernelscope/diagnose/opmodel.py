"""Compulsory bytes and FLOPs of one decode step (or one prefill chunk) per operation class.

Weights are read once and activations are read and written once per use, so the totals are lower
bounds on real traffic; an achieved bandwidth derived from them is a lower bound too. Attention reuses
kernelscope.analytic. No torch here: this module also runs on the GPU-less verification path.
"""
from dataclasses import dataclass

from kernelscope.analytic import DTYPE_BYTES, attention_flops, attention_traffic
from kernelscope.workload import Workload

OP_CLASSES = ("embed", "norm", "qkv_proj", "rope", "attention", "o_proj", "mlp", "lm_head")
LAYER_CLASSES = ("norm", "qkv_proj", "rope", "attention", "o_proj", "mlp")
LOGIT_BYTES = 4      # DecoderModel._logits returns float32


@dataclass(frozen=True)
class OpCost:
    weight_bytes: float
    act_bytes: float
    flops: float

    @property
    def bytes(self) -> float:
        return self.weight_bytes + self.act_bytes


def dtype_bytes(dtype: str) -> int:
    name = str(dtype).removeprefix("torch.")
    if name not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(DTYPE_BYTES)}")
    return DTYPE_BYTES[name]


def op_costs(cfg, dtype: str, phase: str, tokens: int, lens) -> dict:
    """Decode: tokens = batch size, lens = KV length of each row after the append.
    Prefill: tokens = chunk length, lens = (keys visible after the chunk,)."""
    if phase not in ("decode", "prefill"):
        raise ValueError(f"phase must be 'decode' or 'prefill', got {phase!r}")
    lens = tuple(int(n) for n in lens)
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1 or not lens or min(lens) < 1:
        raise ValueError("tokens must be a positive int and lens a nonempty tuple of positive ints")
    name = str(dtype).removeprefix("torch.")
    b = dtype_bytes(name)
    T, H, I, V, L = tokens, cfg.hidden, cfg.intermediate, cfg.vocab, cfg.n_layers
    Hq, Hkv, d = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    if phase == "decode":
        if len(lens) != T:
            raise ValueError("decode needs one KV length per batch row")
        w = Workload("decode", T, 1, max(lens), Hq, Hkv, d, name, kv_lens=lens if len(set(lens)) > 1 else None)
    else:
        if len(lens) != 1:
            raise ValueError("prefill chunk needs a single visible-key count")
        w = Workload("prefill", 1, T, lens[0], Hq, Hkv, d, name)
    qk = 2 * T * (Hq + Hkv) * d * b if cfg.qk_norm else 0
    per_layer = {
        "norm": OpCost(2 * H * b, 2 * (2 * T * H * b) + qk, 0),
        "qkv_proj": OpCost(H * (Hq + 2 * Hkv) * d * b, T * H * b + T * (Hq + 2 * Hkv) * d * b, 2 * T * H * (Hq + 2 * Hkv) * d),
        "rope": OpCost(0, 2 * T * (Hq + Hkv) * d * b, 0),
        "attention": OpCost(0, attention_traffic(w)["total_bytes"], attention_flops(w)),
        "o_proj": OpCost(Hq * d * H * b, T * Hq * d * b + 2 * T * H * b, 2 * T * Hq * d * H),
        "mlp": OpCost(3 * H * I * b, T * b * (3 * H + 6 * I), 2 * T * 3 * H * I),
    }
    costs = {"embed": OpCost(T * H * b, T * H * b, 0)}
    costs.update({k: OpCost(c.weight_bytes * L, c.act_bytes * L, c.flops * L) for k, c in per_layer.items()})
    costs["lm_head"] = OpCost(H * b + V * H * b, 2 * T * H * b + T * V * LOGIT_BYTES, 2 * T * V * H)
    return costs
