"""What a flash-attn decode variant launches: kernel kind, split count, and each CTA's key count
in hardware linear block order (sources: flash_api.cpp set_params_splitkv / num_splits_heuristic,
flash_fwd_launch_template.h grid shapes, flash_fwd_kernel.h split ranges; spec F2, F3, F15, F17).

Decode with GQA packs a KV head's query heads into one CTA, so a split launch has
S * B * H_kv CTAs, linear id = s + S * (b * H_kv + h), and the non-split launch has B * H_kv
CTAs, linear id = b + B * h. Split ranges are sized from the cache CAPACITY (dense: L_kv; paged:
whole pages), so CTAs whose range starts past a sequence's live length do no work.
"""
import math
import re
from dataclasses import dataclass

import numpy as np

BLOCK_N = 128
PAGE = 256
KIND_RESOURCES = {"nonsplit": (128, 255, 49152), "split": (128, 244, 81920), "split_paged": (128, 244, 81920)}


@dataclass(frozen=True)
class Variant:
    name: str
    num_splits: int          # 0 = the library heuristic
    paged: bool


def parse_variant(name: str) -> Variant:
    paged = name.endswith("_paged")
    base = name[: -len("_paged")] if paged else name
    if base == "fa2":
        return Variant(name, 1, paged)
    if base == "flashdecoding":
        return Variant(name, 0, paged)
    m = re.fullmatch(r"fd_s(\d+)", base)
    if m:
        return Variant(name, int(m.group(1)), paged)
    raise ValueError(f"{name!r} is not a flash-attn decode variant the model covers")


def check_scope(w) -> None:
    if w.phase != "decode" or w.L_q != 1:
        raise ValueError("the model covers decode with L_q=1 only")
    if w.d != 128 or w.dtype not in ("float16", "bfloat16"):
        raise ValueError(f"the model is calibrated for d=128 fp16/bf16 only, got d={w.d} {w.dtype}")


def num_splits_heuristic(batch_nheads_mblocks: int, num_sms: int, num_n_blocks: int, max_splits: int = 128) -> int:
    """Exact port of flash-attn 2.8.3 num_splits_heuristic (flash_api.cpp)."""
    if batch_nheads_mblocks >= 0.8 * num_sms:
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    eff = [0.0] * (max_splits + 1)
    best = 0.0
    for s in range(1, max_splits + 1):
        if s > 1 and math.ceil(num_n_blocks / s) == math.ceil(num_n_blocks / (s - 1)):
            continue
        n_waves = batch_nheads_mblocks * s / num_sms
        eff[s] = n_waves / math.ceil(n_waves)
        best = max(best, eff[s])
    for s in range(1, max_splits + 1):
        if (s == 1 or math.ceil(num_n_blocks / s) != math.ceil(num_n_blocks / (s - 1))) and eff[s] >= 0.85 * best:
            return s
    return 1


def capacity(w, paged: bool) -> int:
    return math.ceil(w.L_kv / PAGE) * PAGE if paged else w.L_kv


def resolve_splits(v: Variant, w, n_sm: int) -> int:
    if v.num_splits > 0:
        return v.num_splits
    m_blocks = math.ceil((w.H_q // w.H_kv) / 64)       # the GQA-packed query rows, 64 per m-block
    return num_splits_heuristic(w.B * w.H_kv * m_blocks, 2 * n_sm, math.ceil(capacity(w, v.paged) / BLOCK_N), 128)


@dataclass(frozen=True)
class Launch:
    kind: str                # "nonsplit" | "split" | "split_paged"
    splits: int
    keys: np.ndarray         # keys per CTA, linear block order


def build_launch(w, v: Variant, n_sm: int) -> Launch:
    check_scope(w)
    lens = np.asarray(w.lens(), dtype=np.int64)
    S = resolve_splits(v, w, n_sm)
    if not v.paged and S == 1:
        ids = np.arange(w.B * w.H_kv)
        return Launch("nonsplit", 1, lens[ids % w.B])
    per_split = math.ceil(math.ceil(capacity(w, v.paged) / BLOCK_N) / S) * BLOCK_N
    ids = np.arange(S * w.B * w.H_kv)
    s = ids % S
    b = (ids // S) // w.H_kv
    lo = s * per_split
    keys = np.clip(np.minimum(lens[b], lo + per_split) - lo, 0, None)
    return Launch("split_paged" if v.paged else "split", S, keys)
