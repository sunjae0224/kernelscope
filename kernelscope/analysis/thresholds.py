"""The batch size from which flash-attn 2.8.3's decode heuristic stops splitting, per GPU and head shape.

``set_params_splitkv`` (csrc/flash_attn/flash_api.cpp:317) calls ``num_splits_heuristic`` with
``batch * num_heads * num_m_blocks`` CTAs and ``2 * SM`` slots, and the heuristic returns 1 as soon as
the CTAs reach 80 % of the slots. Decode with GQA first swaps the query groups into the sequence
dimension (flash_api.cpp:1278), so ``num_heads`` is the KV head count and one m-block holds up to 64
query heads per KV head. The threshold is therefore a property of (SM count, KV heads per GPU) only;
this is the library's rule evaluated, not a measurement. ``launch_agreement`` checks the rule against
recorded RTX 4090 launches: a split run launches the split kernel and the combine kernel (2 launches).
"""
import json
import math
from pathlib import Path

from kernelscope.model.geometry import parse_variant, resolve_splits
from kernelscope.workload import Workload

HEURISTIC_PLUGINS = ("flashdecoding", "flashdecoding_paged")


def no_split_batch(n_sm: int, n_kv_heads: int, n_heads: int | None = None) -> int:
    """Smallest decode batch whose B * H_kv * m-blocks reaches 0.8 * (2 * SM), the heuristic's no-split branch."""
    if n_sm < 1 or n_kv_heads < 1:
        raise ValueError("need at least one SM and one KV head")
    m_blocks = math.ceil((n_heads // n_kv_heads) / 64) if n_heads else 1
    batch = 1
    while batch * n_kv_heads * m_blocks < 0.8 * (2 * n_sm):     # the same comparison as num_splits_heuristic
        batch += 1
    return batch


def threshold_table(spec: dict) -> dict:
    """{gpu name: {KV heads per GPU: threshold batch}} for the GPUs and head counts listed in ``spec``."""
    return {g["name"]: {int(h): no_split_batch(g["sm"], int(h)) for h in spec["kv_heads_per_gpu"]} for g in spec["gpus"]}


def load_spec(path) -> dict:
    return json.loads(Path(path).read_text())


def launch_agreement(summaries, n_sm: int = 128) -> dict:
    """Recorded heuristic decode runs whose launch count (2 = split + combine) matches the ported rule."""
    rows = agree = split = 0
    heads = set()
    for path in summaries:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r.get("status") != "ok" or r.get("plugin") not in HEURISTIC_PLUGINS:
                continue
            w = Workload.from_key(r["workload_key"])
            if w.phase != "decode" or w.L_q != 1 or w.d != 128:
                continue
            predicted_split = resolve_splits(parse_variant(r["plugin"]), w, n_sm) > 1
            rows += 1
            split += predicted_split
            agree += predicted_split == (r["launches"] >= 2)
            heads.add((w.H_q, w.H_kv))
    return {"rows": rows, "agree": agree, "split": split, "no_split": rows - split, "head_shapes": sorted(heads)}
