import numpy as np
import pytest

from kernelscope.model.geometry import (Variant, build_launch, capacity, check_scope, num_splits_heuristic,
                                        parse_variant, resolve_splits)
from kernelscope.workload import Workload


def _w(lens, H_q=32, H_kv=8):
    return Workload(phase="decode", B=len(lens), L_q=1, L_kv=max(lens), H_q=H_q, H_kv=H_kv, d=128, kv_lens=lens)


def test_parse_variant_covers_the_flash_family():
    assert parse_variant("fa2") == Variant("fa2", 1, False)
    assert parse_variant("flashdecoding") == Variant("flashdecoding", 0, False)
    assert parse_variant("fd_s16") == Variant("fd_s16", 16, False)
    assert parse_variant("fa2_paged") == Variant("fa2_paged", 1, True)
    assert parse_variant("flashdecoding_paged") == Variant("flashdecoding_paged", 0, True)
    assert parse_variant("fd_s4_paged") == Variant("fd_s4_paged", 4, True)
    with pytest.raises(ValueError):
        parse_variant("sdpa_flash")


@pytest.mark.parametrize("bnh, nblocks, expected", [
    (8, 8, 8),        # B1 L1K: 64 CTAs (spec STATUS)
    (8, 64, 32),      # B1 L8K: 256 CTAs (gate report)
    (128, 8, 2),      # B16 L1K
    (256, 256, 1),    # B32: at/above 0.8 * 256 -> no split (spec F15)
    (4, 64, 64),      # S2 B1 L8K
])
def test_heuristic_port_matches_measured_split_counts(bnh, nblocks, expected):
    assert num_splits_heuristic(bnh, 256, nblocks, 128) == expected


def test_paged_capacity_rounds_up_to_whole_pages():
    w = _w([1000, 300])
    assert capacity(w, paged=False) == 1000
    assert capacity(w, paged=True) == 1024


def test_resolve_splits_uses_the_heuristic_only_for_num_splits_zero():
    w = _w([8192])
    assert resolve_splits(parse_variant("flashdecoding"), w, 128) == 32
    assert resolve_splits(parse_variant("fd_s4"), w, 128) == 4
    assert resolve_splits(parse_variant("flashdecoding"), w, 64) == 16       # fewer SMs, fewer splits


def test_nonsplit_launch_orders_blocks_batch_fastest():
    L = build_launch(_w([100, 7], H_kv=2, H_q=4), parse_variant("fa2"), 128)
    assert L.kind == "nonsplit" and L.splits == 1
    assert list(L.keys) == [100, 7, 100, 7]          # linear id = b + B*h


def test_split_launch_follows_capacity_based_ranges():
    L = build_launch(_w([1000, 200], H_kv=1, H_q=1), parse_variant("fd_s4"), 128)
    # n_blocks = ceil(1000/128) = 8, per split = 2 blocks = 256 keys; linear id = s + S*(b*H_kv + h)
    assert L.kind == "split" and L.splits == 4
    assert list(L.keys) == [256, 256, 256, 232, 200, 0, 0, 0]


def test_paged_variants_always_use_the_split_kernel():
    L = build_launch(_w([512, 512]), parse_variant("fa2_paged"), 128)
    assert L.kind == "split_paged" and L.splits == 1 and len(L.keys) == 16


def test_scope_is_enforced():
    with pytest.raises(ValueError, match="d=128"):
        check_scope(Workload(phase="decode", B=1, L_q=1, L_kv=64, H_q=8, H_kv=2, d=64))
    with pytest.raises(ValueError, match="decode"):
        check_scope(Workload(phase="prefill", B=1, L_q=64, L_kv=64, H_q=8, H_kv=2, d=128))
