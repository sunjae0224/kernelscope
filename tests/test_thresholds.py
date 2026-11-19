import json

import pytest

from kernelscope.analysis.thresholds import launch_agreement, no_split_batch, threshold_table
from kernelscope.model.geometry import num_splits_heuristic


def test_rtx4090_qwen3_threshold_is_26_and_matches_the_ported_heuristic():
    assert no_split_batch(128, 8, 32) == 26
    assert num_splits_heuristic(26 * 8, 2 * 128, 256) == 1          # 32K keys = 256 n-blocks
    assert num_splits_heuristic(25 * 8, 2 * 128, 256) > 1
    assert no_split_batch(128, 4, 28) == 52                         # the S2 head grid


def test_threshold_scales_with_sms_and_inverse_kv_heads():
    assert no_split_batch(108, 8) == 22 and no_split_batch(132, 8) == 27
    assert no_split_batch(128, 1) == 205
    assert no_split_batch(128, 2, 2 * 65) == 52                     # 65 query heads per KV head need 2 m-blocks
    with pytest.raises(ValueError):
        no_split_batch(0, 8)


def test_threshold_table_reads_gpus_and_kv_heads():
    spec = {"kv_heads_per_gpu": [8, 4], "gpus": [{"name": "a", "sm": 128}, {"name": "b", "sm": 58}]}
    assert threshold_table(spec) == {"a": {8: 26, 4: 52}, "b": {8: 12, 4: 24}}


def test_launch_agreement_counts_split_runs_by_launches(tmp_path):
    rows = [{"status": "ok", "plugin": "flashdecoding_paged", "launches": 1,
             "workload_key": "decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal"},     # B >= 26: no split
            {"status": "ok", "plugin": "flashdecoding_paged", "launches": 2,
             "workload_key": "decode_B16_Lq1_Lkv8192+512x15_Hq28_Hkv4_d128_float16_causal"},      # B < 52: split
            {"status": "ok", "plugin": "fa2_paged", "launches": 1,
             "workload_key": "decode_B16_Lq1_Lkv8192+512x15_Hq28_Hkv4_d128_float16_causal"}]      # not the heuristic
    path = tmp_path / "summaries.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = launch_agreement([path])
    assert got["rows"] == 2 and got["agree"] == 2 and got["split"] == 1 and got["head_shapes"] == [(28, 4), (32, 8)]
