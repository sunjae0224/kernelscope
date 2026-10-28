import gzip
import json

import pytest

from scripts.vllm_profile_step import (analyze, classify_kernel, decode_rows, decompose, merged_length, parse_step_name,
                                       summarize_rows, top_kernels)


def test_step_annotation_parses_simple_and_detailed_forms():
    assert parse_step_name("execute_context_0(0)_generation_32(32)") == (0, 32)
    assert parse_step_name("execute_48640_context_2(sq48640sk48640sqsq1sqsk2)_generation_0(sq0sk0sqsq0sqsk0)") == (2, 0)
    assert parse_step_name("cudaLaunchKernel") is None


@pytest.mark.parametrize("name,expected", [
    ("void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128,...>>", "attention"),
    ("void flash::flash_fwd_splitkv_combine_kernel<...>", "attention"),
    ("void flashinfer::BatchDecodeWithPagedKVCacheKernel<...>", "attention"),
    ("void flashinfer::PersistentVariableLengthMergeStatesKernel<...>", "attention"),
    ("void vllm::reshape_and_cache_flash_kernel<...>", "kv_write"),
    ("sm80_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x64x64", "gemm"),
    ("nvjet_sm89_tst_128x256_64x4_2x1_v_bz_coopA_TNT", "gemm"),
    ("void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>", "gemm"),
    ("void vllm::rms_norm_kernel<c10::BFloat16>", "other"),
    ("void vllm::act_and_mul_kernel<...>", "other"),
])
def test_kernel_classes(name, expected):
    assert classify_kernel(name) == expected


def test_extra_attention_regex_reclassifies():
    assert classify_kernel("my_custom_kernel") == "other"
    assert classify_kernel("my_custom_kernel", extra_attention="custom") == "attention"


def test_merged_length_counts_overlap_once():
    assert merged_length([(0, 10), (5, 12), (20, 25)]) == 17
    assert merged_length([]) == 0


def _event(ph, name, ts, dur=0, cat="user_annotation", **args):
    return {"ph": ph, "name": name, "ts": ts, "dur": dur, "cat": cat, "args": args}


def _trace():
    ev = [_event("X", "execute_context_2(100)_generation_0(0)", 0, 50, "user_annotation")]
    ev += [_event("X", "execute_context_0(0)_generation_32(32)", 1000, 200), _event("X", "execute_context_0(0)_generation_32(32)", 3000, 200),
           _event("X", "execute_context_0(0)_generation_32(32)", 5000, 200)]
    for i, (launch, kstart, name, dur) in enumerate([(1010, 1100, "flash_fwd_splitkv_kernel", 400), (1020, 1520, "sm80_xmma_gemm", 300),
                                                    (1030, 1830, "rms_norm_kernel", 50), (3010, 3100, "flash_fwd_splitkv_kernel", 900)]):
        ev.append(_event("X", "cudaLaunchKernel", launch, 5, "cuda_runtime", correlation=i))
        ev.append(_event("X", name, kstart, dur, "kernel", correlation=i))
    return {"traceEvents": ev}


def test_decompose_assigns_kernels_by_launch_and_measures_host_gap():
    rows, unassigned = decompose(_trace())
    assert unassigned == 0 and [r["context_requests"] for r in rows] == [2, 0, 0, 0]
    first = rows[1]
    assert first["wall_us"] == 2000 and first["attention_us"] == 400 and first["gemm_us"] == 300 and first["other_us"] == 50
    assert first["gpu_busy_us"] == 750 and first["host_gap_us"] == 1250 and first["kernels"] == 3
    assert rows[2]["attention_us"] == 900 and rows[3]["wall_us"] is None
    assert len(decode_rows(rows)) == 2                      # the last iteration has no successor, the prefill is not decode


def test_summary_is_the_median_over_pure_decode_iterations():
    s = summarize_rows(decompose(_trace())[0])
    assert s["decode_iterations"] == 2 and s["batch"] == 32
    assert s["attention_us_median"] == 650 and s["wall_us_median"] == 2000
    assert 0 < s["gpu_busy_share_of_wall"] < 1
    assert summarize_rows([]) == {"decode_iterations": 0}


def test_top_kernels_rank_by_total_time():
    top = top_kernels(_trace(), n=2)
    assert top[0]["name"] == "flash_fwd_splitkv_kernel" and top[0]["total_us"] == 1300 and top[0]["calls"] == 2


def test_analyze_reads_traces_and_reports_ragged_over_uniform(tmp_path, capsys):
    for scenario, scale in (("ragged", 2), ("uniform", 1)):
        d = tmp_path / "traces" / f"flash_attn_eager_{scenario}"
        d.mkdir(parents=True)
        trace = _trace()
        for e in trace["traceEvents"]:
            if e["cat"] == "kernel" and "flash" in e["name"]:
                e["dur"] *= scale
        with gzip.open(d / "w.pt.trace.json.gz", "wt") as handle:
            json.dump(trace, handle)
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": {
        f"flash_attn_eager_{s}": {"backend": "FLASH_ATTN", "mode": "eager", "scenario": s} for s in ("ragged", "uniform")}}))
    result = analyze(tmp_path)
    ratio = result["ragged_over_uniform"]["FLASH_ATTN/eager"]
    assert ratio["attention_us"] == pytest.approx(2.0) and ratio["gemm_us"] == pytest.approx(1.0)
    assert (tmp_path / "step_breakdown.json").exists()


def test_kv_cache_kwargs_fixes_size_only_when_given():
    from scripts.vllm_profile_step import kv_cache_kwargs
    assert kv_cache_kwargs(None) == {}
    assert kv_cache_kwargs(8) == {"kv_cache_memory_bytes": 8 * 2**30}


def test_fa2_launches_reads_grid_split_flag_and_combine_kernels(tmp_path):
    from kernelscope.analysis.vllm_steps import fa2_launches
    traits = "void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t, " \
             "Flash_kernel_traits<128, 64, 64, 4, cutlass::bfloat16_t> >, "
    kernel = lambda flags, grid, graph, dur: {"cat": "kernel", "name": traits + flags + ">(flash::Flash_fwd_params)", "dur": dur,
                                              "args": {"grid": grid, "graph id": graph}}
    events = [kernel("false, false, false, false, true, false, false, false", [1, 32, 8], 7, 10.0),
              kernel("false, false, false, false, true, false, false, false", [1, 32, 8], 0, 30.0),
              kernel("false, false, false, false, true, false, true, false", [1, 16, 256], 0, 5.0),
              {"cat": "kernel", "name": "void flash::flash_fwd_splitkv_combine_kernel<...>", "dur": 1.0, "args": {}},
              {"cat": "kernel", "name": "_combine_sampled_and_draft_tokens_kernel", "dur": 1.0, "args": {}}]
    path = tmp_path / "t.pt.trace.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump({"traceEvents": events}, handle)
    got = fa2_launches(path)
    assert got["combine_launches"] == 1
    assert got["groups"][0] == {"grid": [1, 32, 8], "split": False, "launches": 2, "graph_launches": 1, "dur_us_median": 20.0}
    assert got["groups"][1]["split"] is True and got["groups"][1]["grid"] == [1, 16, 256]
