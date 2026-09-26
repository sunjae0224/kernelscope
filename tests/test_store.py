import pandas as pd

from kernelscope.results.store import ResultStore


ROWS = [
    {"workload_key": "decode_B1_Lq1_Lkv4096_Hq32_Hkv8_d128_float16_causal", "kernel": "fa2",
     "backend": "ncu", "metric": "gpu__time_duration.sum", "unit": "usecond", "value": 45.1,
     "launch_idx": 0},
    {"workload_key": "decode_B1_Lq1_Lkv4096_Hq32_Hkv8_d128_float16_causal", "kernel": "fa2",
     "backend": "ncu", "metric": "launch__grid_size", "unit": "", "value": 32.0,
     "launch_idx": 0},
]


def test_write_then_load_round_trips_rows(tmp_path):
    store = ResultStore(tmp_path)
    store.write(ROWS, tag="ncu_fa2")
    df = store.load()
    assert len(df) == 2
    assert set(df.columns) >= {"workload_key", "kernel", "backend", "metric", "unit", "value", "launch_idx"}
    assert df.loc[df.metric == "gpu__time_duration.sum", "value"].item() == 45.1


def test_each_write_is_its_own_file_so_parallel_writers_do_not_clobber(tmp_path):
    store = ResultStore(tmp_path)
    store.write(ROWS, tag="a")
    store.write(ROWS, tag="a")
    assert len(list(tmp_path.glob("*.parquet"))) == 2
    assert len(store.load()) == 4


def test_load_on_empty_store_returns_empty_frame_with_schema(tmp_path):
    df = ResultStore(tmp_path).load()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "workload_key" in df.columns


def test_write_stamps_extra_columns_on_every_row(tmp_path):
    store = ResultStore(tmp_path)
    store.write(ROWS, tag="x", extra={"host": "a100-box", "run_id": "r1"})
    df = store.load()
    assert (df["host"] == "a100-box").all()
    assert (df["run_id"] == "r1").all()


def test_load_dirs_reads_a_portable_summaries_only_directory(tmp_path):
    import json
    from kernelscope.results.store import load_dirs
    key = "decode_B1_Lq1_Lkv4096_Hq32_Hkv8_d128_float16_causal"
    lines = [
        json.dumps({"status": "ok", "plugin": "fa2", "workload_key": key, "cache_state": "cold", "kernel_time_us": 40.0}),
        json.dumps({"status": "error", "plugin": "fd_s2", "workload_key": key, "cache_state": "cold"}),
        '{"status": "ok", "plugin": "fd_s4"',  # truncated trailing line from an interrupted writer
    ]
    (tmp_path / "summaries.jsonl").write_text("\n".join(lines) + "\n")
    df = load_dirs([tmp_path])
    assert len(df) == 1
    row = df.iloc[0]
    assert (row.kernel, row.backend, row.metric, row.value, row.cache_state) == (
        "fa2", "profile", "kernel_time_us", 40.0, "cold")


def test_load_dirs_prefers_parquet_rows_when_present(tmp_path):
    import json
    from kernelscope.results.store import load_dirs
    ResultStore(tmp_path).write(ROWS, tag="ncu_fa2")
    (tmp_path / "summaries.jsonl").write_text(json.dumps(
        {"status": "ok", "plugin": "fa2", "workload_key": ROWS[0]["workload_key"], "kernel_time_us": 40.0}) + "\n")
    assert len(load_dirs([tmp_path])) == len(ROWS)
