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
