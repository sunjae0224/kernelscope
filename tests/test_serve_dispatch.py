import pandas as pd
import pytest

from kernelscope.model.machine import MachineSpec
from kernelscope.model.params import SPIKE_DEFAULTS
from kernelscope.serve.dispatch import FixedPolicy, ModelPolicy, TablePolicy, make_policy


def test_fixed_and_invalid_specs():
    assert make_policy("heuristic").choose([1024], 32, 8) == 0
    assert make_policy("fixed:8").name == "fixed8"
    for value in (-1, 129, 1.5, True):
        with pytest.raises(ValueError):
            FixedPolicy(value)
    with pytest.raises(ValueError, match="requires"):
        make_policy("model")


def test_table_filters_shape_and_caches_page_configuration(tmp_path):
    path = tmp_path / "table.csv"
    pd.DataFrame([
        {"workload_key": "decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal", "best_kernel": "fd_s4_paged"},
        {"workload_key": "decode_B1_Lq1_Lkv1024_Hq8_Hkv2_d128_float16_causal", "best_kernel": "fd_s16_paged"},
        {"workload_key": "decode_B32_Lq1_Lkv32768x2+1024x30_Hq32_Hkv8_d128_float16_causal", "best_kernel": "fd_s8_paged"}
    ]).to_csv(path, index=False)
    policy = TablePolicy(path)
    assert policy.choose([1000], 32, 8) == 4
    assert not policy.last_cache_hit
    assert policy.choose([1001], 32, 8) == 4 and policy.last_cache_hit
    assert policy.choose([1000], 8, 2) == 16
    assert policy.choose([32768] * 2 + [1024] * 30, 32, 8) == 8
    with pytest.raises(ValueError, match="no measured shape"):
        policy.choose([1000], 4, 1)


def test_table_rejects_dense_layout(tmp_path):
    path = tmp_path / "dense.csv"
    pd.DataFrame([{"workload_key": "decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal",
                   "best_kernel": "fd_s8"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="paged"):
        TablePolicy(path)


def test_model_cache_is_ordered_and_page_quantized(monkeypatch):
    from types import SimpleNamespace
    import kernelscope.serve.dispatch as dispatch
    calls = []
    def prediction(candidate, workload, *args):
        calls.append((candidate, workload.lens()))
        return SimpleNamespace(time_us=1 if candidate == "fd_s8_paged" else 2)
    monkeypatch.setattr(dispatch, "predict", prediction)
    policy = ModelPolicy(SimpleNamespace(n_sm=128), SPIKE_DEFAULTS, candidates=["fa2_paged", "fd_s8_paged"])
    assert policy.choose([32100, 1000], 32, 8) == 8
    assert policy.choose([32101, 1001], 32, 8) == 8 and len(calls) == 2
    policy.choose([1001, 32101], 32, 8)
    assert len(calls) == 4
    policy.choose([32300, 1001], 32, 8)
    assert len(calls) == 6
