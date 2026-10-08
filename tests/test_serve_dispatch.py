from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from kernelscope.model.machine import MachineSpec
from kernelscope.model.params import SPIKE_DEFAULTS
from kernelscope.serve.dispatch import FixedPolicy, ModelPolicy, TablePolicy, make_policy
from kernelscope.workload import Workload


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


def test_flashinfer_policies_name_their_attention_backend_and_leave_num_splits_to_it():
    for spec in ("flashinfer", "flashinfer_cudacore"):
        policy = make_policy(spec)
        assert policy.name == spec and policy.attention == spec and policy.choose([5, 300], 4, 2) == 0
    assert make_policy("heuristic").attention == "fa2" and make_policy("fixed:8").attention == "fa2"


R = "decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal"     # CUDA-core is the overall best here
U = "decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal"               # tensor-core is the overall best here
F = "decode_B2_Lq1_Lkv4096_Hq32_Hkv8_d128_float16_causal"               # an FA2 split count is the overall best here
NAN = float("nan")


def _any_table(tmp_path, rows=None):
    """Rows shaped like static_default_losses: the overall best plus every candidate's own time."""
    base = [(R, "flashinfer_paged_cudacore", 270.0, "fd_s16_paged", 280.0, 650.0, 270.0),
            (U, "flashinfer_paged", 49.0, "fd_s4_paged", 50.0, 49.0, 55.0),
            (F, "fd_s8_paged", 60.0, "fd_s8_paged", 60.0, 70.0, 80.0)]
    columns = ["workload_key", "best_kernel", "best_us", "fa2_best_kernel", "fa2_best_us", "flashinfer_tc_us",
               "flashinfer_cc_us"]
    path = tmp_path / "any.csv"
    pd.DataFrame(base if rows is None else rows, columns=columns).assign(complete=True).to_csv(path, index=False)
    return path


ALL = ("fa2", "flashinfer", "flashinfer_cudacore")
LENS_R, LENS_U, LENS_F = [32768] + [512] * 31, [1024], [4096, 4096]


def test_any_table_picks_the_overall_best_backend_and_reports_it(tmp_path):
    policy = TablePolicy(_any_table(tmp_path), backends=ALL, name="table_any")
    assert policy.choose(LENS_R, 32, 8) == 0 and policy.attention == "flashinfer_cudacore"
    assert policy.last["kernel"] == "flashinfer_paged_cudacore" and policy.last["attention"] == "flashinfer_cudacore"
    assert policy.last["workload_key"] == R and policy.last["distance"] == 0 and policy.last["cost_us"] == 270.0
    assert policy.choose(LENS_U, 32, 8) == 0 and policy.attention == "flashinfer"
    assert policy.last["kernel"] == "flashinfer_paged"
    assert policy.choose(LENS_F, 32, 8) == 8 and policy.attention == "fa2" and policy.last["kernel"] == "fd_s8_paged"


def test_plan_cost_is_weighed_against_the_per_layer_kernel_gain(tmp_path):
    table = _any_table(tmp_path)
    # R: 36 layers x 10 us saved by CUDA-core = 360 us per step against the plan() call.
    cheap = TablePolicy(table, backends=ALL, plan_us=300.0)
    cheap.n_layers = 36
    assert cheap.choose(LENS_R, 32, 8) == 0 and cheap.attention == "flashinfer_cudacore"
    assert cheap.last["cost_us"] == pytest.approx(36 * 270 + 300)
    dear = TablePolicy(table, backends=ALL, plan_us=400.0)
    dear.n_layers = 36
    assert dear.choose(LENS_R, 32, 8) == 16 and dear.attention == "fa2"
    assert dear.last["kernel"] == "fd_s16_paged" and dear.last["cost_us"] == pytest.approx(36 * 280)
    assert dear.choose(LENS_U, 32, 8) == 4 and dear.attention == "fa2"          # 36 x 1 us gain < 400 us plan


def test_plan_cost_without_n_layers_is_an_error_but_zero_plan_needs_none(tmp_path):
    table = _any_table(tmp_path)
    assert TablePolicy(table, backends=ALL).n_layers is None
    assert TablePolicy(table, backends=ALL).choose(LENS_R, 32, 8) == 0
    with pytest.raises(ValueError, match="n_layers"):
        TablePolicy(table, backends=ALL, plan_us=100.0).choose(LENS_R, 32, 8)


def test_fa2_only_policy_reads_fa2_best_kernel_from_an_any_table(tmp_path):
    policy = TablePolicy(_any_table(tmp_path))
    assert policy.name == "table" and policy.attention == "fa2"
    assert policy.choose(LENS_R, 32, 8) == 16 and policy.attention == "fa2"
    assert policy.last["kernel"] == "fd_s16_paged" and policy.last["cost_us"] == 280.0
    assert policy.choose(LENS_U, 32, 8) == 4 and policy.last["kernel"] == "fd_s4_paged"


def test_flashinfer_best_kernel_without_fa2_best_kernel_is_rejected_for_fa2(tmp_path):
    path = tmp_path / "old.csv"
    pd.DataFrame([{"workload_key": R, "best_kernel": "flashinfer_paged"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="fa2_best_kernel"):
        TablePolicy(path)


def test_candidates_without_a_time_or_in_incomplete_rows_are_left_out(tmp_path):
    table = _any_table(tmp_path, rows=[(R, "flashinfer_paged_cudacore", 270.0, "fd_s16_paged", 280.0, 650.0, NAN),
                                       (U, "flashinfer_paged", 49.0, "fd_s4_paged", 50.0, 49.0, 55.0)])
    with pytest.raises(ValueError, match="no candidate"):
        TablePolicy(table, backends=("flashinfer_cudacore",))                   # R has no CUDA-core time
    policy = TablePolicy(table, backends=ALL)
    assert policy.choose(LENS_R, 32, 8) == 16 and policy.attention == "fa2"        # no CUDA-core time, tc 650 > 280
    assert policy.choose(LENS_U, 32, 8) == 0 and policy.attention == "flashinfer"
    frame = pd.read_csv(table)
    frame.loc[frame.workload_key == U, "complete"] = False
    frame.to_csv(table, index=False)
    assert TablePolicy(table, backends=ALL).choose(LENS_U, 32, 8) == 16         # the U cell is gone; R is nearest


def test_table_policy_validates_backends_and_plan_cost(tmp_path):
    table = _any_table(tmp_path)
    for bad in ((), ("fa3",), ("fa2", "fa2")):
        with pytest.raises(ValueError, match="backends"):
            TablePolicy(table, backends=bad)
    for bad in (-1.0, float("nan"), True):
        with pytest.raises(ValueError, match="plan_us"):
            TablePolicy(table, plan_us=bad)


def test_cache_hit_restores_the_backend_of_that_page_configuration(tmp_path):
    policy = TablePolicy(_any_table(tmp_path), backends=ALL)
    assert policy.choose(LENS_R, 32, 8) == 0 and policy.attention == "flashinfer_cudacore"
    first = dict(policy.last)
    assert policy.choose(LENS_F, 32, 8) == 8 and policy.attention == "fa2"
    assert policy.choose([32768 - 1] + [511] * 31, 32, 8) == 0                     # same pages as LENS_R
    assert policy.last_cache_hit and policy.attention == "flashinfer_cudacore" and policy.last == first
    policy.reset()
    assert policy.last is None and not policy.last_cache_hit


def test_make_policy_table_any_specs(tmp_path):
    path = _any_table(tmp_path)
    plain, priced = make_policy(f"table_any:{path}"), make_policy(f"table_any:{path}:400")
    assert (plain.name, plain.plan_us, plain.backends) == ("table_any", 0.0, ALL)
    assert (priced.name, priced.plan_us, priced.backends) == ("table_any_p400", 400.0, ALL)
    assert make_policy(f"table_any:{path}:412.5").name == "table_any_p412.5"
    assert make_policy(f"table_any:{path}:0").name == "table_any_p0"
    table = make_policy(f"table:{path}")
    assert (table.name, table.plan_us, table.backends) == ("table", 0.0, ("fa2",))
    with pytest.raises(ValueError, match="table_any"):
        make_policy("nonsense")


def test_hybrid_keeps_a_fa2_only_table_even_when_given_an_any_table(tmp_path):
    from kernelscope.serve.dispatch import HybridPolicy
    hybrid = HybridPolicy(SimpleNamespace(n_sm=128), SPIKE_DEFAULTS, _any_table(tmp_path))
    assert hybrid.table.backends == ("fa2",) and hybrid.table.choose(LENS_R, 32, 8) == 16 and hybrid.attention == "fa2"


def test_bundled_any_table_with_free_plan_reproduces_its_best_kernel():
    path = Path(__file__).resolve().parents[1] / "demo_data" / "dispatch_paged_cold_any.csv"
    if not path.exists():
        pytest.skip("run scripts/package_demo.py (or --tables-only) to build the any table")
    table = pd.read_csv(path)
    best = dict(zip(table.workload_key, table.best_kernel))
    policy = TablePolicy(path, backends=ALL)
    for key in table.workload_key:
        w = Workload.from_key(key)
        policy.choose(list(w.lens()), w.H_q, w.H_kv)
        assert policy.last["kernel"] == best[policy.last["workload_key"]]
