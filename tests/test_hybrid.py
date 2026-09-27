from pathlib import Path

import pytest

from kernelscope.model.hybrid import hybrid_pick, table_neighbor, variant_for_splits
from kernelscope.workload import Workload

REPO = Path(__file__).resolve().parents[1]


def test_hybrid_takes_the_table_pick_when_the_model_predicts_it_close_to_its_best():
    predicted = {"fa2_paged": 19.5, "fd_s4_paged": 21.0, "fd_s8_paged": 25.0}
    assert hybrid_pick(predicted, "fd_s4_paged", delta=0.10) == "fd_s4_paged"     # 21.0 <= 1.1 * 19.5
    assert hybrid_pick(predicted, "fd_s8_paged", delta=0.10) == "fa2_paged"       # 25.0 >  1.1 * 19.5
    assert hybrid_pick(predicted, None, delta=0.10) == "fa2_paged"
    assert hybrid_pick(predicted, "fd_s2_paged", delta=0.10) == "fa2_paged"       # not a candidate


def test_hybrid_with_zero_delta_is_the_model_and_with_infinite_delta_is_the_table():
    predicted = {"fa2_paged": 19.5, "fd_s4_paged": 21.0}
    assert hybrid_pick(predicted, "fd_s4_paged", delta=0.0) == "fa2_paged"
    assert hybrid_pick(predicted, "fd_s4_paged", delta=float("inf")) == "fd_s4_paged"


def test_table_neighbor_uses_the_nearest_measured_shape_with_the_same_heads():
    def w(key):
        return Workload.from_key(key)
    table = [(w("decode_B1_Lq1_Lkv512_Hq32_Hkv8_d128_float16_causal"), "fd_s4_paged"),
             (w("decode_B4_Lq1_Lkv2048_Hq32_Hkv8_d128_float16_causal"), "fd_s2_paged"),
             (w("decode_B1_Lq1_Lkv512_Hq28_Hkv4_d128_float16_causal"), "fa2_paged")]
    variant, distance = table_neighbor(w("decode_B2_Lq1_Lkv512_Hq32_Hkv8_d128_float16_causal"), table)
    assert (variant, distance) == ("fd_s4_paged", pytest.approx(1.0))            # one doubling of B away
    assert table_neighbor(w("decode_B2_Lq1_Lkv512_Hq16_Hkv2_d128_float16_causal"), table) == (None, None)


def test_variant_for_splits_matches_the_serving_convention():
    assert [variant_for_splits(n) for n in (0, 1, 8)] == ["flashdecoding_paged", "fa2_paged", "fd_s8_paged"]


def test_hybrid_policy_composes_model_and_table_on_cpu():
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    from kernelscope.serve.dispatch import HybridPolicy, ModelPolicy, TablePolicy, make_policy
    machine = MachineSpec.from_json(REPO / "machines" / "rtx4090.json")
    params = ModelParams.from_json(REPO / "models" / "rtx4090.json")
    table = REPO / "demo_data" / "dispatch_paged_cold.csv"
    lens = [32768] + [512] * 31
    model_choice = ModelPolicy(machine, params).choose(lens, 32, 8)
    table_choice = TablePolicy(table).choose(lens, 32, 8)
    everything = HybridPolicy(machine, params, table, delta=float("inf"))
    nothing = HybridPolicy(machine, params, table, delta=0.0)
    assert everything.choose(lens, 32, 8) == table_choice
    assert nothing.choose(lens, 32, 8) == model_choice
    assert everything.last["source"] == "table" and nothing.last["source"] == "model"
    p = make_policy(f"hybrid:{table}:0.1", machine=machine, params=params)
    assert p.name == "hybrid" and p.delta == 0.1


def test_hybrid_policy_falls_back_to_the_model_when_the_table_lacks_the_head_shape():
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    from kernelscope.serve.dispatch import HybridPolicy, ModelPolicy
    machine = MachineSpec.from_json(REPO / "machines" / "rtx4090.json")
    params = ModelParams.from_json(REPO / "models" / "rtx4090.json")
    lens = [512] * 4
    hybrid = HybridPolicy(machine, params, REPO / "demo_data" / "dispatch_paged_cold.csv", delta=0.2)
    assert hybrid.choose(lens, 16, 2) == ModelPolicy(machine, params).choose(lens, 16, 2)   # Hq=16/Hkv=2 not in the table
    assert hybrid.last["source"] == "model" and hybrid.last["table"] is None
