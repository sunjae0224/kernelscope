import json

import pytest

from kernelscope.model.machine import MachineSpec
from kernelscope.model.params import SPIKE_DEFAULTS, KindParams, ModelParams, StateParams
from kernelscope.model.predict import DENSE_VARIANTS, bandwidth_bytes_per_us, predict, rank_variants
from kernelscope.workload import Workload

M = MachineSpec(name="t", n_sm=128, max_threads_sm=1536, max_ctas_sm=24, regs_sm=65536, smem_sm=102400,
                reserved_smem_per_block=1024, l2_bytes=72 * 2**20, dram_gbps=953.0, l2_gbps=4850.0,
                l2_curve=((16 * 2**20, 1.0), (72 * 2**20, 1.0), (96 * 2**20, 0.75), (128 * 2**20, 0.0)),
                l2_curve_ref_bytes=72 * 2**20)
K = KindParams(cost_us_per_key=0.05, t0_us=2.0, t_empty_us=0.5, gamma=0.5, t_fixed_us=1.0)
P = ModelParams(machine="t", states={s: StateParams(kinds={"nonsplit": K, "split": K, "split_paged": K},
                                                     comb_a_us=5.0, comb_b_us=0.001) for s in ("cold", "warm")})
W1 = Workload(phase="decode", B=1, L_q=1, L_kv=8192, H_q=32, H_kv=8, d=128)


def test_bandwidth_by_cache_state():
    assert bandwidth_bytes_per_us(M, "cold", 1e6) == pytest.approx(953e3)
    assert bandwidth_bytes_per_us(M, "warm", 32 * 2**20) == pytest.approx(4850e3)
    assert bandwidth_bytes_per_us(M, "warm", 96 * 2**20) == pytest.approx(953e3 / 0.25)
    assert bandwidth_bytes_per_us(M, "warm", 1 << 30) == pytest.approx(953e3)


def test_fa2_single_sequence_is_eight_lonely_ctas():
    p = predict("fa2", W1, M, P, "cold")
    assert p.kind == "nonsplit" and p.ctas == 8 and p.slots_per_sm == 2 and p.limiter == "regs"
    assert p.main_us == pytest.approx(2.0 + 8192 * 0.05 + 1.0)
    assert p.combine_us == 0.0 and p.time_us == p.main_us


def test_split_variant_adds_a_combine_cost():
    p = predict("fd_s8", W1, M, P, "cold")
    assert p.kind == "split" and p.splits == 8 and p.ctas == 64 and p.slots_per_sm == 1
    assert p.combine_us == pytest.approx(5.0 + 0.001 * 1 * 32 * 8)
    assert p.time_us < predict("fa2", W1, M, P, "cold").time_us


def test_more_shared_memory_lets_the_split_kernel_pair_up():
    assert predict("fd_s8", W1, M.scaled(smem=1.64), P, "cold").slots_per_sm == 2


def test_rank_variants_sorts_by_predicted_time():
    r = rank_variants(W1, DENSE_VARIANTS, M, P, "cold")
    assert [x.time_us for x in r] == sorted(x.time_us for x in r)
    assert r[-1].plugin == "fa2"


def test_out_of_scope_workloads_raise():
    with pytest.raises(ValueError):
        predict("fa2", Workload(phase="decode", B=1, L_q=1, L_kv=64, H_q=8, H_kv=2, d=64), M, P, "cold")


def test_params_round_trip(tmp_path):
    SPIKE_DEFAULTS.to_json(tmp_path / "p.json")
    assert ModelParams.from_json(tmp_path / "p.json") == SPIKE_DEFAULTS
    assert set(json.loads((tmp_path / "p.json").read_text())["states"]) == {"cold", "warm"}
