import json

import numpy as np
import pytest

from kernelscope.backends.realhw.kprofile import blocks_per_sm_limit
from kernelscope.model.machine import MachineSpec

SPEC = {
    "name": "NVIDIA GeForce RTX 4090", "cc": "8.9", "n_sm": 128, "max_threads_sm": 1536, "max_ctas_sm": 24,
    "regs_sm": 65536, "smem_sm": 102400, "reserved_smem_per_block": 1024, "l2_bytes": 75497472,
    "dram_gbps": 952.6, "l2_gbps": 4561.0, "cta_dram_gbps": 26.0, "cta_l2_gbps": 46.4,
    "l2_hit_curve": [{"mib": 16, "gbps": 4850, "hit": 1.0}, {"mib": 72, "gbps": 4793, "hit": 1.0},
                     {"mib": 96, "gbps": 3866, "hit": 0.754}, {"mib": 128, "gbps": 953, "hit": 0.0}],
    "tc_tflops": 168.6, "clock_mhz": 3105.0, "block_placement": None, "provenance": {},
}


@pytest.fixture
def m(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(SPEC))
    return MachineSpec.from_json(p)


def test_loads_the_measured_fields(m):
    assert m.n_sm == 128 and m.max_ctas_sm == 24 and m.smem_sm == 102400
    assert m.dram_gbps == 952.6 and m.l2_gbps == 4561.0
    assert m.l2_curve[0] == (16 * 2**20, 1.0) and m.l2_curve_ref_bytes == 75497472


def test_props_feed_the_occupancy_calculator(m):
    assert blocks_per_sm_limit(128, 244, 81920, m.props()) == (1, "smem")
    assert blocks_per_sm_limit(128, 255, 49152, m.props()) == (2, "regs")


def test_sm_order_puts_even_sms_first(m):
    o = m.sm_order()
    assert list(o[:4]) == [0, 2, 4, 6] and list(o[64:66]) == [1, 3] and sorted(o) == list(range(128))


def test_l2_hit_interpolates_and_clamps(m):
    assert m.l2_hit(1 * 2**20) == 1.0
    assert m.l2_hit(72 * 2**20) == 1.0
    assert m.l2_hit(84 * 2**20) == pytest.approx((1.0 + 0.754) / 2)
    assert m.l2_hit(1 << 30) == 0.0


def test_scaling_changes_one_resource_each(m):
    half = m.scaled(sm=0.5)
    assert half.n_sm == 64 and list(half.sm_order()[:2]) == [0, 2]
    assert m.scaled(dram=2).dram_gbps == pytest.approx(1905.2)
    big = m.scaled(l2=2)
    assert big.l2_bytes == 2 * m.l2_bytes
    assert big.l2_hit(144 * 2**20) == 1.0        # the curve scales with capacity
    assert m.scaled(smem=1.64).smem_sm == 167936
    assert blocks_per_sm_limit(128, 244, 81920, m.scaled(smem=1.64).props())[0] == 2
    assert "sm=0.5" in half.name


def test_tensor_core_ceiling_and_ridge(m):
    assert m.tc_tflops == 168.6
    assert m.ridge_flop_per_byte() == pytest.approx(168.6e12 / 952.6e9)
    assert m.scaled(dram=0.5).ridge_flop_per_byte() == pytest.approx(2 * m.ridge_flop_per_byte())


def test_missing_tensor_core_ceiling_is_none_and_ridge_refuses(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({k: v for k, v in SPEC.items() if k != "tc_tflops"}))
    m = MachineSpec.from_json(p)
    assert m.tc_tflops is None
    with pytest.raises(ValueError, match="tc_tflops"):
        m.ridge_flop_per_byte()
