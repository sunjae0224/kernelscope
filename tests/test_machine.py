import pytest

from kernelscope.bench import machine
from kernelscope.bench.stream import num_chunks


def test_num_chunks_refuses_a_single_chunk_per_program():
    assert num_chunks(1 << 20, G=128) == 8                   # 1 Mi elems / (1024 * 128)
    with pytest.raises(ValueError, match="hoist"):
        num_chunks(1024 * 128, G=128)


def test_hit_fraction_inverts_concurrent_l2_and_dram_service():
    assert machine.hit_fraction(1000.0, 1000.0, 5000.0) == 0.0
    assert machine.hit_fraction(5000.0, 1000.0, 5000.0) == 1.0
    assert machine.hit_fraction(2000.0, 1000.0, 5000.0) == pytest.approx(0.5)   # misses half the bytes at a 1000 GB/s DRAM limit -> 2000 GB/s
    assert machine.hit_fraction(9000.0, 1000.0, 5000.0) == 1.0          # clipped


def test_measure_machine_assembles_the_spec_from_injected_measurements(monkeypatch):
    from kernelscope.backends.realhw.kprofile import RTX4090_PROPS
    monkeypatch.setattr(machine, "props_from_torch", lambda device: dict(RTX4090_PROPS))

    def fake_stream(nbytes, G, **kw):
        if G == 1:
            return 50.0 if nbytes <= 8 << 20 else 27.0
        return 4800.0 if nbytes <= 36 << 20 else (1500.0 if nbytes <= 128 << 20 else 958.0)

    spec = machine.measure_machine("cuda", l2_curve_mib=(16, 512), stream=fake_stream,
                                   tc_tflops=lambda: 150.0, clocks=lambda: (2520.0, 10501.0), placement=None)
    assert spec["n_sm"] == 128 and spec["max_ctas_sm"] == 24 and spec["smem_sm"] == 102400
    assert spec["dram_gbps"] == 958.0 and spec["l2_gbps"] == 4800.0
    assert spec["cta_dram_gbps"] == 27.0 and spec["cta_l2_gbps"] == 50.0
    assert [p["mib"] for p in spec["l2_hit_curve"]] == [16, 512]
    assert spec["l2_hit_curve"][0]["hit"] == 1.0 and spec["l2_hit_curve"][1]["hit"] == 0.0
    assert spec["tc_tflops"] == 150.0 and spec["clock_mhz"] == 2520.0
    assert spec["block_placement"] is None
    assert "date" in spec["provenance"]


@pytest.mark.gpu
def test_stream_bandwidths_on_the_4090():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from kernelscope.bench.stream import stream_gbps
    dram = stream_gbps(1 << 30, 512)
    l2 = stream_gbps(36 << 20, 128)
    assert 800 < dram < 1100, dram            # rated 1008 GB/s; probes measured 958
    assert l2 > 2.5 * dram, (l2, dram)
    assert stream_gbps(512 << 20, 256) < 1.3 * dram
