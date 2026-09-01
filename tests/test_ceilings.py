import json

from kernelscope.bench.ceilings import DEFAULT_COPY_BYTES, gbps, measure_ceilings, tflops


def test_unit_helpers():
    assert gbps(1e9, 1.0) == 1.0
    assert gbps(2e9, 0.5) == 4.0
    assert tflops(1e12, 1.0) == 1.0


def test_default_copy_stays_under_torch_32bit_indexing_limit():
    # > 2^31 bytes per tensor makes torch fall back to 64-bit indexing kernels (~2x slower)
    assert DEFAULT_COPY_BYTES <= 1 << 30


def test_measure_ceilings_runs_on_cpu_with_tiny_sizes_and_records_device(tmp_path):
    out = measure_ceilings(device="cpu", copy_bytes=1 << 20, matmul_ns=(32, 64), iters=2)
    assert out["hbm_gbps"] > 0
    assert out["fp16_matmul_tflops"] > 0
    assert out["matmul_best_n"] in (32, 64)
    assert set(out["matmul_tflops_by_n"]) == {"32", "64"}
    assert out["device"] == "cpu"
    assert out["copy_bytes"] == 1 << 20
    assert "l2" not in " ".join(out)          # no launch-bound pseudo-ceiling
    p = tmp_path / "ceilings.json"
    p.write_text(json.dumps(out))
    assert json.loads(p.read_text())["fp16_matmul_tflops"] == out["fp16_matmul_tflops"]
