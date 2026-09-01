import pandas as pd
import pytest

from kernelscope.analysis.roofline import plot_roofline, roofline_points

K1 = "decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal"
K2 = "decode_B16_Lq1_Lkv8192_Hq32_Hkv8_d128_float16_causal"


def _an(kernel, key, ai, tf, gb):
    return [{"workload_key": key, "kernel": kernel, "backend": "analytic", "metric": m, "unit": "",
             "value": v, "launch_idx": 0, "note": None}
            for m, v in (("arithmetic_intensity", ai), ("achieved_tflops", tf), ("achieved_gbps", gb))]


DF = pd.DataFrame(_an("fa2", K1, 3.98, 0.34, 85.5) + _an("flashdecoding", K1, 3.98, 1.23, 309.0)
                  + _an("fa2", K2, 3.99, 5.0, 1250.0)
                  + [{"workload_key": K1, "kernel": "fa2", "backend": "latency", "metric": "median_s", "unit": "s",
                      "value": 55e-6, "launch_idx": 0, "note": None}])
CEIL = {"hbm_gbps": 1791.0, "fp16_matmul_tflops": 260.6}


def test_roofline_points_one_per_cell_with_intensity_and_throughput():
    pts = roofline_points(DF)
    assert len(pts) == 3
    row = pts[(pts.kernel == "fa2") & (pts.workload_key == K1)].iloc[0]
    assert row["ai"] == 3.98 and row["tflops"] == 0.34 and row["gbps"] == 85.5
    assert set(pts.columns) >= {"kernel", "workload_key", "ai", "tflops", "gbps", "label"}
    assert row["label"] == "B1 L1024"


def test_plot_roofline_writes_a_png_and_returns_the_roof_at_each_point(tmp_path):
    out = tmp_path / "roofline.png"
    roof = plot_roofline(roofline_points(DF), CEIL, out)
    assert out.exists() and out.stat().st_size > 1000
    # memory roof = AI * HBM GB/s / 1000 (TFLOPS), capped at the tensor-core ceiling
    assert roof[("fa2", K1)] == pytest.approx(min(3.98 * 1791.0 / 1000, 260.6))
    assert roof[("fa2", K2)] == pytest.approx(min(3.99 * 1791.0 / 1000, 260.6))


def test_plot_roofline_without_ceilings_still_plots_points(tmp_path):
    out = tmp_path / "r.png"
    roof = plot_roofline(roofline_points(DF), None, out)
    assert out.exists() and roof == {}
