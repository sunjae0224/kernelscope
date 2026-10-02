import pytest

pytest.importorskip("matplotlib")

from kernelscope.diagnose.figures import op_breakdown
from kernelscope.diagnose.opmodel import OP_CLASSES


def _summary():
    def policy(scale):
        ops = [{"op_class": c, "gpu_us": scale * (i + 1) * 10.0, "share": 0.05 * (i + 1), "verdict": "below_ceiling_unknown"}
               for i, c in enumerate(OP_CLASSES)]
        return {"ops": ops, "step_waterfall": {"step_us_mean": scale * 500.0}}
    return {"policies": {"heuristic": policy(1.0), "table": policy(0.5)}}


def test_renders_png_and_svg(tmp_path):
    png = op_breakdown(_summary(), tmp_path / "b.png", tmp_path / "b.svg")
    assert png.exists() and png.stat().st_size > 1000 and (tmp_path / "b.svg").stat().st_size > 1000


def test_svg_is_optional(tmp_path):
    assert op_breakdown(_summary(), tmp_path / "only.png").exists()
