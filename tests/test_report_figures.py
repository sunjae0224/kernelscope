from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from scripts.report_figures import FIGURES, MAX_HEIGHT_IN, MAX_WIDTH_IN, render_all  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def test_every_report_figure_renders_from_the_committed_evidence_within_half_a_page(tmp_path):
    written = render_all(REPO, tmp_path, font=None)
    assert [p.stem for p in written] == list(FIGURES)
    for path in written:
        assert path.stat().st_size > 10_000, path
    for name, build in FIGURES.items():
        fig = build(REPO)
        w, h = fig.get_size_inches()
        assert w <= MAX_WIDTH_IN and h <= MAX_HEIGHT_IN, (name, w, h)
        matplotlib.pyplot.close(fig)
