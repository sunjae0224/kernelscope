import pandas as pd
import pytest

pytest.importorskip("plotly")
from kernelscope.dashboard import figures, style


def test_heatmap_units_winners_and_missing_cells():
    rows = pd.DataFrame([
        dict(B=1, L_kv=1024, ragged=False, heuristic_regret=.5, best_kernel="fd_s8", best_us=10., heuristic_us=15.),
        dict(B=2, L_kv=2048, ragged=False, heuristic_regret=0., best_kernel="fa2", best_us=20., heuristic_us=20.),
    ])
    fig = figures.regret_heatmap(rows)
    assert fig.data[0].text[0][0] == "s8"
    assert fig.data[0].text[0][1] == ""
    assert fig.data[0].zmin == 0
    assert "µs" in fig.data[0].hovertemplate
    with pytest.raises(ValueError, match="geometry"):
        figures.regret_heatmap(pd.concat([rows, rows]))


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_serving_series_keep_policy_identity(theme):
    serving = {p: {"steps": pd.DataFrame({"step": [0, 1], "attn_us": [1000., 2000.]})} for p in ["model", "heuristic"]}
    fig = figures.serving_steps(serving, theme)
    assert fig.data[0].line.color == style.policy_color("model", theme)
    assert fig.data[1].line.color == style.policy_color("heuristic", theme)
    assert list(fig.data[0].y) == [1., 2.]
    assert fig.layout.paper_bgcolor == style.SURFACE[theme]
    assert fig.layout.yaxis.title.text == "Time (ms)"


def test_hypothetical_base_is_explicitly_predicted():
    frame = pd.DataFrame({"plugin": ["fa2"], "time_us": [10.]})
    fig = figures.whatif_compare(frame, frame)
    assert all("predicted" in trace.name for trace in fig.data)
    fig = figures.variants_bar(pd.DataFrame(), pd.Series({"fa2": 10.}), "fa2")
    assert fig.data[0].name == "GPU measured"
    assert fig.data[0].y[0].endswith("← library heuristic")


def test_cta_work_curves_and_backend_bar():
    from kernelscope.dashboard import figures
    fig = figures.cta_work_curves([{"label": "a", "color": "#2a78d6", "keys": [5, 0, 3, 1]},
                                   {"label": "b", "color": "#1baf7a", "keys": [2, 2, 2, 2]}], "light")
    assert [trace.name for trace in fig.data] == ["a", "b"]
    assert list(fig.data[0].y) == [5, 3, 1]                      # zero-work CTAs are dropped, sorted busiest first
    row = {"heuristic_us": 1071.8, "best_fa2_kernel": "fd_s16_paged", "best_fa2_us": 279.4,
           "flashinfer_tensorcore_us": 670.9, "flashinfer_cudacore_us": float("nan")}
    bar = figures.backend_bar(row, "dark")
    assert len(bar.data) == 1 and len(bar.data[0].x) == 3        # the NaN backend is left out
    assert bar.layout.paper_bgcolor == "#1a1a19"
