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
