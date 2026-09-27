"""Consistent, entity-based chart colors and accessible chart surfaces."""
import math

CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
TEXT = {"light": ("#0b0b0b", "#52514e"), "dark": ("#ffffff", "#c3c2b7")}
GRID = {"light": "#e8e7e3", "dark": "#2e2e2b"}
_SLOT = {"heuristic": 0, "model": 1, "table": 2, "fa2": 3}
_DASH = ["dot", "dash", "dashdot", "longdash", "longdashdot"]


def policy_color(policy: str, theme: str = "light") -> str:
    return CATEGORICAL[theme][_SLOT.get(policy, 6)]


def policy_dash(policy: str) -> str:
    suffix = policy.removeprefix("fixed")
    if policy.startswith("fixed") and suffix.isdigit() and int(suffix) > 0:
        return _DASH[int(math.log2(int(suffix))) % len(_DASH)]
    return "solid"


def layout(fig, theme: str, title: str, left: int = 45, right: int = 35):
    ink, muted = TEXT[theme]
    fig.update_layout(
        title={"text": title, "font": {"color": ink, "size": 16}},
        paper_bgcolor=SURFACE[theme], plot_bgcolor=SURFACE[theme],
        font={"color": muted, "family": "Inter, Noto Sans KR, sans-serif", "size": 12},
        legend={"orientation": "h", "y": 1.13, "font": {"color": ink}},
        margin={"l": left, "r": right, "t": 90, "b": 45}, bargap=0.25,
        hoverlabel={"bgcolor": SURFACE[theme], "font": {"color": ink}},
    )
    fig.update_xaxes(gridcolor=GRID[theme], zeroline=False, linecolor=GRID[theme], automargin=True)
    fig.update_yaxes(gridcolor=GRID[theme], zeroline=False, linecolor=GRID[theme], automargin=True)
    return fig
