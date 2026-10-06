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


PAGE_CSS = """<style>
  .block-container { max-width: 1440px; padding-top: 2.4rem; padding-bottom: 3rem; }
  h1 { letter-spacing: -0.065em; font-weight: 750 !important; }
  h2,h3 { letter-spacing: -0.025em; }
  [data-testid="stMetric"] { border: 1px solid #b8bfcc55; border-radius: 14px; padding: 17px 20px; }
  [data-testid="stMetricLabel"] { font-size: .84rem; }
  [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
  [data-testid="stTabs"] button { font-weight: 650; padding-left: 18px; padding-right: 18px; }
  .eyebrow { font-size: .73rem; letter-spacing: .18em; font-weight: 750; color: #56729b; margin-bottom: 10px; }
  .hero-note { max-width: 850px; font-size: 1.07rem; line-height: 1.7; opacity: .78; margin-bottom: 22px; }
  .evidence { display: inline-block; border: 1px solid #8b97aa55; border-radius: 999px;
     padding: 5px 11px; font-size: .75rem; margin: 0 6px 8px 0; }
  .footnote { font-size: .78rem; opacity: .66; line-height: 1.6; }
</style>"""


def current_theme(default: str = "light") -> str:
    """The theme the browser reports, else the configured base; ``default`` outside Streamlit.

    Streamlit may report the browser's default light scheme on the first render, before it
    receives the configured app colors, so the first call in a session returns the configured base."""
    try:
        import streamlit as st
    except ImportError:
        return default
    configured = st.get_option("theme.base") or default
    try:
        theme = st.context.theme.type or configured
    except (AttributeError, KeyError):
        theme = configured
    theme = theme if theme in ("light", "dark") else configured
    if not st.session_state.get("_dashboard_theme_initialized"):
        theme = configured
        st.session_state["_dashboard_theme_initialized"] = True
    return theme
