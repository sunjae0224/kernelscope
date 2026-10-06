"""Run: python -m streamlit run dashboard/app.py --server.address 127.0.0.1

Two pages: Demo (fixed-order scenes for the presentation, the default) and Lab (the five
exploration tabs for Q&A drill-down). Streamlit executes this file by path; a source checkout
need not be installed."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="KernelScope · GPU Attention Observatory", page_icon="◈", layout="wide")
pages = [st.Page("demo_page.py", title="Demo", default=True), st.Page("lab_page.py", title="Lab")]
st.navigation(pages).run()
