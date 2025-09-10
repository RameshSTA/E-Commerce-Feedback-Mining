# app/main_app.py
from __future__ import annotations

# Ensure src/ is importable
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

import streamlit as st
st.set_page_config(page_title="ECommerceFeedbackAI", page_icon="🛍️", layout="wide", initial_sidebar_state="collapsed")

from src.ui.components import inject_css, page_header, top_nav, footer, safe_logo_data_url

# Robust lazy loader
import importlib
def _load_and_render(module_path: str, func_name: str, label: str) -> None:
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        st.error(f"Could not import `{module_path}` for **{label}**."); st.exception(e); return
    fn = getattr(mod, func_name, None)
    if not callable(fn):
        st.error(f"`{func_name}()` not found in `{module_path}` for **{label}**."); return
    try:
        fn()
    except Exception as e:
        st.error(f"**{label}** crashed during rendering:"); st.exception(e)

# Global CSS + header
inject_css()
logo_url = safe_logo_data_url("app/assets/logo.svg")
page_header(title="ECommerceFeedbackAI", logo=logo_url, subtitle="Mine • Understand • Act")

# Modern icon nav
labels = ["Overview", "Ingest", "Explore", "Modeling", "Monitoring", "Settings"]
icons  = ["house",   "upload", "search",  "cpu",     "activity",   "settings"]
selected = top_nav(labels, icons, default_index=0)

# Routing
if selected == "Overview":
    _load_and_render("app.page_content.page_overview", "render_overview", "Overview")
elif selected == "Ingest":
    _load_and_render("app.page_content.page_ingest", "render_ingest", "Ingest")
elif selected == "Explore":
    _load_and_render("app.page_content.page_explore", "render_explore", "Explore")
elif selected == "Modeling":
    _load_and_render("app.page_content.page_modeling", "render_modeling", "Modeling")
elif selected == "Monitoring":
    _load_and_render("app.page_content.page_monitoring", "render_monitoring", "Monitoring")
elif selected == "Settings":
    _load_and_render("app.page_content.page_settings", "render_settings", "Settings")

footer("© 2025 ECommerceFeedbackAI · Built for Product, CX & Ops")