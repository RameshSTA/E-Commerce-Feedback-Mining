# app/page_content/page_settings.py
from __future__ import annotations
import streamlit as st

def render_settings():
    st.subheader("Settings")
    st.caption("Basic app preferences and info.")
    st.toggle("Dark mode (app theme)", value=False, help="Set Streamlit theme in .streamlit/config.toml for global behavior.")
    st.text_input("Contact email", "ops@yourcompany.com")
    st.caption("Changes here are illustrative; persist to a config store in production.")