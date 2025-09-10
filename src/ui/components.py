# src/ui/components.py
from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable, Optional

import streamlit as st


# -----------------------------
# Small helpers
# -----------------------------
def _b64svg(svg_path: Path) -> Optional[str]:
    if not svg_path.exists():
        return None
    data = svg_path.read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode("utf-8")


def safe_logo_data_url(path: str | Path) -> Optional[str]:
    """Return data-url for an SVG/PNG logo if present; else None."""
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix.lower() == ".svg":
        return _b64svg(p)
    # PNG/JPG fallback
    data = p.read_bytes()
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode("utf-8")


# -----------------------------
# Global CSS
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
<style>
/* Layout & fonts (neutral, professional) */
html, body, [class^="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji"; }
:root {
  --ink: #0f172a;      /* very dark slate */
  --muted: #6b7280;    /* gray-500 */
  --accent: #ef4444;   /* red-500 */
  --line: #e5e7eb;     /* gray-200 */
  --hover: #f9fafb;    /* gray-50 */
}

/* Sticky header wrapper */
.efai-sticky { position: sticky; top: 0; z-index: 999; background: #ffffff; }

/* Header row */
.efai-header {
  display: flex; align-items: center; gap: 14px;
  padding: 10px 4px; border-bottom: 1px solid var(--line);
}

/* Logo + title block */
.efai-logo { width: 28px; height: 28px; display:flex; align-items:center; justify-content:center; }
.efai-title { font-weight: 800; font-size: 20px; color: var(--ink); letter-spacing: 0.2px; }

/* Top nav (pills with icons) */
.efai-nav {
  display: flex; gap: 6px; flex-wrap: wrap; padding: 6px 0 0 0; margin-bottom: 4px;
}
.efai-tab {
  border: 1px solid var(--line);
  background: #ffffff;
  padding: 8px 12px;
  border-radius: 999px;
  color: var(--ink);
  font-weight: 600;
  font-size: 13px;
  display: inline-flex; gap: 8px; align-items: center;
  cursor: pointer;
  transition: all .15s ease;
}
.efai-tab:hover { background: var(--hover); }
.efai-tab.active {
  border-color: var(--accent);
  color: var(--accent);
  box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.12);
}

/* Footer */
.efai-footer { color: var(--muted); font-size: 12px; padding: 22px 0 10px; border-top: 1px solid var(--line); margin-top: 18px; }

/* Slim down default button padding for nav buttons inside columns */
button[kind="secondary"] { padding: 0.4rem 0.75rem !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Header (logo + title + nav below)
# -----------------------------
def page_header(title: str, logo: str | None = None, subtitle: str | None = None) -> None:
    st.markdown('<div class="efai-sticky">', unsafe_allow_html=True)
    c1, c2 = st.columns([0.06, 0.94])
    with c1:
        if logo:
            st.markdown(f'<img src="{logo}" class="efai-logo" />', unsafe_allow_html=True)
        else:
            st.markdown('<div class="efai-logo">🛍️</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"""
<div class="efai-header">
  <div class="efai-title">{title}</div>
  {'<div style="color:#6b7280;font-weight:500;margin-left:6px;">'+subtitle+'</div>' if subtitle else ''}
</div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Icon helpers
# -----------------------------
# Pass emojis (👍, ⚙️, 📊) OR use these short names below:
_ICON_MAP = {
    "house": "🏠",
    "upload": "📤",
    "search": "🔎",
    "cpu": "🧠",
    "activity": "📈",
    "settings": "⚙️",
    "shield": "🛡️",
    "model": "🤖",
    "spark": "✨",
}


def _iconify(icon: str) -> str:
    if not icon:
        return ""
    # If it's a single unicode emoji, return as is
    if len(icon) <= 2:
        return icon
    return _ICON_MAP.get(icon, "•")


# -----------------------------
# Top navigation
# -----------------------------
def top_nav(labels: list[str], icons: list[str], default_index: int = 0) -> str:
    """
    A simple, responsive top nav with icon + label 'pills'.
    Uses buttons for state; looks like a horizontal nav bar.
    """
    if "efai_selected_tab" not in st.session_state:
        st.session_state.efai_selected_tab = labels[default_index]

    st.markdown('<div class="efai-nav">', unsafe_allow_html=True)
    cols = st.columns(len(labels))
    for i, (col, lab, ico) in enumerate(zip(cols, labels, icons)):
        with col:
            active = (st.session_state.efai_selected_tab == lab)
            icon_txt = _iconify(ico)
            btn_label = f"{icon_txt} {lab}" if icon_txt else lab
            # We use a button but wrap it in a style class via container
            c = st.container()
            with c:
                if st.button(btn_label, key=f"efai_tab_{i}", type="secondary"):
                    st.session_state.efai_selected_tab = lab
            # Inject the active style on the container's root element
            st.markdown(
                f"""
<script>
const last = window.parent.document.querySelectorAll('button[kind="secondary"]')[{i}];
if (last) {{
  const wrap = last.closest('div[data-testid="stHorizontalBlock"]') || last.parentElement;
  if (wrap) {{
    wrap.classList.add('efai-tab');
    {'wrap.classList.add("active");' if active else 'wrap.classList.remove("active");'}
  }}
}}
</script>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state.efai_selected_tab


# -----------------------------
# Footer
# -----------------------------
def footer(text: str) -> None:
    st.markdown(f'<div class="efai-footer">{text}</div>', unsafe_allow_html=True)