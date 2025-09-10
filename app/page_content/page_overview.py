# app/page_content/page_overview.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Lightweight helpers (no deep coupling to utils to keep this page robust) ---

ROOT = Path(__file__).resolve().parents[2]

@st.cache_data(show_spinner=False)
def _auto_pick_processed() -> Optional[Path]:
    """
    Try to locate a processed dataset the app can use on first load.
    Priority:
      1) data/processed/*processed*.parquet
      2) data/processed/*.parquet
      3) data/processed/*.csv
    Returns a Path or None.
    """
    proc = ROOT / "data" / "processed"
    if not proc.exists():
        return None
    # 1) processed parquet
    m = sorted(proc.glob("*processed*.parquet"))
    if m:
        return m[0]
    # 2) any parquet
    m = sorted(proc.glob("*.parquet"))
    if m:
        return m[0]
    # 3) any csv
    m = sorted(proc.glob("*.csv"))
    if m:
        return m[0]
    return None


@st.cache_data(show_spinner=False)
def _load_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {ext} for {path.name}")


def _derive_label(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Best-effort label derivation for headline metrics:
    - If 'label' exists (0/1 or neg/pos), use it.
    - Else if ratings exist (rating/rate/star overall between 1..5), define >=4 => 1 else 0.
    Returns a 0/1 Series or None if not derivable.
    """
    # Existing binary label
    for col in ["label", "sentiment", "target"]:
        if col in df.columns:
            s = df[col]
            # normalize to 0/1 where possible
            if s.dtype == bool:
                return s.astype(int)
            if s.dtype.kind in "biuf":
                # already numeric 0/1 or -1/1
                vals = pd.to_numeric(s, errors="coerce")
                if set(np.unique(vals.dropna())) <= {0, 1}:
                    return vals.astype(int)
                # Map {-1,1} -> {0,1}
                if set(np.unique(vals.dropna())) <= {-1, 0, 1}:
                    return (vals > 0).astype(int)
            if s.dtype == object:
                low = s.astype(str).str.lower()
                if set(np.unique(low.dropna())) <= {"pos", "neg"}:
                    return (low == "pos").astype(int)

    # Rating-based derivation
    candidates = [c for c in df.columns if "rating" in c.lower() or "star" in c.lower() or c.lower() in {"overall","score"}]
    for col in candidates:
        try:
            r = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            continue
        # sanity: looks like a 1..5 distribution?
        if r.dropna().between(1, 5).mean() > 0.9:  # mostly in range
            return (r >= 4).astype(int)
    return None


def _pick_text_col(df: pd.DataFrame) -> Optional[str]:
    """
    Try to pick a reasonable text column for display.
    """
    for cand in ["text", "review_text", "review", "comment", "body", "content", "summary", "headline"]:
        if cand in df.columns:
            return cand
    # fallback: the longest average-length string column
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        return None
    lengths = {c: float(pd.Series(df[c].dropna().astype(str)).str.len().mean()) for c in obj_cols}
    return max(lengths, key=lengths.get) if lengths else None


def _make_month(df: pd.DataFrame) -> Optional[pd.Series]:
    for cand in ["review_date", "date", "timestamp", "created_at", "time"]:
        if cand in df.columns:
            dt = pd.to_datetime(df[cand], errors="coerce")
            if dt.notna().any():
                return dt.dt.to_period("M").dt.to_timestamp()
    return None


# --- Page renderer ---

def render_overview() -> None:
    # HERO
    st.markdown(
        """
        <div class="efai-section">
          <div class="efai-h1">Overview</div>
          <div class="efai-sub">
            Turn qualitative feedback into quantified insight — at product speed.
            This workspace helps you ingest reviews, mine topics & aspects, model sentiment,
            and monitor drift so CX and Product can act before issues become churn.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Try to locate a current/processed dataset
    path = _auto_pick_processed()
    if path is None:
        with st.container():
            st.warning(
                "No processed dataset found. Go to **Ingest** to upload and process a file "
                "(CSV or Parquet). Once processed, it will appear here automatically."
            )
            _show_value_props()
            return

    # Load data
    try:
        df = _load_any(path)
    except Exception as e:
        st.error(f"Failed to load dataset at `{path}`.")
        st.exception(e)
        _show_value_props()
        return

    st.caption(f"Active dataset: `{path.relative_to(ROOT)}`")

    if df.empty:
        st.warning("Dataset is empty after loading. Please reprocess the data from **Ingest**.")
        _show_value_props()
        return

    # Basic normalization for KPIs
    text_col = _pick_text_col(df)
    label = _derive_label(df)
    month = _make_month(df)

    # KPIs
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total feedback", f"{len(df):,}")
        with c2:
            if label is not None:
                pos_rate = float(label.mean()) * 100.0
                st.metric("Positive share", f"{pos_rate:.1f}%")
            else:
                st.metric("Positive share", "—")
        with c3:
            if text_col:
                avg_len = float(pd.Series(df[text_col].dropna().astype(str)).str.len().mean())
                st.metric("Avg text length", f"{avg_len:.0f} chars")
            else:
                st.metric("Avg text length", "—")
        with c4:
            # rough % with images if a column hints at images
            img_cols = [c for c in df.columns if "image" in c.lower() or "photo" in c.lower()]
            pct = "—"
            if img_cols:
                any_img = pd.Series(np.zeros(len(df), dtype=bool))
                for c in img_cols:
                    any_img = any_img | df[c].notna()
                pct = f"{(any_img.mean()*100):.1f}%"
            st.metric("Reviews w/ media", pct)

    st.divider()

    # Charts row
    colA, colB = st.columns([1.3, 1])
    with colA:
        st.markdown("#### Volume over time")
        if month is not None:
            vol = pd.DataFrame({"month": month}).dropna()
            vol = vol.value_counts("month").sort_index()
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.plot(vol.index, vol.values, linewidth=2)
            ax.set_xlabel("Month")
            ax.set_ylabel("Reviews")
            ax.grid(True, alpha=0.25)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No parsable date column found to plot volume. If you include a `date` column, we'll chart trends here.")

    with colB:
        st.markdown("#### Sentiment snapshot")
        if label is not None:
            pos = int(label.sum())
            neg = int((1 - label).sum())
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.bar(["Positive", "Negative"], [pos, neg])
            ax.set_ylabel("Count")
            for i, v in enumerate([pos, neg]):
                ax.text(i, v, f"{v:,}", ha="center", va="bottom")
            ax.grid(axis="y", alpha=0.25)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No label or rating available to compute sentiment. Add a `label` (0/1) or `rating` (1–5).")

    st.divider()

    # Recent samples (quick qualitative glance)
    st.markdown("#### Recent samples")
    if text_col:
        view_cols = [text_col]
        # Optional metadata columns
        for c in ["rating", "overall", "label", "sentiment", "product", "category", "reviewer", "title", "summary", "date"]:
            if c in df.columns and c not in view_cols:
                view_cols.append(c)
        st.dataframe(
            df[view_cols].tail(12).iloc[::-1].reset_index(drop=True),
            use_container_width=True,
            height=360,
        )
    else:
        st.info("No obvious text column found. Common names: `text`, `review_text`, `review`, `comment`.")

    st.divider()

    _show_value_props()


def _show_value_props() -> None:
    """Business framing + what each module does. Makes the page self-explanatory."""
    st.markdown(
        """
        <div class="efai-h2">Why this matters</div>
        <div class="efai-grid">
          <div class="efai-card">
            <div class="efai-card-title">Reduce blind spots</div>
            <div class="efai-card-body">
              Mine themes and sentiment across tens of thousands of reviews to uncover issues
              before they hit NPS, CSAT or revenue.
            </div>
          </div>
          <div class="efai-card">
            <div class="efai-card-title">Prioritise with evidence</div>
            <div class="efai-card-body">
              Quantify impact by volume and polarity; link themes to SKU, platform or release to
              feed a clear, data-backed roadmap.
            </div>
          </div>
          <div class="efai-card">
            <div class="efai-card-title">Ship improvements faster</div>
            <div class="efai-card-body">
              A lightweight pipeline from ingest → mine → model → monitor helps Product & CX iterate
              without waiting for a full data warehouse project.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="efai-h2">How to use this workspace</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        1. **Ingest** — Upload CSV/Parquet (raw reviews). The app normalises columns.
        2. **Explore** — Inspect distributions, top terms, co-occurrences, and quick slices.
        3. **Modeling** — Train a baseline text classifier (LogReg + TF-IDF) with small-data guardrails.
        4. **Monitoring** — Build a reference profile and watch for drift in volume, sentiment & text stats.
        5. **Settings** — (Optional) Configure thresholds, sampling or export paths.
        """
    )