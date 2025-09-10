# app/page_content/page_monitoring.py
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from src.utils.datasets import pick_active_processed, load_any
from src.monitoring.drift import build_reference_profile, compare_to_reference


def _pick_text_and_label_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    text_candidates = [c for c in df.columns if c.lower() in ("text", "review_text", "review", "content", "comment")]
    text_col = text_candidates[0] if text_candidates else df.select_dtypes("object").columns.tolist()[0]

    label_candidates = [c for c in df.columns if c.lower() in ("label", "rating", "sentiment", "category")]
    label_col = label_candidates[0] if label_candidates else None
    return text_col, label_col


def _slice_windows(df: pd.DataFrame, ref_rows: int, cur_rows: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref_rows = max(1, min(ref_rows, len(df)))
    cur_rows = max(1, min(cur_rows, len(df)))
    ref_df = df.head(ref_rows).copy()
    cur_df = df.tail(cur_rows).copy()
    return ref_df, cur_df


def render_monitoring() -> None:
    st.markdown("### Monitoring")
    st.caption("Detect data drift and monitor model health signals on text feedback streams.")

    # 1) Active processed dataset
    try:
        path, label = pick_active_processed(prefer_uploaded=True)
    except Exception:
        path, label = None, None

    if not path:
        st.info("**No processed dataset to monitor.** Use **Ingest** to create one.")
        return

    try:
        df = load_any(path)
    except Exception as e:
        st.error(f"Could not load processed dataset from `{path}`.")
        st.exception(e)
        return

    if df.empty:
        st.warning("Processed dataset is empty.")
        return

    n = len(df)
    st.markdown(f"**Dataset:** `{label}` · **Rows:** {n:,} · **Columns:** {len(df.columns)}")

    if n < 2:
        st.warning("Need at least 2 rows to compare reference vs current windows.")
        st.dataframe(df.head(10), use_container_width=True)
        return

    # 2) Choose columns
    try:
        guess_text, guess_label = _pick_text_and_label_columns(df)
    except Exception:
        st.error("Could not infer a text column from the dataset.")
        return

    c1, c2 = st.columns([1.2, 1.2])
    with c1:
        text_col = st.selectbox("Text column", options=df.columns.tolist(), index=df.columns.get_loc(guess_text))
    with c2:
        label_options = ["<none>"] + df.columns.tolist()
        label_idx = 0
        if guess_label and guess_label in df.columns:
            label_idx = label_options.index(guess_label)
        label_choice = st.selectbox("Label column (optional)", options=label_options, index=label_idx)
        label_col = None if label_choice == "<none>" else label_choice

    # 3) Window selection — dynamic bounds for tiny datasets
    # Ensure at least one row remains for the other window
    ref_min, ref_max = 1, max(1, n - 1)
    cur_min, cur_max = 1, max(1, n - 1)

    # Sensible defaults: ~60% reference, rest current
    ref_default = max(ref_min, min(int(0.6 * n), ref_max))
    cur_default = max(cur_min, min(n - ref_default, cur_max))
    if cur_default < cur_min:  # if n is very small, ensure at least 1 row
        cur_default = cur_min
        ref_default = max(ref_min, n - cur_default)

    c3, c4 = st.columns(2)
    with c3:
        ref_rows = st.slider(
            "Reference window (rows)",
            min_value=ref_min,
            max_value=ref_max,
            value=ref_default,
        )
    with c4:
        cur_rows = st.slider(
            "Current window (rows)",
            min_value=cur_min,
            max_value=cur_max,
            value=cur_default,
        )

    ref_df, cur_df = _slice_windows(df, ref_rows=ref_rows, cur_rows=cur_rows)

    # 4) Build reference profile (fits TF-IDF on reference only)
    try:
        ref_profile = build_reference_profile(ref_df, text_col=text_col, label_col=label_col)
    except Exception as e:
        st.error("Failed to build reference profile.")
        st.exception(e)
        return

    # 5) Compare current vs reference (returns a single-row DataFrame of metrics)
    try:
        report_df = compare_to_reference(ref_profile, cur_df, text_col=text_col, label_col=label_col)
    except Exception as e:
        st.error("Failed to compute drift report.")
        st.exception(e)
        return

    # 6) KPIs
    st.subheader("KPIs")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Reference rows", f"{len(ref_df):,}")
    with k2:
        st.metric("Current rows", f"{len(cur_df):,}")
    with k3:
        st.metric("Vocab size", f"{ref_profile.vocab_size:,}")

    # 7) Drift diagnostics (table)
    st.subheader("Drift diagnostics")
    st.caption("Cosine similarity uses TF-IDF centroids (fitted on reference). OOV uses reference vocabulary.")
    st.dataframe(report_df, use_container_width=True)

    # 8) Quick cues
    row = report_df.iloc[0]
    cos = float(row.get("cosine_similarity", 0.0))
    len_diff = float(row.get("len_diff_abs", 0.0))
    oov = float(row.get("oov_rate", 0.0))
    kl = row.get("kl_div", float("nan"))

    st.markdown("**Signals**")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Cosine similarity ↑", f"{cos:.3f}")
    with s2:
        st.metric("Length drift (abs words) ↓", f"{len_diff:.2f}")
    with s3:
        st.metric("OOV rate ↓", f"{oov:.2%}")
    with s4:
        st.metric("KL(label) ↓", "—" if pd.isna(kl) else f"{float(kl):.4f}")

    st.caption(
        "Rules of thumb: cosine < 0.85 may indicate distribution shift; OOV > 10% suggests new vocabulary; "
        "large length drift may mean UX or platform change; KL requires a label column."
    )

    # 9) Peek at windows (optional)
    with st.expander("Preview windows", expanded=False):
        st.write("**Reference head**")
        st.dataframe(ref_df.head(5), use_container_width=True)
        st.write("**Current head**")
        st.dataframe(cur_df.head(5), use_container_width=True)