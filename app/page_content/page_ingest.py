# app/page_content/page_ingest.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from src.utils.datasets import (
    ensure_dirs,
    load_any,
    save_parquet,
    mark_uploaded_active,
    mark_default_active,
    dataset_badge,
)

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Unify common columns
    ren = {
        "Review Text": "text",
        "Review_Text": "text",
        "reviewText": "text",
        "Review": "text",
        "Text": "text",
        "Title": "title",
        "Rating": "rating",
        "Overall": "rating",
        "Score": "rating",
        "Category": "category",
        "Department Name": "category",
    }
    df.rename(columns={k: v for k, v in ren.items() if k in df.columns}, inplace=True)

    # Ensure text column
    if "text" not in df.columns:
        obj_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
        if obj_cols:
            df["text"] = df[obj_cols[0]].astype(str)
        else:
            df["text"] = ""
    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    # Light cleanup
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Nice column order
    front = [c for c in ["review_id", "date", "title", "text", "rating", "category"] if c in df.columns]
    others = [c for c in df.columns if c not in front]
    return df[front + others]

def render_ingest() -> None:
    st.markdown("### Ingest")
    st.caption("Upload or select a dataset, we’ll normalize it and make it the **active** dataset for Explore / Modeling / Monitoring.")

    paths = ensure_dirs()

    with st.expander("Default project dataset", expanded=True):
        default_raw = paths.raw_dir / "realdata.csv"  # change if your default file differs
        default_processed = paths.processed_dir / "preprocessed_reviews.parquet"
        if default_raw.exists() and default_processed.exists():
            mark_default_active(default_raw, default_processed)
            st.markdown(dataset_badge(f"default: {default_processed.name}"))
        else:
            st.info("No default files found yet under data/raw or data/processed.")

    st.markdown("#### Upload a file (CSV or Parquet)")
    up = st.file_uploader("Upload file", type=["csv", "parquet", "pq"])
    if up is not None:
        tmp_path = paths.uploaded_dir / up.name
        tmp_path.write_bytes(up.read())
        st.success(f"Uploaded to: `{tmp_path}`")

        # Load & preview
        try:
            raw_df = load_any(tmp_path)
        except Exception as e:
            st.error(f"Failed to read: {e}")
            return

        with st.expander("Preview uploaded data", expanded=False):
            st.dataframe(raw_df.head(200), use_container_width=True)

        # Normalize & save
        norm = _normalize(raw_df)
        out = paths.processed_dir / "uploaded_preprocessed.parquet"
        save_parquet(norm, out)

        # Mark active (persist + current session)
        mark_uploaded_active(tmp_path, out)
        st.session_state["active_processed_path"] = str(out)

        st.success(f"Processed & active: `{out}`")
        st.button("Use this dataset now", on_click=lambda: st.experimental_rerun())