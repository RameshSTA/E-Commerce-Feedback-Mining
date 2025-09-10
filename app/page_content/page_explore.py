# app/page_content/page_explore.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import re
from collections import Counter

# Optional sklearn (for nicer tokenisation/ngrams). We fall back gracefully.
_HAS_SK = True
try:
    from sklearn.feature_extraction.text import CountVectorizer
except Exception:
    _HAS_SK = False


ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- Helpers ----------------------------- #
@st.cache_data(show_spinner=False)
def _list_processed() -> List[Path]:
    """Return candidate processed parquet files."""
    if not PROC_DIR.exists():
        return []
    files = sorted(PROC_DIR.glob("*.parquet"))
    # prefer ones we created via ingest (_processed suffix)
    files_sorted = sorted(
        files,
        key=lambda p: (not p.name.endswith("_processed.parquet"), p.stat().st_mtime),
    )
    return files_sorted


@st.cache_data(show_spinner=False)
def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _guess_has_label(df: pd.DataFrame) -> bool:
    return "label" in df.columns and df["label"].dropna().isin([0, 1]).all()


def _guess_has_rating(df: pd.DataFrame) -> bool:
    return "rating" in df.columns and pd.to_numeric(df["rating"], errors="coerce").notna().any()


def _default_text_col(df: pd.DataFrame) -> str:
    candidates = ["text", "review_text", "review", "comment", "body", "content"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first object column
    obj = [c for c in df.columns if df[c].dtype == object]
    return obj[0] if obj else df.columns[0]


def _apply_filters(
    df: pd.DataFrame,
    text_col: str,
    keyword: str,
    min_len: int,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    g = df.copy()
    # keyword filter
    if keyword.strip():
        kw = keyword.strip().lower()
        g = g[g[text_col].astype(str).str.lower().str.contains(re.escape(kw), na=False)]
    # min length
    if min_len > 0:
        g = g[g[text_col].astype(str).str.len() >= min_len]
    # date range
    if date_range and "date" in g.columns:
        start, end = date_range
        dt = pd.to_datetime(g["date"], errors="coerce")
        g = g[(dt >= start) & (dt <= end)]
    return g


def _simple_tokenize(series: pd.Series) -> List[str]:
    """Very simple tokenizer if sklearn isn't available."""
    toks = []
    for txt in series.dropna().astype(str):
        # keep letters/numbers, split on non-alnum
        words = re.findall(r"[A-Za-z0-9']+", txt.lower())
        toks.extend(words)
    return toks


def _top_terms(
    texts: pd.Series,
    stop_words: Optional[List[str]],
    n_top: int = 25,
    ngram_range: Tuple[int, int] = (1, 1),
) -> List[Tuple[str, int]]:
    """Return top n terms; prefer sklearn CountVectorizer; fallback to simple counts."""
    if _HAS_SK:
        try:
            vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range, min_df=2)
            X = vectorizer.fit_transform(texts.dropna().astype(str).values)
            freqs = np.asarray(X.sum(axis=0)).ravel()
            vocab = np.array(vectorizer.get_feature_names_out())
            order = np.argsort(freqs)[::-1][:n_top]
            return list(zip(vocab[order].tolist(), freqs[order].astype(int).tolist()))
        except Exception:
            pass

    # fallback simple counts
    tokens = _simple_tokenize(texts)
    if stop_words:
        sw = set(stop_words)
        tokens = [t for t in tokens if t not in sw]
    # build ngrams
    n, m = ngram_range
    if n == 1 and m == 1:
        ctr = Counter(tokens)
    else:
        grams = []
        for k in range(n, m + 1):
            for i in range(len(tokens) - k + 1):
                grams.append(" ".join(tokens[i : i + k]))
        ctr = Counter(grams)
    return ctr.most_common(n_top)


def _plot_bar(items: List[Tuple[str, int]], title: str, xlabel: str):
    if not items:
        st.info("No terms to show for current filters.")
        return
    terms, counts = zip(*items)
    fig, ax = plt.subplots()
    y = np.arange(len(terms))
    ax.barh(y, counts)
    ax.set_yticks(y)
    ax.set_yticklabels(terms)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.25)
    st.pyplot(fig, use_container_width=True)


# ----------------------------- Page ----------------------------- #
def render_explore() -> None:
    st.markdown(
        """
        <div class="efai-section">
          <div class="efai-h1">Explore</div>
          <div class="efai-sub">
            Understand what customers talk about, how sentiment breaks down, and how feedback trends over time.
            Use filters to focus on segments that matter (e.g., a product line or campaign period).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    files = _list_processed()
    if not files:
        st.warning("No processed dataset found. Go to **Ingest** to create one.")
        return

    names = [f.name for f in files]
    idx_default = 0
    chosen_name = st.selectbox("Dataset", options=names, index=idx_default)
    path = files[names.index(chosen_name)]

    try:
        df = _load_parquet(path)
    except Exception as e:
        st.error(f"Failed to load `{path.name}`.")
        st.exception(e)
        return

    if df.empty:
        st.warning("Dataset is empty after load.")
        return

    # Required text column
    text_col_default = _default_text_col(df)
    text_col = st.selectbox("Text column", options=list(df.columns), index=list(df.columns).index(text_col_default))

    # KPI row
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows", f"{len(df):,}")
        with c2:
            avg_len = float(pd.Series(df[text_col].dropna().astype(str)).str.len().mean())
            st.metric("Avg text length", f"{avg_len:.0f} chars")
        with c3:
            has_label = _guess_has_label(df)
            if has_label:
                pos_rate = float(df["label"].mean()) if df["label"].notna().any() else np.nan
                st.metric("Positive rate", f"{pos_rate*100:.1f}%")
            else:
                st.metric("Positive rate", "—")
        with c4:
            has_rating = _guess_has_rating(df)
            if has_rating:
                r = pd.to_numeric(df["rating"], errors="coerce")
                st.metric("Avg rating", f"{r.mean():.2f} / 5")
            else:
                st.metric("Avg rating", "—")

    st.divider()

    # Filters
    with st.expander("Filters", expanded=True):
        left, right = st.columns([1.2, 1])
        with left:
            keyword = st.text_input("Contains keyword (case-insensitive)", "")
            min_len = st.slider("Minimum text length", 0, 1000, 20, step=10)
        with right:
            date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
            if "date" in df.columns:
                dt = pd.to_datetime(df["date"], errors="coerce")
                if dt.notna().any():
                    mn, mx = dt.min().to_pydatetime(), dt.max().to_pydatetime()
                    start, end = st.date_input("Date range", (mn, mx))
                    if isinstance(start, tuple) or isinstance(end, tuple):
                        # Streamlit returns tuple before selection; ignore
                        date_range = None
                    else:
                        date_range = (pd.to_datetime(start), pd.to_datetime(end))

    fdf = _apply_filters(df, text_col, keyword, min_len, date_range)
    st.caption(f"{len(fdf):,} row(s) match current filters.")

    # Quick preview (paged)
    with st.expander("Sample rows", expanded=False):
        cols_to_show = [c for c in ["review_id", text_col, "label", "rating", "date", "source"] if c in fdf.columns]
        st.dataframe(fdf[cols_to_show].head(500), use_container_width=True, height=360)

    st.divider()

    # Distributions row
    cA, cB = st.columns(2)
    with cA:
        if _guess_has_label(fdf):
            vc = fdf["label"].fillna(-1).map({1: "Positive", 0: "Negative", -1: "Unknown"}).value_counts()
            fig, ax = plt.subplots()
            ax.bar(vc.index, vc.values)
            ax.set_title("Label distribution")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            ax.grid(True, axis="y", alpha=0.25)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No `label` column found. Add a label via rating threshold on Ingest.")

    with cB:
        if _guess_has_rating(fdf):
            r = pd.to_numeric(fdf["rating"], errors="coerce").dropna()
            fig, ax = plt.subplots()
            ax.hist(r, bins=np.arange(0.5, 5.6, 0.5))
            ax.set_title("Rating distribution")
            ax.set_xlabel("Stars")
            ax.set_ylabel("Count")
            ax.grid(True, axis="y", alpha=0.25)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No `rating` column found.")

    st.divider()

    # Top terms (unigram & bigram)
    with st.container():
        st.markdown("### What customers talk about")
        # optional stopwords (simple)
        stop = ["the", "and", "a", "an", "of", "to", "is", "it", "for", "in", "on", "this", "that", "with", "was", "are", "as", "but", "if", "or", "be"]
        left, right = st.columns(2)
        with left:
            uni = _top_terms(fdf[text_col], stop_words=stop, n_top=25, ngram_range=(1, 1))
            _plot_bar(uni, "Top unigrams", "Frequency")
        with right:
            bi = _top_terms(fdf[text_col], stop_words=stop, n_top=25, ngram_range=(2, 2))
            _plot_bar(bi, "Top bigrams", "Frequency")

    st.divider()

    # Volume over time (if date exists)
    if "date" in fdf.columns:
        dt = pd.to_datetime(fdf["date"], errors="coerce")
        if dt.notna().any():
            by = st.selectbox("Aggregate by", options=["Week", "Month"], index=1)
            if by == "Week":
                t = dt.dt.to_period("W").dt.start_time
            else:
                t = dt.dt.to_period("M").dt.to_timestamp()
            volume = fdf.assign(_t=t).groupby("_t").size()
            fig, ax = plt.subplots()
            ax.plot(volume.index, volume.values, marker="o")
            ax.set_title("Feedback volume over time")
            ax.set_xlabel("Date")
            ax.set_ylabel("# Reviews")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)

    # Download filtered
    st.markdown("### Export current view")
    out_cols = [c for c in ["review_id", text_col, "label", "rating", "date", "source"] if c in fdf.columns]
    csv = fdf[out_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered (CSV)",
        data=csv,
        file_name=f"{path.stem}__filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )