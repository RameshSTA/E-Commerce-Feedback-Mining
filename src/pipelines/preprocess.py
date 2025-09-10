# src/pipelines/preprocess.py
from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd

# Use scikit-learn's built-in English stop words (no external downloads)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# PorterStemmer does NOT require downloading NLTK corpora.
try:
    from nltk.stem import PorterStemmer  # lightweight, no data files needed
    _STEMMER = PorterStemmer()
    _USE_STEM = True
except Exception:
    _STEMMER = None
    _USE_STEM = False

from src.utils.paths import project_root

STOP = set(ENGLISH_STOP_WORDS)

def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def clean_text(s: str) -> str:
    """
    Minimal, fast, download-free text cleaner:
      - lowercase
      - strip URLs
      - keep letters/digits/space
      - remove stop words (sklearn)
      - optional Porter stemming (no downloads needed)
    """
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)     # URLs -> space
    s = re.sub(r"[^a-z0-9\s]", " ", s)          # keep alnum + spaces
    tokens = [t for t in s.split() if t and t not in STOP]
    if _USE_STEM and _STEMMER is not None:
        tokens = [_STEMMER.stem(t) for t in tokens]
    return _normalize_whitespace(" ".join(tokens))

def label_from_rating(r: float) -> int:
    """
    Binary sentiment:
      - 1 for rating >= 4 (positive)
      - 0 for rating <= 2 (negative)
      - ignore rating == 3 (neutral) by returning -1 and dropping later
    """
    if pd.isna(r):
        return -1
    if r >= 4:
        return 1
    if r <= 2:
        return 0
    return -1

def main(inp: str, out: str) -> None:
    root = project_root()
    src = root / inp
    dst = root / out
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src)

    # Deduplicate and clean
    df = df.drop_duplicates(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    # Clean text (download-free)
    df["text_clean"] = df["text"].apply(clean_text)

    # Build labels from rating (binary)
    df["sentiment"] = df["rating"].apply(label_from_rating)
    df = df[df["sentiment"] >= 0].reset_index(drop=True)

    # (Optional) basic sanity checks
    n = len(df)
    pos = int((df["sentiment"] == 1).sum())
    neg = int((df["sentiment"] == 0).sum())
    print(f"[preprocess] rows={n}, pos={pos}, neg={neg}, stem={'on' if _USE_STEM else 'off'}")

    df.to_parquet(dst, index=False)
    print(f"[preprocess] wrote: {dst}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="processed parquet from ingest")
    ap.add_argument("--out", required=True, help="output parquet path")
    a = ap.parse_args()
    main(a.input, a.out)