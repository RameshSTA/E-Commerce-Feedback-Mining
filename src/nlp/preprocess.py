from __future__ import annotations
import re
from typing import Iterable, List
import pandas as pd

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<.*?>")
_NONALNUM = re.compile(r"[^a-z0-9\s]")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = _URL.sub(" ", s)
    s = _HTML.sub(" ", s)
    s = s.lower()
    s = _NONALNUM.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def build_text(df: pd.DataFrame, text_cols: Iterable[str]) -> pd.Series:
    cols = [c for c in text_cols if c in df.columns]
    if not cols:
        raise ValueError(f"No text columns present from: {text_cols}")
    raw = df[cols].astype(str).agg(" ".join, axis=1)
    return raw.fillna("").astype(str).map(normalize_text)