# src/utils/datasets.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# ---------- Active dataset selector ----------
ACTIVE_JSON = ".active_dataset.json"

@dataclass(frozen=True)
class DataPaths:
    root: Path
    raw_dir: Path
    processed_dir: Path
    uploaded_dir: Path

def get_paths() -> DataPaths:
    # repo root = file -> src -> utils -> (repo)
    root = Path(__file__).resolve().parents[2]
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    uploaded = root / "data" / "uploaded"
    for p in (raw, processed, uploaded):
        p.mkdir(parents=True, exist_ok=True)
    return DataPaths(root=root, raw_dir=raw, processed_dir=processed, uploaded_dir=uploaded)

def _active_file() -> Path:
    return get_paths().root / ACTIVE_JSON

def _read_active() -> Dict[str, str]:
    f = _active_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}

def _write_active(d: Dict[str, str]) -> None:
    _active_file().write_text(json.dumps(d, indent=2), encoding="utf-8")

# ---------- Mark / pick active datasets ----------
def mark_uploaded_active(raw_path: Path | str, processed_path: Path | str) -> None:
    d = _read_active()
    d["uploaded_raw"] = str(Path(raw_path))
    d["uploaded_processed"] = str(Path(processed_path))
    _write_active(d)

def mark_default_active(raw_path: Path | str, processed_path: Path | str) -> None:
    d = _read_active()
    d["default_raw"] = str(Path(raw_path))
    d["default_processed"] = str(Path(processed_path))
    _write_active(d)

def pick_active_processed(prefer_uploaded: bool = True) -> Tuple[Path, str]:
    # 1) session override
    try:
        import streamlit as st
        if "active_processed_path" in st.session_state:
            p = Path(st.session_state["active_processed_path"])
            if p.exists():
                return p, f"session: {p.name}"
    except Exception:
        pass

    # 2) persisted selection
    d = _read_active()
    keys = ["uploaded_processed", "default_processed"] if prefer_uploaded else ["default_processed", "uploaded_processed"]
    for k in keys:
        v = d.get(k)
        if v and Path(v).exists():
            p = Path(v)
            return p, f"{k}: {p.name}"

    # 3) fallback: first processed file
    paths = list(get_paths().processed_dir.glob("*.parquet")) + list(get_paths().processed_dir.glob("*.csv"))
    if paths:
        return paths[0], f"fallback: {paths[0].name}"

    raise FileNotFoundError("No processed dataset found. Please ingest first.")

def pick_active_raw(prefer_uploaded: bool = True) -> Tuple[Path, str]:
    d = _read_active()
    keys = ["uploaded_raw", "default_raw"] if prefer_uploaded else ["default_raw", "uploaded_raw"]
    for k in keys:
        v = d.get(k)
        if v and Path(v).exists():
            p = Path(v)
            return p, f"{k}: {p.name}"
    paths = list(get_paths().raw_dir.glob("*"))
    if paths:
        return paths[0], f"fallback: {paths[0].name}"
    raise FileNotFoundError("No raw dataset found.")

# ---------- I/O helpers ----------
def load_any(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Could not resolve a data file from: {path}")
    ext = p.suffix.lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if ext == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file extension: {ext}")

def save_parquet(df: pd.DataFrame, out: Path | str) -> Path:
    """
    Save to Parquet (requires pyarrow). Creates parent dirs.
    """
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p

# ---------- UI helpers ----------
def ensure_dirs() -> DataPaths:
    return get_paths()

def dataset_badge(label: str) -> str:
    return f"**Dataset:** {label}"

def guess_text_col(df: pd.DataFrame) -> str:
    # Best-effort guess for a text column
    for c in ["text", "review_text", "review", "comment", "body", "content", "message", "title"]:
        if c in df.columns:
            return c
    obj = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
    if obj:
        scores = [(c, df[c].astype(str).str.len().mean()) for c in obj]
        return sorted(scores, key=lambda x: x[1], reverse=True)[0][0]
    return df.columns[0]