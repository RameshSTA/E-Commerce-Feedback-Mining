# src/pipelines/ingest.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.utils.paths import project_root

COLUMN_MAP = {
    "Review Text": "text",
    "Title": "title",
    "Rating": "rating",
    "Recommended IND": "recommended",
    "Positive Feedback Count": "helpful",
    "Class Name": "class_name",
    "Department Name": "department_name",
    "Division Name": "division_name",
    "Age": "age",
}

def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Map known columns; keep unknowns if present
    for raw, std in COLUMN_MAP.items():
        if raw in df.columns:
            df.rename(columns={raw: std}, inplace=True)
    return df

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only columns we use downstream + drop empties
    cols = ["text","title","rating","recommended","helpful","class_name","department_name","division_name","age"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[cols].copy()
    # Coerce types
    df["text"] = df["text"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["recommended"] = pd.to_numeric(df["recommended"], errors="coerce")
    df["helpful"] = pd.to_numeric(df["helpful"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    # drop totally empty texts
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df

def main(input: str, out: str) -> None:
    root = project_root()
    src = root / input
    dst = root / out
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw(src)
    df = normalize(df)
    df.to_parquet(dst, index=False)
    print(f"[ingest] wrote: {dst}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path under project root to raw CSV (e.g., data/raw/realdata.csv)")
    ap.add_argument("--out", required=True, help="Path under project root to output parquet")
    a = ap.parse_args()
    main(a.input, a.out)