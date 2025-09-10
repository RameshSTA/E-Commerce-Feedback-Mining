# src/pipelines/featurize.py
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

from src.utils.paths import project_root

def main(inp: str, out_dir: str, max_features: int = 30000) -> None:
    root = project_root()
    src = root / inp
    out = root / out_dir
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src)
    texts = df["text_clean"].tolist()

    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), min_df=3)
    X = vect.fit_transform(texts)

    dump(vect, out / "vectorizer.pkl")
    np.save(out / "X_shape.npy", np.array(X.shape))
    df_out = df[["text","text_clean","sentiment","rating","recommended","helpful","class_name","department_name","division_name","age"]]
    df_out.to_parquet(out / "dataset.parquet", index=False)
    print(f"[featurize] vectorizer.pkl, X_shape.npy and dataset.parquet saved under {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="preprocessed parquet")
    ap.add_argument("--out_dir", required=True, help="artifact output dir, e.g., artifacts/nlp/2025-09-10")
    ap.add_argument("--max_features", type=int, default=30000)
    a = ap.parse_args()
    main(a.input, a.out_dir, a.max_features)