# src/pipelines/evaluate.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from src.utils.paths import project_root

def main(data_dir: str, model_dir: str) -> None:
    root = project_root()
    ddir = root / data_dir
    mdir = root / model_dir

    vect: TfidfVectorizer = load(ddir / "vectorizer.pkl")
    df = pd.read_parquet(ddir / "dataset.parquet")
    X = vect.transform(df["text_clean"].tolist())
    y = df["sentiment"].astype(int).to_numpy()

    clf = load(mdir / "model_sentiment.pkl")
    ypred = clf.predict(X)

    report = classification_report(y, ypred, output_dict=True)
    cm = confusion_matrix(y, ypred).tolist()

    out = {"classification_report": report, "confusion_matrix": cm}
    (mdir / "evaluate_full.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[evaluate] wrote: {mdir / 'evaluate_full.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir produced by featurize.py")
    ap.add_argument("--model_dir", required=True, help="dir with trained model")
    a = ap.parse_args()
    main(a.data_dir, a.model_dir)