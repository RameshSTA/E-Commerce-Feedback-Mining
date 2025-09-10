# src/pipelines/train.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump, load

from src.utils.paths import project_root

def nmf_topics(model: NMF, vect: TfidfVectorizer, topn: int = 12) -> list[list[str]]:
    feat_names = vect.get_feature_names_out()
    topics = []
    for comp in model.components_:
        idx = np.argsort(comp)[::-1][:topn]
        topics.append([feat_names[i] for i in idx])
    return topics

def main(data_dir: str, out_dir: str, n_topics: int = 12, test_size: float = 0.2, seed: int = 42) -> None:
    root = project_root()
    ddir = root / data_dir
    out = root / out_dir
    out.mkdir(parents=True, exist_ok=True)

    vect: TfidfVectorizer = load(ddir / "vectorizer.pkl")
    df = pd.read_parquet(ddir / "dataset.parquet")

    X = vect.transform(df["text_clean"].tolist())
    y = df["sentiment"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    clf = LogisticRegression(max_iter=200, n_jobs=None, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))

    # Topics via NMF on all X (unsupervised)
    nmf = NMF(n_components=n_topics, random_state=seed)
    W = nmf.fit_transform(X)

    dump(clf, out / "model_sentiment.pkl")
    dump(nmf, out / "nmf.pkl")
    # metrics + shapes
    (out / "metrics.json").write_text(json.dumps({"accuracy": acc, "f1": f1}, indent=2), encoding="utf-8")

    # Save topics to CSV
    topics = nmf_topics(nmf, vect, topn=15)
    topics_df = pd.DataFrame(
        {"topic_id": list(range(len(topics))), "top_terms": [", ".join(t) for t in topics]}
    )
    topics_df.to_csv(out / "topics.csv", index=False)

    print(f"[train] accuracy={acc:.3f}, f1={f1:.3f}")
    print(f"[train] wrote artifacts to: {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir with vectorizer.pkl and dataset.parquet (from featurize)")
    ap.add_argument("--out_dir", required=True, help="artifact dir to write model/metrics/topics")
    ap.add_argument("--n_topics", type=int, default=12)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.data_dir, a.out_dir, a.n_topics, a.test_size, a.seed)