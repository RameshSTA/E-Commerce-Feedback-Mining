from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
from langdetect import detect, LangDetectException

from src.nlp.preprocess import build_text
from src.nlp.sentiment import score_sentiment
from src.nlp.topics import fit_topics, infer_topics
from src.nlp.aspects import tag_aspects

def detect_lang_safe(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unk"

def run_mining(input_path: Path, out_dir: Path, n_topics: int = 8) -> Path:
    df = pd.read_parquet(input_path) if input_path.suffix.lower() in {".parquet",".pq"} else pd.read_csv(input_path)
    df = df.copy()
    text = build_text(df, text_cols=["title","review"])

    # language filter (optional: keep only English for modeling)
    df["lang"] = text.map(detect_lang_safe)
    eng_mask = df["lang"].eq("en") | df["lang"].eq("unk")
    df_model = df[eng_mask].reset_index(drop=True)
    text_model = text.loc[df_model.index]

    # sentiment
    sent = score_sentiment(text_model)
    df_model = pd.concat([df_model, sent], axis=1)

    # topics
    nmf, vec = fit_topics(text_model, n_topics=n_topics)
    topic_labels, topic_terms = infer_topics(nmf, vec, text_model, topn_terms=6)
    df_model["topic"] = topic_labels

    # aspects
    df_model["aspects"] = tag_aspects(text_model, max_tags=3)

    # artifact folder
    ts = datetime.now().strftime("%Y-%m-%d")
    out_dir = out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # write enriched
    enriched_path = out_dir / "mined_reviews.parquet"
    df_model.to_parquet(enriched_path, index=False)

    # also export summaries
    terms_df = pd.DataFrame({
        "topic": [i for i, _ in enumerate(topic_terms)],
        "top_terms": [", ".join(t) for t in topic_terms],
    })
    terms_df.to_csv(out_dir / "topics.csv", index=False)

    kpi = df_model.groupby("sentiment", as_index=False).size()
    kpi.to_csv(out_dir / "sentiment_kpis.csv", index=False)

    print(f"[mine] wrote {enriched_path} ({len(df_model)} english/unknown rows)")
    return out_dir

def parse_args():
    ap = argparse.ArgumentParser(description="Feedback mining pipeline")
    ap.add_argument("--input", default="data/processed/normalized_reviews.parquet")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--topics", type=int, default=8)
    return ap.parse_args()

if __name__ == "__main__":
    a = parse_args()
    run_mining(Path(a.input), Path(a.outdir), n_topics=a.topics)