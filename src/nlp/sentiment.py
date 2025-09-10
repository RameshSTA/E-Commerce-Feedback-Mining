from __future__ import annotations
from typing import Tuple
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# Ensure lexicon once (safe if cached layerless)
def _ensure_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

_Analyzer: SentimentIntensityAnalyzer | None = None

def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _Analyzer
    if _Analyzer is None:
        _ensure_vader()
        _Analyzer = SentimentIntensityAnalyzer()
    return _Analyzer

def score_sentiment(texts: pd.Series) -> pd.DataFrame:
    sid = _get_analyzer()
    rows = []
    for t in texts.fillna(""):
        s = sid.polarity_scores(t)
        # compound ∈ [-1,1]; map to label
        if s["compound"] >= 0.2:
            label = "positive"
        elif s["compound"] <= -0.2:
            label = "negative"
        else:
            label = "neutral"
        rows.append((s["compound"], s["pos"], s["neu"], s["neg"], label))
    return pd.DataFrame(rows, columns=["compound", "pos", "neu", "neg", "sentiment"])