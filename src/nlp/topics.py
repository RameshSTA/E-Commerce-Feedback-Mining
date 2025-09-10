from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def fit_topics(
    corpus: pd.Series,
    n_topics: int = 8,
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 3
) -> Tuple[NMF, TfidfVectorizer]:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    X = vec.fit_transform(corpus.tolist())
    model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=400)
    model.fit(X)
    return model, vec

def infer_topics(
    model: NMF,
    vec: TfidfVectorizer,
    corpus: pd.Series,
    topn_terms: int = 6
) -> Tuple[pd.Series, List[List[str]]]:
    X = vec.transform(corpus.tolist())
    W = model.transform(X)  # (n_docs, n_topics)
    labels = W.argmax(axis=1)
    # topic terms
    terms = np.array(vec.get_feature_names_out())
    H = model.components_
    topic_terms: List[List[str]] = []
    for k in range(H.shape[0]):
        idx = np.argsort(H[k])[::-1][:topn_terms]
        topic_terms.append(terms[idx].tolist())
    return pd.Series(labels, index=corpus.index), topic_terms