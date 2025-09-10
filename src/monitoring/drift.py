# src/monitoring/drift.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


# ------------------------- utilities ------------------------- #
def _clean_texts(texts: Iterable[str]) -> List[str]:
    out: List[str] = []
    for t in texts:
        s = "" if t is None else str(t)
        s = s.strip()
        out.append(s)
    return out


def _safe_min_df(n_docs: int) -> int | float:
    """
    Choose a min_df that won't exceed doc count. For very small sets, use 1.
    """
    if n_docs <= 5:
        return 1
    return 2


def _safe_max_df(n_docs: int) -> float | int:
    """
    Keep max_df < n_docs implied threshold. For tiny sets just use 1.0 (no cap).
    """
    if n_docs <= 5:
        return 1.0
    return 0.95


def _is_fitted(v: TfidfVectorizer) -> bool:
    try:
        check_is_fitted(v)
        return True
    except NotFittedError:
        return False


def _tfidf_centroid(v: TfidfVectorizer, texts: List[str]) -> np.ndarray:
    """
    Embed texts with a *fitted* vectorizer and return L2-normalized centroid.
    """
    if not _is_fitted(v):
        raise NotFittedError("TF-IDF vectorizer is not fitted.")
    if len(texts) == 0:
        return np.zeros((v.vocabulary_.__len__(),), dtype=float)
    X = v.transform(texts)  # sparse
    if X.shape[0] == 0:
        # no rows after preprocessing
        return np.zeros((X.shape[1],), dtype=float)
    # mean over rows -> dense
    c = np.asarray(X.mean(axis=0)).ravel()
    # L2 normalize
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _avg_len(texts: List[str]) -> float:
    if not texts:
        return 0.0
    return float(np.mean([len(t.split()) for t in texts]))


def _oov_rate(v: TfidfVectorizer, texts: List[str]) -> float:
    """
    % tokens in current texts not present in the reference vocabulary.
    """
    if not _is_fitted(v):
        raise NotFittedError("TF-IDF vectorizer is not fitted.")
    vocab = set(v.vocabulary_.keys())
    toks: List[str] = []
    for t in texts:
        toks.extend(str(t).split())
    if not toks:
        return 0.0
    oov = sum(1 for tok in toks if tok not in vocab)
    return float(oov) / float(len(toks))


def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p_ = p + eps
    q_ = q + eps
    p_ /= p_.sum()
    q_ /= q_.sum()
    return float(np.sum(p_ * (np.log(p_) - np.log(q_))))


# ------------------------- profile & core ------------------------- #
@dataclass
class ReferenceProfile:
    """
    A fitted reference profile built from a 'reference' (training) dataset.
    """
    vectorizer: TfidfVectorizer
    centroid: np.ndarray                # L2-normalized TF-IDF centroid
    vocab_size: int
    mean_len: float
    label_dist: Optional[Dict[str, float]] = None  # normalized frequency


def build_reference_profile(
    ref_df: pd.DataFrame,
    text_col: str,
    label_col: Optional[str] = None,
    stop_words: Optional[str] = "english",
) -> ReferenceProfile:
    """
    Fit a TF-IDF vectorizer *on the reference texts* and compute its centroid, length
    and optional label distribution. This profile is later used to compare current data.
    """
    if text_col not in ref_df.columns:
        raise ValueError(f"Column {text_col!r} not found in reference dataframe.")

    ref_texts = _clean_texts(ref_df[text_col].astype(str).tolist())
    n_docs = len(ref_texts)

    # defensively choose thresholds to avoid max_df < min_df errors
    min_df = _safe_min_df(n_docs)
    max_df = _safe_max_df(n_docs)

    vec = TfidfVectorizer(
        lowercase=True,
        stop_words=stop_words,
        min_df=min_df,
        max_df=max_df,
        max_features=5000,
        ngram_range=(1, 2),
    )
    # Fit on reference texts
    if n_docs == 0:
        # empty reference: build a degenerate vectorizer with no vocab
        # (scikit-learn doesn't allow empty fit; we'll fall back to zeros)
        # Users should avoid this by providing some data.
        # For safety, create a dummy token.
        vec = TfidfVectorizer(vocabulary={"__dummy__": 0})
        centroid = np.zeros((1,), dtype=float)
        vocab_size = 1
    else:
        vec.fit(ref_texts)
        centroid = _tfidf_centroid(vec, ref_texts)
        vocab_size = len(vec.vocabulary_) if hasattr(vec, "vocabulary_") else 0

    mean_len = _avg_len(ref_texts)

    label_dist: Optional[Dict[str, float]] = None
    if label_col and label_col in ref_df.columns:
        counts = ref_df[label_col].astype(str).value_counts(dropna=False)
        total = counts.sum()
        if total > 0:
            label_dist = {str(k): float(v) / float(total) for k, v in counts.items()}

    return ReferenceProfile(
        vectorizer=vec,
        centroid=centroid,
        vocab_size=vocab_size,
        mean_len=mean_len,
        label_dist=label_dist,
    )


def compare_to_reference(
    ref: ReferenceProfile,
    current_df: pd.DataFrame,
    text_col: str,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute drift diagnostics by comparing 'current' data to the fitted reference.

    Returns a 1-row DataFrame with:
      - cosine_similarity
      - mean_len_ref, mean_len_cur, len_diff_abs
      - oov_rate
      - kl_div (if labels available), else NaN
    """
    if text_col not in current_df.columns:
        raise ValueError(f"Column {text_col!r} not found in current dataframe.")

    cur_texts = _clean_texts(current_df[text_col].astype(str).tolist())

    # Ensure the reference vectorizer is fitted (and use it)
    vec = ref.vectorizer
    if not _is_fitted(vec):
        # Refit on reference-equivalent assumptions (shouldn't happen if built via build_reference_profile)
        # Fall back to fitting on current texts if reference fitting was impossible
        if len(cur_texts) > 0:
            vec.fit(cur_texts)

    cur_centroid = _tfidf_centroid(vec, cur_texts)
    cos = _cosine(ref.centroid, cur_centroid)
    mean_len_cur = _avg_len(cur_texts)
    len_diff_abs = abs(mean_len_cur - ref.mean_len)
    oov = _oov_rate(vec, cur_texts)

    # Label drift (optional)
    kl_value = np.nan
    if label_col and (label_col in current_df.columns) and (ref.label_dist is not None):
        cur_counts = current_df[label_col].astype(str).value_counts(dropna=False)
        # union of labels
        labels = sorted(set(list(ref.label_dist.keys()) + list(cur_counts.index.astype(str).tolist())))
        ref_probs = np.array([ref.label_dist.get(lbl, 0.0) for lbl in labels], dtype=float)
        if ref_probs.sum() == 0:
            ref_probs = np.ones_like(ref_probs) / len(ref_probs)
        cur_probs = np.array([float(cur_counts.get(lbl, 0.0)) for lbl in labels], dtype=float)
        if cur_probs.sum() > 0:
            cur_probs = cur_probs / cur_probs.sum()
            kl_value = _kl_div(ref_probs, cur_probs)

    out = pd.DataFrame(
        [{
            "cosine_similarity": cos,
            "mean_len_ref": ref.mean_len,
            "mean_len_cur": mean_len_cur,
            "len_diff_abs": len_diff_abs,
            "oov_rate": oov,
            "kl_div": kl_value,
        }]
    )
    return out