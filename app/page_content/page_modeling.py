# app/page_content/page_modeling.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

_HAS_XGB = True
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    _HAS_XGB = False

from src.utils.datasets import pick_active_processed, load_any, dataset_badge, guess_text_col
from src.utils.paths import project_root

_HAS_TRACKER = True
try:
    from src.monitoring.tracker import Tracker  # type: ignore
except Exception:
    _HAS_TRACKER = False
    Tracker = None  # type: ignore


@dataclass
class TrainConfig:
    text_col: str
    label_col: str
    test_size: float
    random_state: int
    stratify_ok: bool
    model_name: str
    max_features: int
    ngram_max: int
    class_weight: Optional[str]
    cv_folds: int


def _pick_label_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["label", "target", "sentiment", "rating", "stars", "category"]:
        if c in df.columns:
            return c
    # low-cardinality fallback
    candidates = [(c, df[c].nunique()) for c in df.columns if c != "text"]
    candidates = [(c, k) for c, k in candidates if 2 <= k <= 10]
    if candidates:
        return sorted(candidates, key=lambda x: x[1])[0][0]
    return None

def _can_stratify(y: pd.Series) -> bool:
    vc = y.value_counts()
    return (vc.min() >= 2) and (vc.nunique() >= 2)

def _build_estimator(name: str, class_weight: Optional[str]):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=200, solver="liblinear" if class_weight else "lbfgs", class_weight=class_weight)
    if name == "LinearSVC":
        return LinearSVC(class_weight=class_weight)
    if name == "MultinomialNB":
        return MultinomialNB(alpha=0.5)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight=class_weight)
    if name == "XGBoost" and _HAS_XGB:
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            tree_method="hist", n_jobs=-1, eval_metric="mlogloss"
        )
    raise ValueError(f"Unknown or unavailable model: {name}")

def _build_pipeline(cfg: TrainConfig) -> Pipeline:
    vect = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=(1, cfg.ngram_max),
        strip_accents="unicode",
        lowercase=True,
        min_df=1,
    )
    est = _build_estimator(cfg.model_name, cfg.class_weight)
    return Pipeline([("tfidf", vect), ("clf", est)])

def _save_artifacts(pipe: Pipeline, cfg: TrainConfig, label_map: Dict[str, int], run_tag: str) -> Optional[Path]:
    outdir = project_root() / "artifacts" / "models"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        model_path = outdir / f"{cfg.model_name}_{run_tag}.joblib"
        meta_path = outdir / f"{cfg.model_name}_{run_tag}.json"
        joblib.dump(pipe, model_path)
        meta = {
            "model": cfg.model_name,
            "text_col": cfg.text_col,
            "label_col": cfg.label_col,
            "max_features": cfg.max_features,
            "ngram_max": cfg.ngram_max,
            "class_weight": cfg.class_weight,
            "label_map": label_map,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return model_path
    except Exception:
        return None

def render_modeling() -> None:
    st.markdown("### Modeling")
    st.caption("Train robust classifiers on the **active** dataset. Upload on **Ingest** to switch datasets.")

    # Prefer session-state active (set by Ingest) → uploaded → default
    path, label = pick_active_processed(prefer_uploaded=True)
    df = load_any(path)

    st.markdown(dataset_badge(label))

    if df.empty or len(df) < 20:
        st.info("You need at least ~20 labeled rows to train. Ingest more data or choose a different label.")
        st.dataframe(df.head(100), use_container_width=True)
        st.stop()

    text_col_guess = "text" if "text" in df.columns else guess_text_col(df)
    label_col_guess = _pick_label_col(df)

    c1, c2, c3 = st.columns(3)
    with c1:
        text_col = st.selectbox("Text column", df.columns.tolist(), index=df.columns.get_loc(text_col_guess))
    with c2:
        if label_col_guess is None:
            st.error("No suitable label column found (try 'label', 'sentiment', 'rating', 'category').")
            st.stop()
        label_col = st.selectbox("Label column", df.columns.tolist(), index=df.columns.get_loc(label_col_guess))
    with c3:
        models = ["LogisticRegression", "LinearSVC", "MultinomialNB", "RandomForest"] + (["XGBoost"] if _HAS_XGB else [])
        model_name = st.selectbox("Model", models, index=0)

    with st.expander("Advanced"):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: max_features = st.selectbox("TF-IDF max features", [500, 2000, 5000, 10000], index=2)
        with c2: ngram_max = st.selectbox("N-gram max", [1, 2], index=1)
        with c3: test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        with c4:
            cw_opt = st.selectbox("Class weight", ["(none)", "balanced"], index=1)
            class_weight = None if cw_opt == "(none)" else "balanced"
        with c5: cv_folds = st.selectbox("CV folds (0=skip)", [0, 3, 5, 10], index=1)
        random_state = 42

    X_text = df[text_col].astype(str).fillna("")
    y_raw = df[label_col].astype(str).fillna("")

    classes_sorted = sorted(y_raw.unique().tolist())
    label_to_id = {c: i for i, c in enumerate(classes_sorted)}
    y = y_raw.map(label_to_id).astype(int)

    can_stratify = (y.value_counts().min() >= 2) and (y.nunique() >= 2)
    if not can_stratify:
        st.warning("Some classes have < 2 samples — proceeding **without** stratified split.")

    cfg = TrainConfig(
        text_col=text_col,
        label_col=label_col,
        test_size=float(test_size),
        random_state=random_state,
        stratify_ok=bool(can_stratify),
        model_name=model_name,
        max_features=int(max_features),
        ngram_max=int(ngram_max),
        class_weight=class_weight,
        cv_folds=int(cv_folds),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=(y if cfg.stratify_ok else None)
    )

    pipe = _build_pipeline(cfg)

    # Optional CV
    if cfg.cv_folds > 0 and len(np.unique(y_train)) > 1 and len(y_train) >= cfg.cv_folds:
        try:
            skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state) if cfg.stratify_ok else cfg.cv_folds
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
            st.caption(f"CV f1_macro ({cfg.cv_folds}-fold): mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}")
        except Exception:
            st.caption("CV skipped (dataset too small / non-stratifiable).")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }

    # ROC-AUC (binary only)
    if len(classes_sorted) == 2:
        try:
            # Try proba, fallback decision_function
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                proba = pipe.named_steps["clf"].predict_proba(pipe.named_steps["tfidf"].transform(X_test))
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
            elif hasattr(pipe.named_steps["clf"], "decision_function"):
                decision = pipe.named_steps["clf"].decision_function(pipe.named_steps["tfidf"].transform(X_test))
                if decision.ndim == 1:
                    decision = np.vstack([-decision, decision]).T
                exp = np.exp(decision - decision.max(axis=1, keepdims=True))
                proba = exp / exp.sum(axis=1, keepdims=True)
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
        except Exception:
            pass

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with c2: st.metric("F1 (macro)", f"{metrics['f1_macro']:.3f}")
    with c3: st.metric("Classes", str(len(classes_sorted)) if "roc_auc" not in metrics else f"ROC-AUC {metrics['roc_auc']:.3f}")

    try:
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes_sorted))))
        cm_df = pd.DataFrame(cm, index=[f"true:{c}" for c in classes_sorted], columns=[f"pred:{c}" for c in classes_sorted])
        st.markdown("#### Confusion matrix")
        st.dataframe(cm_df, use_container_width=True)
    except Exception:
        pass

    try:
        rep = classification_report(y_test, y_pred, target_names=classes_sorted, digits=3)
        with st.expander("Classification report"):
            st.code(rep)
    except Exception:
        pass

    # Save artifacts + track
    run_tag = f"{label_col}_{model_name}"
    mp = _save_artifacts(pipe, cfg, label_to_id, run_tag)
    if mp:
        st.success(f"Model saved: `{mp}`")

    if _HAS_TRACKER:
        try:
            tracker = Tracker(project_root())
            with tracker.run(tags={"page": "modeling", "dataset": label}):
                tracker.log_params({
                    "model": cfg.model_name,
                    "text_col": cfg.text_col,
                    "label_col": cfg.label_col,
                    "max_features": cfg.max_features,
                    "ngram_max": cfg.ngram_max,
                    "class_weight": cfg.class_weight or "(none)",
                    "cv_folds": cfg.cv_folds,
                })
                tracker.log_metric("accuracy", float(metrics["accuracy"]))
                tracker.log_metric("f1_macro", float(metrics["f1_macro"]))
                if "roc_auc" in metrics:
                    tracker.log_metric("roc_auc", float(metrics["roc_auc"]))
            st.caption("Run logged under `runs/`.")
        except Exception:
            st.caption("Run logging skipped.")