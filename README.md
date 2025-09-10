# E-Commerce Feedback Mining: Sentiment, Topics & Monitoring

<p align="left">
  <a href="https://ecommercefeedbackai.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit" alt="Live Demo">
  </a>
  <a href="#-license">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License: MIT">
  </a>
</p>

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/NLTK-%230C59A3.svg?style=for-the-badge&logo=nltk&logoColor=white" />
  <img src="https://img.shields.io/badge/spaCy-%2309A3D5.svg?style=for-the-badge&logo=spacy&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/NetworkX-%230D5A9A.svg?style=for-the-badge&logo=networkx&logoColor=white" />
</p>

**ECommerceFeedbackAI** turns noisy customer reviews into actions. It mines sentiment, discovers topics, builds co-occurrence networks, and monitors data/model drift — in a **production-style** Streamlit app with an opinionated folder structure, CLI pipelines, and lightweight experiment tracking.

---

## Table of Contents

1. [Why this project](#-why-this-project)
2. [Live demo](#-live-demo)
3. [What you get](#-what-you-get)
4. [Architecture](#-architecture)
5. [Project structure](#-project-structure)
6. [Data flow](#-data-flow)
7. [Install & run](#-install--run)
8. [Use your own data](#-use-your-own-data)
9. [Pipelines & CLI](#-pipelines--cli)
10. [Monitoring & drift](#-monitoring--drift)
11. [Modeling](#-modeling)
12. [Tech stack](#-tech-stack)
13. [Results & screenshots](#-results--screenshots)
14. [Troubleshooting](#-troubleshooting)
15. [Resume bullets (XYZ)](#-resume-bullets-xyz)
16. [License](#-license)
17. [Author](#-author)

---

## 🧠 Why this project

Modern CX/PM teams drown in feedback. This app:
- **Scales** review mining (thousands → minutes).
- **Quantifies** themes & sentiment (P/N/Neu).
- **Surfaces relationships** between topics (co-occurrence graph).
- **Monitors drift** so your models don’t silently degrade.

It’s engineered for **real usage**: modular `src/`, lazy-loaded pages, robust dataset switching, CLI pipelines, and simple local “MLflow-lite” tracking.

---

## 🚀 Live Demo

If deployed, open: **https://ecommercefeedbackai.streamlit.app/**  
(Or run locally — instructions below.)

---

## 📦 What you get

- **Streamlit app** with 5 tabs: Overview, Ingest, Explore, Modeling, Monitoring.
- **Use your data** (CSV/Parquet) or default sample data.
- **NLP pipeline**: cleaning → TF-IDF → LDA topics → VADER sentiment.
- **Co-occurrence network** (NetworkX + Plotly).
- **Drift detection**: text length/lexicality/TF-IDF centroid shift, label ratios, KS tests on numerics.
- **Pipelines**: `ingest`, `preprocess`, `run_all`.
- **Tiny tracker** (optional): write-once JSON/CSV runs for metrics/artifacts.

---

## 🏗 Architecture

- **UI:** Streamlit, component helpers, sticky header, icon nav.
- **Data utils:** consistent path resolution, active dataset registry, safe loading/saving.
- **Pipelines:** `src/pipelines/*.py` for batch runs (ingest → preprocess → artifacts).
- **Monitoring:** `src/monitoring/drift.py` (reference profile, compare reports).
- **Modeling:** baseline classical models (LogReg, LinearSVC), vectorizers with guard-rails.

---

## 📁 Project structure
ECommerceFeedbackAI/
├── app/
│   ├── main_app.py                   # router + header + nav
│   └── page_content/
│       ├── page_overview.py
│       ├── page_ingest.py
│       ├── page_explore.py
│       ├── page_modeling.py
│       └── page_monitoring.py
├── src/
│   ├── ui/components.py              # CSS, header, nav, cards, KPIs
│   ├── utils/
│   │   ├── paths.py                  # project_root(), ensure_dirs…
│   │   └── datasets.py               # load_any(), registry, defaults
│   ├── nlp/                          # preprocessors, tokenizers (if extended)
│   ├── monitoring/drift.py           # profiles, comparisons, KS tests
│   ├── pipelines/
│   │   ├── ingest.py
│   │   ├── preprocess.py
│   │   └── run_all.py
│   └── modeling/                     # model helpers (optional)
├── data/
│   ├── raw/realdata.csv              # your dataset (example)
│   └── processed/                    # normalized/preprocessed parquet
├── runs/                             # (optional) tracker outputs
├── requirements.txt
└── README.md

---

## 🔁 Data flow

1. **Ingest**  
   CSV/Parquet → normalize columns (`text`, optional `rating`/`label`) → save parquet.
2. **Preprocess**  
   clean text (lower/strip/punct/stopwords) → lemma (spaCy) → TF-IDF → artifacts.
3. **Explore**  
   sentiment (VADER), topic prevalence, per-topic sentiment, sample reviews.
4. **Modeling**  
   TF-IDF → LogReg / LinearSVC (with guardrails for small classes & tiny corpora).
5. **Monitoring**  
   build reference (first N rows) → compare current window → drift KPIs & alerts.

---

## ⚙️ Install & run

```bash
# 1) Clone
git clone https://github.com/<your-username>/ECommerceFeedbackAI.git
cd ECommerceFeedbackAI

# 2) Virtual env
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3) Upgrade pip + install deps
python -m pip install -U pip
pip install -r requirements.txt

# 4) NLTK data (first run needs these)
python - <<'PY'
import nltk
for p in ['vader_lexicon','stopwords','punkt','wordnet','omw-1.4']:
    nltk.download(p)
print("Downloaded NLTK data.")
PY

# 5) Run the app
streamlit run app/main_app.py
📤 Use your own data
	•	CSV/Parquet with at least one column that is textual.
	•	On Ingest tab:
	•	Upload file → select text column (auto-guessed) → optional label/rating.
	•	App normalizes and registers it as active. Other pages switch automatically.

Accepted text column names (auto-guessing): text, review_text, Review Text, comment, body, content.
If unsure, you’ll be prompted to select a column.
