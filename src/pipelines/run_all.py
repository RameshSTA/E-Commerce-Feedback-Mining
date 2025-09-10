# src/pipelines/run_all.py
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
from src.utils.paths import project_root

def run(cmd: list[str]):
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main(raw_csv: str, run_tag: str | None = None) -> None:
    root = project_root()
    tag = run_tag or datetime.now().strftime("%Y-%m-%d")

    # Paths
    ing_out = f"data/processed/normalized_reviews.parquet"
    pre_out = f"data/processed/preprocessed_reviews.parquet"
    art_dir = f"artifacts/nlp/{tag}"

    # 1) ingest
    run(["python", "-m", "src.pipelines.ingest", "--input", raw_csv, "--out", ing_out])

    # 2) preprocess
    run(["python", "-m", "src.pipelines.preprocess", "--input", pre_out.replace("preprocessed", "normalized"), "--out", pre_out])

    # 3) featurize
    run(["python", "-m", "src.pipelines.featurize", "--input", pre_out, "--out_dir", art_dir])

    # 4) train
    run(["python", "-m", "src.pipelines.train", "--data_dir", art_dir, "--out_dir", art_dir])

    # 5) evaluate
    run(["python", "-m", "src.pipelines.evaluate", "--data_dir", art_dir, "--model_dir", art_dir])

    print(f"[run_all] completed. See {root / art_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run full NLP pipeline on raw CSV")
    ap.add_argument("--raw_csv", default="data/raw/realdata.csv")
    ap.add_argument("--run_tag", default=None, help="Artifacts folder suffix (default: YYYY-MM-DD)")
    a = ap.parse_args()
    main(a.raw_csv, a.run_tag)