# src/monitoring/tracker.py
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


# ---------- Storage layout ----------
# runs/
#   <run_id>/
#       run.json             # {run_id, start_time, end_time, status, tags, params, metrics}
#       metrics.csv          # step, metric, value, ts
#       params.json          # flat params dict
#       artifacts/           # any files logged here


def _runs_root(project_root: Path) -> Path:
    p = project_root / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_ts() -> float:
    return time.time()


@dataclass
class RunInfo:
    run_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running|finished|failed
    tags: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


class Tracker:
    """
    Tiny experiment tracker (MLflow-like) that writes simple JSON/CSV files.
    """

    def __init__(self, root: Path):
        self.root = _runs_root(root)
        self.active_run: Optional[RunInfo] = None

    # ----- Run lifecycle -----
    def start_run(self, tags: Optional[Dict[str, Any]] = None) -> RunInfo:
        run_id = str(int(_now_ts() * 1000))
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        info = RunInfo(run_id=run_id, start_time=_now_ts(), tags=tags or {})
        self.active_run = info
        self._save_run(info)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        return info

    def end_run(self, status: str = "finished") -> None:
        if not self.active_run:
            return
        self.active_run.status = status
        self.active_run.end_time = _now_ts()
        self._save_run(self.active_run)
        self.active_run = None

    @contextmanager
    def run(self, tags: Optional[Dict[str, Any]] = None):
        info = self.start_run(tags=tags)
        try:
            yield info
            self.end_run("finished")
        except Exception:
            self.end_run("failed")
            raise

    # ----- Logging -----
    def log_params(self, params: Dict[str, Any]) -> None:
        run = self._need_run()
        run.params.update(params)
        self._save_run(run)
        with open(self._run_dir(run) / "params.json", "w", encoding="utf-8") as f:
            json.dump(run.params, f, indent=2)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        run = self._need_run()
        run.metrics[key] = float(value)
        self._save_run(run)
        mpath = self._run_dir(run) / "metrics.csv"
        df = pd.DataFrame(
            [{"step": step if step is not None else 0, "metric": key, "value": float(value), "ts": _now_ts()}]
        )
        if mpath.exists():
            df.to_csv(mpath, mode="a", header=False, index=False)
        else:
            df.to_csv(mpath, index=False)

    def log_artifact(self, local_path: Path, artifact_name: Optional[str] = None) -> Path:
        run = self._need_run()
        adir = self._run_dir(run) / "artifacts"
        adir.mkdir(parents=True, exist_ok=True)
        dest = adir / (artifact_name or local_path.name)
        dest.write_bytes(Path(local_path).read_bytes())
        return dest

    # ----- Query -----
    def list_runs(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for p in sorted(self.root.iterdir()):
            if not p.is_dir():
                continue
            meta = p / "run.json"
            if not meta.exists():
                continue
            try:
                info = json.loads(meta.read_text())
                rows.append(
                    {
                        "run_id": info.get("run_id"),
                        "status": info.get("status"),
                        "start_time": info.get("start_time"),
                        "end_time": info.get("end_time"),
                        "tags": info.get("tags", {}),
                        "params": info.get("params", {}),
                        "metrics": info.get("metrics", {}),
                    }
                )
            except Exception:
                continue
        if not rows:
            return pd.DataFrame(columns=["run_id", "status", "start_time", "end_time", "tags", "params", "metrics"])
        df = pd.DataFrame(rows)
        df["start_time"] = pd.to_datetime(df["start_time"], unit="s", errors="coerce")
        df["end_time"] = pd.to_datetime(df["end_time"], unit="s", errors="coerce")
        return df.sort_values("start_time", ascending=False)

    def load_metrics_long(self, run_id: str) -> pd.DataFrame:
        mpath = self.root / run_id / "metrics.csv"
        if not mpath.exists():
            return pd.DataFrame(columns=["step", "metric", "value", "ts"])
        df = pd.read_csv(mpath)
        df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
        return df

    def artifacts_dir(self, run_id: str) -> Path:
        return self.root / run_id / "artifacts"

    # ----- internals -----
    def _need_run(self) -> RunInfo:
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        return self.active_run

    def _run_dir(self, run: RunInfo) -> Path:
        return self.root / run.run_id

    def _save_run(self, run: RunInfo) -> None:
        meta = self._run_dir(run) / "run.json"
        meta.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")