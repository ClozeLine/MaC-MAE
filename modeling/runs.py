"""
Run folders: every experiment (baseline or network) writes the same files, so all
analysis can be done later from disk without rerunning anything.

    runs/<name>/
        config.json        settings, seed, git commit, package versions, start/end time
        predictions.csv    one row per crater and split: label, prediction, class probabilities
        metrics.json       metrics computed from predictions.csv (can always be recomputed)
"""
import json
import platform
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_creation.extract import ROOT

RUNS_DIR = ROOT / "runs"
CLASSES = [1, 2, 3, 4]
PROB_COLS = [f"p{c}" for c in CLASSES]
PACKAGES = ["numpy", "pandas", "scikit-learn", "torch", "timm"]


def _git_commit() -> str | None:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout
    return (out.stdout.strip() + ("-dirty" if dirty.strip() else "")) or None


def _versions() -> dict:
    found = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            found[p] = version(p)
        except PackageNotFoundError:
            pass
    return found


def run_dir(name: str) -> Path:
    path = RUNS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def finished_runs() -> list[str]:
    """Names of all runs that have predictions (creates runs/ if it does not exist yet)."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in RUNS_DIR.iterdir() if (p / "predictions.csv").exists())


def write_config(path: Path, config: dict):
    """Save the settings plus everything needed to reproduce the run."""
    full = dict(config, git_commit=_git_commit(), versions=_versions(), host=platform.node(),
                started=datetime.now().isoformat(timespec="seconds"))
    (path / "config.json").write_text(json.dumps(full, indent=2, default=str))


def mark_finished(path: Path):
    cfg = json.loads((path / "config.json").read_text())
    cfg["finished"] = datetime.now().isoformat(timespec="seconds")
    (path / "config.json").write_text(json.dumps(cfg, indent=2, default=str))


def save_predictions(path: Path, crater_id, split, label, probs: np.ndarray):
    """One row per crater: true label, predicted class and the probability of each class."""
    probs = np.asarray(probs, dtype=float)
    df = pd.DataFrame({"crater_id": crater_id, "split": split, "label": np.asarray(label, dtype=int)})
    df["pred"] = np.array(CLASSES)[probs.argmax(axis=1)]
    df[PROB_COLS] = probs
    df.to_csv(path / "predictions.csv", index=False, float_format="%.6f")
    return df


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path) / "predictions.csv")
    df[PROB_COLS] = df[PROB_COLS].div(df[PROB_COLS].sum(axis=1), axis=0)   # undo rounding from the csv
    return df