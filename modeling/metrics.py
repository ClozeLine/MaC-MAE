"""
Metrics, computed only from a predictions table (crater_id, split, label, pred, p1..p4),
so every model and baseline is scored the same way.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss, precision_recall_fscore_support)

from modeling.runs import CLASSES, PROB_COLS

SIZE_BINS = [3, 5, 10, 20, np.inf]
SIZE_NAMES = ["3-5 km", "5-10 km", "10-20 km", ">20 km"]

# small subgroups may lack a class; the metrics handle that (zero_division=0), so the warning is noise
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found")
warnings.filterwarnings("ignore", message=".*cohen_kappa_score.*undefined")


def scores(y: np.ndarray, pred: np.ndarray, probs: np.ndarray | None = None) -> dict:
    """All metrics for one set of predictions. Ordinal metrics treat classes 1-4 as ordered."""
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    out = {
        "n": int(len(y)),
        "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "qwk": cohen_kappa_score(y, pred, labels=CLASSES, weights="quadratic"),   # ordinal agreement
        "mae": float(np.abs(y - pred).mean()),                                       # mean class distance
        "within_one": float((np.abs(y - pred) <= 1).mean()),                         # off by at most one class
        "per_class": {c: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(n[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion": confusion_matrix(y, pred, labels=CLASSES).tolist(),             # rows = true, cols = predicted
    }
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float64)
        out["log_loss"] = log_loss(y, probs / probs.sum(axis=1, keepdims=True), labels=CLASSES)  # exact sum 1
    return out


def evaluate(pred: pd.DataFrame, groups: pd.DataFrame | None = None) -> dict:
    """Scores per split, and per subgroup (e.g. size bin, period) if groups is given.
    groups: crater_id plus one column per grouping."""
    if groups is not None:
        pred = pred.merge(groups, on="crater_id", how="left")
    result = {}
    for split, part in pred.groupby("split"):
        y, p = part.label.to_numpy(), part.pred.to_numpy()
        result[split] = {"all": scores(y, p, part[PROB_COLS].to_numpy())}
        for col in ([] if groups is None else [c for c in groups.columns if c != "crater_id"]):
            result[split][col] = {str(g): scores(sub.label.to_numpy(), sub.pred.to_numpy())
                                  for g, sub in part.groupby(col, observed=True)}
    return result


def size_bin(diam_km: pd.Series) -> pd.Series:
    return pd.cut(diam_km, SIZE_BINS, labels=SIZE_NAMES, right=False)


def summary(result: dict, split: str = "test") -> pd.Series:
    """Headline numbers of one run, for tables."""
    s = result[split]["all"]
    return pd.Series({k: s[k] for k in ["macro_f1", "accuracy", "balanced_accuracy", "qwk", "mae", "within_one"]})