"""
Non-image properties of each crater (diameter, geological period, size bin), computed once
and cached. Used by the baselines and to break down results by subgroup.
"""
import numpy as np
import pandas as pd

from dataset_creation.analysis import assign_eras
from dataset_creation.extract import OUT_DIR
from dataset_creation.split import DATASET_CSV
from modeling.metrics import size_bin

FEATURES_CSV = OUT_DIR / "crater_features.csv"


def crater_features(refresh: bool = False) -> pd.DataFrame:
    """crater_id, split, label, diam_km, log_diam, period, size_bin."""
    if FEATURES_CSV.exists() and not refresh:
        return pd.read_csv(FEATURES_CSV)
    d = pd.read_csv(DATASET_CSV)
    eras = assign_eras(d)
    period = eras["era"] if "era" in eras else eras["period"]
    f = pd.DataFrame({
        "crater_id": d.crater_id, "split": d.split, "label": d.label, "diam_km": d.diam_km,
        "log_diam": np.log(d.diam_km), "period": period.to_numpy(), "size_bin": size_bin(d.diam_km),
    })
    f.to_csv(FEATURES_CSV, index=False)
    return f


def groups() -> pd.DataFrame:
    """Subgroups used to break down every run's results."""
    return crater_features()[["crater_id", "size_bin", "period"]]