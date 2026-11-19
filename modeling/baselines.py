"""
No-image baselines: logistic regression on crater properties only.

    uv run python -m modeling.baselines

Writes runs/baseline_<name>/ (config, predictions for all splits, metrics) and prints
the test scores next to chance level.
"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modeling.features import crater_features, groups
from modeling.metrics import evaluate, summary
from modeling.runs import mark_finished, run_dir, save_predictions, write_config

BASELINES = {                      # name -> (numeric columns, categorical columns)
    "diameter": (["log_diam"], []),
    "terrain": ([], ["period"]),
    "combined": (["log_diam"], ["period"]),
}


def make_model(numeric: list[str], categorical: list[str]):
    """Class-weighted logistic regression, same weighting idea as the image models."""
    steps = []
    if numeric:
        steps.append(("num", StandardScaler(), numeric))
    if categorical:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return make_pipeline(ColumnTransformer(steps),
                         LogisticRegression(class_weight="balanced", max_iter=1000))


def run_baseline(name: str, f: pd.DataFrame) -> dict:
    numeric, categorical = BASELINES[name]
    out = run_dir(f"baseline_{name}")
    write_config(out, {"type": "baseline", "name": name, "model": "logistic_regression",
                       "class_weight": "balanced", "numeric": numeric, "categorical": categorical})

    train = f[f.split == "train"]
    model = make_model(numeric, categorical).fit(train[numeric + categorical], train.label)
    probs = model.predict_proba(f[numeric + categorical])       # columns ordered as model.classes_ = 1..4
    assert list(model.classes_) == [1, 2, 3, 4]
    pred = save_predictions(out, f.crater_id, f.split, f.label, probs)

    result = evaluate(pred, groups())
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
    mark_finished(out)
    return result


def chance(f: pd.DataFrame) -> float:
    """Expected test macro-F1 of guessing classes in proportion to their frequency (= 1 / n_classes)."""
    shares = f[f.split == "test"].label.value_counts(normalize=True)
    return float(np.mean([2 * q * q / (q + q) for q in shares]))


if __name__ == "__main__":
    f = crater_features()
    rows = {name: summary(run_baseline(name, f)) for name in BASELINES}
    table = pd.DataFrame(rows).T
    print(table.round(4).to_string())
    print(f"\nchance (frequency-proportional guessing), macro-F1: {chance(f):.4f}")