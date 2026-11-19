"""
(Re)compute metrics for any runs from their predictions.csv, and compare them.

    uv run python -m modeling.evaluate                      # all runs, test split
    uv run python -m modeling.evaluate baseline_diameter    # selected runs
    uv run python -m modeling.evaluate --split val
    uv run python -m modeling.evaluate --by size_bin        # also per subgroup (size_bin or period)
"""
import argparse
import json

import pandas as pd

from modeling.features import groups
from modeling.metrics import SIZE_NAMES, evaluate, summary
from modeling.runs import RUNS_DIR, finished_runs, load_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run names (default: all in runs/)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--by", choices=["size_bin", "period"], help="also show macro-F1 per subgroup")
    args = ap.parse_args()

    names = args.runs or finished_runs()
    if not names:
        print(f"no runs with predictions in {RUNS_DIR} yet; run e.g. `python -m modeling.baselines` first")
        return
    g = groups()
    rows, by_rows = {}, {}
    for name in names:
        result = evaluate(load_predictions(RUNS_DIR / name), g)
        (RUNS_DIR / name / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
        rows[name] = summary(result, args.split)
        if args.by:
            by_rows[name] = {k: v["macro_f1"] for k, v in result[args.split][args.by].items()}

    print(f"--- {args.split} ---")
    print(pd.DataFrame(rows).T.round(4).to_string())
    if args.by:
        print(f"\nmacro-F1 by {args.by}:")
        table = pd.DataFrame(by_rows).T
        if args.by == "size_bin":
            table = table[[c for c in SIZE_NAMES if c in table]]
        print(table.round(4).to_string())


if __name__ == "__main__":
    main()