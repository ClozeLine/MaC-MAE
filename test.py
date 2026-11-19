import pandas as pd
from dataset_creation.split import DATASET_CSV

d = pd.read_csv(DATASET_CSV)
print(d.groupby("label").diam_km.agg(["count", "mean", "median"]).round(2))
print(f"\nSpearman rank correlation (diameter vs class): {d.diam_km.corr(d.label, method='spearman'):.2f}")