import numpy as np
import pandas as pd

from dataset_creation.extract import OUT_DIR, METADATA_CSV

DATASET_CSV = OUT_DIR / "dataset.csv"
SPLIT_FRAC = (0.70, 0.15, 0.15)


def block_id(meta: pd.DataFrame, block_deg: float = 10) -> pd.Series:
    """Grid cell of each crater: block_deg x block_deg squares over the ±30° band."""
    col = np.floor((meta.lon + 180) / block_deg).clip(0, 360 / block_deg - 1)
    row = np.floor((meta.lat + 30) / block_deg).clip(0, 60 / block_deg - 1)
    return (row * 1000 + col).astype(int)


def assign_split(meta: pd.DataFrame, seed: int, block_deg: float = 10) -> pd.Series:
    """Randomly assign whole blocks to train/val/test, by crater count."""
    blocks = block_id(meta, block_deg)
    counts = blocks.value_counts().sample(frac=1, random_state=seed)  # shuffled blocks
    cum = counts.cumsum() / counts.sum()
    split_of = pd.Series(np.select([cum <= SPLIT_FRAC[0], cum <= SPLIT_FRAC[0] + SPLIT_FRAC[1]],
                                   ["train", "val"], "test"), index=counts.index)
    return blocks.map(split_of)


def imbalance(meta: pd.DataFrame, split: pd.Series) -> float:
    """Largest difference (percentage points) between a split's class share and the overall share."""
    overall = meta.label.value_counts(normalize=True)
    per_split = pd.crosstab(split, meta.label, normalize="index")
    return float(100 * (per_split - overall).abs().max().max())


def best_split(meta: pd.DataFrame, n_seeds: int = 500) -> tuple[pd.Series, int]:
    """Try many seeds; keep the most class-balanced split"""
    scores = {s: imbalance(meta, assign_split(meta, s)) for s in range(n_seeds)}
    seed = min(scores, key=scores.get)
    print(f"seed {seed}: max class-share difference {scores[seed]:.2f} pp")
    return assign_split(meta, seed), seed

def complete_only(m: pd.DataFrame) -> pd.DataFrame:
    """Quality control, keep only crops without missing pixels"""
    return m[m.nodata_frac == 0]

if __name__ == "__main__":

    meta = pd.read_csv(METADATA_CSV)
    n_all = len(meta)
    meta = complete_only(meta).reset_index(drop=True)
    print(f"removed {n_all - len(meta)} of {n_all} crops with missing pixels")

    meta["split"], seed = best_split(meta)
    print((100 * meta.split.value_counts(normalize=True)).round(2))
    print((100 * pd.crosstab(meta.split, meta.label, normalize="index")).round(2))

    meta.to_csv(DATASET_CSV, index=False)
    print(f"saved {DATASET_CSV} ({len(meta)} craters, seed {seed})")