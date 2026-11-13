import pandas as pd
from pathlib import Path

COLUMNS = {
    "CRATER_ID": "crater_id",
    "LATITUDE_CIRCLE_IMAGE": "lat",
    "LONGITUDE_CIRCLE_IMAGE": "lon",
    "DIAM_CIRCLE_IMAGE": "diam_km",
    "DEGRADATION_STATE": "label"
}

def load_catalogue(path: str | Path) -> pd.DataFrame:
    """Load R&H catalogue. Keep only columns needed (COLUMNS)"""
    cat = pd.read_csv(path, sep="\t", encoding="latin-1", usecols=list(COLUMNS), low_memory=False)
    return cat.rename(columns=COLUMNS)

def filter_region(cat: pd.DataFrame, lat_range: tuple, lon_range: tuple,
                  min_diam_km: float = 3.0) -> pd.DataFrame:
    """Find craters inside lat/lon box above a min diameter (km)"""
    mask = (
        cat.lat.between(*lat_range)
        & cat.lon.between(*lon_range)
        & (cat.diam_km > min_diam_km)
        & cat.label.notna()
    )
    return cat[mask]

def sample_craters(cat: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    return cat.groupby("label").sample(n=n_per_class, random_state=seed)
