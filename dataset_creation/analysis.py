import geopandas as gpd
import pandas as pd

from dataset_creation.extract import ROOT, METADATA_CSV
from dataset_creation.split import DATASET_CSV
from dataset_creation.plots_and_figures import unit_to_era

GEOLOGY_SHP = ROOT / "data" / "geology" / "SIM3292_Global_Geology.shp"


def assign_eras(meta: pd.DataFrame, geology_path=GEOLOGY_SHP) -> pd.DataFrame:
    """Add the geologic unit and era of the terrain under each crater"""
    geo = gpd.read_file(geology_path)[["Unit", "geometry"]].to_crs("+proj=longlat +R=3396190 +no_defs")
    pts = gpd.GeoDataFrame(meta, geometry=gpd.points_from_xy(meta.lon, meta.lat), crs=geo.crs)
    joined = gpd.sjoin(pts, geo, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]  # a point on a shared border can match twice
    joined["era"] = joined.Unit.map(unit_to_era)
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))


if __name__ == "__main__":
    meta = pd.read_csv(DATASET_CSV)
    joined = assign_eras(meta)

    print(joined.era.isna().sum(), "craters without an era")
    print(pd.crosstab(joined.era, joined.label, margins=True))

    pct = 100 * pd.crosstab(joined.era, joined.label, normalize="index")
    print(pct.round(2))
    print("classes 1+2 per era:")
    print((pct[1] + pct[2]).round(2))