import json
import hashlib
import threading
import time
import numpy as np
import pandas as pd
import rasterio
import subprocess

from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm

from dataset_creation.catalogue import load_catalogue, filter_region
from dataset_creation.imagery import COG_BASE, NODATA, quad_name, quad_url, read_crop, nodata_fraction

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "data" / "robbins_hynek_2014_v2.txt"
OUT_DIR = ROOT / "data" / "crops"
METADATA_CSV = OUT_DIR / "metadata.csv"
CONTEXT = 3.0
OUT_PX = 1024
WORKERS = 16
_local = threading.local()

FIELDS = ["crater_id", "lat", "lon", "diam_km", "label", "quad", "status", "error", "nodata_frac",
          "crosses_quad", "side_km", "m_per_px", "p2", "p98", "mean", "std", "path", "seconds"]

def get_ds(quad: str):
    """Per-thread cache of open quads"""
    if not hasattr(_local, "ds"):
        _local.ds = {}
    if quad not in _local.ds:
        _local.ds[quad] = rasterio.open(quad_url(quad))
    return _local.ds[quad]

def write_manifest():
    """Run settings. This is so data can be reproduced"""
    path = OUT_DIR / "manifest.json"
    if path.exists():
        return
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    path.write_text(json.dumps({
        "catalogue": str(CATALOGUE.name),
        "mosaic": COG_BASE,
        "context": CONTEXT,
        "out_px": OUT_PX,
        "projection": "x-window widened by 1/cos(lat), resampled to square",
        "nodata": f"{NODATA} (uint16), boundless reads past quad edges",
        "git_commit": commit or None,
        "started": datetime.now().isoformat(timespec="seconds"),
        "catalogue_sha256": hashlib.sha256(CATALOGUE.read_bytes()).hexdigest(),
    }, indent=2))

def valid_stats(img: np.ndarray) -> dict:
    """Intensity stats over pixels with data (NaN if crop empty)"""
    v = img[img != NODATA]
    if v.size == 0:
        return dict(p2=np.nan, p98=np.nan, mean=np.nan, std=np.nan)
    p2, p98 = np.percentile(v, [2, 98])
    return dict(p2=float(p2), p98=float(p98), mean=float(v.mean()), std=float(v.std()))

def extract_one(ds, row) -> dict:
    img, crosses = read_crop(ds, row.lat, row.lon, row.diam_km, CONTEXT, OUT_PX)
    rel_path = f"images/{row.crater_id}.png"

    # 16-bit png, raw vals
    Image.fromarray(img).save(OUT_DIR / rel_path)

    side_km = CONTEXT * row.diam_km
    return dict(status="ok", error="", nodata_frac=nodata_fraction(img), crosses_quad=crosses,
                side_km=side_km, m_per_px=side_km * 1000 / OUT_PX, **valid_stats(img), path=rel_path)

def append_row(rec: dict):
    pd.DataFrame([rec], columns=FIELDS).to_csv(
        METADATA_CSV, mode="a", header=not METADATA_CSV.exists(),index=False)

def already_done() -> set:
    """Crater ids that were successfully extracted already (erroneous ones get rerun)"""
    if not METADATA_CSV.exists():
        return set()
    metadata = pd.read_csv(METADATA_CSV) # noqa: PyCharm overload false positive
    return set(metadata.loc[metadata.status == "ok", "crater_id"])

def process(row) -> dict:
    base = dict(crater_id=row.crater_id, lat=row.lat, lon=row.lon, diam_km=row.diam_km,
                label=int(row.label), quad=row.quad)
    t0 = time.perf_counter()
    try:
        rec = base | extract_one(get_ds(row.quad), row)
    except Exception as e:
        rec = base | dict(status="error", error=repr(e))
    return rec | dict(seconds=round(time.perf_counter() - t0, 2))

def main(limit: int | None = None):
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    write_manifest()

    band = filter_region(load_catalogue(CATALOGUE), lat_range=(-30, 30), lon_range=(-180, 180))
    band = band.assign(quad=[quad_name(la, lo) for la, lo in zip(band.lat, band.lon)])
    todo = band[~band.crater_id.isin(already_done())]

    if limit:
        todo = todo.sample(min(limit, len(todo)), random_state=42)
    print(f"{len(band)} labelled craters, {len(todo)} to extract")

    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(process, row) for row in todo.itertuples()]
        try:
            for f in tqdm(as_completed(futures), total=len(futures)):
                append_row(f.result())
        except KeyboardInterrupt:
            print("\nStopping: cancelling queued craters (in-flight ones will finish)...")
            pool.shutdown(cancel_futures=True)
            raise

if __name__ == "__main__":
    main(limit=None)