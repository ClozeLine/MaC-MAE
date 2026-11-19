"""
Image cache: every crop of the final dataset resized once and stored in a single uint16 array
on disk, so training reads ~3.5 GB instead of decoding 53 GB of 1024 px PNGs every epoch.

    uv run python -m modeling.cache                 # 224 px (default)
    uv run python -m modeling.cache --size 384      # another resolution, kept side by side

Writes data/cache/
    images_<size>.npy       (N, size, size) uint16, original brightness values (no normalisation)
    index_<size>.csv        crater_id of each row of the array
    manifest_<size>.json    settings, source and completion time
Resumable: an interrupted build continues where it stopped. The cache does not depend on the
split or labels (those are looked up by crater_id), so re-splitting never requires a rebuild.
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from dataset_creation.extract import OUT_DIR, ROOT
from dataset_creation.imagery import NODATA
from dataset_creation.split import DATASET_CSV

CACHE_DIR = ROOT / "data" / "cache"
FLUSH_EVERY = 1000


def _paths(size: int) -> dict:
    return {k: CACHE_DIR / f"{k}_{size}.{ext}"
            for k, ext in [("images", "npy"), ("index", "csv"), ("done", "npy"), ("manifest", "json")]}


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Area-average downsampling (box filter) of a 16-bit crop; keeps the original value range."""
    small = np.asarray(Image.fromarray(img.astype(np.float32)).resize((size, size), Image.BOX))
    return np.clip(np.rint(small), 0, 65535).astype(np.uint16)


def build(size: int = 224, workers: int = 8):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _paths(size)
    data = pd.read_csv(DATASET_CSV)

    if p["index"].exists(): # resuming -> keep the existing row order
        index = pd.read_csv(p["index"])
        missing = set(data.crater_id) - set(index.crater_id)
        if missing:
            raise RuntimeError(f"{len(missing)} craters in dataset.csv are not in the cache; "
                               f"delete data/cache/*_{size}.* and rebuild")
        data = index.merge(data[["crater_id", "path"]], on="crater_id", how="left")
    else:
        data[["crater_id"]].to_csv(p["index"], index=False)

    n = len(data)
    if p["images"].exists():
        images = np.load(p["images"], mmap_mode="r+")
    else:
        images = np.lib.format.open_memmap(p["images"], mode="w+", dtype=np.uint16, shape=(n, size, size))
    done = np.load(p["done"]) if p["done"].exists() else np.zeros(n, dtype=bool)
    todo = np.flatnonzero(~done)
    print(f"{n} craters, {done.sum()} already cached, {len(todo)} to do")

    def work(i: int) -> int:
        img = np.array(Image.open(OUT_DIR / data.path.iat[i]))
        if (img == NODATA).any():
            raise ValueError(f"{data.crater_id.iat[i]} contains missing pixels; it should not be in dataset.csv")
        images[i] = resize(img, size)
        return i

    with ThreadPoolExecutor(workers) as pool:
        for k, i in enumerate(tqdm(pool.map(work, todo), total=len(todo)), start=1):
            done[i] = True
            if k % FLUSH_EVERY == 0: # save progress for resuming
                images.flush()
                np.save(p["done"], done)
    images.flush()
    np.save(p["done"], done)

    if done.all():
        p["manifest"].write_text(json.dumps({
            "size": size, "n": n, "dtype": "uint16", "resampling": "box (area average) from 1024 px",
            "values": "original 16-bit brightness, not normalised",
            "dataset_sha256": hashlib.sha256(DATASET_CSV.read_bytes()).hexdigest(),
            "finished": datetime.now().isoformat(timespec="seconds"),
        }, indent=2))
        print(f"done: {p['images']} ({p['images'].stat().st_size / 1e9:.2f} GB)")


def load(size: int = 224) -> tuple[np.ndarray, pd.Series]:
    """The cached images (memory-mapped, read lazily) and a crater_id -> row lookup."""
    p = _paths(size)
    if not p["manifest"].exists():
        raise FileNotFoundError(f"no complete cache for size {size}; run `python -m modeling.cache --size {size}`")
    index = pd.read_csv(p["index"]).crater_id
    return np.load(p["images"], mmap_mode="r"), pd.Series(np.arange(len(index)), index=index)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    build(args.size, args.workers)