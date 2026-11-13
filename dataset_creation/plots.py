import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from dataset_creation.imagery import read_crop, stretch, nodata_fraction, quad_url, quad_name
from dataset_creation.extract import OUT_DIR, METADATA_CSV
from PIL import Image

def plot_sample(sample, n_per_class: int):
    """Grid with one column per class, rows sorted by diameter."""

    sample = sample.sort_values(["label", "diam_km"])
    fig, axes = plt.subplots(n_per_class, 4, figsize=(12, 3 * n_per_class))
    for ax, row in zip(axes.T.flat, sample.itertuples()):
        with rasterio.open(quad_url(quad_name(row.lat, row.lon))) as ds:
            img, _ = read_crop(ds, row.lat, row.lon, row.diam_km, out_px=256)
            lo, hi = stretch(img)
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(f"class {int(row.label)} | {row.diam_km:.1f} km | nd {nodata_fraction(img):.0%}", fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    plt.show()

def show_crop(path):
    """Display an 16-bit crop with a 2-98% contrast stretch."""
    img = np.array(Image.open(path))
    lo, hi = stretch(img)
    plt.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":

    meta = pd.read_csv(METADATA_CSV)
    for row in meta[meta.status == "ok"].head(3).itertuples():
        print(row.crater_id, f"class {row.label}", f"{row.diam_km} km", f"nd {row.nodata_frac:.0%}")
        show_crop(OUT_DIR / row.path)