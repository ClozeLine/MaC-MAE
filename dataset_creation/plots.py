import rasterio
import matplotlib.pyplot as plt
from dataset_creation.imagery import read_crop, stretch, nodata_fraction, quad_url, quad_name

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