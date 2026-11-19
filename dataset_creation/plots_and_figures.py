import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from dataset_creation.imagery import read_crop, stretch, nodata_fraction, quad_url, quad_name
from dataset_creation.extract import CONTEXT, OUT_DIR, ROOT
from dataset_creation.split import DATASET_CSV, block_id
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import box
from PIL import Image

ERA_ORDER = {"N": 0, "H": 1, "A": 2}
ERA_NAMES = {"N": "Noachian", "H": "Hesperian", "A": "Amazonian"}
ERA_HATCH = {"Noachian": "", "Hesperian": "////", "Amazonian": "...."}
REGIONS_APPROX = {"Olympus Mons": (-134, 18), "Tharsis": (-112, 2), "Amazonis\nPlanitia": (-160, 8),
                  "Valles Marineris": (-70, -9), "Chryse\nPlanitia": (-42, 25), "Arabia Terra": (5, 20),
                  "Syrtis Major": (66, 2), "Isidis\nPlanitia": (93, 19), "Elysium": (147, 23),
                  "Hesperia\nPlanum": (110, -22)}
TEXT_WIDTH_IN = 4.8          # \the\textwidth in pt / 72.27
FIG_DIR = ROOT / "figures"
SPLIT_COLORS = {"train": "#6666FF", "val": "#FF9933", "test": "#FF4D4D"}
SPLIT_NAMES = {"train": "Training", "val": "Validation", "test": "Test"}

# class examples
BINS = [3, 5, 10, 20, np.inf]
BIN_NAMES = ["3–5 km", "5–10 km", "10–20 km", "> 20 km"]
CLASS_NAMES = {1: "1 heavily\ndegraded", 2: "2 moderately\ndegraded", 3: "3 lightly\ndegraded", 4: "4 fresh"}
DISPLAY_PX = 384             # downsampled for the PDF (about 300 dpi at this size)
N_APPENDIX = 6               # craters per class in the appendix gallery


def unit_to_era(unit) -> str | None:
    """Oldest era letter in a unit code, e.g. 'lHv' -> Hesperian"""
    eras = [c for c in str(unit) if c in ERA_ORDER]
    return ERA_NAMES[min(eras, key=ERA_ORDER.get)] if eras else None


def load_eras(geology_path) -> gpd.GeoDataFrame:
    """Geologic map units merged into the three eras, in plain lon/lat degrees"""
    geo = gpd.read_file(geology_path)[["Unit", "geometry"]]
    geo = geo.to_crs("+proj=longlat +R=3396190 +no_defs")
    geo["era"] = geo.Unit.map(unit_to_era)
    return geo.dropna(subset=["era"]).dissolve(by="era").reset_index()


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

def plot_nodata_gallery(meta: pd.DataFrame, n: int = 16, seed: int = 0):
    """Random crops with more than 5% NODATA"""
    bad = meta[meta.nodata_frac > 0.05].sample(n, random_state=seed).sort_values("nodata_frac")
    fig, axes = plt.subplots(n // 4, 4, figsize=(12, 3 * n // 4))
    for ax, row in zip(axes.flat, bad.itertuples()):
        img = np.array(Image.open(OUT_DIR / row.path))
        lo, hi = stretch(img)
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"class {row.label} | {row.diam_km:.1f} km | nd {row.nodata_frac:.0%}"
                     f"{' | edge' if row.crosses_quad else ''}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def plot_map(meta: pd.DataFrame, basemap_path, geology_path, title: str = "", seed: int = 42):
    """Crater locations by class on a Mars map, with geologic periods in the +-30 deg band.
    Drawn at the printed width, so font sizes are the sizes in the paper."""
    class_colors = {1: "#A50F15",  # dark red (heavily degraded)
                    2: "#D95F02",  # dark orange
                    3: "#08519C",  # dark blue
                    4: "#006D2C"}  # dark green (fresh)
    class_names = {1: "1 (heavily degraded)", 2: "2 (moderately degraded)",
                   3: "3 (lightly degraded)", 4: "4 (fresh)"}
    plt.rcParams.update({"font.size": 7, "legend.fontsize": 6, "legend.title_fontsize": 6,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "hatch.linewidth": 0.25})

    with rasterio.open(basemap_path) as ds:
        base = ds.read(1)
    base = np.roll(base, base.shape[1] // 2, axis=1)   # basemap starts at 0 deg E

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN / 2))
    ax.imshow(base, cmap="gray", extent=(-180, 180, -90, 90))

    for row in load_eras(geology_path).clip(box(-180, -30, 180, 30)).itertuples():
        gpd.GeoSeries([row.geometry]).plot(ax=ax, facecolor="none", edgecolor=(1, 1, 1, 0.45),
                                           hatch=ERA_HATCH[row.era], linewidth=0.3)
    for name, (lon, lat) in REGIONS_APPROX.items():
        ax.text(lon, lat, name, color="white", fontsize=4.5, ha="center", va="center", linespacing=0.9,
                path_effects=[pe.withStroke(linewidth=1, foreground="black")])

    pts = meta.sample(frac=1, random_state=seed)        # shuffle so no class is drawn on top
    ax.scatter(pts.lon, pts.lat, s=0.15, c=pts.label.map(class_colors), linewidths=0, rasterized=True)
    for lat in (-30, 30):
        ax.axhline(lat, color="white", lw=0.5, ls="--")

    class_handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=3, label=class_names[k])
                     for k, c in class_colors.items()]
    era_handles = [Patch(facecolor="#4d4d4d", edgecolor="white", hatch=h, hatch_linewidth=0.5,
                         linewidth=0.3, label=e) for e, h in ERA_HATCH.items()]

    LEG = dict(framealpha=0.9, fontsize=5, title_fontsize=5.5, borderpad=0.3,
               labelspacing=0.25, handletextpad=0.4, borderaxespad=0.3)

    leg = ax.legend(handles=class_handles, title="Degradation state", loc="lower left", **LEG)
    ax.add_artist(leg)
    ax.legend(handles=era_handles, title="Geologic period", loc="lower right",
              handlelength=1.6, handleheight=1.0, **LEG)

    ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)", title=title,
           xticks=range(-180, 181, 60), yticks=range(-90, 91, 30))

    out = FIG_DIR / "crater_map.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")


def _display_image(row) -> np.ndarray:
    """Saved crop with a 2-98% stretch, downsampled for print."""
    img = np.array(Image.open(OUT_DIR / row.path))
    lo, hi = stretch(img)
    img8 = (255 * np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)).astype(np.uint8)
    return np.array(Image.fromarray(img8).resize((DISPLAY_PX, DISPLAY_PX), Image.LANCZOS))


def _bar_km(width_km: float) -> float:
    """Round scale-bar length: the largest of 1, 2, 5, 10, ... km up to ~30% of the image width."""
    nice = [n * 10 ** e for e in range(-1, 4) for n in (1, 2, 5)]
    return max(n for n in nice if n <= 0.3 * width_km)


def _blank(ax):
    """No ticks or frame, but keeps the axis labels (unlike axis("off"))."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _show_crater(ax, row):
    """Crater image with its diameter (top left) and a scale bar (bottom right)."""
    ax.imshow(_display_image(row), cmap="gray", vmin=0, vmax=255)
    _blank(ax)

    stroke = [pe.withStroke(linewidth=1, foreground="black")]
    ax.text(0.04, 0.96, f"$D$ = {row.diam_km:.1f} km", transform=ax.transAxes, color="white",
            fontsize=5.5, va="top", path_effects=stroke)

    width_km = CONTEXT * row.diam_km                 # image is CONTEXT diameters wide (and tall)
    length = _bar_km(width_km)
    x1 = 0.95 * DISPLAY_PX
    x0 = x1 - length / width_km * DISPLAY_PX
    y = 0.93 * DISPLAY_PX
    ax.plot([x0, x1], [y, y], color="white", lw=1.5, solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=2.3, foreground="black")])
    ax.text(x1, y - 0.03 * DISPLAY_PX, f"{length:g} km", color="white", fontsize=5,
            ha="right", va="bottom", path_effects=stroke)


def _test_with_bins(data: pd.DataFrame) -> pd.DataFrame:
    test = data[data.split == "test"]
    return test.assign(bin=pd.cut(test.diam_km, BINS, labels=BIN_NAMES, right=False))


def plot_class_examples(data: pd.DataFrame, seed: int = 0) -> list[str]:
    """Main-text figure: one random test crater per class (rows) and diameter bin (columns).
    Returns the crater IDs shown."""
    test = _test_with_bins(data)
    cell = TEXT_WIDTH_IN / 4.6                       # room for the row labels
    fig, axes = plt.subplots(4, 4, figsize=(TEXT_WIDTH_IN, 4 * cell),
                             gridspec_kw=dict(wspace=0.04, hspace=0.04))
    used = []
    for i, label in enumerate(CLASS_NAMES):
        for j, b in enumerate(BIN_NAMES):
            ax = axes[i, j]
            _blank(ax)
            cand = test[(test.label == label) & (test.bin == b)]
            if len(cand):
                row = cand.sample(1, random_state=seed).iloc[0]
                _show_crater(ax, row)
                used.append(row.crater_id)
            if i == 0:
                ax.set_title(b, fontsize=7)
            if j == 0:
                ax.set_ylabel(CLASS_NAMES[label], fontsize=7)
    out = FIG_DIR / "class_examples.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return used


def plot_class_gallery(data: pd.DataFrame, seed: int = 0, exclude: list[str] = ()):
    """Appendix figure: N_APPENDIX random test craters per class (columns), sorted by diameter."""
    test = data[(data.split == "test") & ~data.crater_id.isin(exclude)]
    cell = TEXT_WIDTH_IN / 4
    fig, axes = plt.subplots(N_APPENDIX, 4, figsize=(TEXT_WIDTH_IN, N_APPENDIX * cell),
                             gridspec_kw=dict(wspace=0.04, hspace=0.04))
    for j, label in enumerate(CLASS_NAMES):
        pick = test[test.label == label].sample(N_APPENDIX, random_state=seed + 1).sort_values("diam_km")
        for i, row in enumerate(pick.itertuples()):
            _show_crater(axes[i, j], row)
        axes[0, j].set_title(CLASS_NAMES[label], fontsize=7)
    out = FIG_DIR / "class_gallery.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def plot_split_map(data: pd.DataFrame, basemap_path, block_deg: int = 10):
    """The +-30 deg band on the basemap, each 10x10 deg block coloured by its split."""
    plt.rcParams.update({"font.size": 7, "legend.fontsize": 6,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    with rasterio.open(basemap_path) as ds:
        base = ds.read(1)
    base = np.roll(base, base.shape[1] // 2, axis=1)  # basemap starts at 0 deg E
    rows = base.shape[0]
    band = base[int(rows * 60 / 180): int(rows * 120 / 180)]  # 30N .. 30S

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN / 5))
    ax.imshow(band, cmap="gray", extent=(-180, 180, -30, 30))
    blocks = data.assign(block=block_id(data, block_deg)).groupby("block").split.first()
    for b, split in blocks.items():
        row, col = divmod(int(b), 1000)
        ax.add_patch(Rectangle((-180 + col * block_deg, -30 + row * block_deg), block_deg, block_deg,
                               facecolor=SPLIT_COLORS[split], alpha=0.55, edgecolor="white", linewidth=0.3))

    handles = [Patch(facecolor=c, alpha=0.55, label=SPLIT_NAMES[s]) for s, c in SPLIT_COLORS.items()]
    (ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False),
     ax.set(xlim=(-180, 180), ylim=(-30, 30), xticks=range(-180, 181, 60), yticks=(-30, 0, 30),
           xlabel="Longitude (°E)", ylabel="Latitude (°N)"))
    out = FIG_DIR / "split_map.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    data = pd.read_csv(DATASET_CSV)

    plot_split_map(data, basemap_path=ROOT / "data" / "mars_basemap.tif")