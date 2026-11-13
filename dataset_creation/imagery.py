import os
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "YES"
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_VERSION"] = "2"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["VSI_CACHE"] = "TRUE"

import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import from_bounds

COG_BASE = "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Mars/Mars_MRO_CTX_Equi_Mosaics_Robbins/6mpp"
NODATA = 32768

def quad_name(lat: float, lon: float) -> str:
    """Mosaic quad containing (lat, lon) (e.g., 'MC15'. 45° wide; MC08-15 north, MC16-23 south). 8 slots"""
    lon = (lon + 180) % 360 - 180
    slot = min(int((lon + 180) // 45 ), 7)
    return f"MC{8 + slot if lat >= 0 else 16 + slot:02d}"

def quad_url(name: str) -> str:
    return f"/vsicurl/{COG_BASE}/{name}_6mpp.16bit.tif"

def read_crop(ds, lat: float, lon: float, diam_km: float,
              context: float = 3.0, out_px: int = 1024) -> tuple[np.ndarray, bool]:
    """Square crop of side context x diameter centered on the crater (as uint16)"""
    xs, ys = transform("+proj=longlat +R=3396190 +no_defs", ds.crs, [lon], [lat])

    # undo east-west stretch of the projection (half_x)
    half_y = context * diam_km * 1000 / 2
    half_x = half_y / np.cos(np.radians(lat))

    window = from_bounds(xs[0] - half_x, ys[0] - half_y, xs[0] + half_x, ys[0] + half_y, ds.transform)

    # does the crop window stick out past the edge of the quad tile? (T/F)
    crosses = (
            window.col_off < 0
            or window.row_off < 0
            or window.col_off + window.width > ds.width
            or window.row_off + window.height > ds.height
    )

    # if img extends out of bounds, fill it with NODATA so it can be tracked
    img = ds.read(1, window=window, out_shape=(out_px, out_px), boundless=crosses, fill_value=-32768)

    return img.view(np.uint16), crosses

def stretch(img: np.ndarray) -> tuple[float, float]:
    """2-98 percentile of valid pixels, for display"""
    return tuple(np.percentile(img[img != NODATA], [2, 98]))

def nodata_fraction(img: np.ndarray) -> float:
    """Share of pixels with no CTX coverage."""
    return float((img == NODATA).mean())