#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc

# ---- Project imports ----
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))
from src.paths.paths import predictions_dir, masks_dir
from src.dataset.variables import var_names  # use your lists

# grid (0.5° global)
LAT_ALL = np.arange(-89.75, 90.0, 0.5, dtype="float32")   # 360
LON_ALL = np.arange(0.0, 360.0, 0.5, dtype="float32")     # 720
NY, NX = len(LAT_ALL), len(LON_ALL)

# daily outputs (you said: predict daily for all vars)
DAILY_OUT = list(var_names["outputs"])

# chunking / period
TILE_T, TILE_Y, TILE_X = 365, 30, 30
PERIOD = ("1901-01-01", "2023-12-31")

FORCING_MASK = masks_dir / "forcing_nan_mask.nc"

def make_time_axis_days_since_1901(start: str, end: str) -> np.ndarray:
    import cftime
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    dates = xr.cftime_range(start=start, end=end, freq="D", calendar="noleap")
    return np.asarray([(d - ref).days for d in dates], dtype="int32")

def load_forcing_mask(mask_nc: Path):
    if not mask_nc.exists(): return None
    with xr.open_dataset(mask_nc, decode_times=False) as ds:
        name = next((v for v in ds.data_vars if ds[v].ndim == 2), None)
        if name is None: return None
        arr = ds[name].values
    return arr.astype(bool)  # 1=valid, 0=invalid

def tile_windows(ny, nx, ysize, xsize):
    for y0 in range(0, ny, ysize):
        y1 = min(y0 + ysize, ny)
        for x0 in range(0, nx, xsize):
            x1 = min(x0 + xsize, nx)
            yield slice(y0, y1), slice(x0, x1)

def keep_tile(mask, ys: slice, xs: slice) -> bool:
    if mask is None: return True
    return np.any(mask[ys, xs])

def ensure_store_root(root_store_path: Path, variables, lat, lon) -> None:
    """
    Create/open a Zarr store and ensure root contains:
      - time/lat/lon coords
      - one array per variable
    """
    import zarr
    from numcodecs import Blosc

    t0, t1 = PERIOD
    time_days = make_time_axis_days_since_1901(t0, t1)
    T, Y, X = len(time_days), len(lat), len(lon)

    store = zarr.DirectoryStore(str(root_store_path))
    root  = zarr.open_group(store=store, mode="a")   # <-- root (no subpath)

    # coords (tiny, uncompressed)
    if "time" not in root:
        root.create_dataset("time", shape=(T,), chunks=(T,), dtype="i4",
                            compressor=None, overwrite=False)[:] = time_days
        root["time"].attrs.update({"units": "days since 1901-01-01 00:00:00",
                                   "calendar": "noleap"})
    if "lat" not in root:
        root.create_dataset("lat", shape=(Y,), chunks=(Y,), dtype="f4",
                            compressor=None, overwrite=False)[:] = lat.astype("float32")
    if "lon" not in root:
        root.create_dataset("lon", shape=(X,), chunks=(X,), dtype="f4",
                            compressor=None, overwrite=False)[:] = lon.astype("float32")

    comp = Blosc(cname="zstd", clevel=4, shuffle=Blosc.SHUFFLE)
    tchunk, ychunk, xchunk = TILE_T, TILE_Y, TILE_X
    for v in variables:
        if v in root:
            continue
        d = root.create_dataset(
            v, shape=(T, Y, X), chunks=(tchunk, ychunk, xchunk),
            dtype="f4", compressor=comp, fill_value=np.float32(np.nan),
            overwrite=False,
        )
        d.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]

    root.attrs.update({"dimensions": ["time", "lat", "lon"]})
    zarr.consolidate_metadata(store)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_name", required=True)
    args = ap.parse_args()

    RUN_ROOT = predictions_dir / args.job_name
    ZARR_DIR = RUN_ROOT / "zarr"
    ZARR_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_STORE = ZARR_DIR / f"{args.job_name}.zarr"
    print(f"[INFO] Output Zarr (root): {ROOT_STORE}")

    mask = load_forcing_mask(FORCING_MASK)

    ensure_store_root(ROOT_STORE, DAILY_OUT, LAT_ALL, LON_ALL)

    tiles = []
    for ys, xs in tile_windows(NY, NX, TILE_Y, TILE_X):
        if keep_tile(mask, ys, xs):
            tiles.append((ys.start, ys.stop, xs.start, xs.stop))

    tiles_path = ZARR_DIR / f"tiles_{TILE_Y}x{TILE_X}.json"
    with open(tiles_path, "w") as f:
        json.dump({
            "ny": NY, "nx": NX,
            "tile_lat": TILE_Y, "tile_lon": TILE_X,
            "tiles": tiles,
            "zarr_store": str(ROOT_STORE)
        }, f)
    print(f"[OK] {len(tiles)} tiles -> {tiles_path}")

if __name__ == "__main__":
    main()