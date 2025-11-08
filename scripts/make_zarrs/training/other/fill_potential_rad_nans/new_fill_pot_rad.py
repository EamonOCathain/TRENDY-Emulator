#!/usr/bin/env python3
"""
Fill NaNs in 'potential_radiation' for a shuffled-location training Zarr
by repeating a single 365-day template year per (location, scenario) that
contains any NaNs.

Target Zarr:
  /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/
    training_new/test/train_location_test_period_early/daily.zarr

Assumptions
-----------
- Variable name: 'potential_radiation'
- Variable dims: (time, scenario, location)   <-- IMPORTANT
- Coords 'lat' and 'lon' exist, both 1-D of length = location
- Zarr time length T is an exact multiple of 365
- Template NetCDF (decode_times=False):
    /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/
    preprocessed/1x1/historical/annual_files/potential_radiation/
    potential_radiation_1901.nc
  with dims (time=365, lat, lon)

Parallelism
-----------
- Per-pixel writes can be fanned out with threads via VAR_WORKERS (default 1).
"""

from __future__ import annotations
import os
import sys
import math
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import xarray as xr
import zarr
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------- CONFIG ----------------------
ZARR_PATH = Path(
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/"
    "training_new/test/train_location_test_period_early/daily.zarr"
)
VAR_NAME = "potential_radiation"

TEMPLATE_NC = Path(
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/"
    "preprocessed/1x1/historical/annual_files/potential_radiation/"
    "potential_radiation_1901.nc"
)

# Threading (per-pixel writes: each is one (location, scenario))
VAR_WORKERS = int(os.getenv("VAR_WORKERS", "1"))

# Tolerance for nearest-neighbour lat/lon match
LATLON_ATOL = float(os.getenv("LATLON_ATOL", "1e-8"))
# ----------------------------------------------------


def setup_logging() -> logging.Logger:
    log = logging.getLogger("fill_prad_nans")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                         datefmt="%Y-%m-%d %H:%M:%S"))
        log.addHandler(h)
    return log


log = setup_logging()
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


# ---------------------- ZARR I/O ----------------------
def open_zarr(zarr_path: Path):
    """Open group, return (root, arr, lats_by_loc, lons_by_loc)."""
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr not found: {zarr_path}")

    root = zarr.open_group(str(zarr_path), mode="a")

    if VAR_NAME not in root:
        raise KeyError(f"Variable '{VAR_NAME}' not found in {zarr_path}")
    arr = root[VAR_NAME]

    if arr.ndim != 3:
        raise RuntimeError(f"Expected (time, scenario, location) array; got shape {arr.shape}")

    T, S, L = arr.shape
    log.info("Opened '%s' with shape (time=%d, scenario=%d, location=%d)", VAR_NAME, T, S, L)

    if T % 365 != 0:
        raise RuntimeError(f"time length {T} is not a multiple of 365 (required).")

    # Coordinates written by ensure_training_skeleton: 1-D per-location
    if "lat" not in root or "lon" not in root:
        raise KeyError("Expected 1-D 'lat' and 'lon' coordinates in Zarr.")
    lats = np.array(root["lat"][:], dtype=np.float64)
    lons = np.array(root["lon"][:], dtype=np.float64)

    if lats.ndim != 1 or lons.ndim != 1 or lats.shape[0] != L or lons.shape[0] != L:
        raise RuntimeError("lat/lon must be 1-D and length == location dimension.")

    return root, arr, lats, lons


def scan_nan_pixels(arr: zarr.Array) -> List[Tuple[int, int]]:
    """
    Return list of (loc, scen) pairs where ANY NaN exists across time.
    arr is (time, scenario, location).
    """
    T, S, L = arr.shape
    flagged: List[Tuple[int, int]] = []
    for s in range(S):
        count_s = 0
        for loc in range(L):
            v = arr.oindex[0:T, s, loc]  # 1-D time series
            if np.isnan(v).any():
                flagged.append((loc, s))
                count_s += 1
        log.info("[scan] scenario %d: %d / %d pixels contain NaNs", s, count_s, L)
    log.info("[scan] total NaN pixels: %d", len(flagged))
    return flagged


# ---------------------- TEMPLATE NC ----------------------
def load_template_year(nc_path: Path):
    """Load template 365-day data + lat/lon arrays. Handles lon convention."""
    if not nc_path.exists():
        raise FileNotFoundError(f"Template NC not found: {nc_path}")

    ds = xr.open_dataset(nc_path, decode_times=False)
    if VAR_NAME not in ds:
        raise KeyError(f"'{VAR_NAME}' missing in {nc_path.name}")

    v = ds[VAR_NAME]
    if v.ndim != 3:
        raise RuntimeError(f"Template var must be 3-D (time,lat,lon); got dims {tuple(v.dims)}")
    tname, latname, lonname = v.dims
    if int(v.sizes[tname]) != 365:
        raise RuntimeError(f"Template time length must be 365; got {int(v.sizes[tname])}")

    lat_nc = np.asarray(ds[latname].values, dtype=np.float64)
    lon_nc = np.asarray(ds[lonname].values, dtype=np.float64)

    # Normalize lon grid to 0..360 (Zarr lon is 0..360 from your builder)
    def to_0360(a):
        a = np.asarray(a, dtype=np.float64)
        # If grid looks like -180..180, convert to 0..360
        if a.min() < 0:
            a = (a + 360.0) % 360.0
        return a

    lon_nc = to_0360(lon_nc)

    data_365 = np.asarray(v.values, dtype=np.float32)  # (365, Ny, Nx)
    ds.close()
    return data_365, lat_nc, lon_nc


def nearest_latlon_idx(lat_nc: np.ndarray, lon_nc: np.ndarray, lat: float, lon: float) -> Tuple[int, int]:
    """
    Nearest-neighbour index on the template grid for a given (lat, lon).
    Zarr lon is 0..360; ensure target lon is 0..360 before matching.
    """
    # Map lon to 0..360 to match template
    if lon < 0.0:
        lon = (lon + 360.0) % 360.0

    iy = int(np.argmin(np.abs(lat_nc - lat)))
    ix = int(np.argmin(np.abs(lon_nc - lon)))

    if (abs(lat_nc[iy] - lat) > LATLON_ATOL) or (abs(lon_nc[ix] - lon) > LATLON_ATOL):
        # Accept nearest; tolerance is just informative.
        pass
    return iy, ix


def tiled_series_for_pixel(data_365: np.ndarray, iy: int, ix: int, T: int) -> np.ndarray:
    daily = data_365[:, iy, ix].astype(np.float32)  # (365,)
    reps = T // 365
    out = np.tile(daily, reps)
    if out.shape[0] != T:
        out = out[:T]
    return out


# ---------------------- WRITES & VERIFICATION ----------------------
def write_pixel(arr: zarr.Array, loc: int, s: int, series: np.ndarray):
    """Overwrite one pixel series for (loc, scen). arr is (time, scenario, location)."""
    T = arr.shape[0]
    if series.shape[0] != T:
        raise RuntimeError(f"series length {series.shape[0]} != time length {T}")
    arr.oindex[0:T, s, loc] = series  # 1-D write


def final_nan_count(arr: zarr.Array) -> int:
    """Count remaining NaNs across the full array (scans per pixel)."""
    T, S, L = arr.shape
    total = 0
    for s in range(S):
        for loc in range(L):
            v = arr.oindex[0:T, s, loc]
            cnt = int(np.isnan(v).sum())
            if cnt:
                total += cnt
    return total


# ---------------------- MAIN ----------------------
def main():
    log.info("Starting NaN fill for '%s'", VAR_NAME)

    root, arr, lats, lons = open_zarr(ZARR_PATH)
    T, S, L = arr.shape

    flagged = scan_nan_pixels(arr)
    if not flagged:
        log.info("No NaN pixels found — nothing to do.")
        print(0)
        return

    data_365, lat_nc, lon_nc = load_template_year(TEMPLATE_NC)
    Ny, Nx = data_365.shape[1], data_365.shape[2]
    log.info("Loaded template year: data=(365, %d, %d); lat=%d; lon=%d", Ny, Nx, lat_nc.size, lon_nc.size)

    def _process(loc: int, s: int) -> Tuple[int, int, bool, str]:
        lat = float(lats[loc])
        lon = float(lons[loc])
        try:
            iy, ix = nearest_latlon_idx(lat_nc, lon_nc, lat, lon)
            series = tiled_series_for_pixel(data_365, iy, ix, T)
            write_pixel(arr, loc, s, series)
            return loc, s, True, ""
        except Exception as e:
            return loc, s, False, str(e)

    ok = 0
    fail = 0
    if VAR_WORKERS > 1 and len(flagged) > 1:
        log.info("Writing with ThreadPoolExecutor(max_workers=%d) over %d pixels", VAR_WORKERS, len(flagged))
        with ThreadPoolExecutor(max_workers=VAR_WORKERS) as ex:
            futs = [ex.submit(_process, loc, s) for (loc, s) in flagged]
            for f in as_completed(futs):
                loc, s, success, err = f.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    log.warning("[write-fail] loc=%d scen=%d: %s", loc, s, err)
    else:
        for (loc, s) in flagged:
            _, _, success, err = _process(loc, s)
            if success:
                ok += 1
            else:
                fail += 1
                log.warning("[write-fail] loc=%d scen=%d: %s", loc, s, err)

    log.info("Overwrite complete: success=%d, failed=%d, total=%d", ok, fail, len(flagged))

    remaining = final_nan_count(arr)
    if remaining == 0 and fail == 0:
        log.info("[SUCCESS] All NaNs filled for '%s'.", VAR_NAME)
    else:
        log.warning("[CHECK] Remaining NaNs: %d (write failures=%d)", remaining, fail)

    # Print bare number last for easy parsing
    print(remaining)


if __name__ == "__main__":
    main()