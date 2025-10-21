#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict
from contextlib import contextmanager
import json
import time
import os
import xarray as xr
import numpy as np
import zarr
from numcodecs import Blosc
import cftime
from datetime import timedelta
import cftime

# Public constants (used by main; override if you like)
TILE_T_DEFAULT = 365  # noleap daily

# ==== TIME AXES (noleap) ====
def _days_since_1901_noleap_daily(start: str, end: str) -> np.ndarray:
    import cftime, xarray as xr
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    dates = xr.date_range(start=start, end=end, freq="D", calendar="noleap", use_cftime=True)
    return np.asarray([(d - ref).days for d in dates], dtype="int32")


def _days_since_1901_noleap_monthly(start_year: int, end_year: int) -> np.ndarray:
    """Use the first of each month as the timestamp (days since 1901-01-01)."""
    import cftime
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    vals = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            dt = cftime.DatetimeNoLeap(y, m, 1)
            vals.append((dt - ref).days)
    return np.asarray(vals, dtype="int32")


def _days_since_1901_noleap_annual(start_year: int, end_year: int) -> np.ndarray:
    """Use Jan-01 of each year as the timestamp (days since 1901-01-01)."""
    import cftime
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    vals = []
    for y in range(start_year, end_year + 1):
        dt = cftime.DatetimeNoLeap(y, 1, 1)
        vals.append((dt - ref).days)
    return np.asarray(vals, dtype="int32")


def _period_meta(p0: str, p1: str):
    """Return (start_year, end_year, n_days_total, n_months_total, n_years_total)."""
    sy, ey = int(p0[:4]), int(p1[:4])
    n_years = ey - sy + 1
    n_months = n_years * 12
    n_days = n_years * 365  # noleap
    return sy, ey, n_days, n_months, n_years


# ==== GRID ====
def _global_halfdeg_grid():
    lat = np.arange(-89.75, 90.0, 0.5, dtype="float32")  # 360
    lon = np.arange(0.0, 360.0, 0.5, dtype="float32")    # 720
    return lat, lon


# ==== CORE INIT HELPERS ====
def _ensure_coords_(root, *, time_arr: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                    calendar_note: str):
    T, Y, X = len(time_arr), len(lat), len(lon)

    # --- time --- (int32, NO fill value)
    if "time" not in root:
        d = root.create_dataset(
            "time", shape=(T,), chunks=(T,), dtype="i4",
            compressor=None, fill_value=None, overwrite=False
        )
    else:
        d = root["time"]
        if d.shape != (T,):
            d.resize((T,))
        # ensure no fill value on coords
        if getattr(d, "fill_value", None) is not None:
            d.fill_value = None
    d[:] = np.asarray(time_arr, dtype="int32")
    d.attrs["units"] = "days since 1901-01-01 00:00:00"
    d.attrs["calendar"] = calendar_note
    d.attrs["_ARRAY_DIMENSIONS"] = ["time"]

    def _ensure_1d(name, target_vals, length):
        if name not in root:
            a = root.create_dataset(
                name, shape=(length,), chunks=(length,), dtype="f4",
                compressor=None, fill_value=None, overwrite=False
            )
        else:
            a = root[name]
            need_fix = (a.shape != (length,))
            if not need_fix:
                cur = np.asarray(a[:])
                need_fix = (cur.size != length) or (not np.isfinite(cur).all())
            if need_fix:
                a.resize((length,))
            # make SURE coords have no fill value
            if getattr(a, "fill_value", None) is not None:
                a.fill_value = None
        a[:] = target_vals.astype("float32")
        root[name].attrs["_ARRAY_DIMENSIONS"] = [name]

    _ensure_1d("lat", lat, Y)
    _ensure_1d("lon", lon, X)
    
def _time_axis_noleap(time_res: str, length: int) -> np.ndarray:
    """
    Build a noleap 'days since 1901-01-01' time axis of given length.
    - daily: every day starting 1901-01-01
    - monthly: 1st of each month starting 1901-01-01
    - annual: Jan 1 each year starting 1901
    """
    ref = cftime.DatetimeNoLeap(1901, 1, 1)

    if time_res == "daily":
        # successive days
        vals = [(ref + timedelta(days=i) - ref).days for i in range(length)]

    elif time_res == "monthly":
        # first day of each month
        y, m = 1901, 1
        vals = []
        for _ in range(length):
            dt = cftime.DatetimeNoLeap(y, m, 1)
            vals.append((dt - ref).days)
            m += 1
            if m > 12:
                m, y = 1, y + 1

    elif time_res == "annual":
        # Jan-01 each year
        vals = [(cftime.DatetimeNoLeap(1901 + i, 1, 1) - ref).days for i in range(length)]

    else:
        raise ValueError(f"Unsupported time_res: {time_res}")

    return np.asarray(vals, dtype="int32")


def repair_coords(store_path: str | Path, time_res: str):
    store_path = Path(store_path)
    root = zarr.open_group(store=zarr.DirectoryStore(str(store_path)), mode="a")

    DEFAULTS = {"daily": 44895, "monthly": 1476, "annual": 123}
    T = None
    if "time" in root:
        try:
            T = int(root["time"].shape[0])
        except Exception:
            T = None
    if T is None:
        for _, arr in root.arrays():
            if arr.ndim == 3:
                T = int(arr.shape[0])
                break
    if T is None:
        T = DEFAULTS.get(time_res)
    if T is None:
        raise RuntimeError("Could not infer time length and no default for this time_res.")

    time_arr = _time_axis_noleap(time_res, T)
    lat = np.arange(-89.75, 90.0, 0.5, dtype="float32")   # 360
    lon = np.arange(0.0, 360.0, 0.5, dtype="float32")     # 720
    calendar = "noleap"

    # time
    t = root.require_dataset(
        "time", shape=(len(time_arr),), chunks=(len(time_arr),),
        dtype="i4", compressor=None, fill_value=None
    )
    if t.shape != (len(time_arr),):
        t.resize((len(time_arr),))
    # ensure no coord fill value
    if getattr(t, "fill_value", None) is not None:
        t.fill_value = None
    t[:] = time_arr
    t.attrs["units"] = "days since 1901-01-01 00:00:00"
    t.attrs["calendar"] = calendar
    t.attrs["_ARRAY_DIMENSIONS"] = ["time"]

    def ensure_coord(name: str, vals: np.ndarray):
        d = root.require_dataset(
            name, shape=(len(vals),), chunks=(len(vals),),
            dtype="f4", compressor=None, fill_value=None
        )
        if d.shape != (len(vals),):
            d.resize((len(vals),))
        if getattr(d, "fill_value", None) is not None:
            d.fill_value = None
        d[:] = vals.astype("float32")
        d.attrs["_ARRAY_DIMENSIONS"] = [name]

    ensure_coord("lat", lat)
    ensure_coord("lon", lon)

    for v, arr in root.arrays():
        if v in ("time", "lat", "lon"):
            continue
        arr.attrs.setdefault("_ARRAY_DIMENSIONS", ["time", "lat", "lon"])

    print(f"[OK] Repaired coords in {store_path} (fill_value cleared) | time_res={time_res} | T={T}")


def ensure_all_stores(
    *,
    run_root: Path,
    period: tuple[str, str],
    daily_vars: List[str],
    monthly_vars: List[str],
    annual_vars: List[str],
    tile_h: int,
    tile_w: int,
    tile_t: int = TILE_T_DEFAULT,
):
    """
    Create/open three zarr stores directly under `run_root`:
      - <run_root>/daily.zarr
      - <run_root>/monthly.zarr
      - <run_root>/annual.zarr
      and a tiles JSON: <run_root>/tiles_<h>x<w>.json
    """
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    p0, p1 = period
    sy, ey, N_days, N_months, N_years = _period_meta(p0, p1)
    lat, lon = _global_halfdeg_grid()
    NY, NX = len(lat), len(lon)

    # --- daily store ---
    daily_store = run_root / "daily.zarr"
    ds = zarr.open_group(store=zarr.DirectoryStore(str(daily_store)), mode="a")
    time_daily = _days_since_1901_noleap_daily(p0, p1)
    _ensure_coords_(ds, time_arr=time_daily, lat=lat, lon=lon, calendar_note="noleap")
    _ensure_vars_(ds, var_names=daily_vars,
                  shape=(N_days, NY, NX),
                  chunks=(tile_t, tile_h, tile_w))
    zarr.consolidate_metadata(zarr.DirectoryStore(str(daily_store)))

    # --- monthly store ---
    monthly_store = run_root / "monthly.zarr"
    ms = zarr.open_group(store=zarr.DirectoryStore(str(monthly_store)), mode="a")
    time_monthly = _days_since_1901_noleap_monthly(sy, ey)
    _ensure_coords_(ms, time_arr=time_monthly, lat=lat, lon=lon, calendar_note="noleap")
    _ensure_vars_(ms, var_names=monthly_vars,
                  shape=(N_months, NY, NX),
                  chunks=(12, tile_h, tile_w))
    zarr.consolidate_metadata(zarr.DirectoryStore(str(monthly_store)))

    # --- annual store ---
    annual_store = run_root / "annual.zarr"
    as_ = zarr.open_group(store=zarr.DirectoryStore(str(annual_store)), mode="a")
    time_annual = _days_since_1901_noleap_annual(sy, ey)
    _ensure_coords_(as_, time_arr=time_annual, lat=lat, lon=lon, calendar_note="noleap")
    _ensure_vars_(as_, var_names=annual_vars,
                  shape=(N_years, NY, NX),
                  chunks=(1, tile_h, tile_w))
    zarr.consolidate_metadata(zarr.DirectoryStore(str(annual_store)))

    # --- tiles json ---
    tiles_json = run_root / f"tiles_{tile_h}x{tile_w}.json"
    if not tiles_json.exists():
        tiles = []
        for y0 in range(0, NY, tile_h):
            y1 = min(y0 + tile_h, NY)
            for x0 in range(0, NX, tile_w):
                x1 = min(x0 + tile_w, NX)
                tiles.append((y0, y1, x0, x1))
        with open(tiles_json, "w") as f:
            json.dump({"ny": NY, "nx": NX,
                       "tile_lat": tile_h, "tile_lon": tile_w,
                       "tiles": tiles}, f)

    meta = {
        "start_year": sy,
        "end_year": ey,
        "n_days": N_days,
        "n_months": N_months,
        "n_years": N_years,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "tile_t": tile_t,
    }
    return daily_store, monthly_store, annual_store, tiles_json, meta

def _ensure_vars_(root, *, var_names: List[str], shape, chunks, clevel=4):
    comp = Blosc(cname="zstd", clevel=int(clevel), shuffle=Blosc.SHUFFLE)
    for v in var_names:
        if v in root:
            root[v].attrs.setdefault("_ARRAY_DIMENSIONS", ["time", "lat", "lon"])
            if tuple(root[v].shape) != tuple(shape):
                raise RuntimeError(
                    f"Existing variable '{v}' has shape {tuple(root[v].shape)}; expected {tuple(shape)}."
                )
            continue
        d = root.create_dataset(
            v, shape=shape, chunks=chunks, dtype="f4",
            compressor=comp, fill_value=np.float32(np.nan), overwrite=False
        )
        d.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]


# ---------------- Lock helper ---------------- #
@contextmanager
def simple_lock(lock_path: Path, timeout=120):
    lock_f = lock_path.open("w")
    start = time.time()
    waited = False
    while True:
        try:
            os.lockf(lock_f.fileno(), os.F_TLOCK, 0)
            if waited:
                print(f"[LOCK] Acquired: {lock_path}")
            break
        except OSError:
            if not waited:
                print(f"[LOCK] Waiting for {lock_path} ...")
                waited = True
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout acquiring lock {lock_path}")
            time.sleep(0.5)
    try:
        yield
    finally:
        os.lockf(lock_f.fileno(), os.F_ULOCK, 0)
        lock_f.close()
        
# ------------------ NETCDF Helpers ----------------
def _list_zarr_vars(zarr_path: Path) -> List[str]:
    """Return all data variable names in a consolidated Zarr dataset."""
    ds = xr.open_zarr(zarr_path, consolidated=True, decode_times=False)
    try:
        return list(ds.data_vars.keys())
    finally:
        ds.close()


def _build_var_encoding(is_monthly: bool) -> dict:
    """
    Hardcoded encodings:
      - dtype: float32
      - zlib compression, complevel 3
      - chunks: time=12 for monthly, time=1 for annual; lat=90, lon=90
    """
    t_chunk = 120 if is_monthly else 10
    return {
        "zlib": True,
        "complevel": 3,
        "dtype": "float32",
        "_FillValue": np.nan,
        "chunksizes": (t_chunk, 90, 90),
    }


def _apply_time_encoding(ds: xr.Dataset) -> None:
    """
    Force time encoding to 'days since 1901-01-01 00:00:00' and 365_day calendar.
    Values themselves are taken from the Zarr (ds) — we just set encoding.
    """
    if "time" in ds.coords:
        ds["time"].encoding = {"units": "days since 1901-01-01 00:00:00", "calendar": "noleap"}


def _export_one_var(
    *,
    var: str,
    zarr_path: Path,
    out_nc: Path,
    is_monthly: bool,
    overwrite: bool,
) -> str:
    """
    Export a single variable from a consolidated Zarr store to NetCDF:
      - keep time as integer 'days since 1901-01-01' (noleap)
      - write float32 data with zlib compression and sensible chunks
      - ensure coords (lat/lon/time) have no _FillValue and are finite
      - stamp canonical half-degree lat/lon grid to avoid any drift/NaNs
    """
    if out_nc.exists() and not overwrite:
        return f"[SKIP] {out_nc.name} exists"

    ds = xr.open_zarr(zarr_path, consolidated=True, decode_times=False)
    try:
        if var not in ds.data_vars:
            return f"[WARN] Variable {var} not found in {zarr_path.name}; skipping"

        # Select only the variable and ensure float32 payload
        dsv = ds[[var]].astype("float32")

        # ---- Stamp canonical coords (robust against pre-existing NaNs/misalignment) ----
        # Half-degree global grid:
        lat = np.arange(-89.75, 90.0, 0.5, dtype="float32")   # 360
        lon = np.arange(0.0, 360.0, 0.5, dtype="float32")     # 720
        if "lat" in dsv.dims:
            dsv = dsv.assign_coords(lat=("lat", lat))
        if "lon" in dsv.dims:
            dsv = dsv.assign_coords(lon=("lon", lon))

        # ---- Keep time numeric, set CF attrs, and disable coordinate fill ----
        _apply_time_encoding(dsv)  # sets units/calendar on time
        if "time" in dsv.coords:
            # Also ensure coords never get a fill value on write
            enc_time = dict(dsv["time"].encoding) if hasattr(dsv["time"], "encoding") else {}
            enc_time.update({"_FillValue": None})
            dsv["time"].encoding = enc_time

        if "lat" in dsv.coords:
            dsv["lat"].encoding = {"_FillValue": None}
            # helpful CF-ish attrs (won't hurt if already present)
            dsv["lat"].attrs.setdefault("standard_name", "latitude")
            dsv["lat"].attrs.setdefault("long_name", "latitude")
            dsv["lat"].attrs.setdefault("units", "degrees_north")

        if "lon" in dsv.coords:
            dsv["lon"].encoding = {"_FillValue": None}
            dsv["lon"].attrs.setdefault("standard_name", "longitude")
            dsv["lon"].attrs.setdefault("long_name", "longitude")
            dsv["lon"].attrs.setdefault("units", "degrees_east")

        # ---- Data var encoding (compression, dtype, chunks) ----
        enc = {var: _build_var_encoding(is_monthly)}

        out_nc.parent.mkdir(parents=True, exist_ok=True)
        dsv.to_netcdf(out_nc, engine="netcdf4", encoding=enc, mode="w", format="NETCDF4")
        return f"[OK]   {out_nc.name}"
    finally:
        ds.close()


def export_all_netcdf_for_scenario(
    *,
    scenario: str,          # kept for filenames/logs; not used for path assembly
    nc_root: Path,          # <- exact root directory to write into
    monthly_zarr: Path,
    annual_zarr: Path,
    overwrite: bool,
) -> None:
    """
    Export all variables from monthly & annual Zarrs to per-variable NetCDFs into `nc_root`.
    Files: <nc_root>/<scenario>_<var>.nc
    """
    nc_root = Path(nc_root)
    nc_root.mkdir(parents=True, exist_ok=True)

    monthly_vars = _list_zarr_vars(monthly_zarr)
    annual_vars  = _list_zarr_vars(annual_zarr)

    tasks = []
    for v in monthly_vars:
        out_nc = nc_root / f"{scenario}_{v}.nc"
        tasks.append(("monthly", v, monthly_zarr, out_nc))
    for v in annual_vars:
        out_nc = nc_root / f"{scenario}_{v}.nc"
        tasks.append(("annual", v, annual_zarr, out_nc))

    if not tasks:
        print(f"[EXPORT] No variables found to export for scenario {scenario}.")
        return

    print(f"[EXPORT] Starting NetCDF export for {scenario}: {len(tasks)} files → {nc_root}")
    results = []
    for kind, v, zpath, out_nc in tasks:
        r = _export_one_var(
            var=v,
            zarr_path=zpath,
            out_nc=out_nc,
            is_monthly=(kind == "monthly"),
            overwrite=overwrite,
        )
        print(f"[EXPORT] {r}")
        results.append(r)

    ok = sum(1 for r in results if r.startswith("[OK]"))
    skip = sum(1 for r in results if r.startswith("[SKIP]"))
    warn = sum(1 for r in results if r.startswith("[WARN]"))
    err = sum(1 for r in results if r.startswith("[ERR]"))

    for r in results:
        print(f"[EXPORT] {r}")

    print(f"[EXPORT] Finished {scenario}: {ok} written, {skip} skipped, {warn} warnings, {err} errors.")