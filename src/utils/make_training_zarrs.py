# src/utils/make_training_zarrs.py
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc

# ---------------- Project root ----------------
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

# Paths / variable sets
from src.paths.paths import preprocessed_dir as _PREPROC_ROOT
from src.dataset.variables import climate_vars, land_use_vars, var_names

# Make it explicit we’re working under 1x1
PREPROC_ROOT = _PREPROC_ROOT / "1x1"

# ---------------- Constants ----------------
# Default compressor for data arrays (coords usually uncompressed)
DEFAULT_DATA_COMPRESSOR = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
DEFAULT_COORDS_COMPRESSOR = None  # coords are tiny

# Variable buckets (used to pick historical vs preindustrial by scenario)
CLIMATE_VARS = set(climate_vars)
LAND_USE_VARS = set(land_use_vars)
CO2_VARS = {"co2"}
S0_VARS = CLIMATE_VARS | LAND_USE_VARS | CO2_VARS
S1_VARS = CLIMATE_VARS | LAND_USE_VARS
S2_VARS = LAND_USE_VARS

# Training Zarr layout / coords
SCENARIOS = ("S0", "S1", "S2", "S3")
TIME_RESES = ("daily", "monthly", "annual")
LAT_ALL = np.arange(-89.75, 90.0, 0.5, dtype="float32")  # 360
LON_ALL = np.arange(0.0, 360.0, 0.5, dtype="float32")    # 720

__all__ = [
    "make_time_axis_days_since_1901", "build_indices_from_mask",
    "flat_to_ij", "ensure_training_skeleton", "ensure_variable_in_training_store",
    "variables_for_time_res", "open_source_for_var",
    "load_batch_from_daily", "load_batch_from_full",
    "check_target_region_filled", "assert_no_nans", "scenario_index",
    "out_store_path",
]

# ---------------- Paths / helpers ----------------
def out_store_path(root: Path, set_name: str, loc_key :str, period_key: str, time_res: str) -> Path:
    """
    Output store path: {set}/{set}_location_{period}/{time_res}.zarr
    e.g. .../val/val_location_whole_period/daily.zarr
    """
    return root / set_name / f"{loc_key}_location_{period_key}" / f"{time_res}.zarr"


def make_time_axis_days_since_1901(time_res: str, start: str, end: str) -> np.ndarray:
    import cftime
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    if time_res == "daily":
        dates = xr.cftime_range(start=start, end=end, freq="D", calendar="noleap")
    elif time_res == "monthly":
        dates = xr.cftime_range(start=start, end=end, freq="MS", calendar="noleap")
    elif time_res == "annual":
        dates = xr.cftime_range(start=start, end=end, freq="YS", calendar="noleap")
    else:
        raise ValueError("time_res must be daily|monthly|annual")
    return np.asarray([(d - ref).days for d in dates], dtype="int32")


def build_indices_from_mask(mask_nc: Path | str, code: int, *, shuffle: bool = True, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    with xr.open_dataset(mask_nc, decode_times=False) as ds:
        varname = next((v for v in ds.data_vars if ds[v].ndim == 2), None)
        if varname is None:
            raise ValueError("Mask file must contain a 2D variable (lat, lon).")
        mask = ds[varname].values
    flat = np.flatnonzero(mask.ravel(order="C") == code).astype(np.int64)
    if shuffle:
        rng.shuffle(flat)
    return flat


def flat_to_ij(flat_idx: np.ndarray, nx: int = len(LON_ALL)) -> tuple[np.ndarray, np.ndarray]:
    iy = (flat_idx // nx).astype(np.int64)
    ix = (flat_idx % nx).astype(np.int64)
    return iy, ix


def ensure_training_skeleton(
    store: Path,
    *,
    time_days: np.ndarray,
    loc_idx: np.ndarray,
    overwrite: bool = False,
    coords_compressor: Blosc | None = DEFAULT_COORDS_COMPRESSOR,
) -> None:
    """
    Create the base Zarr group with coordinate arrays and metadata.
    - Uses no compression (or very light) on coords.
    - Safe to call repeatedly; respects `overwrite`.
    """
    if store.exists():
        if not overwrite:
            return
        import shutil
        shutil.rmtree(store)

    lat_idx, lon_idx = flat_to_ij(loc_idx)
    ds = xr.Dataset(coords=dict(
        time=("time", np.asarray(time_days, dtype="int32")),
        scenario=("scenario", np.arange(len(SCENARIOS), dtype="int32")),
        location=("location", np.arange(len(loc_idx), dtype="int32")),
        lat=("location", LAT_ALL[lat_idx].astype("float32")),
        lon=("location", LON_ALL[lon_idx].astype("float32")),
    ))
    ds["time"].attrs.update({
        "units": "days since 1901-01-01 00:00:00",
        "calendar": "noleap",
    })

    enc = {}
    if coords_compressor is not None:
        for k in ds.coords:
            enc[k] = {"compressor": coords_compressor}

    store.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(store, mode="w", encoding=enc)
    zarr.consolidate_metadata(str(store))


def ensure_variable_in_training_store(
    store: Path,
    var_name: str,
    *,
    n_time: int,
    n_location: int,
    location_chunk: int | None = None,
    overwrite: bool = False,
    compressor: Blosc | None = DEFAULT_DATA_COMPRESSOR,
) -> None:
    """
    Ensure a 3D array (time, scenario, location) exists for `var_name`.
    - Chunks: (n_time, 1, location_chunk) — time contiguous, scenario=1 per chunk,
      locations tiled by `location_chunk`.
    """
    if location_chunk is None:
        location_chunk = 70

    root = zarr.open_group(str(store), mode="a")
    want_shape = (n_time, len(SCENARIOS), n_location)

    if var_name in root:
        arr = root[var_name]
        if overwrite or arr.shape != want_shape:
            del root[var_name]
        else:
            arr.attrs.setdefault("_ARRAY_DIMENSIONS", ["time", "scenario", "location"])
            return

    arr = root.create_dataset(
        name=var_name,
        shape=want_shape,
        chunks=(n_time, 1, location_chunk),
        dtype="f4",
        fill_value=np.nan,
        compressor=compressor,
        overwrite=False,
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "scenario", "location"]


# ---------------- Source selection ----------------
def variables_for_time_res(time_res: str) -> list[str]:
    return list(var_names[time_res])


def choose_source_dir(scenario: str, var_name: str) -> Path:
    """
    Decide historical vs preindustrial root for a given (scenario, var).
    """
    historical_dir = PREPROC_ROOT / "historical"
    preindustrial_dir = PREPROC_ROOT / "preindustrial"
    if scenario == "S0":
        return preindustrial_dir if (var_name in S0_VARS) else historical_dir
    if scenario == "S1":
        return preindustrial_dir if (var_name in S1_VARS) else historical_dir
    if scenario == "S2":
        return preindustrial_dir if (var_name in S2_VARS) else historical_dir
    return historical_dir


def _get_daily_files_mode() -> str:
    # Late binding so env var changes are honoured in long jobs
    return os.getenv("DAILY_FILES_MODE", "annual").lower()


def _daily_dir_for_mode(base_root: Path, mode: str, var_name: str) -> Path:
    """
    Directory that holds daily inputs for `mode`:
      - 'annual'  -> base_root/annual_files/<var>
      - 'decade'  -> base_root/decade_files/<var>
      - 'twenty'  -> base_root/twenty_year_files/<var>
    """
    mode = mode.lower()
    if mode == "annual":
        return base_root / "annual_files" / var_name
    if mode == "decade":
        return base_root / "decade_files" / var_name
    if mode == "twenty":
        return base_root / "twenty_year_files" / var_name
    raise ValueError(f"Unknown DAILY_FILES_MODE='{mode}' (expected 'annual'|'decade'|'twenty')")


_YEAR_ONLY_RE = re.compile(r"(\d{4})(?!\d)")
_YEAR_RANGE_RE = re.compile(r"(\d{4})-(\d{4})(?!\d)")


def _stem_year(path: Path) -> int | None:
    m = _YEAR_ONLY_RE.search(path.stem)
    return int(m.group(1)) if m else None


def _stem_range(path: Path) -> tuple[int, int] | None:
    m = _YEAR_RANGE_RE.search(path.stem)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _filter_files_by_year_window(files: list[Path], y0: int, y1: int) -> list[Path]:
    """
    Keep files whose (year or year-range) intersects [y0, y1], sorted by start-year.
    Works for both annual files and range files (decade/twenty-year).
    """
    spans = []
    for p in files:
        yr = _stem_year(p)
        if yr is not None:
            s, e = yr, yr
        else:
            rng = _stem_range(p)
            if rng is None:
                continue
            s, e = rng
        if e < y0 or s > y1:
            continue
        spans.append((s, e, p))
    spans.sort(key=lambda t: (t[0], t[1], t[2].name))
    return [p for _, _, p in spans]


def open_source_for_var(scen: str, var_name: str, time_res: str, *, daily_mode: str | None = None):
    """
    Return ('daily'|'full', src_obj).

    - time_res == 'daily':
        returns list[Path] pointing to annual/decade/twenty-year files, chosen by
        `daily_mode` or env var DAILY_FILES_MODE ('annual' default).
    - time_res == 'monthly'|'annual':
        returns path to a single full-span NetCDF file.
    """
    base_root = choose_source_dir(scen, var_name)

    if time_res == "daily":
        mode = (daily_mode or _get_daily_files_mode())
        var_dir = _daily_dir_for_mode(base_root, mode, var_name)
        if not var_dir.is_dir():
            raise FileNotFoundError(f"Missing dir for {mode} daily files: {var_dir}")
        files = sorted(var_dir.glob(f"{var_name}_*.nc"))
        if not files:
            raise FileNotFoundError(f"No files for {var_name} under {var_dir}")
        return ("daily", files)

    full_fp = base_root / "full_time" / f"{var_name}.nc"
    if not full_fp.exists():
        raise FileNotFoundError(f"No {time_res} file for '{var_name}' at {full_fp}")
    return ("full", full_fp)


# ---------------- Reading helpers ----------------
def _detect_lat_lon_dims(da: xr.DataArray) -> tuple[str, str]:
    cand_lat = ("lat", "latitude", "y")
    cand_lon = ("lon", "longitude", "x")
    lat_dim = next((d for d in cand_lat if d in da.dims), None)
    lon_dim = next((d for d in cand_lon if d in da.dims), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Could not find lat/lon dims in {da.dims}")
    return lat_dim, lon_dim


def _isel_points(da: xr.DataArray, iy: np.ndarray, ix: np.ndarray) -> xr.DataArray:
    """
    Vectorized point selection producing shape (time, points), without .vindex.
    Uses isel with a shared 'points' dim on both indexers to pair (iy[k], ix[k]).
    """
    lat_dim, lon_dim = _detect_lat_lon_dims(da)
    iy = np.asarray(iy, dtype=int).ravel()
    ix = np.asarray(ix, dtype=int).ravel()
    pts = xr.DataArray(np.arange(iy.size), dims=("points",))
    sel = da.isel({lat_dim: (pts.dims[0], iy), lon_dim: (pts.dims[0], ix)})
    return sel  # (time, points)

def load_batch_from_daily_tiles(
    files: list[Path],
    var_name: str,
    iy: np.ndarray,
    ix: np.ndarray,
    start_str: str | None = None,
    end_str: str | None = None,
) -> np.ndarray:
    """
    Sequential, HDF5-safe reader:
      - optional pre-filter by years
      - open ONE file at a time (engine='netcdf4', decode_times=False, cache=False)
      - convert its time → days since 1901-01-01 (noleap)
      - slice to [start:end]
      - gather (time, batch) and concat as numpy
    Returns (T_total, B) float32.
    """
    import re, gc
    import cftime
    import numpy as np
    import xarray as xr
    import logging

    log = logging.getLogger(__name__)
    ref = cftime.DatetimeNoLeap(1901, 1, 1)

    def parse_ymd(s: str) -> tuple[int, int, int]:
        y, m, d = s.split("-"); return int(y), int(m), int(d)

    if start_str and end_str:
        sY, sM, sD = parse_ymd(start_str)
        eY, eM, eD = parse_ymd(end_str)
        s_days = (cftime.DatetimeNoLeap(sY, sM, sD) - ref).days
        e_days = (cftime.DatetimeNoLeap(eY, eM, eD) - ref).days
        log.debug("[daily-tiles] %s: target days [%d..%d] (%s..%s)",
                  var_name, s_days, e_days, start_str, end_str)
    else:
        s_days = e_days = None
        log.debug("[daily-tiles] %s: no explicit day window; using all available time", var_name)

    def to_ref_days(time_da: xr.DataArray) -> np.ndarray:
        vals  = time_da.values
        units = (time_da.attrs.get("units") or "").strip()
        cal   = (time_da.attrs.get("calendar") or "noleap").lower()
        if np.issubdtype(vals.dtype, np.number) and units.startswith("days since"):
            m = re.match(r"days since\s+(\d{4}-\d{2}-\d{2})", units)
            if m:
                oy, om, od = map(int, m.group(1).split("-"))
                origin = cftime.DatetimeNoLeap(oy, om, od) if ("noleap" in cal or cal == "365_day") \
                         else cftime.DatetimeGregorian(oy, om, od)
                return vals.astype(np.int64) + (origin - ref).days
        if np.issubdtype(vals.dtype, np.datetime64):
            base = np.datetime64("1901-01-01")
            return (vals - base).astype("timedelta64[D]").astype(np.int64)
        dates = cftime.num2date(vals, units=units or "days since 1901-01-01",
                                calendar=cal or "noleap", only_use_cftime_datetimes=True)
        out = np.empty(len(dates), dtype=np.int64)
        for i, dt in enumerate(dates):
            d = 28 if (dt.month == 2 and dt.day == 29) else dt.day
            out[i] = (cftime.DatetimeNoLeap(dt.year, dt.month, d) - ref).days
        return out

    def overlaps_years(fp: Path, y0: int, y1: int) -> bool:
        ms = re.findall(r"(?:^|[_-])((?:19|20)\d{2})(?:-((?:19|20)\d{2}))?", fp.name)
        if not ms:
            return True
        yrs = []
        for a, b in ms:
            yrs.append(int(a))
            if b: yrs.append(int(b))
        fmin, fmax = min(yrs), max(yrs)
        return not (fmax < y0 or fmin > y1)

    # --- Pre-filter file list (best-effort) ---
    n_before = len(files)
    if start_str and end_str:
        y0, y1 = int(start_str[:4]), int(end_str[:4])
        files = [f for f in files if overlaps_years(f, y0, y1)]
    log.info("[daily-tiles] %s: files %d -> %d after year-filter",
             var_name, n_before, len(files))

    if not files:
        raise FileNotFoundError(f"No daily files for {var_name} in requested period.")

    pieces = []
    t_total = 0
    for i, fp in enumerate(files):
        # Probe quickly
        try:
            with xr.open_dataset(fp, engine="netcdf4", decode_times=False, cache=False) as p:
                if var_name not in p:
                    log.debug("[daily-tiles] %s: skip %s (var missing)", var_name, fp.name)
                    continue
        except Exception as e:
            log.warning("[daily-tiles] %s: skip %s (probe failed: %s)", var_name, fp.name, e)
            continue

        # Open, slice, extract, close immediately
        log.info("[daily-tiles] %s: reading %s (%d/%d)", var_name, fp.name, i+1, len(files))
        with xr.open_dataset(fp, engine="netcdf4", decode_times=False, cache=False) as ds:
            da = ds[var_name]
            if "time" not in da.dims:
                log.debug("[daily-tiles] %s: %s has no 'time' dim; skip", var_name, fp.name)
                continue
            da = da.transpose("time", "lat", "lon")
            td = to_ref_days(ds["time"])
            if s_days is not None and e_days is not None:
                keep = np.where((td >= s_days) & (td <= e_days))[0]
                if keep.size == 0:
                    log.debug("[daily-tiles] %s: %s no overlap with target window; skip",
                              var_name, fp.name)
                    continue
                da = da.isel(time=keep)

            block = _isel_points(da, iy, ix)  # (time, batch)
            arr = np.asarray(block.values, dtype=np.float32)
            pieces.append(arr)
            t_total += arr.shape[0]
            log.info("[daily-tiles] %s: added %d timesteps from %s (cum T=%d)",
                     var_name, arr.shape[0], fp.name, t_total)

        # drop refs, force GC to close any lingering handles promptly
        del da, ds, block, arr
        gc.collect()

    if not pieces:
        raise FileNotFoundError(
            f"No overlapping daily data found for {var_name} in [{start_str},{end_str}] across {n_before} files."
        )

    out = np.concatenate(pieces, axis=0)
    log.info("[daily-tiles] %s: concatenated shape %s (T=%d, B=%d)",
             var_name, tuple(out.shape), out.shape[0], out.shape[1] if out.ndim == 2 else -1)
    return out

def load_batch_from_daily(
    files: list[Path],
    var_name: str,
    iy: np.ndarray,
    ix: np.ndarray,
    start_str: str | None = None,
    end_str: str | None = None,
) -> np.ndarray:
    """
    Read (time, batch) for var_name from a list of annual or range NetCDFs.
    - Filters files to those intersecting the requested years (if start/end given).
    - Within each file, slices the time axis to the requested day range (noleap).
    - Returns float32 array (T_total, B).
    """
    import cftime
    ref = cftime.DatetimeNoLeap(1901, 1, 1)

    if start_str is not None and end_str is not None:
        y0, y1 = int(start_str[:4]), int(end_str[:4])
        files = _filter_files_by_year_window(files, y0, y1)

    if not files:
        raise FileNotFoundError(f"No daily files for {var_name} in requested period.")

    s_days = e_days = None
    if start_str is not None and end_str is not None:
        sY, sM, sD = map(int, start_str.split("-"))
        eY, eM, eD = map(int, end_str.split("-"))
        s_days = (cftime.DatetimeNoLeap(sY, sM, sD) - ref).days
        e_days = (cftime.DatetimeNoLeap(eY, eM, eD) - ref).days

    def to_days(vals):
        arr = np.asarray(vals)
        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(np.int64)
        if np.issubdtype(arr.dtype, np.datetime64):
            base = np.datetime64("1901-01-01")
            return (arr - base).astype("timedelta64[D]").astype(np.int64)
        return np.array([(v - ref).days for v in arr], dtype=np.int64)

    pieces = []
    for fp in files:
        with xr.open_dataset(fp, decode_times=False) as ds:
            if var_name not in ds:
                raise KeyError(f"{var_name} missing in {fp}; have {list(ds.data_vars)}")
            da = ds[var_name]

            if s_days is not None and e_days is not None and "time" in da.dims:
                td = to_days(ds["time"].values)
                mask = (td >= s_days) & (td <= e_days)
                if not mask.any():
                    continue
                da = da.isel(time=np.where(mask)[0])

            sel = _isel_points(da, iy, ix).astype("float32")
            pieces.append(np.asarray(sel.values))

    if not pieces:
        raise FileNotFoundError(f"No overlapping daily data found for {var_name} in [{start_str},{end_str}].")

    return np.concatenate(pieces, axis=0)

def load_batch_from_full(
    file_path: Path,
    var_name: str,
    iy: np.ndarray,
    ix: np.ndarray,
    start_str: str | None = None,
    end_str: str | None = None,
) -> np.ndarray:
    """
    Read a (time, batch) slice from a full-span file (monthly/annual).
    If start_str/end_str are given, time-slice to that period first.
    """
    import cftime
    ref = cftime.DatetimeNoLeap(1901, 1, 1)

    with xr.open_dataset(file_path, decode_times=False) as ds:
        if var_name not in ds:
            raise KeyError(f"{var_name} missing in {file_path}; have {list(ds.data_vars)}")

        da = ds[var_name]

        if start_str is not None and end_str is not None and "time" in da.dims:
            t = ds["time"].values

            def to_days(vals):
                arr = np.asarray(vals)
                if np.issubdtype(arr.dtype, np.integer):
                    return arr.astype(np.int64)
                if np.issubdtype(arr.dtype, np.datetime64):
                    base = np.datetime64("1901-01-01")
                    return (arr - base).astype("timedelta64[D]").astype(np.int64)
                return np.array([(v - ref).days for v in arr], dtype=np.int64)

            td = to_days(t)
            s = (cftime.DatetimeNoLeap(*map(int, start_str.split("-"))) - ref).days
            e = (cftime.DatetimeNoLeap(*map(int, end_str.split("-"))) - ref).days
            mask = (td >= s) & (td <= e)
            if not mask.any():
                raise ValueError(f"{file_path}: no time points within [{start_str},{end_str}]")
            da = da.isel(time=np.where(mask)[0])

        sel = _isel_points(da, iy, ix).astype("float32")
        return np.asarray(sel.values)


# ---------------- Integrity checks ----------------
def check_target_region_filled(arr: zarr.Array, t0: int, t1: int, s_idx: int, loc0: int, loc1: int) -> bool:
    block = arr[t0:t1, s_idx:s_idx+1, loc0:loc1]
    return np.all(np.isfinite(block))


def assert_no_nans(name: str, data_tb: np.ndarray, iy: np.ndarray, ix: np.ndarray):
    if np.any(~np.isfinite(data_tb)):
        t, b = np.argwhere(~np.isfinite(data_tb))[0]
        raise RuntimeError(
            f"[NAN] {name}: t={int(t)}, batch_index={int(b)}, "
            f"lat={float(LAT_ALL[int(iy[b])])}, lon={float(LON_ALL[int(ix[b])])}, "
            f"(iy,ix)=({int(iy[b])},{int(ix[b])})"
        )


def scenario_index(scen: str) -> int:
    try:
        return SCENARIOS.index(scen)
    except ValueError:
        raise ValueError(f"Unknown scenario {scen}; expected one of {SCENARIOS}")