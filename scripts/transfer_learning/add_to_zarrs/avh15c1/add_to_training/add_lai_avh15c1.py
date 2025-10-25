#!/usr/bin/env python3
"""
Add new variable(s) to existing training Zarrs wherever the source time coverage
overlaps each period, without overwriting existing data or touching coords.

Key points
----------
- Assumes the Zarr stores already exist (coords/time/scenario/location).
- Does NOT call ensure_training_skeleton / ensure_variable_in_training_store.
- Creates the new dataset in the Zarr group if it doesn't exist yet, by copying
  shape/chunks from an existing variable in the same store (fill=NaN).
- Writes only overlapping timesteps to the single configured scenario; all other
  scenarios remain NaN.
- Reads the store’s *own* time coordinate for alignment.
- Uses the *same shuffled* location order as the original training Zarrs
  (shuffle=True, seed=42).
- After writing, consolidates metadata so xarray.open_zarr(..., consolidated=True) works.
"""

from __future__ import annotations
import argparse, sys, logging, json
from pathlib import Path
from typing import Dict, Tuple
from cftime import num2date
import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc
import cftime

# ---------- Project setup ----------
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.utils.make_training_zarrs import (
    out_store_path,
    scenario_index,
    build_indices_from_mask,
    flat_to_ij,
)
from src.paths.paths import masks_dir, zarr_dir, preprocessed_dir

# Paths to Seasonality Mask
SEASONAL_PATH = preprocessed_dir / Path("transfer_learning/avh15c1/lai_avh15c1_seasonality.zarr")
SEASONAL_VAR  = "lai_avh15c1"

# ---------- Logging ----------
def setup_logging() -> logging.Logger:
    log = logging.getLogger("addvar_allperiods_noensure")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                         datefmt="%-Y-%m-%d %H:%M:%S"))
        log.addHandler(h)
    return log

logger = setup_logging()

# ---------- Config ----------
FINAL_LOCATION_CHUNK = 70
DEFAULT_COMP = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
OUT_ROOT = zarr_dir / "training_new"

MASK_PATH = masks_dir / "tvt_mask.nc"
SCENARIOS = ("S0", "S1", "S2", "S3")
TIME_RESES = ("monthly",)  # we only write monthly here

# Map mask_code -> location key
LOC_FOR_MASK = {0: "train", 1: "val", 2: "test"}

# Fixed mapping of sets/periods to directory structure
SET_SPECS = [
    ("train", 0, ["train_period"]),
    ("val",   1, ["whole_period", "val_period_early", "val_period_late"]),
    ("test",  2, ["whole_period", "test_period_early", "test_period_late"]),
]

# --- New variable configuration ---
NEW_VAR_MODE = {
    "var_names": ["lai_avh15c1"],
    "scenario": "S3",  # only this scenario will be populated
    "source_paths": {
        "lai_avh15c1": "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/lai_avh15c1.nc",
    },
}

# ---------- Helpers ----------
def detect_source_time_res(tvals: np.ndarray) -> str:
    """Infer cadence from day deltas."""
    arr = np.asarray(tvals)
    if arr.size < 3:
        return "monthly"
    dif = np.diff(arr.astype(np.int64))
    md = int(np.median(dif))
    if 360 <= md <= 370:
        return "annual"
    if md == 1:
        return "daily"
    return "monthly"

def days_to_iso(d: int) -> str:
    base = np.datetime64("1901-01-01")
    return str(base + np.timedelta64(int(d), "D"))

def to_days_since_1901(ds_time: xr.DataArray) -> np.ndarray:
    """Convert any CF/np/cftime time axis to days since 1901-01-01 (noleap)."""
    import cftime
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    vals = ds_time.values
    if np.issubdtype(vals.dtype, np.number):
        return vals.astype(np.int64)
    if np.issubdtype(vals.dtype, np.datetime64):
        base = np.datetime64("1901-01-01")
        return (vals - base).astype("timedelta64[D]").astype(np.int64)
    out = np.empty(vals.shape[0], dtype=np.int64)
    for i, v in enumerate(vals):
        out[i] = (cftime.DatetimeNoLeap(v.year, v.month, min(v.day, 28)) - ref).days
    return out

def overlap_days(src_days: np.ndarray, ref_days: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices (src_idx, ref_idx) for days in common between arrays."""
    where = {int(d): i for i, d in enumerate(ref_days)}
    src_idx, ref_idx = [], []
    for i, d in enumerate(src_days):
        j = where.get(int(d))
        if j is not None:
            src_idx.append(i); ref_idx.append(j)
    return np.asarray(src_idx, dtype=np.int64), np.asarray(ref_idx, dtype=np.int64)

def pick_reference_array(root: zarr.hierarchy.Group) -> zarr.core.Array:
    """Pick any existing data array in the store to copy shape/chunks."""
    for name, obj in root.arrays():
        if name not in ("time", "location", "scenario", "lat", "lon"):
            return obj
    raise RuntimeError("No reference data array found in store for shape/chunks; store must contain at least one data var.")

def _array_dims_from_ref(root: zarr.hierarchy.Group, ref_name: str | None) -> list[str]:
    """
    Return the dimension names to stamp into _ARRAY_DIMENSIONS.
    Prefer the reference array's attribute if present; otherwise default.
    """
    if ref_name and ref_name in root:
        try:
            attrs = root[ref_name].attrs.asdict()
            dims = attrs.get("_ARRAY_DIMENSIONS", None)
            if isinstance(dims, (list, tuple)) and len(dims) == 3:
                return list(dims)
        except Exception:
            pass
    # training arrays are [time, scenario, location]
    return ["time", "scenario", "location"]

def ensure_new_array_in_group(
    root: zarr.hierarchy.Group,
    var_name: str,
    shape: Tuple[int, int, int],
    chunks: Tuple[int, int, int],
    compressor=DEFAULT_COMP,
    array_dims: list[str] | None = None,
):
    """
    Create a zarr array if missing, and stamp `_ARRAY_DIMENSIONS`.
    (xarray can then skip introspection that sometimes triggers json.loads issues.)
    """
    if var_name in root:
        # still ensure dims attr is present
        try:
            attrs = root[var_name].attrs.asdict()
            if "_ARRAY_DIMENSIONS" not in attrs and array_dims:
                root[var_name].attrs["_ARRAY_DIMENSIONS"] = array_dims
        except Exception:
            pass
        return

    arr = root.create_dataset(
        var_name,
        shape=shape,
        chunks=chunks,
        dtype="f4",
        compressor=compressor,
        fill_value=np.float32(np.nan),
        overwrite=False,
        dimension_separator=".",  # keep consistent with existing stores
        filters=None,
        order="C",
    )
    # Add dims attribute for xarray
    try:
        arr.attrs["_ARRAY_DIMENSIONS"] = array_dims or ["time", "scenario", "location"]
    except Exception:
        pass

def read_points_from_src(src_path: Path, var_name: str, t0: int, t1: int, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    """Read (t1-t0, B) from (time, lat, lon) source for given point list."""
    with xr.open_dataset(src_path, decode_times=False) as ds:
        da = ds[var_name]
        da = da.isel(time=slice(t0, t1))  # (t_sel, lat, lon)
        lat_dim = next(d for d in ("lat", "latitude", "y") if d in da.dims)
        lon_dim = next(d for d in ("lon", "longitude", "x") if d in da.dims)
        pts = xr.DataArray(np.arange(iy.size), dims=("points",))
        sub = da.transpose("time", lat_dim, lon_dim).isel(
            {lat_dim: (pts.dims[0], iy), lon_dim: (pts.dims[0], ix)}
        )
        return np.asarray(sub.values, dtype=np.float32)  # (t_sel, B)
    
def read_points_from_clim(clim_path: Path, var_name: str, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    """
    Returns (12, B): monthly climatology (month 1..12) for B points.
    Assumes dims are ('time', 'lat', 'lon') with time length 12.
    """
    # Try consolidated metadata first; fall back to non-consolidated
    try:
        ds = xr.open_zarr(str(clim_path), consolidated=True)
    except Exception:
        ds = xr.open_zarr(str(clim_path), consolidated=False)

    da = ds[var_name]
    lat_dim = next(d for d in ("lat", "latitude", "y") if d in da.dims)
    lon_dim = next(d for d in ("lon", "longitude", "x") if d in da.dims)
    pts = xr.DataArray(np.arange(iy.size), dims=("points",))
    sub = da.transpose("time", lat_dim, lon_dim).isel(
        {lat_dim: (pts.dims[0], iy), lon_dim: (pts.dims[0], ix)}
    )
    return np.asarray(sub.values, dtype=np.float32)  # (12, B)

# ---------- Main ----------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Add variable(s) to existing Zarrs (monthly only), no skeleton/ensure."
    )
    p.add_argument(
        "--overwrite-var",
        action="store_true",
        help="Delete the existing target variable in each store before writing (only affects listed var(s)).",
    )
    args = p.parse_args(argv)

    # Build masked indices with the SAME shuffle/seed as the main training Zarrs
    masked_idx_by_code: Dict[int, np.ndarray] = {
        code: build_indices_from_mask(MASK_PATH, code=code, shuffle=True, seed=42)
        for code in (0, 1, 2)
    }

    for var_name in NEW_VAR_MODE["var_names"]:
        src_path = Path(NEW_VAR_MODE["source_paths"][var_name])
        if not src_path.exists():
            raise FileNotFoundError(f"Missing source file for {var_name}: {src_path}")

        # Inspect source time axis + detect cadence
        with xr.open_dataset(src_path, decode_times=False) as ds:
            if var_name not in ds.variables:
                raise KeyError(f"{var_name} missing in {src_path}; have {list(ds.data_vars)}")
            src_days_all = to_days_since_1901(ds["time"])
        src_time_res = detect_source_time_res(src_days_all)
        logger.info(
            f"[VAR] {var_name}: src={src_path.name} cadence={src_time_res} "
            f"span=[{int(src_days_all.min())}..{int(src_days_all.max())}] len={src_days_all.size}"
        )

        if src_time_res != "monthly":
            logger.info(f"[SKIP] {var_name}: source cadence {src_time_res} != monthly")
            continue

        scen_idx = scenario_index(NEW_VAR_MODE["scenario"])

        # Work list (no SLURM sharding)
        work_list = [
            (set_name, mask_code, period_key)
            for set_name, mask_code, period_keys in SET_SPECS
            for period_key in period_keys
        ]
        logger.info(f"[INFO] Processing {len(work_list)} set/period targets (no sharding).")

        for set_name, mask_code, period_key in work_list:
            # Which location mask to use for this period?
            if period_key == "whole_period":
                loc_key = set_name
                loc_mask_code = mask_code
            else:
                loc_key = "train"
                loc_mask_code = 0

            # Align locations to chunk size
            loc_idx_full = masked_idx_by_code[loc_mask_code]
            n_locations = (len(loc_idx_full) // FINAL_LOCATION_CHUNK) * FINAL_LOCATION_CHUNK
            if n_locations == 0:
                logger.warning(f"[WARN] No locations for set={set_name} period={period_key} (mask={loc_mask_code}); skip.")
                continue
            loc_idx = loc_idx_full[:n_locations]
            iy, ix = flat_to_ij(loc_idx)

            time_res = "monthly"  # fixed here
            store_path = out_store_path(OUT_ROOT, set_name, loc_key, period_key, time_res)

            if not Path(store_path).exists():
                logger.warning(f"[MISS] Store does not exist, skipping: {store_path}")
                continue
            root = zarr.open_group(str(store_path), mode="a")

            # Use the store's own time axis (must exist)
            if "time" not in root:
                logger.warning(f"[BAD] No 'time' coord in store, skipping: {store_path}")
                continue
            ref_days = np.asarray(root["time"][:], dtype=np.int64)
            n_time = ref_days.size

            ref_dt = num2date(ref_days, units="days since 1901-01-01", calendar="noleap")
            month_idx = np.asarray([dt.month - 1 for dt in ref_dt], dtype=np.int64)
            
            # restrict filling to 1982..2018 inclusive
            years = np.asarray([dt.year for dt in ref_dt], dtype=np.int16)
            fill_time_mask = (years >= 1982) & (years <= 2018)   # shape (n_time,)
            
            # Overlap with source
            src_idx, tgt_idx = overlap_days(src_days_all, ref_days)
            if src_idx.size == 0:
                logger.info(f"[SKIP] No time overlap for {store_path}")
                continue

            # Log human-readable overlap
            tgt_start_day = int(ref_days[tgt_idx.min()])
            tgt_end_day   = int(ref_days[tgt_idx.max()])
            src_start_day = int(src_days_all[src_idx.min()])
            src_end_day   = int(src_days_all[src_idx.max()])
            logger.info(
                f"[OVERLAP] {var_name} → {store_path}\n"
                f"          source: {days_to_iso(src_start_day)} .. {days_to_iso(src_end_day)} (len={src_idx.size})\n"
                f"          target: {days_to_iso(tgt_start_day)} .. {days_to_iso(tgt_end_day)} (len={tgt_idx.size})"
            )

            # Reference array to copy shape/chunks + dims attr
            try:
                ref_arr = pick_reference_array(root)
                ref_name = ref_arr.path.split("/")[-1] if hasattr(ref_arr, "path") else None
            except RuntimeError as e:
                logger.warning(f"[MISS] {e} Store: {store_path}")
                continue

            T_ref, S_ref, L_ref = ref_arr.shape
            if T_ref != n_time or L_ref != n_locations:
                logger.warning(
                    f"[MISMATCH] {store_path}: ref var shape {ref_arr.shape} "
                    f"!= expected (T={n_time}, S=?, L={n_locations}). Skipping."
                )
                continue

            # Prepare target array (create if needed) and set _ARRAY_DIMENSIONS
            array_dims = _array_dims_from_ref(root, ref_name)
            
            # If requested, drop only this variable from the store, then recreate it fresh
            if args.overwrite_var and var_name in root:
                try:
                    del root[var_name]
                    logger.info(f"[OVERWRITE] Deleted existing '{var_name}' in {store_path}")
                except Exception as e:
                    logger.warning(f"[OVERWRITE] Failed to delete '{var_name}' in {store_path}: {e}")
            
            ensure_new_array_in_group(
                root,
                var_name,
                shape=(n_time, S_ref, n_locations),
                chunks=ref_arr.chunks,
                compressor=DEFAULT_COMP,
                array_dims=array_dims,
            )
            
            # Read current array block (may already have values from previous runs)
            arr = root[var_name]  # (time, scenario, location)
            block = arr.oindex[0:n_time, scen_idx, 0:n_locations]  # (T, L)

            # Read source window and scatter into a working buffer
            t0, t1 = int(src_idx.min()), int(src_idx.max()) + 1
            data_sel = read_points_from_src(src_path, var_name, t0, t1, iy, ix)  # (t_sel, L)

            full_block = block.copy()  # start from existing (don’t lose prior data)
            # Put the new source values at overlapping timesteps
            rel = src_idx - t0
            full_block[tgt_idx, :] = data_sel[rel, :]

            # Map each time to its month’s climatology
            clim12 = read_points_from_clim(SEASONAL_PATH, SEASONAL_VAR, iy, ix)  # (12, L)
            clim_mapped = clim12[month_idx, :]                                   # (T,  L)
            
            # sanity check
            assert clim12.shape[0] == 12, f"Seasonality time axis must be 12, got {clim12.shape[0]}"

            # Fill ONLY NaNs with corresponding monthly climatology, and ONLY in 1982..2018
            nan_mask = np.isnan(full_block) & fill_time_mask[:, None]
            if nan_mask.any():
                full_block = np.where(nan_mask, clim_mapped, full_block).astype(np.float32, copy=False)
                logger.info(
                    f"[FILL] Filled {int(nan_mask.sum())} NaNs with monthly climatology (years 1982–2018) in {store_path}"
                )
            else:
                logger.info(f"[FILL] No NaNs to fill in 1982–2018 window for {store_path}")

            # Write back once, after merging source + fills
            arr.oindex[0:n_time, scen_idx:scen_idx+1, 0:n_locations] = full_block[:, None, :]

            # Write a simple "done" flag
            done_dir = Path(store_path) / ".done"
            done_dir.mkdir(exist_ok=True)
            flag = done_dir / f"{var_name}__{NEW_VAR_MODE['scenario']}__{period_key}__{time_res}.done"
            flag.write_text(json.dumps({
                "var": var_name,
                "scenario": NEW_VAR_MODE["scenario"],
                "period": period_key,
                "time_res": time_res
            }), encoding="utf-8")

            # Consolidate metadata so xarray.open_zarr(..., consolidated=True) is robust
            try:
                zarr.consolidate_metadata(str(store_path))
            except Exception as e:
                logger.warning(f"[WARN] consolidate_metadata failed for {store_path}: {e}")

            logger.info(f"[OK] Wrote {var_name} → {store_path} (scenario={NEW_VAR_MODE['scenario']})")

    logger.info("[DONE] All variables processed.")

if __name__ == "__main__":
    main()