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
"""

from __future__ import annotations
import argparse, os, sys, logging, json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc

# ---------- Project setup ----------
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.utils.make_training_zarrs import (
    out_store_path,
    scenario_index,
    build_indices_from_mask,
    flat_to_ij,
)
from src.paths.paths import masks_dir, zarr_dir
from src.utils.tools import slurm_shard
# ---------- Logging ----------
def setup_logging() -> logging.Logger:
    log = logging.getLogger("addvar_allperiods_noensure")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                         datefmt="%Y-%m-%d %H:%M:%S"))
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

# IMPORTANT: map periods to the correct SET directory (fixed)
SET_SPECS = [
    ("train", 0, ["train_period"]),
    ("val",   1, ["whole_period", "val_period_early", "val_period_late"]),
    ("test",  2, ["whole_period", "test_period_early", "test_period_late"]),
]

# --- New variable configuration ---
NEW_VAR_MODE = {
    "var_names": ["avh15c1_lai"],
    "scenario": "S3",  # only this scenario will be populated
    "source_paths": {
        "avh15c1_lai": "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/avh15c1_lai.nc",
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
    # Fallback: if no data arrays yet, raise — we won't create skeletons here.
    raise RuntimeError("No reference data array found in store for shape/chunks; store must contain at least one data var.")

def ensure_new_array_in_group(
    root: zarr.hierarchy.Group,
    var_name: str,
    shape: Tuple[int, int, int],
    chunks: Tuple[int, int, int],
    compressor=DEFAULT_COMP,
):
    """Create a zarr array if missing, otherwise do nothing."""
    if var_name in root:
        return
    root.create_dataset(
        var_name,
        shape=shape,
        chunks=chunks,
        dtype="f4",
        compressor=compressor,
        fill_value=np.float32(np.nan),
        overwrite=False,
        dimension_separator=".",
        filters=None,
        order="C",
    )
    # Note: coords/attrs not touched

def read_points_from_src(src_path: Path, var_name: str, t0: int, t1: int, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    """Read (t1-t0, B) from (time, lat, lon) source for given point list."""
    with xr.open_dataset(src_path, decode_times=False) as ds:
        da = ds[var_name]
        da = da.isel(time=slice(t0, t1))  # (t_sel, lat, lon)
        # figure lat/lon dim names
        lat_dim = next(d for d in ("lat", "latitude", "y") if d in da.dims)
        lon_dim = next(d for d in ("lon", "longitude", "x") if d in da.dims)
        # pull exact points
        pts = xr.DataArray(np.arange(iy.size), dims=("points",))
        sub = da.transpose("time", lat_dim, lon_dim).isel(
            {lat_dim: (pts.dims[0], iy), lon_dim: (pts.dims[0], ix)}
        )
        return np.asarray(sub.values, dtype=np.float32)  # (t_sel, B)

# ---------- Main ----------
def main(argv=None):
    _ = argparse.ArgumentParser(description="Add variable(s) to existing Zarrs (monthly only), no skeleton/ensure.").parse_args(argv)

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
        logger.info(f"[VAR] {var_name}: src={src_path.name} cadence={src_time_res} "
                    f"span=[{int(src_days_all.min())}..{int(src_days_all.max())}] "
                    f"len={src_days_all.size}")

        # We only write monthly
        if src_time_res != "monthly":
            logger.info(f"[SKIP] {var_name}: source cadence {src_time_res} != monthly")
            continue

        scen_idx = scenario_index(NEW_VAR_MODE["scenario"])

        # Loop sets/periods we care about (with corrected mapping)
        # Prepare a work list of (set_name, mask_code, period_key)
        work_list = [
            (set_name, mask_code, period_key)
            for set_name, mask_code, period_keys in SET_SPECS
            for period_key in period_keys
        ]

        # Split across SLURM array jobs if running under sbatch
        total_items = len(work_list)
        work_list = slurm_shard(work_list)
        logger.info(f"[SLURM] This shard has {len(work_list)} of {total_items} total items.")

        for set_name, mask_code, period_key in work_list:
            # locations aligned to FINAL_LOCATION_CHUNK
            # directory naming + which mask to use for L
            if period_key == "whole_period":
                loc_key = set_name                 # val_location_* or test_location_* or train_location_*
                loc_mask_code = mask_code          # use the set's mask
            else:
                loc_key = "train"                  # train_location_{period} under val/test dirs
                loc_mask_code = 0                  # BUT locations are the TRAIN set

            # locations aligned to FINAL_LOCATION_CHUNK (use loc_mask_code, not mask_code)
            loc_idx_full = masked_idx_by_code[loc_mask_code]
            n_locations = (len(loc_idx_full) // FINAL_LOCATION_CHUNK) * FINAL_LOCATION_CHUNK
            if n_locations == 0:
                logger.warning(f"[WARN] No locations for set={set_name} period={period_key} (mask={loc_mask_code}); skip.")
                continue
            loc_idx = loc_idx_full[:n_locations]
            iy, ix = flat_to_ij(loc_idx)


            time_res = "monthly"  # fixed
            store = out_store_path(OUT_ROOT, set_name, loc_key, period_key, time_res)

            # Open existing store; do NOT create skeletons
            if not Path(store).exists():
                logger.warning(f"[MISS] Store does not exist, skipping: {store}")
                continue
            root = zarr.open_group(str(store), mode="a")

            # Use the store's own time axis (must exist)
            if "time" not in root:
                logger.warning(f"[BAD] No 'time' coord in store, skipping: {store}")
                continue
            ref_days = np.asarray(root["time"][:], dtype=np.int64)
            n_time = ref_days.size

            # Overlap with source
            src_idx, tgt_idx = overlap_days(src_days_all, ref_days)
            if src_idx.size == 0:
                logger.info(f"[SKIP] No time overlap for {store}")
                continue

            # --- New: print the time window that will be written ---
            tgt_start_day = int(ref_days[tgt_idx.min()])
            tgt_end_day   = int(ref_days[tgt_idx.max()])
            src_start_day = int(src_days_all[src_idx.min()])
            src_end_day   = int(src_days_all[src_idx.max()])
            logger.info(
                f"[OVERLAP] {var_name} → {store}\n"
                f"          source: {days_to_iso(src_start_day)} .. {days_to_iso(src_end_day)} "
                f"(len={src_idx.size})\n"
                f"          target: {days_to_iso(tgt_start_day)} .. {days_to_iso(tgt_end_day)} "
                f"(len={tgt_idx.size})"
            )

            # Ensure the target array exists (create once from a reference var)
            try:
                ref_arr = pick_reference_array(root)
            except RuntimeError as e:
                logger.warning(f"[MISS] {e} Store: {store}")
                continue

            T_ref, S_ref, L_ref = ref_arr.shape
            if T_ref != n_time or L_ref != n_locations:
                logger.warning(f"[MISMATCH] {store}: ref var shape {ref_arr.shape} "
                                f"!= expected (T={n_time}, S=?, L={n_locations}). Skipping.")
                continue

            # Copy chunks from reference (keep S chunk=1, L chunk=FINAL_LOCATION_CHUNK usually)
            chunks = ref_arr.chunks
            if var_name not in root:
                ensure_new_array_in_group(
                    root,
                    var_name,
                    shape=(n_time, S_ref, n_locations),
                    chunks=chunks,
                    compressor=DEFAULT_COMP,
                )

            # Read minimal source window and scatter into full n_time
            t0, t1 = int(src_idx.min()), int(src_idx.max()) + 1
            data_sel = read_points_from_src(src_path, var_name, t0, t1, iy, ix)  # (t_sel, B)

            full_block = np.full((n_time, n_locations), np.nan, dtype=np.float32)
            rel = src_idx - t0
            full_block[tgt_idx, :] = data_sel[rel, :]

            # Write into the single scenario; leave others as is (NaN prefill)
            arr = root[var_name]
            arr.oindex[0:n_time, scen_idx:scen_idx+1, 0:n_locations] = full_block[:, None, :]

            # Optional flag
            done_dir = Path(store) / ".done"
            done_dir.mkdir(exist_ok=True)
            flag = done_dir / f"{var_name}__{NEW_VAR_MODE['scenario']}__{period_key}__{time_res}.done"
            flag.write_text(json.dumps({"var": var_name,
                                        "scenario": NEW_VAR_MODE["scenario"],
                                        "period": period_key,
                                        "time_res": time_res}), encoding="utf-8")

            try:
                zarr.consolidate_metadata(str(store))
            except Exception as e:
                logger.warning(f"[WARN] consolidate_metadata failed for {store}: {e}")

            logger.info(f"[OK] Wrote {var_name} → {store} (scenario={NEW_VAR_MODE['scenario']})")

    logger.info("[DONE] All variables processed.")

if __name__ == "__main__":
    main()