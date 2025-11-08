#!/usr/bin/env python3
# scripts/overwrite_prad_test_early_twenty.py
"""
Overwrite ONLY 'potential_radiation' for the test early period (daily) training Zarr.

Changes from previous version:
- Always reads from the fixed file:
    /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/1x1/historical/twenty_year_files/potential_radiation/potential_radiation_1981-2000.nc
- Ignores the file's time axis completely. Assumes it starts at 1901-01-01,
  daily, noleap, and simply writes the first n_time rows needed by the target period.
- Overwrites actual values in the store for the target period; store's own time
  coordinate is created from 1901-01-01 via make_time_axis_days_since_1901.

Array-friendly:
- Builds per-tile tasks (disjoint location slices).
- Uses src.utils.tools.slurm_shard(tasks) to select tasks for this SLURM array element.
- Each shard overwrites only its own tiles (safe concurrent writes to disjoint regions).
- Only shard 0 consolidates metadata and validates NaNs at the end.

Target:
- tensor_type='test', mask_code=2, period='test_period_early', time_res='daily'
- variable: 'potential_radiation'
"""

from __future__ import annotations
import os
import sys
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc

# --- Logging ---------------------------------------------------------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("overwrite_prad_test_early_twenty")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                         datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(h)
    return logger

logger = setup_logging()
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# --- Project imports -------------------------------------------------------
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.utils.make_training_zarrs import (
    ensure_training_skeleton,
    ensure_variable_in_training_store,
    out_store_path,
    scenario_index,
    make_time_axis_days_since_1901,
    build_indices_from_mask,
    flat_to_ij,
)
from src.paths.paths import masks_dir, zarr_dir
from src.utils.tools import slurm_shard

# --- Config ---------------------------------------------------------------
FINAL_LOCATION_CHUNK = 70
VAR_WORKERS = int(os.getenv("VAR_WORKERS", "1"))
DEFAULT_COMP = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)

MASK_PATH = masks_dir / "tvt_mask.nc"
SCENARIOS = ("S0", "S1", "S2", "S3")

# Target scope
TENSOR_TYPE = "test"
MASK_CODE   = 2              # test mask
PERIOD_KEY  = "test_period_early"
TIME_RES    = "daily"
VAR_NAME    = "potential_radiation"

PERIODS = {
    "test_period_early":  ("1901-01-01", "1918-12-31"),
}

LOC_FOR_MASK = {0: "train", 1: "val", 2: "test"}

# Where to write (can override with OUT_ROOT_OVERRIDE env)
OUT_ROOT_DEFAULT = zarr_dir / "training_new"
OUT_ROOT = Path(os.getenv("OUT_ROOT_OVERRIDE", str(OUT_ROOT_DEFAULT)))

# Fixed source file (always used)
FIXED_SOURCE_FILE = Path(
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/1x1/historical/twenty_year_files/potential_radiation/potential_radiation_1981-2000.nc"
)

def is_shard0() -> bool:
    """Return True if this is SLURM array task 0 (or no array set)."""
    tid = os.getenv("SLURM_ARRAY_TASK_ID")
    if tid is None:
        return True  # non-array run acts as shard 0
    try:
        return int(tid) == 0
    except Exception:
        return False

def load_fixed_source_array() -> np.ndarray:
    """
    Load VAR_NAME from the fixed NetCDF file as a numpy array [time, lat, lon].
    Ignores and does not decode the file's time axis.
    """
    if not FIXED_SOURCE_FILE.exists():
        raise FileNotFoundError(f"Fixed source file not found: {FIXED_SOURCE_FILE}")
    ds = xr.open_dataset(FIXED_SOURCE_FILE, decode_times=False)
    if VAR_NAME not in ds:
        raise KeyError(f"Variable '{VAR_NAME}' not found in {FIXED_SOURCE_FILE.name}")
    # Load into memory (thread-safe indexing afterward)
    arr = ds[VAR_NAME].astype("float32").values  # shape [T, Y, X]
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected shape for {VAR_NAME}: {arr.shape} (expected 3D [time, lat, lon])")
    return arr  # numpy float32

def process_prad_tile_from_fixed(
    target_arr,            # zarr array [time, scenario, location]
    n_time: int,
    loc_start: int,
    loc_end: int,
    iy: np.ndarray,
    ix: np.ndarray,
    src_full: np.ndarray,  # numpy [Tsrc, Ny, Nx]
) -> None:
    """
    Overwrite this tile for all scenarios using the fixed source array.
    - Uses the first n_time rows of src_full (ignores src time values).
    - Selects elementwise (iy[k], ix[k]) per location into a [n_time, Lchunk] block.
    """
    Tsrc, Ny, Nx = src_full.shape
    Lchunk = int(loc_end - loc_start)
    if Tsrc < n_time:
        raise RuntimeError(
            f"Source has too few timesteps for requested period: src={Tsrc}, needed={n_time}"
        )

    # Pairwise gather: convert to flat spatial index, slice first n_time days
    flat_idx = (iy.astype(np.int64) * Nx + ix.astype(np.int64))  # [Lchunk]
    block = src_full[:n_time].reshape(n_time, Ny * Nx)[:, flat_idx]  # [n_time, Lchunk]
    block = np.ascontiguousarray(block, dtype="float32")

    # Write the same block for each scenario (S0..S3)
    for scen in SCENARIOS:
        s_idx = scenario_index(scen)
        target_arr.oindex[0:n_time, s_idx:s_idx+1, loc_start:loc_end] = block[:, None, :]

def validate_and_count_nans(store: Path) -> int:
    root = zarr.open_group(str(store), mode="r")
    arr = root[VAR_NAME]
    T, S, L = arr.shape
    total_nans = 0
    for s_idx, _scen in enumerate(SCENARIOS):
        for loc_start in range(0, L, FINAL_LOCATION_CHUNK):
            loc_end = min(loc_start + FINAL_LOCATION_CHUNK, L)
            block = arr.oindex[0:T, s_idx, loc_start:loc_end]
            total_nans += int(np.isnan(block).sum())
    return total_nans

def main():
    # --- indices & time axis (store time is authoritative) ---
    start_str, end_str = PERIODS[PERIOD_KEY]
    time_days = make_time_axis_days_since_1901(TIME_RES, start_str, end_str)
    n_time = len(time_days)

    # masked test set locations (aligned to chunk)
    loc_idx_full = build_indices_from_mask(MASK_PATH, code=MASK_CODE, shuffle=True, seed=42)
    n_locations = (len(loc_idx_full) // FINAL_LOCATION_CHUNK) * FINAL_LOCATION_CHUNK
    loc_idx = loc_idx_full[:n_locations]
    lat_idx_all, lon_idx_all = flat_to_ij(loc_idx)

    loc_key  = {0: "train", 1: "val", 2: "test"}[MASK_CODE]
    set_name = "test"
    set_name = "test"  # top-level tensor
    period_dir = f"train_location_{PERIOD_KEY}"  # literally "train_location_test_period_early"
    leaf = f"{TIME_RES}.zarr"                    # "daily.zarr"
    store = OUT_ROOT / set_name / period_dir / leaf
    store.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Target store: %s", store)
    logger.info("Locations: %d (chunk=%d); Timesteps: %d; Period: %s -> %s",
                n_locations, FINAL_LOCATION_CHUNK, n_time, start_str, end_str)

    # Ensure skeleton/variable exist (safe for concurrent shards; overwrite=False)
    ensure_training_skeleton(
        store,
        time_days=time_days,   # <-- store's time axis: daily noleap from 1901-01-01
        loc_idx=loc_idx,
        overwrite=False,
    )
    ensure_variable_in_training_store(
        store,
        VAR_NAME,
        n_time=n_time,
        n_location=n_locations,
        location_chunk=FINAL_LOCATION_CHUNK,
        overwrite=False,
        compressor=DEFAULT_COMP,
    )

    # Open target zarr once per shard
    root = zarr.open_group(str(store), mode="a")
    arr = root[VAR_NAME]

    # Load the fixed source once per shard (numpy array)
    src_full = load_fixed_source_array()  # [Tsrc, Ny, Nx]
    logger.info("Loaded fixed source array %s with shape %s", FIXED_SOURCE_FILE.name, src_full.shape)

    # Build full tile list
    total_tiles = n_locations // FINAL_LOCATION_CHUNK
    tiles: List[Tuple[int, int]] = []
    for ci in range(total_tiles):
        loc_start = ci * FINAL_LOCATION_CHUNK
        loc_end   = min(loc_start + FINAL_LOCATION_CHUNK, n_locations)
        tiles.append((loc_start, loc_end))

    # Shard across SLURM array
    tiles_this_shard = slurm_shard(tiles)
    logger.info("Total tiles: %d | This shard will process: %d", len(tiles), len(tiles_this_shard))

    # Process our assigned tiles
    wrote_count = 0
    if tiles_this_shard:
        with ThreadPoolExecutor(max_workers=VAR_WORKERS) as ex:
            futs = []
            for (loc_start, loc_end) in tiles_this_shard:
                local = slice(loc_start, loc_end)
                iy = lat_idx_all[local]
                ix = lon_idx_all[local]
                futs.append(ex.submit(
                    process_prad_tile_from_fixed, arr, n_time, loc_start, loc_end, iy, ix, src_full
                ))
            for f in as_completed(futs):
                f.result()
                wrote_count += 1

    logger.info("[OK] Overwrote %d/%d tiles on this shard for %s.", wrote_count, len(tiles_this_shard), VAR_NAME)

    # Only shard 0 consolidates metadata and validates NaNs
    if is_shard0():
        try:
            zarr.consolidate_metadata(str(store))
            logger.info("[FINALIZE] consolidated %s", store)
        except Exception as e:
            logger.warning("[FINALIZE] consolidate_metadata failed for %s: %s", store, e)

        nan_count = validate_and_count_nans(store)
        if nan_count > 0:
            logger.warning("[NaN REPORT] '%s' total NaNs in %s: %d", VAR_NAME, store, nan_count)
        else:
            logger.info("[NaN REPORT] No NaNs for '%s' in %s", VAR_NAME, store)
        # Print bare number last for easy parsing
        print(nan_count)

if __name__ == "__main__":
    main()