#!/usr/bin/env python3
# scripts/overwrite_prad_test_early_twenty.py
"""
Overwrite ONLY 'potential_radiation' for the test early period (daily) training Zarr.

Array-friendly:
- Builds per-tile tasks (disjoint location slices).
- Uses src.utils.tools.slurm_shard(tasks) to select tasks for this SLURM array element.
- Each shard overwrites only its own tiles (safe concurrent writes to disjoint regions).
- Only shard 0 consolidates metadata and validates NaNs at the end.

Details:
- Target:  tensor_type='test', mask_code=2, period='test_period_early', time_res='daily'
- Variable: 'potential_radiation'
- Source layout: daily "twenty"-year files (daily_files_mode='twenty')
- Overwrite: unconditional (ignores any .done flags)
- Validation: shard 0 counts total NaNs for 'potential_radiation' and prints the number
"""

from __future__ import annotations
import os
import sys
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List

import numpy as np
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
    open_source_for_var,
    load_batch_from_daily_tiles,
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

# Force daily twenty-year layout for sources
DAILY_FILES_MODE = "twenty"  # same meaning as --daily_files_mode=twenty


def is_shard0() -> bool:
    """Return True if this is SLURM array task 0 (or no array set)."""
    tid = os.getenv("SLURM_ARRAY_TASK_ID")
    if tid is None:
        return True  # non-array run acts as shard 0
    try:
        return int(tid) == 0
    except Exception:
        return False


def process_prad_tile(
    arr,                    # zarr array [time, scenario, location]
    n_time: int,
    loc_start: int,
    loc_end: int,
    iy: np.ndarray,
    ix: np.ndarray,
    start_str: str,
    end_str: str,
) -> None:
    """
    Unconditionally read from daily-twenty-year sources and overwrite this tile for all scenarios.
    """
    for scen in SCENARIOS:
        s_idx = scenario_index(scen)

        mode, src_obj = open_source_for_var(scen, VAR_NAME, TIME_RES, daily_mode=DAILY_FILES_MODE)
        if mode != "daily":
            raise RuntimeError(f"Expected daily mode for {VAR_NAME}, got {mode}")

        data_tb = load_batch_from_daily_tiles(src_obj, VAR_NAME, iy, ix, start_str, end_str)

        if data_tb.shape[0] != n_time:
            raise RuntimeError(
                f"Time mismatch {VAR_NAME} scen={scen}: source T={data_tb.shape[0]} vs target T={n_time}"
            )

        data_tb = np.ascontiguousarray(data_tb, dtype="float32")  # enforce dtype

        # overwrite write into disjoint location slice
        arr.oindex[0:n_time, s_idx:s_idx+1, loc_start:loc_end] = data_tb[:, None, :]


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
    # --- indices & time axis ---
    start_str, end_str = PERIODS[PERIOD_KEY]
    time_days = make_time_axis_days_since_1901(TIME_RES, start_str, end_str)
    n_time = len(time_days)

    # masked test set locations (aligned to chunk)
    loc_idx_full = build_indices_from_mask(MASK_PATH, code=MASK_CODE, shuffle=True, seed=42)
    n_locations = (len(loc_idx_full) // FINAL_LOCATION_CHUNK) * FINAL_LOCATION_CHUNK
    loc_idx = loc_idx_full[:n_locations]
    lat_idx_all, lon_idx_all = flat_to_ij(loc_idx)

    loc_key  = LOC_FOR_MASK[MASK_CODE]
    set_name = "test"
    store = out_store_path(OUT_ROOT, set_name, loc_key, PERIOD_KEY, TIME_RES)

    logger.info("Target store: %s", store)
    logger.info("Locations: %d (chunk=%d); Timesteps: %d; Period: %s -> %s",
                n_locations, FINAL_LOCATION_CHUNK, n_time, start_str, end_str)

    # Ensure skeleton/variable exist (safe for concurrent shards; overwrite=False)
    ensure_training_skeleton(
        store,
        time_days=time_days,
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

    # Open once per shard
    root = zarr.open_group(str(store), mode="a")
    arr = root[VAR_NAME]

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
                    process_prad_tile, arr, n_time, loc_start, loc_end, iy, ix, start_str, end_str
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
    else:
        # Non-zero tasks print nothing (avoid double-reporting)
        pass


if __name__ == "__main__":
    main()