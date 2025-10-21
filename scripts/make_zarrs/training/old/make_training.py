#!/usr/bin/env python3
import os, sys, argparse
from pathlib import Path
import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc
import logging

# -----------------------------------------------------------------------------
# Unbuffered logs
# -----------------------------------------------------------------------------
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# -----------------------------------------------------------------------------
# Logging (timestamps + levels)
# -----------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("make_training")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    return logger

logger = setup_logging()

# -----------------------------------------------------------------------------
# Performance knobs
# -----------------------------------------------------------------------------
FINAL_LOCATION_CHUNK = 70                 # target Zarr chunk along location
NUM_LOCATION_CHUNKS_PER_WRITE = 16        # n*70 = locations per write batch
DAILY_DAYS_TOTAL = 365 * 123              # guard/check for whole-period daily

# -----------------------------------------------------------------------------
# Project paths & imports
# -----------------------------------------------------------------------------
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

# Import the new helpers (now support annual|decade|twenty)
from src.utils.make_training_zarrs import (
    out_store_path,
    make_time_axis_days_since_1901,
    build_indices_from_mask,
    flat_to_ij,
    ensure_training_skeleton,
    ensure_variable_in_training_store,
    variables_for_time_res,
    scenario_index,
    open_source_for_var,          
    load_batch_from_daily,
    load_batch_from_full,
    check_target_region_filled,
    assert_no_nans,
)
from src.paths.paths import masks_dir, zarr_dir, preprocessed_dir
from src.dataset.variables import var_names, land_use_vars, climate_vars, nfert
from src.utils.tools import slurm_shard

# -----------------------------------------------------------------------------
# Config (single source of truth here)
# -----------------------------------------------------------------------------
OVERWRITE_SKELETON = False
OVERWRITE_VAR_ARRAYS = False

MASK_PATH  = masks_dir / "tvt_mask.nc"
OUT_ROOT_DEFAULT = zarr_dir / "training"

SCENARIOS  = ("S0", "S1", "S2", "S3")
TIME_RESES = ("daily", "monthly", "annual")   # run all three
LAT_ALL    = np.arange(-89.75, 90.0, 0.5, dtype="float32")   # 360
LON_ALL    = np.arange(0.0, 360.0, 0.5, dtype="float32")     # 720
NY, NX     = len(LAT_ALL), len(LON_ALL)

PERIODS = {
    "train_period":      ("1928-01-01", "2013-12-31"),
    "val_period_early":  ("1919-01-01", "1927-12-31"),
    "val_period_late":   ("2014-01-01", "2017-12-31"),
    "test_period_early": ("1901-01-01", "1918-12-31"),
    "test_period_late":  ("2018-01-01", "2023-12-31"),
    "whole_period":      ("1901-01-01", "2023-12-31"),
}

# Tuples for Each Tensor: (tensor_type, mask_code, [periods...])
SET_SPECS = [
    ("val",   1, ["whole_period"]),
    ("test",  2, ["whole_period"]),
    ("train", 0, ["train_period", "val_period_early", "val_period_late",
                  "test_period_early", "test_period_late"]),
]

ALLOW_NAN_FILL_WITH_ZERO = set(nfert)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Build training Zarrs from preprocessed inputs.")
    ap.add_argument(
        "--daily_files_mode",
        choices=["annual", "decade", "twenty"],
        default=os.getenv("DAILY_FILES_MODE", "annual"),
        help="Which daily source layout to read (or set env DAILY_FILES_MODE).",
    )
    ap.add_argument(
        "--out_root",
        type=Path,
        default=Path(os.getenv("OUT_ROOT_OVERRIDE", str(OUT_ROOT_DEFAULT))),
        help="Root of the output training zarrs (default: zarr_dir/training or $OUT_ROOT_OVERRIDE).",
    )
    args = ap.parse_args(argv)

    OUT_ROOT = args.out_root
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("Output root: %s", OUT_ROOT)
    logger.info("Daily files mode: %s", args.daily_files_mode)

    # -------- Build the store-tasks: (set, mask_code, period, time_res) --------
    all_tasks = []
    for tensor_type, mask_code, period_keys in SET_SPECS:
        for period_key in period_keys:
            for time_res in TIME_RESES:
                all_tasks.append((tensor_type, mask_code, period_key, time_res))

    # -------- Shard across SLURM --------
    tasks = slurm_shard(all_tasks)
    logger.info("SLURM shard -> %d store tasks", len(tasks))

    # -------- Pre-compute shuffled flat indices per mask code --------
    masked_idx_by_code = {
        code: build_indices_from_mask(MASK_PATH, code=code, shuffle=True, seed=42)
        for code in (0, 1, 2)
    }

    # -------- Process each store --------
    for tensor_type, mask_code, period_key, time_res in tasks:
        loc_idx_full = masked_idx_by_code[mask_code]
        n_locations_full = len(loc_idx_full)

        # Tail-drop to enforce exact location chunk size
        n_locations = (n_locations_full // FINAL_LOCATION_CHUNK) * FINAL_LOCATION_CHUNK
        if n_locations != n_locations_full:
            logger.warning(
                "Dropping last %d locations (from %d to %d) to enforce chunk size %d",
                n_locations_full - n_locations, n_locations_full, n_locations, FINAL_LOCATION_CHUNK
            )
        loc_idx = loc_idx_full[:n_locations]

        start_str, end_str = PERIODS[period_key]
        time_days = make_time_axis_days_since_1901(time_res, start_str, end_str)
        n_time = len(time_days)

        # sanity guard for daily length (optional)
        if time_res == "daily" and "whole" in period_key and n_time != DAILY_DAYS_TOTAL:
            logger.warning("daily T=%d, expected %d for whole-period", n_time, DAILY_DAYS_TOTAL)

        store = out_store_path(OUT_ROOT, tensor_type, period_key, time_res)
        ensure_training_skeleton(store, time_days=time_days, loc_idx=loc_idx, overwrite=OVERWRITE_SKELETON)

        # Variables for this time resolution
        vars_this_res = variables_for_time_res(time_res)
        logger.info("%s: vars=%d, locations=%d, time=%d", store, len(vars_this_res), n_locations, n_time)

        # Prepare Zarr handle for variable arrays
        root = zarr.open_group(str(store), mode="a")

        # Batch over locations in multiples of FINAL_LOCATION_CHUNK
        batch_size = FINAL_LOCATION_CHUNK * NUM_LOCATION_CHUNKS_PER_WRITE
        total_chunks = n_locations // FINAL_LOCATION_CHUNK
        lat_idx_all, lon_idx_all = flat_to_ij(loc_idx)

        for batch_start in range(0, n_locations, batch_size):
            batch_end = min(batch_start + batch_size, n_locations)
            local_slice = slice(batch_start, batch_end)
            # map to (iy, ix) on the global grid
            iy = lat_idx_all[local_slice]
            ix = lon_idx_all[local_slice]

            # chunk progress window (1-based)
            chunk_start_1b = batch_start // FINAL_LOCATION_CHUNK + 1
            chunk_end_1b   = batch_end   // FINAL_LOCATION_CHUNK

            for var_name in vars_this_res:
                # Ensure var array exists
                ensure_variable_in_training_store(
                    store, var_name,
                    n_time=n_time, n_location=n_locations,
                    location_chunk=FINAL_LOCATION_CHUNK,
                    overwrite=OVERWRITE_VAR_ARRAYS,
                )
                arr = root[var_name]

                # Skip if this region is already filled (no NaNs) for all scenarios
                already = True
                for scen in SCENARIOS:
                    s_idx = scenario_index(scen)
                    if not check_target_region_filled(arr, 0, n_time, s_idx, batch_start, batch_end):
                        already = False
                        break
                if already:
                    logger.info(
                        "[SKIP] %s chunks [%d–%d/%d] already filled for all scenarios",
                        var_name, chunk_start_1b, chunk_end_1b, total_chunks
                    )
                    continue

                # For each scenario, read sources and write (skip if that scenario's region is done)
                for scen in SCENARIOS:
                    s_idx = scenario_index(scen)
                    if check_target_region_filled(arr, 0, n_time, s_idx, batch_start, batch_end):
                        continue  # region for this scenario already filled

                    # NEW: use helper that supports annual|decade|twenty daily sources
                    mode, src_obj = open_source_for_var(scen, var_name, time_res, daily_mode=args.daily_files_mode)

                    # Load (T, B)
                    if mode == "daily":
                        data_tb = load_batch_from_daily(src_obj, var_name, iy, ix, start_str, end_str)
                    else:
                        data_tb = load_batch_from_full(src_obj, var_name, iy, ix, start_str, end_str)

                    # Check time length matches skeleton
                    if data_tb.shape[0] != n_time:
                        raise RuntimeError(
                            f"Time length mismatch for {var_name} scen={scen}: "
                            f"source T={data_tb.shape[0]} vs target T={n_time}"
                        )

                    # Strict NaN handling
                    if var_name in ALLOW_NAN_FILL_WITH_ZERO:
                        data_tb = np.nan_to_num(data_tb, nan=0.0)
                    else:
                        assert_no_nans(f"{var_name} scen={scen}", data_tb, iy, ix)

                    # Write region: (time, scenario, location)
                    data_tsl = data_tb[:, None, :]  # (T,1,B)
                    arr.oindex[0:n_time, s_idx:s_idx+1, batch_start:batch_end] = data_tsl

                logger.info(
                    "Wrote chunks [%d–%d/%d] for %s",
                    chunk_start_1b, chunk_end_1b, total_chunks, var_name
                )

        # consolidate once per store at the end
        try:
            zarr.consolidate_metadata(str(store))
        except Exception as e:
            logger.warning("consolidate_metadata failed for %s: %s", store, e)

    logger.info("[DONE] All shard tasks complete.")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()