#!/usr/bin/env python3
# scripts/make_training_from_preprocessed.py
"""
Make training Zarrs (fresh/resumable) with fast I/O:
- zstd clevel=1 for fast read/write.
- Resumable via per-(var,scenario,chunkrange) .done flags (atomic).
- Optional final full-store NaN validation with --validate.
- Reads **daily inputs** from annual/decade/twenty-year files, selectable via
  --daily_files_mode or env DAILY_FILES_MODE.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import zarr
from numcodecs import Blosc

# ---------------- Logging ----------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("make_training")
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

# ---------------- Project imports ----------------
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.utils.make_training_zarrs import (
    ensure_training_skeleton,
    ensure_variable_in_training_store,
    out_store_path,
    variables_for_time_res,
    scenario_index,
    open_source_for_var,
    load_batch_from_daily,
    load_batch_from_full,
    make_time_axis_days_since_1901,
    build_indices_from_mask,
    flat_to_ij,
)
from src.paths.paths import masks_dir, zarr_dir
from src.dataset.variables import nfert
from src.utils.tools import slurm_shard

# ---------------- Fixed config ----------------
FINAL_LOCATION_CHUNK = 70          # location chunk in the Zarr layout
CHUNKS_PER_WRITE = 48              # write 48 * 70 locations per batch
VAR_WORKERS = 6                    # threads over variables within a batch
DAILY_DAYS_TOTAL = 365 * 123       # guard for daily whole-period

DEFAULT_COMP = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)

MASK_PATH = masks_dir / "tvt_mask.nc"
SCENARIOS = ("S0", "S1", "S2", "S3")
TIME_RESES = ("daily", "monthly", "annual")
PERIODS = {
    "train_period":      ("1928-01-01", "2013-12-31"),
    "val_period_early":  ("1919-01-01", "1927-12-31"),
    "val_period_late":   ("2014-01-01", "2017-12-31"),
    "test_period_early": ("1901-01-01", "1918-12-31"),
    "test_period_late":  ("2018-01-01", "2023-12-31"),
    "whole_period":      ("1901-01-01", "2023-12-31"),
}
SET_SPECS = [
    ("val",   1, ["whole_period"]),
    ("test",  2, ["whole_period"]),
    ("train", 0, ["train_period", "val_period_early", "val_period_late",
                  "test_period_early", "test_period_late"]),
]

ALLOW_NAN_FILL_WITH_ZERO = set(nfert)

OUT_ROOT_DEFAULT = zarr_dir / "training_new"
OUT_ROOT = Path(os.getenv("OUT_ROOT_OVERRIDE", str(OUT_ROOT_DEFAULT)))

# ---------------- Helpers ----------------
def chunk_window_1b(batch_start: int, batch_end: int) -> tuple[int, int]:
    """Return 1-based [start,end] location-chunk indices for this batch."""
    cs = batch_start // FINAL_LOCATION_CHUNK + 1
    ce = batch_end // FINAL_LOCATION_CHUNK
    return cs, ce


def done_flag_path(done_dir: Path, var_name: str, scen: str, cs: int, ce: int) -> Path:
    return done_dir / f"{var_name}__{scen}__{cs}-{ce}.done"


def mark_done_atomic(flag_path: Path, meta: dict):
    tmp = flag_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    os.replace(tmp, flag_path)


@lru_cache(maxsize=None)
def open_source_for_var_cached(scen: str, var_name: str, time_res: str, daily_mode: str):
    return open_source_for_var(scen, var_name, time_res, daily_mode=daily_mode)

# ---------------- Core write ----------------
def process_var_batch(
    var_name: str,
    arr,
    store: Path,
    done_dir: Path,
    n_time: int,
    batch_start: int,
    batch_end: int,
    iy: np.ndarray,
    ix: np.ndarray,
    start_str: str,
    end_str: str,
    time_res: str,
    daily_mode: str,
) -> bool:
    """
    Write one variable's data for [batch_start:batch_end] locations across all scenarios.
    Resumable via .done flags. Returns True if anything was written.
    """
    wrote_any = False
    cs, ce = chunk_window_1b(batch_start, batch_end)
    total_chunks = arr.shape[2] // FINAL_LOCATION_CHUNK

    # Fast global skip if all scenarios already flagged done
    if all(done_flag_path(done_dir, var_name, s, cs, ce).exists() for s in SCENARIOS):
        return False

    for scen in SCENARIOS:
        s_idx = scenario_index(scen)
        flag = done_flag_path(done_dir, var_name, scen, cs, ce)
        if flag.exists():
            continue

        mode, src_obj = open_source_for_var_cached(scen, var_name, time_res, daily_mode)
        if mode == "daily":
            data_tb = load_batch_from_daily(src_obj, var_name, iy, ix, start_str, end_str)
        else:
            data_tb = load_batch_from_full(src_obj, var_name, iy, ix, start_str, end_str)

        if data_tb.shape[0] != n_time:
            raise RuntimeError(
                f"Time mismatch {var_name} scen={scen}: source T={data_tb.shape[0]} vs target T={n_time}"
            )

        if var_name in ALLOW_NAN_FILL_WITH_ZERO:
            data_tb = np.nan_to_num(data_tb, nan=0.0)

        data_tb = np.ascontiguousarray(data_tb, dtype="float32")
        data_tsl = data_tb[:, None, :]  # (time, scenario=1, batch_locs)

        # Write to [time, scenario, location-slice]
        arr.oindex[0:n_time, s_idx:s_idx+1, batch_start:batch_end] = data_tsl

        mark_done_atomic(flag, {
            "var": var_name, "scen": scen, "cs": cs, "ce": ce,
            "time_res": time_res, "range_loc": [int(batch_start), int(batch_end)]
        })
        wrote_any = True

    if wrote_any:
        logger.info("Wrote chunks [%d–%d/%d] for %s", cs, ce, total_chunks, var_name)
    return wrote_any

# ---------------- Validation ----------------
def validate_store_no_nans(store: Path, time_res: str):
    """
    Full-store NaN scan. Iterates vars × scenarios × location-slices.
    Logs first few offending indices if NaNs found and raises at the end.
    """
    root = zarr.open_group(str(store), mode="r")
    errors = []

    vars_this_res = variables_for_time_res(time_res)
    for var_name in vars_this_res:
        arr = root[var_name]
        T, S, L = arr.shape
        loc_chunk = FINAL_LOCATION_CHUNK
        for s_idx, scen in enumerate(SCENARIOS):
            for loc_start in range(0, L, loc_chunk):
                loc_end = min(loc_start + loc_chunk, L)
                block = arr.oindex[0:T, s_idx, loc_start:loc_end]
                if np.isnan(block).any():
                    nan_pos = np.argwhere(np.isnan(block))
                    examples = nan_pos[:3].tolist()
                    errors.append((var_name, scen, loc_start, loc_end, examples))
                    logger.error("[NaN] %s scen=%s loc[%d:%d] e.g. %s",
                                 var_name, scen, loc_start, loc_end, examples[:3])

    if errors:
        raise RuntimeError(f"NaNs found in {len(errors)} block(s). See log above.")
    logger.info("[VALIDATE] No NaNs detected in store %s", store)

# ---------------- Main ----------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Build training Zarrs (fresh, fast, resumable).")
    parser.add_argument("--validate", action="store_true", help="Scan final Zarr for NaNs after writing.")
    parser.add_argument(
        "--daily_files_mode",
        choices=["annual", "decade", "twenty"],
        default=os.getenv("DAILY_FILES_MODE", "annual"),
        help="Which daily source layout to read (or set env DAILY_FILES_MODE).",
    )
    args = parser.parse_args(argv)

    # Masked location indices (pre-shuffled per code)
    masked_idx_by_code = {
        code: build_indices_from_mask(MASK_PATH, code=code, shuffle=True, seed=42)
        for code in (0, 1, 2)
    }

    # Build and shard store tasks across Slurm
    all_tasks = []
    for tensor_type, mask_code, period_keys in SET_SPECS:
        for period_key in period_keys:
            for time_res in TIME_RESES:
                all_tasks.append((tensor_type, mask_code, period_key, time_res))

    tasks = slurm_shard(all_tasks)
    logger.info("SLURM shard -> %d store task(s)", len(tasks))

    # Process stores
    for tensor_type, mask_code, period_key, time_res in tasks:
        loc_idx_full = masked_idx_by_code[mask_code]
        n_locations_full = len(loc_idx_full)

        # Tail-drop to enforce exact location chunk size
        n_locations = (n_locations_full // FINAL_LOCATION_CHUNK) * FINAL_LOCATION_CHUNK
        if n_locations != n_locations_full:
            logger.warning("Dropping last %d locations (from %d to %d) to enforce chunk=%d",
                           n_locations_full - n_locations, n_locations_full, n_locations, FINAL_LOCATION_CHUNK)

        loc_idx = loc_idx_full[:n_locations]
        lat_idx_all, lon_idx_all = flat_to_ij(loc_idx)

        start_str, end_str = PERIODS[period_key]
        time_days = make_time_axis_days_since_1901(time_res, start_str, end_str)
        n_time = len(time_days)

        if time_res == "daily" and "whole" in period_key and n_time != DAILY_DAYS_TOTAL:
            logger.warning("daily T=%d, expected %d for whole-period", n_time, DAILY_DAYS_TOTAL)

        # Prepare store and arrays
        store = out_store_path(OUT_ROOT, tensor_type, period_key, time_res)
        ensure_training_skeleton(store, time_days=time_days, loc_idx=loc_idx, overwrite=False)

        vars_this_res = variables_for_time_res(time_res)
        root = zarr.open_group(str(store), mode="a")
        for var_name in vars_this_res:
            ensure_variable_in_training_store(
                store, var_name,
                n_time=n_time, n_location=n_locations,
                location_chunk=FINAL_LOCATION_CHUNK,
                overwrite=False,
                compressor=DEFAULT_COMP,
            )

        logger.info("%s: vars=%d, locations=%d, time=%d", store, len(vars_this_res), n_locations, n_time)

        # Done flags dir
        done_dir = Path(store) / ".done"
        done_dir.mkdir(exist_ok=True)

        # Batch loop
        batch_size = FINAL_LOCATION_CHUNK * CHUNKS_PER_WRITE
        total_chunks = n_locations // FINAL_LOCATION_CHUNK
        n_batches = (n_locations + batch_size - 1) // batch_size
        logger.info("Work plan — total_chunks=%d, batch_size=%d (=%d×%d), n_batches=%d",
                    total_chunks, batch_size, FINAL_LOCATION_CHUNK, CHUNKS_PER_WRITE, n_batches)

        # Thread pool over variables
        for batch_start in range(0, n_locations, batch_size):
            batch_end = min(batch_start + batch_size, n_locations)
            local = slice(batch_start, batch_end)
            iy = lat_idx_all[local]
            ix = lon_idx_all[local]

            wrote_count = 0
            with ThreadPoolExecutor(max_workers=VAR_WORKERS) as ex:
                futs = []
                for var_name in vars_this_res:
                    arr = root[var_name]
                    futs.append(ex.submit(
                        process_var_batch, var_name, arr, store, done_dir,
                        n_time, batch_start, batch_end, iy, ix,
                        start_str, end_str, time_res,
                        args.daily_files_mode,  # <-- pass mode
                    ))
                for f in as_completed(futs):
                    if f.result():
                        wrote_count += 1

            cs, ce = chunk_window_1b(batch_start, batch_end)
            if wrote_count == 0:
                logger.info("[SKIP] All variables already flagged done for chunks [%d–%d/%d]", cs, ce, total_chunks)

        # Consolidate metadata once per store
        try:
            zarr.consolidate_metadata(str(store))
        except Exception as e:
            logger.warning("consolidate_metadata failed for %s: %s", store, e)

        # Optional final validation
        if args.validate:
            validate_store_no_nans(store, time_res)

    logger.info("[DONE] All shard tasks complete.")


if __name__ == "__main__":
    main()