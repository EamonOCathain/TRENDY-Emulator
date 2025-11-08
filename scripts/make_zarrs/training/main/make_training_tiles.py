#!/usr/bin/env python3
# scripts/make_training_from_preprocessed.py
"""
Tile-parallel Zarr builder with 2D sharding: (task combo) x (location tile)

- Each "work unit" = (tensor_type, mask_code, period_key, time_res, loc_start, loc_end)
- The full worklist is ALL task combos × ALL tiles (aligned to FINAL_LOCATION_CHUNK)
- slurm_shard() splits that flat list across SLURM array elements
- Safe concurrent writes: each unit writes disjoint location chunks
- INIT_ONLY=1 : create skeleton + variables then exit (no writes)
- FINALIZE=1  : consolidate metadata (+ --validate) after writers finish
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
from typing import Tuple, List

import numpy as np
import zarr
from numcodecs import Blosc

import xarray as xr
xr.set_options(file_cache_maxsize=1)
import re
from functools import lru_cache

# timing / memory profiling
import time, resource
def rss_gb():
    try:
        # ru_maxrss is KB on Linux, bytes on macOS; the / (1024**2) normalizes to GB for Linux
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    except Exception:
        return float('nan')
def _mb(nbytes): return f"{nbytes/1e6:.1f} MB"
PERF_ON = os.getenv("PERF", "1") != "0"  # set PERF=0 to disable

# -------- Logging --------
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

# -------- Project imports --------
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.utils.make_training_zarrs import (
    ensure_training_skeleton,
    ensure_variable_in_training_store,
    out_store_path,
    variables_for_time_res,
    scenario_index,
    open_source_for_var,
    load_batch_from_daily_tiles,
    load_batch_from_full,
    make_time_axis_days_since_1901,
    build_indices_from_mask,
    flat_to_ij,
)
from src.paths.paths import masks_dir, zarr_dir
from src.dataset.variables import nfert
from src.utils.tools import slurm_shard  # should support your env overrides

# -------- Config --------
FINAL_LOCATION_CHUNK = 70
VAR_WORKERS = int(os.getenv("VAR_WORKERS", "1"))
DAILY_DAYS_TOTAL = 365 * 123
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

# -------- Helpers --------
# map mask code -> location key bucket
LOC_FOR_MASK = {0: "train", 1: "val", 2: "test"}

def resolve_set_name(tensor_type: str, period_key: str) -> str:
    """
    True set (directory) to write to. Period overrides location:
      - val_period_*  -> 'val'
      - test_period_* -> 'test'
      - whole_period  -> keep tensor_type (val/test in SET_SPECS)
      - train_period  -> 'train'
    """
    if period_key.startswith("val_period"):
        return "val"
    if period_key.startswith("test_period"):
        return "test"
    if period_key == "whole_period":
        return tensor_type  # in SET_SPECS this is 'val' or 'test'
    # train-only spans
    return tensor_type

def chunk_window_1b(start: int, end: int) -> tuple[int, int]:
    cs = start // FINAL_LOCATION_CHUNK + 1
    ce = end   // FINAL_LOCATION_CHUNK
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

_FLAG_RANGE_RE = re.compile(r"__(\d+)-(\d+)\.done$")

@lru_cache(maxsize=None)
def _done_intervals(done_dir: str, var_name: str, scen: str):
    """Return list of (start,end) 1-based chunk ranges from existing flags."""
    d = Path(done_dir)
    out = []
    for p in d.glob(f"{var_name}__{scen}__*.done"):
        m = _FLAG_RANGE_RE.search(p.name)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.append((a, b))
    return out

def tile_is_done(done_dir: Path, var_name: str, scen: str, cs: int, ce: int) -> bool:
    """True if there is an exact flag OR any superset interval that covers [cs,ce]."""
    # exact (new style)
    if (done_dir / f"{var_name}__{scen}__{cs}-{ce}.done").exists():
        return True
    # legacy (old large ranges)
    for a, b in _done_intervals(str(done_dir), var_name, scen):
        if a <= cs and ce <= b:
            return True
    return False

def process_var_tile(
    var_name: str,
    arr,                    # zarr array [time, scenario, location]
    store: Path,
    done_dir: Path,
    n_time: int,
    loc_start: int,
    loc_end: int,
    iy: np.ndarray,
    ix: np.ndarray,
    start_str: str,
    end_str: str,
    time_res: str,
    daily_mode: str,
) -> bool:
    import time, resource  # local import to avoid module-wide changes

    def rss_gb():
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)  # GB on Linux
        except Exception:
            return float("nan")

    def _mb(nbytes: int) -> str:
        return f"{nbytes/1e6:.1f} MB"

    PERF_ON = os.getenv("PERF", "1") != "0"

    wrote_any = False
    cs, ce = chunk_window_1b(loc_start, loc_end)

    # fast skip: if all scenarios already covered by *any* flag that covers this tile
    if all(tile_is_done(done_dir, var_name, s, cs, ce) for s in SCENARIOS):
        return False

    for scen in SCENARIOS:
        s_idx = scenario_index(scen)
        if tile_is_done(done_dir, var_name, scen, cs, ce):
            continue

        # --- timing: open/select sources ---
        t0 = time.perf_counter()
        mode, src_obj = open_source_for_var_cached(scen, var_name, time_res, daily_mode)
        t_open = time.perf_counter()

        logger.info(".. %s scen=%s tile[%d–%d]: reading", var_name, scen, cs, ce)

        # --- timing: read ---
        if mode == "daily":
            data_tb = load_batch_from_daily_tiles(src_obj, var_name, iy, ix, start_str, end_str)
        else:
            data_tb = load_batch_from_full(src_obj, var_name, iy, ix, start_str, end_str)
        t_read = time.perf_counter()

        if data_tb.shape[0] != n_time:
            raise RuntimeError(
                f"Time mismatch {var_name} scen={scen}: source T={data_tb.shape[0]} vs target T={n_time}"
            )

        # --- timing: process (NaN policy + dtype cast) ---
        if var_name in ALLOW_NAN_FILL_WITH_ZERO:
            data_tb = np.nan_to_num(data_tb, nan=0.0)
        data_tb = np.ascontiguousarray(data_tb, dtype="float32")
        t_proc = time.perf_counter()

        logger.info(".. %s scen=%s tile[%d–%d]: read T=%d, B=%d -> writing",
                    var_name, scen, cs, ce, data_tb.shape[0], data_tb.shape[1])

        # --- timing: write ---
        arr.oindex[0:n_time, s_idx:s_idx+1, loc_start:loc_end] = data_tb[:, None, :]
        t_write = time.perf_counter()

        # perf log
        if PERF_ON:
            n_bytes = int(data_tb.size) * 4  # float32
            open_s  = t_open - t0
            read_s  = t_read - t_open
            proc_s  = t_proc - t_read
            write_s = t_write - t_proc
            r_mb_s  = (n_bytes/1e6) / max(1e-6, read_s)
            w_mb_s  = (n_bytes/1e6) / max(1e-6, write_s)
            logger.info(
                "[PERF] %s scen=%s tile[%d-%d] open=%.2fs read=%.2fs proc=%.2fs write=%.2fs "
                "payload=%s rss=%.2fGB read_MBps=%.1f write_MBps=%.1f",
                var_name, scen, cs, ce,
                open_s, read_s, proc_s, write_s,
                _mb(n_bytes), rss_gb(), r_mb_s, w_mb_s
            )

        # after a successful write:
        mark_done_atomic(
            done_flag_path(done_dir, var_name, scen, cs, ce),
            {
                "var": var_name,
                "scen": scen,
                "cs": cs,
                "ce": ce,
                "time_res": time_res,
                "range_loc": [int(loc_start), int(loc_end)],
            },
        )
        wrote_any = True

    return wrote_any

def validate_store_no_nans(store: Path, time_res: str):
    root = zarr.open_group(str(store), mode="r")
    vars_this_res = variables_for_time_res(time_res)
    for var_name in vars_this_res:
        arr = root[var_name]
        T, S, L = arr.shape
        for s_idx, scen in enumerate(SCENARIOS):
            for loc_start in range(0, L, FINAL_LOCATION_CHUNK):
                loc_end = min(loc_start + FINAL_LOCATION_CHUNK, L)
                block = arr.oindex[0:T, s_idx, loc_start:loc_end]
                if np.isnan(block).any():
                    ex = np.argwhere(np.isnan(block))[:3].tolist()
                    raise RuntimeError(f"[NaN] {var_name} scen={scen} loc[{loc_start}:{loc_end}] e.g. {ex}")
    logger.info("[VALIDATE] No NaNs in %s", store)

# -------- Main --------
def main(argv=None):
    p = argparse.ArgumentParser(description="2D-sharded (task × tile) training Zarr builder")
    p.add_argument("--validate", action="store_true",
                   help="(Used with FINALIZE=1) Scan final Zarr for NaNs.")
    p.add_argument("--daily_files_mode", choices=["annual", "decade", "twenty"],
                   default=os.getenv("DAILY_FILES_MODE", "annual"),
                   help="Which daily source layout to read (or set env DAILY_FILES_MODE).")
    args = p.parse_args(argv)

    INIT_ONLY = os.getenv("INIT_ONLY") == "1"
    FINALIZE  = os.getenv("FINALIZE") == "1"

    # Precompute masked location indices for all codes
    masked_idx_by_code = {
        code: build_indices_from_mask(MASK_PATH, code=code, shuffle=True, seed=42)
        for code in (0, 1, 2)
    }
    
    # --- Build ALL task combos (these were "the 21") ---
    all_tasks: List[Tuple[str, int, str, str]] = []
    for tensor_type, mask_code, period_keys in SET_SPECS:
        for period_key in period_keys:
            for time_res in TIME_RESES:
                all_tasks.append((tensor_type, mask_code, period_key, time_res))

    # --- Optional: run only one of the original 21 tasks by index ---
    SELECT_TASK_IDX = os.getenv("SELECT_TASK_IDX")
    if SELECT_TASK_IDX is not None:
        idx = int(SELECT_TASK_IDX)
        if not (0 <= idx < len(all_tasks)):
            raise ValueError(f"SELECT_TASK_IDX={idx} out of range 0..{len(all_tasks)-1}")
        logger.info(f"[FILTER] Restricting to original task index {idx}: {all_tasks[idx]}")
        all_tasks = [all_tasks[idx]]

    # --- Build a FLAT worklist of (task × tile) work units ---
    work_units: List[Tuple[str, int, str, str, int, int]] = []
    task_meta = {}  # cache per-task meta to avoid recomputing
    for (tensor_type, mask_code, period_key, time_res) in all_tasks:
        key = (tensor_type, mask_code, period_key, time_res)

        # Per-task meta
        loc_idx_full = masked_idx_by_code[mask_code]
        n_locations_full = len(loc_idx_full)
        n_locations = (n_locations_full // FINAL_LOCATION_CHUNK) * FINAL_LOCATION_CHUNK
        loc_idx = loc_idx_full[:n_locations]
        lat_idx_all, lon_idx_all = flat_to_ij(loc_idx)

        start_str, end_str = PERIODS[period_key]
        time_days = make_time_axis_days_since_1901(time_res, start_str, end_str)
        n_time = len(time_days)

        task_meta[key] = {
            "n_locations": n_locations,
            "lat_idx_all": lat_idx_all,
            "lon_idx_all": lon_idx_all,
            "start_str": start_str,
            "end_str": end_str,
            "n_time": n_time,
        }
        
        if time_res == "daily":
            years = n_time / 365
        elif time_res == "monthly":
            years = n_time / 12
        else:
            years = n_time
            
        total_years_windows = n_time * n_locations
        
        print(f"For location: {mask_code} and time period {period_key}: n locations = {n_locations}, time res {time_res} and n timesteps {n_time}. Start date {start_str} and end date {end_str}. Years = {years} and total year windows is {total_years_windows}.\n")
        
        

        # Tiles aligned to location chunk
        total_tiles = n_locations // FINAL_LOCATION_CHUNK
        for ci in range(total_tiles):
            loc_start = ci * FINAL_LOCATION_CHUNK
            loc_end   = min(loc_start + FINAL_LOCATION_CHUNK, n_locations)
            work_units.append((tensor_type, mask_code, period_key, time_res, loc_start, loc_end))

    logger.info("Total work units (task × tile) = %d", len(work_units))

    # --- INIT ONLY: create skeletons/variables for all tasks and exit ---
    if INIT_ONLY:
        for (tensor_type, mask_code, period_key, time_res), meta in task_meta.items():
            n_locations = meta["n_locations"]
            start_str, end_str = meta["start_str"], meta["end_str"]
            time_days = make_time_axis_days_since_1901(time_res, start_str, end_str)
            loc_key  = LOC_FOR_MASK[mask_code]
            set_name = resolve_set_name(tensor_type, period_key)
            store = out_store_path(OUT_ROOT, set_name, loc_key, period_key, time_res)

            ensure_training_skeleton(store, time_days=time_days,
                                     loc_idx=masked_idx_by_code[mask_code][:n_locations],
                                     overwrite=False)
            root = zarr.open_group(str(store), mode="a")
            for var_name in variables_for_time_res(time_res):
                ensure_variable_in_training_store(
                    store, var_name,
                    n_time=len(time_days), n_location=n_locations,
                    location_chunk=FINAL_LOCATION_CHUNK,
                    overwrite=False, compressor=DEFAULT_COMP,
                )
            (Path(store) / ".done").mkdir(exist_ok=True)
            logger.info("[INIT_ONLY] ensured skeleton for %s", store)
        logger.info("[INIT_ONLY] done.")
        return

    # --- Shard the FLAT worklist across SLURM array ---
    units_shard = slurm_shard(work_units)
    logger.info("This task will process %d work unit(s).", len(units_shard))

        # --- Process our units ---
    # To reduce re-opens, cache open roots per (set_name, loc_key, period_key, time_res)
    root_cache: dict[Tuple[str, str, str, str], zarr.hierarchy.Group] = {}

    for tensor_type, mask_code, period_key, time_res, loc_start, loc_end in units_shard:
        key = (tensor_type, mask_code, period_key, time_res)
        meta = task_meta[key]
        n_locations = meta["n_locations"]
        lat_idx_all, lon_idx_all = meta["lat_idx_all"], meta["lon_idx_all"]
        start_str, end_str = meta["start_str"], meta["end_str"]
        n_time = meta["n_time"]

        if time_res == "daily" and "whole" in period_key and n_time != DAILY_DAYS_TOTAL:
            logger.warning("daily T=%d, expected %d for whole-period", n_time, DAILY_DAYS_TOTAL)

        # --- new: resolve set_name + loc_key ---
        loc_key  = LOC_FOR_MASK[mask_code]
        set_name = resolve_set_name(tensor_type, period_key)

        store = out_store_path(OUT_ROOT, set_name, loc_key, period_key, time_res)

        # Ensure skeleton/arrays exist (safe if already present)
        ensure_training_skeleton(
            store,
            time_days=make_time_axis_days_since_1901(time_res, start_str, end_str),
            loc_idx=masked_idx_by_code[mask_code][:n_locations],
            overwrite=False,
        )

        cache_key = (set_name, loc_key, period_key, time_res)
        if cache_key not in root_cache:
            root_cache[cache_key] = zarr.open_group(str(store), mode="a")
        root = root_cache[cache_key]

        vars_this_res = variables_for_time_res(time_res)
        for var_name in vars_this_res:
            ensure_variable_in_training_store(
                store, var_name,
                n_time=n_time, n_location=n_locations,
                location_chunk=FINAL_LOCATION_CHUNK,
                overwrite=False, compressor=DEFAULT_COMP,
            )

        done_dir = Path(store) / ".done"
        done_dir.mkdir(exist_ok=True)

        # Compute chunk window
        cs, ce = chunk_window_1b(loc_start, loc_end)
        logger.info("START tile [%d–%d] %s %s (loc %d:%d)",
                    cs, ce, period_key, time_res, loc_start, loc_end)

        # Gather location slice
        local = slice(loc_start, loc_end)
        iy = lat_idx_all[local]
        ix = lon_idx_all[local]

        wrote_count = 0
        with ThreadPoolExecutor(max_workers=VAR_WORKERS) as ex:
            futs = []
            for var_name in vars_this_res:
                arr = root[var_name]
                futs.append(ex.submit(
                    process_var_tile, var_name, arr, store, done_dir,
                    n_time, loc_start, loc_end, iy, ix,
                    start_str, end_str, time_res, args.daily_files_mode
                ))
            for f in as_completed(futs):
                if f.result():
                    wrote_count += 1
                    
        if wrote_count == 0:
            logger.info("[SKIP] Already complete: %s %s tile [%d–%d]",
                        period_key, time_res, cs, ce)
        else:
            logger.info("[OK] Wrote: %s %s tile [%d–%d]", period_key, time_res, cs, ce)
            
        # --- Finalization step only when requested ---
        if FINALIZE:
            # consolidate and (optional) validate for each store touched
            touched = {(sn, lk, pk, tr) for (sn, lk, pk, tr) in root_cache.keys()}
            for set_name, loc_key, period_key, time_res in touched:
                store = out_store_path(OUT_ROOT, set_name, loc_key, period_key, time_res)
                try:
                    zarr.consolidate_metadata(str(store))
                    logger.info("[FINALIZE] consolidated %s", store)
                except Exception as e:
                    logger.warning("[FINALIZE] consolidate_metadata failed for %s: %s", store, e)
                if args.validate:
                    validate_store_no_nans(store, time_res)

    logger.info("[DONE] shard complete.")

if __name__ == "__main__":
    main()