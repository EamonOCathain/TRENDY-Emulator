#!/usr/bin/env python3
"""
Per-variable writer for training Zarr stores (parallel-friendly, no rechunking).

- One process handles one (set_name, period_key, time_res, var) task (via slurm_shard()).
- Writes all 4 scenarios for that variable from inference -> training.
- Respects per-scenario progress masks; safe to resume.
- Never consolidates or rechunks here (do that in a separate script).

Usage:
  python -u write_training_var.py --time-res daily
  # or under SLURM array (auto-shards via src.utils.tools.slurm_shard)
"""

from __future__ import annotations
import os, sys
from pathlib import Path
import argparse
import numpy as np
import xarray as xr
import zarr  

# Unbuffered logs
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# --------------------- Performance knobs ---------------------
CHUNKS_WHILE_WRITING = {
    "annual":  5000,
    "monthly": 500,
    "daily":   7,
}

# --------------------- Project paths & imports ---------------------
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.utils.tools import slurm_shard
from src.paths.paths import masks_dir, zarr_dir
from src.dataset.variables import var_names
from src.utils.zarr_tools import (
    make_time_axis_days_since_1901,
    build_indices_from_mask,
    copy_variables_from_source,
    all_vars_complete,  
)

# --------------------- Config ---------------------
OVERWRITE_DATA = False  # True -> re-write this var (per-scenario masks are cleared)
SCENARIOS      = ("S0", "S1", "S2", "S3")

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

def parse_args():
    ap = argparse.ArgumentParser(description="Per-variable writer for training Zarr stores.")
    ap.add_argument("--time-res", required=True, choices=["annual", "monthly", "daily"],
                    help="Which time resolution to process.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite this variable in the destination (clears progress masks).")
    return ap.parse_args()

def main():
    args = parse_args()
    time_res_to_run = args.time_res
    overwrite_data  = bool(args.overwrite) or OVERWRITE_DATA

    # Build task list = every (set, period, *chosen* time_res, var)
    tasks = []
    for set_name, mask_code, period_keys in SET_SPECS:
        for period_key in period_keys:
            for v in var_names[time_res_to_run]:
                tasks.append((set_name, mask_code, period_key, time_res_to_run, v))
    print(len(tasks))

    # Shard by SLURM task env (or identity if not running under SLURM)
    tasks = slurm_shard(tasks)
    print(f"[INFO] tasks for this shard: {len(tasks)}", flush=True)
    for t in tasks[:3]:
        print(f"[INFO] example task: {t}", flush=True)

    # cache location indices per set
    loc_idx_cache: dict[str, np.ndarray] = {}

    for set_name, mask_code, period_key, time_res, var in tasks:
        # paths
        infer_root = zarr_dir / "inference"
        out_root   = zarr_dir / "training"
        out_store  = out_root / set_name / f"{set_name}_location_{period_key}_{time_res}.zarr"
        if not out_store.exists():
            raise FileNotFoundError(f"Destination training store missing: {out_store}")

        # shuffled location index per set
        if set_name not in loc_idx_cache:
            loc_idx_cache[set_name] = build_indices_from_mask(
                masks_dir / "tvt_mask.nc",
                code=mask_code, shuffle=True, seed=42
            )
        loc_idx = loc_idx_cache[set_name]

        # time axis for this (period, time_res)
        start_str, end_str = PERIODS[period_key]
        time_days = make_time_axis_days_since_1901(time_res, start_str, end_str)

        if overwrite_data:
            root = zarr.open_group(str(out_store), mode="a")
            key = f"complete:{var}"
            if key in root.attrs:
                del root.attrs[key]
        
        # write this var for all scenarios
        for scen_idx, scen_label in enumerate(SCENARIOS):
            src_store = infer_root / scen_label / f"{time_res}.zarr"
            if not src_store.exists():
                print(f"[WARN] Missing inference source: {src_store}", flush=True)
                continue

            with xr.open_zarr(src_store, consolidated=True, decode_times=False) as ds_src:
                src_time_days = np.asarray(ds_src["time"].values, dtype="int64")

            if not np.all(np.diff(src_time_days) > 0):
                raise ValueError(f"{src_store} time axis is not strictly increasing.")

            i0 = int(np.searchsorted(src_time_days, int(time_days[0]),  side="left"))
            i1 = int(np.searchsorted(src_time_days, int(time_days[-1]), side="right"))
            src_time_slice = slice(i0, i1)

            print(f"[START] set={set_name} period={period_key} tres={time_res} scen={scen_label} var={var}", flush=True)

            copy_variables_from_source(
                src_store=src_store,
                dst_store=out_store,
                vars_keep=[var],
                location_index=loc_idx,
                time_slice_src=src_time_slice,
                scenario_index=scen_idx,
                location_block=CHUNKS_WHILE_WRITING[time_res],
                overwrite_data=overwrite_data,
                verbose=True,
            )

        print(f"[DONE] {out_store.name}: wrote var={var}", flush=True)

if __name__ == "__main__":
    main()