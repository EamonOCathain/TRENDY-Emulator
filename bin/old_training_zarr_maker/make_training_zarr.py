from pathlib import Path
import numpy as np
import xarray as xr
import sys, os 
import zarr
from numcodecs import Blosc

# Some Paths
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

# --------------------- Performance knobs ---------------------
# while writing daily train locations, use tiny location chunks to reduce memory pressure
CHUNKS_WHILE_WRITING = {
    "annual": 70_000,  
    "monthly": 7_000,
    "daily": 7,        
}
# the final chunk size you want the training Zarr to have for location
FINAL_LOCATION_CHUNK = 70

# Stop buffering
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from src.utils.zarr_tools import (
    make_time_axis_days_since_1901,
    build_indices_from_mask,
    make_tensor_skeleton,
    copy_variables_from_source,
    all_vars_complete, 
    rechunk_location_store
)
from src.utils.tools import slurm_shard  
from src.paths.paths import (masks_dir, zarr_dir)
from src.dataset.variables import var_names

# --------------------- Overwrite / resume flags ---------------------
OVERWRITE_SKELETON = False  
OVERWRITE_DATA     = False

# --------------------- Paths & configuration ---------------------
MASK_PATH = masks_dir / "tvt_mask.nc"
INFER_ROOT = zarr_dir / "inference"    
OUT_ROOT   = zarr_dir / "training"

SCENARIOS = ("S0", "S1", "S2", "S3")
TIME_RESES = ("annual", "monthly", "daily")
VARS_BY_RES = {
    "annual":  var_names["annual"],
    "monthly": var_names["monthly"],
    "daily":   var_names["daily"],
}
LAT_ALL = np.arange(-89.75, 90.0, 0.5, dtype="float32")
LON_ALL = np.arange(0.0, 360.0, 0.5, dtype="float32")

PERIODS = {
    "train_period":      ("1928-01-01", "2013-12-31"),
    "val_period_early":  ("1919-01-01", "1927-12-31"),
    "val_period_late":   ("2014-01-01", "2017-12-31"),
    "test_period_early": ("1901-01-01", "1918-12-31"),
    "test_period_late":  ("2018-01-01", "2023-12-31"),
    "whole_period":      ("1901-01-01", "2023-12-31"),
}

# (dir_name, mask_code, periods_to_build)
SET_SPECS = [
    ("val",   1, ["whole_period"]),
    ("test",  2, ["whole_period"]),
    ("train", 0, ["train_period", "val_period_early", "val_period_late",
                  "test_period_early", "test_period_late"]),
]
        
# --------------------- Main ---------------------

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Precompute full-span axes (handy if you need checks; not strictly required)
    full_time = {
        "daily":   make_time_axis_days_since_1901("daily",   "1901-01-01", "2023-12-31"),
        "monthly": make_time_axis_days_since_1901("monthly", "1901-01-01", "2023-12-31"),
        "annual":  make_time_axis_days_since_1901("annual",  "1901-01-01", "2023-12-31"),
    }

    # -------- 1) Build flat list of the 21 tasks --------
    tasks = []
    for set_name, mask_code, period_keys in SET_SPECS:
        for period_key in period_keys:
            for time_res in TIME_RESES:
                tasks.append({
                    "set_name": set_name,
                    "mask_code": mask_code,
                    "period_key": period_key,
                    "time_res": time_res,
                })
    # tasks length = 21

    # -------- 2) Shard across SLURM array --------
    tasks = slurm_shard(tasks) 

    # -------- 3) Cache shuffled indices per set --------
    loc_idx_cache = {}

    # -------- 4) Process only my shard --------
    for t in tasks:
        set_name   = t["set_name"]       
        mask_code  = t["mask_code"]
        period_key = t["period_key"]
        time_res   = t["time_res"]

        start_str, end_str = PERIODS[period_key]

        # Get (or build) shuffled location index for this set once
        if set_name not in loc_idx_cache:
            loc_idx_cache[set_name] = build_indices_from_mask(
                MASK_PATH, code=mask_code, shuffle=True, seed=42
            )
        loc_idx = loc_idx_cache[set_name]

        # Period-specific time axis for skeleton
        time_days = make_time_axis_days_since_1901(time_res, start_str, end_str)

        # Output path: <OUT_ROOT>/<train|val|test>/<train|val|test>_location_<period>_<time_res>.zarr
        file_stem = f"{set_name}_location_{period_key}_{time_res}"
        out_store = OUT_ROOT / set_name / f"{file_stem}.zarr"

        loc_chunk_for_skeleton = (CHUNKS_WHILE_WRITING["daily"] if (time_res == "daily") else 70)
        
        make_tensor_skeleton(
            out_store,
            time_days=time_days,
            lat_all=LAT_ALL,
            lon_all=LON_ALL,
            location_index=loc_idx,
            scenario_labels=SCENARIOS,
            chunks=(-1, 1, loc_chunk_for_skeleton),
            overwrite=OVERWRITE_SKELETON,
        )

        # Copy all variables for each scenario
        for scen_idx, scen_label in enumerate(SCENARIOS):
            src_store = INFER_ROOT / f"{scen_label}/{time_res}.zarr"
            if not src_store.exists():
                print(f"[WARN] Missing source: {src_store}", file=sys.stderr, flush=True)
                continue

            with xr.open_zarr(src_store, consolidated=True, decode_times=False) as ds_src_time:
                src_time_days = np.asarray(ds_src_time["time"].values, dtype="int64")

            if not np.all(np.diff(src_time_days) > 0):
                raise ValueError(f"{src_store} time axis is not strictly increasing.")

            i0 = int(np.searchsorted(src_time_days, int(time_days[0]),  side="left"))
            i1 = int(np.searchsorted(src_time_days, int(time_days[-1]), side="right"))
            src_time_slice = slice(i0, i1)

            vars_keep = VARS_BY_RES[time_res]
            
            copy_variables_from_source(
                src_store=src_store,
                dst_store=out_store,
                vars_keep=vars_keep,
                location_index=loc_idx,
                time_slice_src=src_time_slice,
                scenario_index=scen_idx,
                location_block=CHUNKS_WHILE_WRITING[time_res],
                overwrite_data=OVERWRITE_DATA,
                verbose=True,
            )
            
        if time_res == "daily": 

            if all_vars_complete(out_store, VARS_BY_RES["daily"]):
                rechunk_location_store(out_store, target_loc_chunk=FINAL_LOCATION_CHUNK, verbose=True)

        print(f"[OK] Built: {out_store}", file=sys.stderr, flush=True)
    
    print("Finished shard", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()