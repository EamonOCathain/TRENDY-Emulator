#!/usr/bin/env python3
from pathlib import Path
import sys, os
import numpy as np
import zarr

# Unbuffered logs
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# --------------------- Performance knobs ---------------------
CHUNKS_WHILE_WRITING = {
    "annual":  70_000,
    "monthly": 7_000,
    "daily":   7,
}
FINAL_LOCATION_CHUNK = 70

# --------------------- Project paths & imports ---------------------
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.utils.zarr_tools import (
    make_time_axis_days_since_1901,
    build_indices_from_mask,
    make_tensor_skeleton,
    ensure_variable_in_store,
)
from src.paths.paths import masks_dir, zarr_dir
from src.dataset.variables import var_names
from src.utils.tools import slurm_shard  # <-- makes it arrayable

# --------------------- Config ---------------------
OVERWRITE_SKELETON = False      # only recreate skeletons if True
OVERWRITE_VAR_ARRAYS = False    # delete & recreate variable arrays if True

MASK_PATH  = masks_dir / "tvt_mask.nc"
OUT_ROOT   = zarr_dir / "training"

SCENARIOS  = ("S0", "S1", "S2", "S3")
TIME_RESES = ("annual", "monthly", "daily")
LAT_ALL    = np.arange(-89.75, 90.0, 0.5, dtype="float32")
LON_ALL    = np.arange(0.0, 360.0, 0.5, dtype="float32")

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

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # -------- Build the task list at STORE granularity --------
    # One task = (set_name, mask_code, period_key, time_res)
    all_tasks = []
    for set_name, mask_code, period_keys in SET_SPECS:
        for period_key in period_keys:
            for time_res in TIME_RESES:
                all_tasks.append((set_name, mask_code, period_key, time_res))

    # -------- Shard across SLURM array (or no-op if not under SLURM) --------
    tasks = slurm_shard(all_tasks)
    print(f"[INFO] SLURM shard -> {len(tasks)} store tasks", flush=True)

    # cache shuffled location indices per set (shared across tasks on this rank)
    loc_idx_cache = {}

    for set_name, mask_code, period_key, time_res in tasks:
        # locations for this set (shuffled + fixed seed)
        if set_name not in loc_idx_cache:
            loc_idx_cache[set_name] = build_indices_from_mask(
                MASK_PATH, code=mask_code, shuffle=True, seed=42
            )
        loc_idx = loc_idx_cache[set_name]

        start_str, end_str = PERIODS[period_key]
        time_days = make_time_axis_days_since_1901(time_res, start_str, end_str)

        file_stem = f"{set_name}_location_{period_key}_{time_res}"
        out_store = OUT_ROOT / set_name / f"{file_stem}.zarr"
        out_store.parent.mkdir(parents=True, exist_ok=True)

        # daily skeleton uses tiny location chunk (for faster writer); others can be 70 directly
        loc_chunk_for_skeleton = (
            CHUNKS_WHILE_WRITING["daily"] if time_res == "daily" else FINAL_LOCATION_CHUNK
        )

        # 1) ensure coords-only skeleton (idempotent unless OVERWRITE_SKELETON=True)
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
        print(f"[OK] skeleton: {out_store}", flush=True)

        # 2) ensure empty arrays for ALL variables of this time_res (idempotent unless OVERWRITE_VAR_ARRAYS=True)
        vars_this_res = list(var_names[time_res])

        if OVERWRITE_VAR_ARRAYS:
            root = zarr.open_group(str(out_store), mode="a")
            for v in vars_this_res:
                if v in root:
                    del root[v]
                # clear any completion/progress attrs so writers can resume cleanly
                for k in (f"complete:{v}", f"complete:{v}:scen0", f"complete:{v}:scen1",
                          f"complete:{v}:scen2", f"complete:{v}:scen3"):
                    try:
                        if k in root.attrs:
                            del root.attrs[k]
                    except Exception:
                        pass

        for v in vars_this_res:
            ensure_variable_in_store(
                store=out_store,
                var_name=v,
                dtype="float32",
                chunks=(-1, 1, loc_chunk_for_skeleton),
            )
            print(f"[OK] ensured var array '{v}' in {out_store.name} (loc_chunk={loc_chunk_for_skeleton})", flush=True)

        # Consolidate once per store (safe here—one job owns this store task)
        try:
            zarr.consolidate_metadata(str(out_store))
        except Exception:
            pass

    print("[DONE] Training skeletons + empty arrays are ready for writers.", flush=True)

if __name__ == "__main__":
    main()