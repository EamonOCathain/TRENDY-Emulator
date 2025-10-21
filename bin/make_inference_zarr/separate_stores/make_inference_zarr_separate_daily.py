#!/usr/bin/env python3
from numcodecs import Blosc
import sys
from pathlib import Path
import numpy as np
import zarr

# --- Config ---
OVERWRITE_DATA = True
OVERWRITE_SKELETON = True
LAT_CHUNK = 6
LON_CHUNK = 12

# Project paths
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.dataset.variables import var_names, climate_vars, land_use_vars
from src.utils.tools import slurm_shard
from src.paths.paths import (
    zarr_dir,
    data_dir,
    preindustrial_dir,
    historical_dir,
)
from src.utils.zarr_tools import (
    netcdf_to_zarr_var,
    make_zarr_skeleton,
    annual_netcdf_to_zarr,
)

# --- Constants ---
SCENARIOS = ["S1","S2"]
TIME_RES = "daily"
YEARS = np.arange(1901, 2024, 1)

# Output root for per-variable stores
out_dir = zarr_dir / "inference_seperate"
out_dir.mkdir(parents=True, exist_ok=True)

# Build tasks as (scenario, var) over daily variables
daily_vars = list(var_names[TIME_RES])
tasks = [(scenario, var) for scenario in SCENARIOS for var in daily_vars]
tasks = slurm_shard(tasks)
print(f"[INFO] Built {len(SCENARIOS)*len(daily_vars)} total tasks; this shard has {len(tasks)} tasks")

# Sets for routing forcings
S0_vars = set(climate_vars + land_use_vars + ["co2"])
S1_vars = set(climate_vars + land_use_vars)
S2_vars = set(land_use_vars)

preproc_outputs_dir = data_dir / "preprocessed" / "model_outputs"

for scenario, var in tasks:
    # Per-variable store path
    var_store = out_dir / f"{scenario}/{TIME_RES}/{var}.zarr"
    var_store.parent.mkdir(parents=True, exist_ok=True)

    # Make coords-only skeleton for this var store
    make_zarr_skeleton(
        out_path=var_store,
        time_res=TIME_RES,
        start="1901-01-01",
        end="2023-12-31",
        overwrite=OVERWRITE_SKELETON,
        lat_chunk=LAT_CHUNK,
        lon_chunk=LON_CHUNK,
    )

    # Decide source root for forcings vs outputs
    if var in var_names["forcing"]:
        if (scenario == "S0" and var in S0_vars) or \
           (scenario == "S1" and var in S1_vars) or \
           (scenario == "S2" and var in S2_vars):
            src_root = preindustrial_dir
        else:
            src_root = historical_dir

        # Daily = list of per-year files, write by region
        year_files = [src_root / f"annual_files/{var}/{var}_{y}.nc" for y in YEARS]
        year_files.sort()
        annual_netcdf_to_zarr(
            year_files=year_files,
            store=var_store,
            var_name=var,
            lat_chunks=LAT_CHUNK,
            lon_chunks=LON_CHUNK,
            overwrite=OVERWRITE_DATA,
        )

    elif var in var_names["outputs"]:
        # Most outputs are monthly/annual; daily often not present.
        in_path = preproc_outputs_dir / f"ENSMEAN_{scenario}_{var}.nc"
        if in_path.exists():
            # If you do have daily outputs, uncomment below (but verify dims!)
            # netcdf_to_zarr_var(
            #     nc_path=in_path,
            #     zarr_store=var_store,
            #     var_name=var,
            #     overwrite=OVERWRITE_DATA,
            #     lat_chunk=LAT_CHUNK,
            #     lon_chunk=LON_CHUNK,
            # )
            print(f"[SKIP] outputs daily not expected for {var}; found {in_path}, skipping by policy.")
        else:
            print(f"[SKIP] outputs daily not supported or missing for {var}")
    else:
        print(f"[WARN] {var} not in forcing/outputs lists; skipping")
        continue

    # Consolidate per-var store
    try:
        zarr.consolidate_metadata(str(var_store))
        print(f"[INFO] Consolidated metadata: {var_store}")
    except Exception as e:
        print(f"[WARN] consolidate_metadata failed for {var_store}: {e}")

print("[DONE] Per-variable daily stores written. Run your merge step per (scenario, daily) when ready.")