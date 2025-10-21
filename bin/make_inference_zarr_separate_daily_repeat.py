#!/usr/bin/env python3
"""
Daily per-variable writer (per-var Zarr stores) with repeat rules:

- Writes one Zarr store per variable at:
    {zarr_dir}/inference_seperate_repeat/{SCENARIO}/daily/{VAR}.zarr

- For forcings:
    * potential_radiation (exact name): write 1901 only, then repeat that year to the end (S0 & S3).
    * Other vars:
        - S0: write 1901..1920, then repeat that 20-year block to the end.
        - S3: write full 1901..2023.

- Consolidates each per-var store at the end.
- Shards tasks across a SLURM array with slurm_shard((scenario, var) pairs).
"""

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

# Project paths / imports
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
    repeat_first_year_to_end,        
    repeat_first_n_years_to_end,     
)

# --- Constants ---
SCENARIOS = ["S0", "S3"]
TIME_RES = "daily"
YEARS = np.arange(1901, 2024, 1)
POT_RAD_NAME = "potential_radiation" 

# Output root for per-variable stores
out_dir = zarr_dir / "inference_seperate_repeat"
out_dir.mkdir(parents=True, exist_ok=True)

# Build tasks as (scenario, var) over daily variables
daily_vars = list(var_names[TIME_RES])
tasks = [(scenario, var) for scenario in SCENARIOS for var in daily_vars]
tasks = slurm_shard(tasks)
print(f"[INFO] Built {len(SCENARIOS) * len(daily_vars)} total tasks; this shard has {len(tasks)} tasks")

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

        # --- Special case: potential_radiation = write 1901 only, then repeat to end (S0 & S3)
        if var == POT_RAD_NAME:
            y0_file = src_root / f"annual_files/{var}/{var}_1901.nc"
            if not y0_file.exists():
                raise FileNotFoundError(f"Missing source year file: {y0_file}")

            annual_netcdf_to_zarr(
                year_files=[y0_file],
                store=var_store,
                var_name=var,
                lat_chunks=LAT_CHUNK,
                lon_chunks=LON_CHUNK,
                overwrite=OVERWRITE_DATA,
            )
            repeat_first_year_to_end(var_store)  # in-place fill to 2023

        else:
            # --- Regular behavior
            if scenario == "S0":
                first_block = [src_root / f"annual_files/{var}/{var}_{y}.nc" for y in range(1901, 1921)]
                missing = [p for p in first_block if not p.exists()]
                if missing:
                    raise FileNotFoundError(f"Missing first-20y files (sample): {missing[:3]} ...")

                annual_netcdf_to_zarr(
                    year_files=sorted(first_block),
                    store=var_store,
                    var_name=var,
                    lat_chunks=LAT_CHUNK,
                    lon_chunks=LON_CHUNK,
                    overwrite=OVERWRITE_DATA,
                )
                repeat_first_n_years_to_end(var_store, n_years=20)  # in-place fill to 2023

            else:  # S3: full IO 1901..2023
                year_files = [src_root / f"annual_files/{var}/{var}_{y}.nc" for y in YEARS]
                missing = [p for p in year_files if not p.exists()]
                if missing:
                    raise FileNotFoundError(f"Missing daily source files (sample): {missing[:3]} ...")

                annual_netcdf_to_zarr(
                    year_files=sorted(year_files),
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
            print(f"[SKIP] outputs daily not expected for {var}; found {in_path}, skipping by policy.")
        else:
            print(f"[SKIP] outputs daily not supported or missing for {var}")
    else:
        print(f"[WARN] {var} not in forcing/outputs lists; skipping")
        continue

    # Consolidate per-var store (metadata only; data unchanged)
    try:
        zarr.consolidate_metadata(str(var_store))
        print(f"[INFO] Consolidated metadata: {var_store}")
    except Exception as e:
        print(f"[WARN] consolidate_metadata failed for {var_store}: {e}")

print("[DONE] Per-variable daily stores written. "
      "potential_radiation uses 1y repeat; S0 uses 20y repeat for others; S3 full IO.")