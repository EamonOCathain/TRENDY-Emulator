from numcodecs import Blosc
import sys
from pathlib import Path
import numpy as np
import zarr
from cftime import DatetimeNoLeap
import argparse

"""WARNING: Running this script again is destructive."""

OVERWRITE_DATA = True
OVERWRITE_SKELETON = True
LAT_CHUNK = 30
LON_CHUNK = 30
time_chunking_strat = "annual"

# Some Paths
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.dataset.variables import var_names, climate_vars, land_use_vars
from src.utils.tools import slurm_shard

from src.paths.paths import (
    masks_dir,
    dataset_dir,
    make_zarr_dir,
    zarr_dir,
    src_dir,
    data_dir,
    preindustrial_dir,
    historical_dir
)

from src.utils.make_inference_zarrs import (
    netcdf_to_zarr_var,
    make_zarr_skeleton,
    annual_netcdf_to_zarr,
)

# Parse time-res argument (optional)
parser = argparse.ArgumentParser(description="Write inference Zarr for one or all time resolutions.")
parser.add_argument(
    "--time-res",
    choices=["annual", "monthly", "daily"],
    help="If provided, process only this time resolution; otherwise process all."
)
args = parser.parse_args()

# Paths
out_dir = zarr_dir / "inference_30x30"
out_dir.mkdir(parents=True, exist_ok=True)

# Use requested time_res or all of them
time_resolutions = [args.time_res] if args.time_res else ["annual", "monthly", "daily"]

scenarios = ['S0', 'S1', 'S2', 'S3']

# Make skeletons
preprocessed_dir = data_dir / "preprocessed"

# Set up slurm
all_pairs = []
for time_res in time_resolutions:
    for scenario in scenarios:
        all_pairs.append((scenario, time_res))
to_process = slurm_shard(all_pairs)
print(len(all_pairs))

S0_vars = climate_vars + land_use_vars + ['co2']
S1_vars = climate_vars + land_use_vars
S2_vars = land_use_vars

years = np.arange(1901, 2024, 1)

# Add to Zarr
for pair in to_process:
    scenario = pair[0]
    time_res = pair[1]
    
    if time_chunking_strat == "annual":
        if time_res == "annual":
            TIME_CHUNK = 1
        if time_res == "monthly":
            TIME_CHUNK = 12
        if time_res == "daily":
            TIME_CHUNK = 365
    else:
        TIME_CHUNK = -1
        
    # Make the zarr skeleton
    out_path = out_dir / f"{scenario}/{time_res}.zarr"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not out_path.exists() or OVERWRITE_SKELETON:
        make_zarr_skeleton(
            out_path=out_path,
            time_res=time_res,
            start="1901-01-01",
            end="2023-12-31",
            overwrite=True
        )
        print("made skeleton")
    else:
        print("skipping skeleton creation")

    # Write the variables to it
    out_zarr = zarr_dir / f"inference_30x30/{scenario}/{time_res}.zarr"
    for var in var_names[f"{time_res}"]:
        if var in var_names['forcing']:
            # Decide if historical or preindustrial
            if (scenario == 'S0' and var in S0_vars) or \
               (scenario == 'S1' and var in S1_vars) or \
               (scenario == 'S2' and var in S2_vars):
                scenario_dir = preindustrial_dir
            else:
                scenario_dir = historical_dir

            if time_res in ['monthly', 'annual']:
                in_path = scenario_dir / f"full_time/{var}.nc"
                netcdf_to_zarr_var(
                    nc_path=in_path,
                    zarr_store=out_zarr,
                    var_name=var,
                    overwrite=OVERWRITE_DATA,
                    time_chunk = TIME_CHUNK,
                    lat_chunk=LAT_CHUNK,
                    lon_chunk=LON_CHUNK
                )
            else:
                year_files = [scenario_dir / f"annual_files/{var}/{var}_{year}.nc" for year in years]
                year_files.sort()
                annual_netcdf_to_zarr(
                    year_files=year_files,
                    store=out_zarr,
                    var_name=var,
                    time_chunk = TIME_CHUNK,
                    lat_chunks=LAT_CHUNK,
                    lon_chunks=LON_CHUNK
                )

        elif var in var_names['outputs']:
            in_path = preprocessed_dir / f"model_outputs/ENSMEAN_{scenario}_{var}.nc"
            netcdf_to_zarr_var(
                nc_path=in_path,
                zarr_store=out_zarr,
                var_name=var,
                overwrite=OVERWRITE_DATA,
                time_chunk = TIME_CHUNK,
                lat_chunk=LAT_CHUNK,
                lon_chunk=LON_CHUNK
            )
        else:
            print(f"{var} doesn't match any variable list")

    # consolidate once per (scenario, time_res)
    try:
        zarr.consolidate_metadata(str(out_zarr))
        print(f"[INFO] Consolidated metadata for {out_zarr}")
    except Exception as e:
        print(f"[WARN] consolidate_metadata failed for {out_zarr}: {e}")

print("Script finished")