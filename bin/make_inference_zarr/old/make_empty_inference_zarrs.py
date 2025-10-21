from numcodecs import Blosc
import sys
from pathlib import Path
import numpy as np
import zarr
from cftime import DatetimeNoLeap

OVERWRITE_DATA = False
OVERWRITE_SKELETON = False
LAT_CHUNK = 6
LON_CHUNK = 12

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

from src.utils.zarr_tools import (
                                  netcdf_to_zarr_var,
                                  make_zarr_skeleton,
                                  annual_netcdf_to_zarr,
                                  )

# Paths
out_dir = zarr_dir / "inference"
out_dir.mkdir(parents=True, exist_ok=True)

time_resolutions = ['annual', 'monthly', 'daily']

scenarios = ['S0', 'S1', 'S2', 'S3']

# Make skeletons
# Create list of all files to process
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

years = np.arange(1901,2024, 1)

# time chunk
if time_res == 'daily':
    target_chunks = {"time": 365, "lat": 6, "lon": 12}
if time_res == 'monthly':
    target_chunks = {"time": 12, "lat": 6, "lon": 12}
if time_res == "annual":
    target_chunks = {"time": 1, "lat": 6, "lon": 12}

# Add to Zarr
for pair in to_process:
    scenario = pair[0]
    time_res = pair[1]
    # Make the zarr skeleton 
    out_path= out_dir / f"{scenario}/{time_res}.zarr"
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
        
