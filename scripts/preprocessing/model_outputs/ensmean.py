"""You need to run the models.py script linearly to create a paths_df save with all models"""

import xarray as xr
from pathlib import Path
import subprocess
import os
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import sys
import shutil
import cftime
import pandas as pd

OVERWRITE = True

# Some Paths
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.paths.paths import (
    scripts_dir, 
    preprocessing_dir,
    raw_data_dir,
    historical_dir,
    raw_outputs_dir,
    model_outputs_dir
)

# Some Imports
from src.utils.visualisation import finite_mask, first_timestep, plot_mean_seasonal_cycle, plot_timeseries
from src.utils.preprocessing import (
    cdo_ensmean,
    nccopy_chunk,
    make_mask_from_paths_df,
    xr_ensmean
)
from src.utils.tools import slurm_shard
from src.dataset.variables import models, var_names

# Lists of vars
fluxes = var_names['fluxes']
states = var_names['states']
all_vars = var_names['outputs']
carbon_states = ['cVeg', 'cLitter', 'cSoil']
scenarios = ['S0','S1','S2','S3']

# Slurm set-up
pairs = []
for var in all_vars:
    for scenario in scenarios:
        pairs.append((scenario, var))
print(f"Total number of scenario - var pairs is {len(pairs)}")
to_process = slurm_shard(pairs)

# Paths
current_dir = preprocessing_dir / "model_outputs"
ensmean_dir = current_dir / "data/5.ensmean"
final_dir = model_outputs_dir
plot_dir = current_dir / "val_plots"

ensmean_dir.mkdir(parents=True, exist_ok=True)
final_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# Paths df
paths_df = pd.read_csv(current_dir /  "paths_dfs/paths_models_mid_processing.csv")

# loop through each scenario - variable and build list of all paths
ensmean_files = [] 
for pair in to_process:
    scenario = pair[0]
    var = pair[1]
    
    out_path = ensmean_dir / f"ENSMEAN_{scenario}_{var}.nc"
    mask = make_mask_from_paths_df(paths_df, scenario=scenario, variable=var)
    sub_df = paths_df[mask]

    paths = [Path(p) for p in sub_df["path"] if pd.notna(p)]
    if not paths:
        print(f"[INFO] No inputs for {scenario} {var}; skipping ENSMEAN")
        continue

    # run once per (scenario, var) with the full list
    xr_ensmean(paths, out_path, overwrite=OVERWRITE)
    ensmean_files.append(out_path)

# Now chunk them all
chunked_files = []
for path in ensmean_files: 
    out_path = final_dir / path.name
    nccopy_chunk(path, out_path, clevel=3, lat_chunk=20, lon_chunk=20, overwrite=OVERWRITE)
    chunked_files.append(out_path)

# Now plot them
for path in chunked_files:
    if 'S3' in path.name:
        first_timestep(path, output_dir=plot_dir / "ensmean/first_timestep", title=path.stem, overwrite=OVERWRITE)
        finite_mask(path,   output_dir=plot_dir / "ensmean/finite_mask",     title=path.stem, overwrite=OVERWRITE)
        plot_timeseries(path, output_dir=plot_dir / "ensmean/timeseries",    title=path.stem, overwrite=OVERWRITE)
