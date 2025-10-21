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
import pandas as pd

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.utils.visualisation import finite_mask, first_timestep
from src.utils.preprocessing import (
    daily_avg_cdo,
    nccopy_chunk_jsbach,
    print_chunk_sizes,
    floor_time,
    drop_time_bnds,
    run_function,
    run_function_without_tmp,
    rename_to_var,
    _open_ds,
    regrid_file,
    trim_time_xarray,
    repeat_last_timestamp,
    reassign_720_360_grid,
    select_var,
    make_annual_time
)
'''
File comes without time dimension and at grid 768 x 384.
1. Regrid
2. Drop variables (time_bnds, bnds, etc)
3. Repeat to annual timesteps
4. Chunk
'''
# Settings
OVERWRITE = False
NUM_ARRAYS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))  # default to 1 if not under SLURM
clevel = 4 
LAT_CHUNK  = 20
LON_CHUNK  = 20

# Stop Buffering
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Raw Data Directories
raw_file = Path("/Net/Groups/BGI/people/awinkler/share/for_eamon/jsbach_T255TP6M_11tiles_5layers_2015_no-dynveg.nc")
var_names = ["elevation",  "orography_std_dev",  "pore_size_index",  "soil_depth", "soil_porosity"]

# Set the output directory and make it
base_dir = project_root / "preprocessing/jsbach"
data_dir = base_dir / "data"
out_dir_1 = data_dir / "1.single_var"
out_dir_2 = data_dir /"2.regrid"
out_dir_3 = data_dir / "3.repeat_timestep"
out_dir_4 = data_dir / "4.drop_vars"
out_dir_5 = data_dir / "5.chunk"

final_dir = project_root / "data/preprocessed/historical/full_time"
plot_dir = base_dir / "val_plots"

# Make dirs
out_dir_1.mkdir(exist_ok=True, parents=True)
out_dir_2.mkdir(exist_ok=True, parents=True)
out_dir_3.mkdir(exist_ok=True, parents=True)
out_dir_4.mkdir(exist_ok=True, parents=True)
plot_dir.mkdir(exist_ok=True, parents=True)

# Get Slurm task ID
task_id_str = os.getenv("SLURM_ARRAY_TASK_ID")

# Filter the files to process based on task ID
if task_id_str is None:
    print("[Info] Not running inside a SLURM job. Proceeding linearly.")
    files_to_process = var_names
else:
    task_id = int(task_id_str)
    total = len(var_names)
    chunk_size = (total + NUM_ARRAYS - 1) // NUM_ARRAYS 
    start = task_id * chunk_size
    end = min(start + chunk_size, total)
    print(f"[Info] Running inside SLURM array task {task_id}: files {start}:{end} of {total}")
    files_to_process = var_names[start:end]

# Loop through files and take daily average
out_files_1 = []
for var in files_to_process:
    out_path = out_dir_1 / f"{var}.nc"
    select_var(raw_file, out_path, var=var)
    out_files_1.append(out_path)
out_files_2 = run_function(out_files_1, out_dir_2, regrid_file, overwrite=OVERWRITE)
out_files_3 = run_function(out_files_2, out_dir_3, make_annual_time, overwrite=OVERWRITE)
out_files_4 = run_function(out_files_3, out_dir_4, drop_time_bnds, overwrite=OVERWRITE)
final_files = run_function(out_files_4, final_dir, nccopy_chunk_jsbach, overwrite=OVERWRITE, arg1 = clevel, arg2 = LAT_CHUNK, arg3=LON_CHUNK)
    
# ------------------------------------- Validation Plotting -------------------------------------
for path in final_files:
    path = Path(path)
    plot_dir_finished = plot_dir / "processed"
    first_timestep(path, plot_dir_finished / "first_timestep", title=path.stem, overwrite=OVERWRITE)
    finite_mask(path, plot_dir_finished / "finite_mask", title=path.stem, overwrite=OVERWRITE)
    print_chunk_sizes(path)
    
for path in out_files_1:
    path = Path(path)
    plot_dir_raw = plot_dir / "raw"
    first_timestep(path, plot_dir_raw / "first_timestep", title=path.stem, overwrite=OVERWRITE)
    finite_mask(path, plot_dir_raw / "finite_mask", title=path.stem, overwrite=OVERWRITE)
        
print("script finished successfully")