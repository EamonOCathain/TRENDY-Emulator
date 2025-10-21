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
import re 

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))
from src.paths.paths import (
    scripts_dir, 
    preprocessing_dir,
    raw_data_dir
)

from src.utils.visualisation import finite_mask, first_timestep
from src.utils.preprocessing import daily_avg_cdo, nccopy_chunk, print_chunk_sizes, floor_time, drop_time_bnds, run_function, run_function_without_tmp, rename_to_var, _open_ds, extract_year_from_filename, reassign_720_360_grid, process_time_annual_file

'''
123 years * 12 variables = 1476 files

Designed to be run with with 123 slurm arrays (122 with 0 indexing) -> meaning each processes 12 files

1. Daily Average
2. Chunk
3. time floor (round down time values because daily average made them all .375)
4. drop variables (time_bnds, bnds, etc)
'''
# Settings
OVERWRITE = False
NUM_ARRAYS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))  # default to 1 if not under SLURM
clevel = 4 
LAT_CHUNK = 30    
LON_CHUNK = 30

# Stop Buffering
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

raw_data_dir = raw_data_dir / "INPUT"

# Raw Data Directories
raw_climate_dirs = [raw_data_dir / "crujra2.5", raw_data_dir / "Radiation", raw_data_dir / "clouds"]

# Set the output directory and make it
daily_avg_dir = scripts_dir / "preprocessing/climate/data/1.daily_avg"
chunked_dir = scripts_dir / "preprocessing/climate/data/2.chunked"
time_floor_dir = scripts_dir / "preprocessing/climate/data/3.time"
dropped_dir = scripts_dir / "preprocessing/climate/data/4.dropped_vars"
regrid_dir = scripts_dir / "preprocessing/climate/data/5.regrid"


final_dir = project_root / "data/preprocessed/30x30/historical/annual_files"
rechunk_dir = project_root / "data/preprocessed/1x1/historical/annual_files"
plot_dir = scripts_dir / "Validation_plots"

# Make dirs
daily_avg_dir.mkdir(exist_ok=True, parents=True)
chunked_dir.mkdir(exist_ok=True, parents=True)
time_floor_dir.mkdir(exist_ok=True, parents=True)
dropped_dir.mkdir(exist_ok=True, parents=True)
regrid_dir.mkdir(exist_ok=True, parents=True)
plot_dir.mkdir(exist_ok=True, parents=True)
rechunk_dir.mkdir(exist_ok=True, parents=True)

# Gather all files
list_files = []
for d in raw_climate_dirs:
    list_files.extend(d.glob("*.nc"))
list_files = sorted(list_files)
print(f"[INFO] Found {len(list_files)} raw climate files.")

# Get Slurm task ID
task_id_str = os.getenv("SLURM_ARRAY_TASK_ID")

# Filter the files to process based on task ID
if task_id_str is None:
    print("[Info] Not running inside a SLURM job. Proceeding linearly.")
    files_to_process = list_files
else:
    task_id = int(task_id_str)
    total = len(list_files)
    chunk_size = (total + NUM_ARRAYS - 1) // NUM_ARRAYS 
    start = task_id * chunk_size
    end = min(start + chunk_size, total)
    print(f"[Info] Running inside SLURM array task {task_id}: files {start}:{end} of {total}")
    files_to_process = list_files[start:end]

# Loop through files and take daily average
daily_avg_files = run_function(files_to_process, daily_avg_dir,  daily_avg_cdo)
chunked_files = run_function(daily_avg_files, chunked_dir, nccopy_chunk, arg1=clevel, arg2=LAT_CHUNK,arg3=LON_CHUNK)
time_floored_files = run_function(chunked_files, time_floor_dir, process_time_annual_file)
dropped_files = run_function(time_floored_files, dropped_dir, drop_time_bnds)
regrid_files = run_function(dropped_files, regrid_dir, reassign_720_360_grid)

renamed_final_files = []
for file in regrid_files:
    ds, varname = _open_ds(file)
    out_dir = final_dir / varname
    out_dir.mkdir(parents=True, exist_ok=True)
    year = extract_year_from_filename(file)
    out_path = out_dir / f"{varname}_{year}.nc"
    shutil.copy2(file, out_path)
    renamed_final_files.append(out_path)
    print(f"Moved file and renamed to final dir {out_path}")

# Rechunk to lat 1 lon 1
for src in renamed_final_files:
    # keep {var}/{var}_{year}.nc
    rel = src.relative_to(final_dir)
    dst = (rechunk_dir / rel).with_suffix(".nc")
    dst.parent.mkdir(parents=True, exist_ok=True)
    nccopy_chunk(
        in_path=src,
        out_path=dst,
        clevel=clevel,
        lat_chunk=1,
        lon_chunk=1,
        overwrite=OVERWRITE,
    )
    print(f"[OK] 1x1 -> {dst}")
    
# ------------------------------------- Validation Plotting -------------------------------------
for path in renamed_final_files:
    if "1901" not in path.name:
        print(f"Skipping plotting of {path}")
        continue
    else:
        first_timestep(path, plot_dir, title=path.stem)
        finite_mask(path, plot_dir, title=path.stem)
        print_chunk_sizes(path)
        
print("script finished successfully")