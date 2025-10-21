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

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.utils.visualisation import finite_mask, first_timestep
from src.utils.preprocessing import (
    daily_avg_cdo,
    nccopy_chunk,
    print_chunk_sizes,
    floor_time,
    drop_time_bnds,
    run_function,
    run_function_without_tmp,
    rename_to_var,
    _open_ds,
    regrid_file,
    trim_time_xarray,
)
'''
123 years * 12 variables = 1476 files

Designed to be run with with 123 slurm arrays (122 with 0 indexing) -> meaning each processes 12 files

1. Trim
2. regrid
3. chunk
4. drop variables (time_bnds, bnds, etc)
'''
# Settings
OVERWRITE = False
NUM_ARRAYS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))  # default to 1 if not under SLURM
clevel = 4 
LAT_CHUNK  = 5 
LON_CHUNK  = 5

# Stop Buffering
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Raw Data Directories
raw_dir = Path("/Net/Groups/BGI/data/DataStructureMDI/DATA/Incoming/trendy/gcb2024/LAND/INPUT")
ndep_dir = raw_dir / "ndep"
list_files = [ndep_dir / "drynhx_input4MIPs_surfaceFluxes_CMIP_hist_ssp585_NCAR-CCMI-2-0_gn_185001-209912.720x360.nc", 
                    ndep_dir / "drynoy_input4MIPs_surfaceFluxes_CMIP_hist_ssp585_NCAR-CCMI-2-0_gn_185001-209912.720x360.nc",
                    ndep_dir / "wetnhx_input4MIPs_surfaceFluxes_CMIP_hist_ssp585_NCAR-CCMI-2-0_gn_185001-209912.720x360.nc",
                    ndep_dir / "wetnoy_input4MIPs_surfaceFluxes_CMIP_hist_ssp585_NCAR-CCMI-2-0_gn_185001-209912.720x360.nc"]

# Set the output directory and make it
base_dir = project_root / "preprocessing/ndep"
data_dir = base_dir / "data"
out_dir_1 = data_dir / "1.trimmed"
out_dir_2 = data_dir /"2.regrid"
out_dir_3 = data_dir / "3.chunk"
out_dir_4 = data_dir / "4.drop_vars"
final_dir = project_root / "data/preprocessed/historical/full_time"
plot_dir = base_dir / "val_plots"

# Make dirs
out_dir_1.mkdir(exist_ok=True, parents=True)
out_dir_2.mkdir(exist_ok=True, parents=True)
out_dir_3.mkdir(exist_ok=True, parents=True)
out_dir_4.mkdir(exist_ok=True, parents=True)
plot_dir.mkdir(exist_ok=True, parents=True)
final_dir.mkdir(exist_ok=True, parents=True)

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
out_files_1 = run_function(files_to_process, out_dir_1,  trim_time_xarray, 
                           arg1= 612, arg2=2088, arg3 = "monthly")
out_files_2 = run_function(out_files_1, out_dir_2, regrid_file)
out_files_3 = run_function(out_files_2, out_dir_3, nccopy_chunk)
out_files_4 = run_function(out_files_3, out_dir_4, drop_time_bnds)

final_files = []

for file in out_files_4:
    file = Path(file)
    var = file.name[:6]
    out_path = final_dir / f"{var}.nc"
    shutil.copy(file, out_path)
    print(f"copied file to final dir with new file name {out_path}")
    final_files.append(out_path)
    
# ------------------------------------- Validation Plotting -------------------------------------
for path in final_files:
    plot_dir_finished = plot_dir / "processed"
    first_timestep(path, plot_dir_finished / "first_timestep", title=path.stem)
    finite_mask(path, plot_dir_finished / "finite_mask", title=path.stem)
    print_chunk_sizes(path)
    
for path in files_to_process:
    plot_dir_raw = plot_dir / "raw"
    first_timestep(path, plot_dir_raw / "first_timestep", title=path.stem)
    finite_mask(path, plot_dir_raw / "finite_mask", title=path.stem)
        
print("script finished successfully")


            
