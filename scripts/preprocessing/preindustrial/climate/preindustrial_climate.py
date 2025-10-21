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
""" 
Basically we need to:
1. Repeat the first 20 years of the climate, clouds and radiation by just copying and changing the name.
2. Repeat the luh first 20 years.
3. Create pre-industrial constant CO2
"""
# Stop Buffering
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Configs
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.utils.visualisation import finite_mask, first_timestep
from src.utils.preprocessing import (
    _open_ds, nccopy_chunk
)

OVERWRITE = False

# imports
climate_vars = ["pre", 
          "pres", 
          "tmp", 
          "spfh",
          "ugrd", 
          "vgrd", 
          "dlwrf", 
          "tmax", 
          "tmin",
          'cld',
          "tswrf", 
            "fd"]

# Repeat the preindustrial years for scenario 
annual_dir = project_root / "data/preprocessed/historical/annual_files"
preindustrial_dir = project_root /"data/preprocessed/preindustrial"
preindustrial_annual_dir = preindustrial_dir / "annual_files"

# Get Slurm task ID
task_id_str = os.getenv("SLURM_ARRAY_TASK_ID")

# Filter the files to process based on task ID
NUM_ARRAYS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
if task_id_str is None:
    print("[Info] Not running inside a SLURM job. Proceeding linearly.")
    vars_to_process = climate_vars
else:
    task_id = int(task_id_str)
    total = len(climate_vars)
    chunk_size = (total + NUM_ARRAYS - 1) // NUM_ARRAYS 
    start = task_id * chunk_size
    end = min(start + chunk_size, total)
    print(f"[Info] Running inside SLURM array task {task_id}: files {start}:{end} of {total}")
    vars_to_process = climate_vars[start:end]

def preindustrial_climate(in_path: Path, out_path: Path, year: int) -> None:
    """
    Copy everything from in_path but overwrite the time coordinate
    with daily values corresponding to the given year.
    """

    # Open dataset
    ds = xr.open_dataset(in_path)

    # Compute new daily time values
    start = (year - 1901) * 365
    end = start + 365
    new_time = np.arange(start, end, 1, dtype=np.int32)

    # Replace the time values (keep original attrs!)
    time_attrs = ds["time"].attrs.copy() if "time" in ds.coords else {}
    ds = ds.assign_coords(time=("time", new_time))
    ds["time"].attrs = time_attrs  # restore CF metadata

    # Build encoding dictionary (preserve original settings if available)
    enc = {}
    for var in ds.variables:
        if var in ds.encoding:
            enc[var] = ds[var].encoding.copy()

    # Save to NetCDF
    ds.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=enc)
    ds.close()
    print(f"Saved {out_path} with updated time axis for year {year}")

# Make mask to map the repeating patterns to the files
mask = np.tile(np.arange(20), 123 // 20 + 1)[:123]


final_files = []
# Loop through the variables
for var in vars_to_process:
    # Save the input dir for the variable
    input_var_dir = annual_dir / var
    # sort them (relies on continous years and all years present)
    input_annual_files = sorted(input_var_dir.glob("*.nc"))
    # Loop through indexes of mask and copy the appropriate year
    for year_idx, mask_idx in enumerate(mask):
        in_path = input_annual_files[mask_idx]
        year = 1901 + year_idx # Construct the new year 
        out_dir = preindustrial_annual_dir / var 
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{var}_{year}.nc"
        final_files.append(out_path)
        if out_path.exists() and not OVERWRITE:
            print(f"Output path exists at {out_path}, skipping processing climate file")
        else:
            preindustrial_climate(in_path, out_path, year)
            print(f"preindustrial climate file made from {in_path} to {out_path}")

plot_dir = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/preprocessing/preindustrial/climate/val_plots")
plot_dir.mkdir(parents=True, exist_ok=True)
        
for path in final_files:
    if "1901" not in path.name:
        print(f"Skipping plotting of {path}")
        continue
    else:
        if path.exists():
            first_timestep(path, plot_dir / "first_timestep", title=path.stem)
            finite_mask(path, plot_dir / "finite_mask", title=path.stem)
            
# Rechunk to lat 1 lon 1
rechunk_dir = preindustrial_dir / "annual_files_1x1"
for src in final_files:
    # keep {var}/{var}_{year}.nc
    rel = src.relative_to(preindustrial_annual_dir)  
    dst = (rechunk_dir / rel).with_suffix(".nc")
    dst.parent.mkdir(parents=True, exist_ok=True)
    nccopy_chunk(
        in_path=src,
        out_path=dst,
        clevel=4,
        lat_chunk=1,
        lon_chunk=1,
        overwrite=OVERWRITE,
    )
    print(f"[OK] 1x1 -> {dst}")
