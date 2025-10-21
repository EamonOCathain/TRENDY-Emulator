#!/usr/bin/env python3
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
from src.utils.preprocessing import _open_ds, regrid_file, nccopy_chunk, print_chunk_sizes, floor_time, drop_time_bnds, run_function, trim_time_xarray, extract_year_from_filename, run_function_without_tmp, rename_to_var, select_var, nccopy_chunk_jsbach

# --------------------------- Settings ---------------------------
OVERWRITE = False
CLEVEL = 4
LAT_CHUNK = 5     # for nccopy (spatial)
LON_CHUNK = 5

# Unbuffered stdout/stderr (if supported)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

raw_dir = Path("/Net/Groups/BGI/data/DataStructureMDI/DATA/Incoming/trendy/gcb2024/LAND/INPUT/LUH2")
raw_files = [raw_dir / "states.nc", raw_dir / "management.nc"]

base_dir = project_root / "preprocessing/luh2"
selected_vars_dir = base_dir / "data/1.selected_vars"
trimmed_dir = base_dir / "data/2.trimmed"
regrid_dir = base_dir / "data/3.regridded"
dropped_dir = base_dir / "data/4.drop_vars"
chunked_dir = base_dir / "data/5.chunked"
final_dir = project_root / "data/preprocessed/historical/full_time"

dirs = [selected_vars_dir, trimmed_dir, regrid_dir, dropped_dir, chunked_dir, final_dir]
for dir in dirs:
    dir.mkdir(exist_ok=True, parents=True)

# --------------------------- Helpers ---------------------------
def unique_path(base: Path) -> Path:
    """Return a unique path if base exists, appending _1, _2, ..."""
    if not base.exists():
        return base
    i = 1
    while True:
        candidate = base.with_name(f"{base.stem}_{i}{base.suffix}")
        if not candidate.exists():
            return candidate
        i += 1

# =========================== Find Files and Set up Slurm ===========================

# Build (file, var) pairs
list_pairs = []
for f in raw_files:
    ds = xr.open_dataset(f, decode_times=False)
    for v in ds.data_vars:
        if v not in ['time', 'lat', 'lon', 'time_bnds', 'bnds', 'latitude', 'longitude']:
            list_pairs.append((f, v))
    ds.close()
print(f"[INFO] Found {len(list_pairs)} (file, var) pairs.")

# Optional SLURM array sharding
task_id_str = os.getenv("SLURM_ARRAY_TASK_ID")
if task_id_str is None:
    files_to_process = list_pairs
    print("[INFO] Running linearly over all pairs.")
else:
    idx = int(task_id_str)
    if not (0 <= idx < len(list_pairs)):
        raise IndexError(f"SLURM_ARRAY_TASK_ID={idx} out of range for {len(list_pairs)} pairs")
    files_to_process = [list_pairs[idx]]
    print(f"[INFO] Running SLURM task index {idx}: {files_to_process[0]}")

# =========================== PIPELINE ===========================
selected_var_files = []
for tuple in files_to_process:
    file = tuple[0]
    var = tuple[1]
    out_path = selected_vars_dir / f"{var}.nc"
    select_var(file, out_path, var)
    selected_var_files.append(out_path)
    
trimmed_files = run_function(selected_var_files, trimmed_dir, trim_time_xarray, arg1=-124, arg2=-1, arg3 = "annual")
regrid_files = run_function(trimmed_files, regrid_dir, regrid_file)
dropped_files = run_function(regrid_files, dropped_dir, drop_time_bnds)
chunked_files = run_function(dropped_files, chunked_dir, nccopy_chunk_jsbach)
final_files = run_function_without_tmp(chunked_files, final_dir, rename_to_var)

plot_dir = base_dir / "Validation_plots"
        
for path in chunked_files:
    first_timestep(path, plot_dir / "first_timestep", title=path.stem)
    finite_mask(path, plot_dir / "finite_mask", title=path.stem)

print("Script finished successfully", flush=True)