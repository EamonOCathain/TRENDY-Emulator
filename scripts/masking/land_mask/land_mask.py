import subprocess
from pathlib import Path
import pandas as pd
import os
import runpy
import sys
import importlib.util
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import shutil

# Some Paths
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))


from src.utils.visualisation import finite_mask, first_timestep, plot_mean_seasonal_cycle, plot_timeseries
from src.utils.tools import sanity_check, slurm_shard, finite_mask, threshold_mask, combine_masks
from src.utils.preprocessing import time_avg_cdo, standardise_vars, regrid_file

# Import Paths
from src.paths.paths import (
    scripts_dir, 
    preprocessing_dir,
    raw_data_dir,
    historical_dir,
    data_dir
)

# Paths 
current_dir = scripts_dir / "masking/land_mask"
local_data_dir = current_dir / "data"
land_frac_dir = local_data_dir / "1.land_frac"
std_dir = local_data_dir / "2.std_vars"
time_avg_dir = local_data_dir / "3.time_avg"
regrid_dir = local_data_dir / "4.regrid"
thresholds_dir = local_data_dir / "5.threshold"
final_dir = data_dir / "masks"
plot_dir = current_dir / "val_plots"

raw_output_dir = raw_data_dir / "OUTPUT"

# mkdirs
dirs = [data_dir, std_dir, time_avg_dir, regrid_dir, thresholds_dir, final_dir, plot_dir, land_frac_dir]
for d in dirs:
    d.mkdir(exist_ok=True, parents=True)

scenarios = ["S0", "S1", "S2", "S3"]
OVERWRITE = True

# === Find input files ===
possible_names = ['land_mask', 'oceanCoverFrac', 'land_fraction']

models_with_ocean_files = ['CLM5.0', 'ORCHIDEE']
models_with_land_files = {'CLASSIC':'land_fraction', 'ELM':'landmask'}

all_files = []
raw_ocean_files = {}
for model in models_with_ocean_files:
    for scenario in scenarios:
        path = raw_output_dir / model / scenario / f"{model}_{scenario}_oceanCoverFrac.nc"
        if model == 'CLM5.0':
            path = raw_output_dir / model / scenario / f"CLM6.0_{scenario}_oceanCoverFrac.nc"
        if path.exists():
            raw_ocean_files[f"{model}_{scenario}"] = path 
            
raw_land_files = {}
for model, filename in models_with_land_files.items():
    for scenario in scenarios:
        path = raw_output_dir / model / scenario / f"{model}_{scenario}_{filename}.nc"
        if model == 'ELM':
            path = raw_output_dir / model / scenario / f"E3SM_{scenario}_{filename}.nc"
        if path.exists():
            raw_land_files[f"{model}_{scenario}"] = path   
            
# ======== Functions ========
def ocean_to_land_fraction(in_path: str | Path, out_path: str | Path, overwrite: bool=True) -> Path | None:
    """
    Convert oceanCoverFrac (0..1) to land_fraction = 1 - oceanCoverFrac and write to out_path.
    Preserves attrs and writes netCDF4-safe encodings.
    """
    in_path, out_path = Path(in_path), Path(out_path)
    if out_path.exists() and not overwrite:
        print(f"[SKIP] exists: {out_path}")
        return None

    ds = xr.open_dataset(in_path)
    # pick the first data var if name differs
    varname = next(iter(ds.data_vars))
    da = ds[varname]

    land = (1.0 - da).clip(0.0, 1.0)
    land.name = "land_fraction"
    out = land.to_dataset()

    # minimal safe encoding
    enc = {}
    for name, v in out.variables.items():
        is_coord = name in out.coords
        e = dict(getattr(v, "encoding", {}))
        for k in list(e.keys()):
            if k not in {"shuffle","contiguous","endian","dtype","quantize_mode",
                         "szip_pixels_per_block","complevel","chunksizes","compression",
                         "zlib","_FillValue","szip_coding","blosc_shuffle",
                         "fletcher32","significant_digits","least_significant_digit"}:
                e.pop(k, None)
        if is_coord:
            for k in ("_FillValue","chunksizes","compression","zlib","complevel","shuffle","fletcher32"):
                e.pop(k, None)
        enc[name] = e

    out.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=enc)
    ds.close()
    print(f"[OK] wrote land_fraction to {out_path}")
    return out_path

# === STEP 1: Multiply Ocean by -1 ===
for path in raw_ocean_files.values():
    out_path = land_frac_dir / path.name
    ocean_to_land_fraction(path, out_path, overwrite=OVERWRITE)
    all_files.append(out_path)

# copy land mask/fraction files (already land) to same dir
for path in raw_land_files.values():
    out_path = land_frac_dir / path.name
    if out_path.exists() and not OVERWRITE:
        print(f"[SKIP] exists: {out_path}")
    else:
        shutil.copy2(path, out_path)
    all_files.append(out_path)                                  
                                              
# === STEP 1: Standardise Vars ===
std_vars_files = []
for path in all_files:
    out_path = std_dir / path.name
    std_vars_files.append(out_path)
    standardise_vars(path, out_path, overwrite=OVERWRITE)  

# === STEP 2: Time Average ===
time_avg_files = []
for path in std_vars_files:
    out_path = time_avg_dir / path.name
    time_avg_files.append(out_path)
    time_avg_cdo(path, out_path, overwrite = OVERWRITE)

# === STEP 3: Regrid ===
regrid_files = []
for path in time_avg_files: 
    out_path = regrid_dir / path.name
    regrid_files.append(out_path)
    regrid_file(path, out_path, overwrite = OVERWRITE)

# === STEP 4: Threshold Mask - OceanCoverFrac = 0.9 ===
threshold_mask_files = []
for path in regrid_files:
    out_path = thresholds_dir / f"{path.stem}.nc"  
    threshold_mask_files.append(out_path)
    threshold_mask(path, out_path, threshold=0.9, overwrite = OVERWRITE)

# Optional: include external nan mask only if present
nan_mask = final_dir / "training_nan_mask.nc"
if nan_mask.exists():
    threshold_mask_files.append(nan_mask)
    final_dir.mkdir(parents=True, exist_ok=True)
    out_path = final_dir / "land_mask.nc"
    combine_masks(threshold_mask_files, out_path, new_var_name="land_mask", overwrite=True)
else:
    print(f"[WARN] nan mask not found: {nan_mask}")

# === STEP 7: Visualisation ===
# Make the dirs
(plot_dir / "raw").mkdir(parents=True, exist_ok=True)
(plot_dir / "regrid").mkdir(parents=True, exist_ok=True)
(plot_dir / "thresholds").mkdir(parents=True, exist_ok=True)
(plot_dir / "land_mask").mkdir(parents=True, exist_ok=True)

# Raw
'''for file in time_avg_files:
    first_timestep(file, output_path=plot_dir / f"raw/{file.stem}.png", title=file.name, overwrite=OVERWRITE)'''
    
# regrid
for file in regrid_files:
    first_timestep(file, output_path=plot_dir / f"regrid/{file.stem}.png", title=file.name, overwrite=OVERWRITE)

#thresholds        
for file in threshold_mask_files:
    first_timestep(file, output_path=plot_dir / f"thresholds/{file.stem}.png", title=file.name, overwrite=OVERWRITE)

# Final Land Mask
land_mask = final_dir / "land_mask.nc"
if land_mask.exists():
    first_timestep(land_mask, output_path=plot_dir / "land_mask/land_mask.png", title=land_mask.name, overwrite=OVERWRITE)
else:
    "no land mask file to plot"