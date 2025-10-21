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

current_dir = project_root / "scripts/masking/land_mask"

from src.utils.visualisation import finite_mask, first_timestep, plot_mean_seasonal_cycle, plot_timeseries
from src.utils.tools import sanity_check, slurm_shard, finite_mask, threshold_mask, combine_masks
from src.utils.preprocessing import time_avg_cdo, standardise_vars, regrid_file

# Import Paths
from src.paths.paths import (
    masks_dir,
)

regridded_land_mask_dir = current_dir / "data" / "4.regrid"

regridded_files = list(regridded_land_mask_dir.glob("*.nc"))

orchidee_files = [f for f in regridded_files if "ORCHIDEE" in f.name]
clm_files = [f for f in regridded_files if "CLM" in f.name]

to_avg = orchidee_files + clm_files

out_path = masks_dir / "inference_land_mask.nc"

# Fixed r720x360 grid
lat = np.arange(-89.75, 90.0, 0.5)    # 360 values: -89.75 ... 89.75
lon = np.arange(0.0, 360.0, 0.5)      # 720 values: 0.0 ... 359.5

arrays = []
for f in to_avg:
    with xr.open_dataset(f, decode_times=False) as ds:
        da = ds["land_fraction"]

        # Ignore/collapse time
        if "time" in da.dims:
            da = da.isel(time=0) if da.sizes["time"] == 1 else da.mean("time")
        da = da.squeeze(drop=True)

        # Pull raw values and clean
        arr = np.asarray(da.values, dtype=np.float32)

        # Handle fill/missing markers
        fv = da.attrs.get("_FillValue", None)
        mv = da.attrs.get("missing_value", None)
        if fv is not None:
            arr = np.where(arr == fv, np.nan, arr)
        if mv is not None:
            arr = np.where(arr == mv, np.nan, arr)

        # Replace NaNs with 0 and clip to [0,1]
        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.clip(arr, 0.0, 1.0)

        # Threshold: values > 0.9 → 1
        arr = np.where(arr > 0.9, 1.0, arr)

        # Sanity: enforce expected shape
        if arr.shape != (lat.size, lon.size):
            raise RuntimeError(f"Unexpected shape in {f.name}: {arr.shape}, expected {(lat.size, lon.size)}")

        arrays.append(arr)

if not arrays:
    raise RuntimeError("No input masks found to average.")

stack = np.stack(arrays, axis=0)  # (models, 360, 720)

# Rule: any zero -> 0, else average
any_zero = (stack == 0.0).any(axis=0)
mean_vals = stack.mean(axis=0)
out_vals = np.where(any_zero, 0.0, mean_vals).astype(np.float32)

# ---- Package & save (NetCDF + Zarr) ----
out_da = xr.DataArray(
    out_vals, dims=("lat", "lon"),
    coords={"lat": lat, "lon": lon},
    name="land_fraction",
)
out_ds = xr.Dataset({"land_fraction": out_da})

# Paths
out_nc   = masks_dir / "inference_land_mask.nc"
out_zarr = masks_dir / "inference_land_mask.zarr"

# 1) NetCDF (keep this for plotting / portability)
encoding_nc = {"land_fraction": {"zlib": True, "complevel": 4}}
out_ds.to_netcdf(out_nc, engine="netcdf4", format="NETCDF4", encoding=encoding_nc)
print(f"[DONE] Wrote NetCDF land mask to {out_nc}")

# 2) Zarr (fast to read; no netCDF backend needed)
# start fresh
if out_zarr.exists():
    shutil.rmtree(out_zarr)
# write consolidated metadata for faster opens
out_ds.to_zarr(out_zarr, mode="w", consolidated=True)
print(f"[DONE] Wrote Zarr land mask to {out_zarr}")

# ---- Quick plot (use the NetCDF you already had working) ----
if out_nc.exists():
    first_timestep(
        out_nc,
        output_path=current_dir / "val_plots/inference_land_mask/land_mask_avg.png",
        title="Inference Land Mask Average",
        overwrite=True,
        axis_label="Land Fraction",
    )
else:
    print("no land mask file to plot")