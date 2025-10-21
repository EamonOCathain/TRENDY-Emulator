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
1. Repeat the first 20 years of the luh and save as a new file with reconstructed time values
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
    _open_ds,
)
from src.dataset.variables import luh as luh_vars

OVERWRITE = False

# Slurm Logic

# Get Slurm task ID
task_id_str = os.getenv("SLURM_ARRAY_TASK_ID")

# Filter the files to process based on task ID
NUM_ARRAYS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
if task_id_str is None:
    print("[Info] Not running inside a SLURM job. Proceeding linearly.")
    vars_to_process = luh_vars
else:
    task_id = int(task_id_str)
    total = len(luh_vars)
    chunk_size = (total + NUM_ARRAYS - 1) // NUM_ARRAYS 
    start = task_id * chunk_size
    end = min(start + chunk_size, total)
    print(f"[Info] Running inside SLURM array task {task_id}: files {start}:{end} of {total}")
    vars_to_process = luh_vars[start:end]

# function to repeat the first 20 years and save as a new file
def repeat_first_20_years(in_path: str | Path,
                                 out_path: str | Path,
                                 varname: str | None = None) -> None:
    """
    Read a NetCDF with (time, lat, lon), repeat the first 20 timesteps
    to the original time length, keep the original time values & attrs,
    and write with a simple, consistent chunking scheme.

    Encoding is overwritten (zlib+complevel=4), but we *try* to keep the
    same chunk *shape* as the original if it exists; otherwise we fall back
    to (full time, ~20 lat, ~20 lon).
    """
    in_path  = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(in_path, engine="netcdf4", decode_times=False) as ds:
        # pick the data variable if not given
        if varname is None:
            data_vars = list(ds.data_vars)
            if len(data_vars) != 1:
                raise ValueError(f"Provide varname; found variables {data_vars}")
            varname = data_vars[0]
        da = ds[varname]
        if "time" not in da.dims:
            raise ValueError(f"Variable {varname} must have 'time' dimension.")

        tlen = int(ds.sizes["time"])
        if tlen < 20:
            raise ValueError(f"Need at least 20 timesteps; got {tlen}.")

        # Build repeating indices: 0..19 tiled to length tlen
        idx = np.tile(np.arange(20, dtype=np.int32), (tlen // 20) + 1)[:tlen]

        # Repeat data; keep attrs
        da_rep = da.isel(time=idx).astype("float32", copy=False)
        da_rep.attrs = da.attrs

        # Restore original time values & attrs unchanged
        orig_time_vals  = ds["time"].values
        orig_time_attrs = ds["time"].attrs.copy()
        da_rep = da_rep.assign_coords(time=("time", orig_time_vals))
        da_rep["time"].attrs = orig_time_attrs

        # Package to dataset
        ds_out = da_rep.to_dataset(name=varname)

        # ---- Simple encoding (overwrite) ----
        nlat = int(ds_out.sizes.get("lat", 1))
        nlon = int(ds_out.sizes.get("lon", 1))

        # Try to keep original chunk shape if available
        orig_chunks = ds[varname].encoding.get("chunksizes", None)
        if isinstance(orig_chunks, (tuple, list)) and len(orig_chunks) == 3:
            v_chunks = tuple(int(c) for c in orig_chunks)
        else:
            # fallback: full time, ~20x20 spatial (clamped)
            v_chunks = (tlen, min(20, nlat), min(20, nlon))

        enc = {
            varname: {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": v_chunks},
            "time":   {"dtype": "int32", "chunksizes": (tlen,)},
        }
        if "lat" in ds_out.coords:
            enc["lat"] = {"chunksizes": (v_chunks[1],)}
        if "lon" in ds_out.coords:
            enc["lon"] = {"chunksizes": (v_chunks[2],)}

        ds_out.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=enc)

    print(f"[OK] Wrote {out_path} (first 20 years repeated, time unchanged)")

# Repeat the preindustrial years for scenario 
in_dir = project_root / "data/preprocessed/historical/full_time"
out_dir = project_root /"data/preprocessed/preindustrial/full_time"
out_dir.mkdir(parents=True, exist_ok=True)

processed_files = []
for var in luh_vars:
    in_path = in_dir / f"{var}.nc"
    
    if not in_path.exists():
        print(f"Input missing: {in_path}, skipping")
        continue
    
    out_path = out_dir / f"{var}.nc"
    processed_files.append(out_path)
    if out_path.exists() and not OVERWRITE:
        print(f"Skipping var {var}, already exists")
    else:
        repeat_first_20_years(in_path, out_path, varname=var)

plot_dir = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/preprocessing/preindustrial/luh/val_plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# Plotting
for path in processed_files:
    if path.exists():
        first_timestep(path, plot_dir / "first_timestep", title=path.stem)
        finite_mask(path, plot_dir / "finite_mask", title=path.stem)
print("script finished successfully")
