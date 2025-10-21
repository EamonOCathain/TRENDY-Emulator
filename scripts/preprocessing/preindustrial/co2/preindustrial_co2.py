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
    _open_ds,
)

OVERWRITE = False
clevel = 4 
LAT_CHUNK  = 5 
LON_CHUNK  = 5

# Create pre-industrial CO2
def preindustrial_co2(in_path:Path, out_path:Path, lat_chunk = LAT_CHUNK, lon_chunk = LON_CHUNK, clevel = clevel):

    ds, varname = _open_ds(in_path)
    
    da=ds[varname]
    
    first_slice = da.isel(time=0, drop=True)
    
    replicated = xr.concat([first_slice] * 123, dim="time").astype("float32")
    
    # --- build annual noleap time coordinate: 0, 365, 730, ... ---
    time_vals = (np.arange(123, dtype=np.int32) * 365).astype(np.int32)
    replicated = replicated.assign_coords(
        time=xr.DataArray(
            time_vals,
            dims=("time",),
            attrs={
                "units": "days since 1901-01-01 00:00:00",
                "calendar": "noleap",
                "standard_name": "time",
                "axis": "T",
            },
        )
    )

    # --- wrap into a Dataset and carry over attrs ---
    replicated.attrs = ds[varname].attrs
    out_ds = replicated.to_dataset(name=varname)

    # --- encoding: compress + chunk ---
    tlen = 123
    enc = {
        varname: {
            "zlib": True,
            "complevel": clevel,
            "dtype": "float32",
            # If spatial dims exist they'll be chunked; if not, only time applies.
            "chunksizes": tuple(
                [tlen] + [
                    (lat_chunk if d == "lat" else lon_chunk if d == "lon" else out_ds[varname].sizes[d])
                    for d in out_ds[varname].dims if d != "time"
                ]
            ),
        },
        "time": {"dtype": "int32", "chunksizes": (tlen,)},
    }

    # If coords exist, you can keep them contiguous:
    for c in ("lat", "lon"):
        if c in out_ds.coords:
            enc[c] = {}

    out_ds.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=enc)
    ds.close()
    out_ds.close()
    print(f"[OK] Wrote preindustrial CO2 (constant first timestep, 123 years) to {out_path}")

co2_in_path = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/historical/full_time/co2.nc")
co2_out_path = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/preindustrial/full_time/co2.nc")
co2_out_path.parent.mkdir(parents=True, exist_ok=True)
if co2_out_path.exists() and not OVERWRITE:
    print("skipping co2 file, already exists")
else:
    preindustrial_co2(co2_in_path, co2_out_path)

plot_dir = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/preprocessing/preindustrial/co2/val_plots")
plot_dir.mkdir(parents=True, exist_ok=True)
first_timestep(co2_out_path, plot_dir / "first_timestep", title=co2_out_path.stem)
finite_mask(co2_out_path, plot_dir / "finite_mask", title=co2_out_path.stem)