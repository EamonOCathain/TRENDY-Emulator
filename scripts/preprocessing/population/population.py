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
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

import pandas as pd

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.utils.visualisation import finite_mask, first_timestep
from src.utils.preprocessing import run_function_without_tmp, regrid_file
from src.paths.paths import (
    scripts_dir, 
    preprocessing_dir,
    raw_data_dir,
    historical_dir
)

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

# Paths
raw_path = raw_data_dir / "INPUT/pop/population.nc"
current_dir = scripts_dir / "preprocessing/population/"
first_dir = current_dir / "data/first_dir"
plot_dir = scripts_dir / "preprocessing/population/val_plots"
final_dir = historical_dir / "full_time"

dirs = [first_dir, plot_dir, final_dir]
for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

def process_population_to_annual(input_path, output_path, variable, lat_chunk = 5, lon_chunk = 5, overwrite=True):
    """
    Convert population data (irregular intervals) into an ANNUAL time series:
      - 1901–1908: use value at index 0
      - 1909–1948: each decade value (indices 1..4) spans 10 years
      - 1949–2023: one value per year (indices 5..end)
    Output has 123 annual timesteps (1901..2023), no leap years.
    """

    if not overwrite and output_path.exists():
        print("skipping")
        return
    
    print("Processing population (annual)")

    with xr.open_dataset(input_path, decode_times=False) as ds:
        orig_array = ds[variable]

        if orig_array.shape[0] != 127:
            print(f"Expected 127 time steps, got {orig_array.shape[0]}")
            return

        # Keep the last 80 entries (your convention; covers through 2023)
        arr = orig_array[-80:, :, :]  # indices 0..79 used in mapping below

        # Build index mapping per YEAR (length must be 123 for 1901..2023)
        time_idx_yearly = (
            [0] * 8 +                   # 1901–1908 -> arr index 0
            [1] * 10 + [2] * 10 +       # 1909–1928 -> 1,2 each for 10 yrs
            [3] * 10 + [4] * 10 +       # 1929–1948 -> 3,4 each for 10 yrs
            list(range(5, arr.sizes["time"]))  # 1949–2023 -> indices 5..79 (75 yrs)
        )
        years = np.arange(1901, 2024, dtype=int)

        assert len(time_idx_yearly) == len(years) == 123, (
            f"Expected 123 annual steps, got {len(time_idx_yearly)}"
        )

        # Select annual values via index mapping
        annual_array = arr.isel(time=xr.DataArray(time_idx_yearly, dims="time"))

        # Assign annual time coordinate as integer days since 1901-01-01 (noleap)
        import cftime
        dates = xr.cftime_range(start="1901-01-01", periods=len(years), freq="YS", calendar="noleap")
        ref = cftime.DatetimeNoLeap(1901, 1, 1)
        time_vals = np.array([(d - ref).days for d in dates], dtype="i4")

        annual_array = annual_array.assign_coords(time=time_vals)

        # Ensure lat/lon coords present
        if "lat" not in annual_array.coords:
            annual_array = annual_array.assign_coords(lat=orig_array.lat)
        if "lon" not in annual_array.coords:
            annual_array = annual_array.assign_coords(lon=orig_array.lon)

        # Name & dataset
        annual_array = annual_array.rename(variable)
        ds_out = annual_array.to_dataset()

        # Carry over var attrs if present
        if hasattr(orig_array, "attrs"):
            ds_out[variable].attrs = orig_array.attrs

        # Add CF attrs to time
        ds_out["time"].attrs.update({
            "units": "days since 1901-01-01 00:00:00",
            "calendar": "noleap",
        })

        encoding = {
            variable: {
                "zlib": True,
                "complevel": clevel,
                "dtype": "float32",
                "chunksizes": (123, lat_chunk, lon_chunk)
            },
            "time": {"dtype": "i4"},
        }

        # Save
        ds_out.astype("float32").to_netcdf(output_path, engine="netcdf4", encoding=encoding)
        print(f"Saved ANNUAL population data to {output_path}")

regrid_file(raw_path, first_dir / "population.nc", overwrite=OVERWRITE)
process_population_to_annual(first_dir / "population.nc", final_dir /"population.nc", 'population', overwrite=OVERWRITE)
