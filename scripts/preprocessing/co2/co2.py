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
from src.utils.preprocessing import run_function_without_tmp
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

# Raw Data Directories
co2_path = raw_data_dir / "CO2field/global_co2_ann_1700_2023.txt"
out_path = historical_dir / "full_time/co2.nc"
plot_dir = scripts_dir / "preprocessing/co2/val_plots"
plot_dir.mkdir(exist_ok=True, parents=True)
out_path.parent.mkdir(exist_ok=True, parents=True)

def process_co2(input_path: Path, output_path: Path, overwrite: bool = False):
    """
    Load a 1D text file of global annual CO₂ concentration and make it into
    a 720x360 global gridded file at annual resolution.
    Keeps only the last 123 years and writes time with calendar='noleap'.
    """
    variable = "co2"

    if output_path.exists() and not overwrite:
        print("Skipping CO2: output already exists")
        return

    # Load and keep the last 123 annual values
    annual_array = np.loadtxt(input_path, usecols=1).astype(np.float32)
    trimmed_annual = annual_array[-123:]
    if trimmed_annual.shape[0] != 123:
        raise ValueError(f"Expected at least 123 years, got {annual_array.shape[0]}")

    # CF-time coordinate with noleap calendar
    time = xr.cftime_range(start="1901-01-01", periods=123, freq="YS", calendar="noleap")

    # 0.5° grid matching r720x360 (lon: 0..359.5, lat: -89.75..89.75)
    lat = np.linspace(-89.75, 89.75, 360, dtype=np.float32)
    lon = np.arange(0.0, 360.0, 0.5, dtype=np.float32)

    # Broadcast to (time, lat, lon)
    array = trimmed_annual[:, None, None] * np.ones((123, 360, 720), dtype=np.float32)

    da = xr.DataArray(
        array,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name=variable,
        attrs={
            "units": "ppm",
            "long_name": "Atmospheric CO₂ concentration",
            "standard_name": "mole_fraction_of_carbon_dioxide_in_air",
        },
    )

    ds = da.to_dataset()

    ds.attrs = {
        "title": "Annual CO₂ Forcing",
        "source": "Processed from annual text file",
        "note": "Last 123 years only; calendar=noleap (365_day).",
    }

    # Chunk along time fully and 1x1 spatial chunks
    encoding = {
        "co2": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "chunksizes": (123, 1, 1),
        }
    }

    ds.to_netcdf(output_path, engine="netcdf4", format="NETCDF4", encoding=encoding)
    print(f"[OK] Saved annual CO₂ forcing to {output_path}")
    
# Run the function
process_co2(co2_path, out_path, overwrite=OVERWRITE)
finite_mask(out_path, plot_dir / "finite_mask")
first_timestep(out_path, plot_dir / "first_timestep")

