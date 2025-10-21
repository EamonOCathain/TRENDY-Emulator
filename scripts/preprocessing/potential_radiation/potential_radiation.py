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
import cftime
import netCDF4
import shutil

import pandas as pd

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.utils.visualisation import finite_mask, first_timestep
from src.utils.preprocessing import run_function_without_tmp

# Settings
OVERWRITE = False
NUM_ARRAYS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))  # default to 1 if not under SLURM
clevel = 4 
LAT_CHUNK  = 1
LON_CHUNK  = 1

# Stop Buffering
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Raw Data Directories
out_dir = project_root / "data/preprocessed/historical/annual_files/potential_radiation"
out_dir.mkdir(parents=True, exist_ok=True)
plot_dir = project_root / "scripts/preprocessing/potential_radiation/val_plots"
plot_dir.mkdir(exist_ok=True, parents=True)

def compute_potential_radiation_1yr():
    """
    Compute daily mean potential solar radiation for a single 365-day year
    on the r720x360 grid.
    """
    # --- constants & axes ---
    solar_constant = 1.361  # kW m−2
    hours = np.arange(1, 25)               # 1..24
    day_of_year = np.arange(1, 366)        # 365-day (no-leap)
    lat = np.linspace(-89.75, 89.75, 360)  # 360 lat points
    lon = np.arange(0.0, 360.0, 0.5)       # 720 lon points

    # --- daily geometry per latitude (no leap days) ---
    betas = np.zeros((lat.size, day_of_year.size), dtype=np.float64)

    for i, phi in enumerate(lat):
        # declination (deg) for each day
        delta = 23.45 * np.sin(np.deg2rad((360.0 / 365.0) * (284 + day_of_year)))
        # hour angle (deg) for each hour center
        H = 15.0 * (hours - 12.0)

        sin_phi = np.sin(np.deg2rad(phi))
        cos_phi = np.cos(np.deg2rad(phi))
        sin_delta = np.sin(np.deg2rad(delta))
        cos_delta = np.cos(np.deg2rad(delta))

        # beta(day, hour)
        beta = sin_phi * sin_delta[:, None] + cos_phi * cos_delta[:, None] * np.cos(np.deg2rad(H))[None, :]
        beta = np.maximum(beta, 0.0)

        # integrate over hours -> scalar per day
        betas[i, :] = np.trapz(beta, hours, axis=1)

    # Convert to daily energy. (kept consistent with your original formula)
    I_lat_day = solar_constant * betas * 3600.0  # kWh m−2 per day (as per your earlier code)

    # Broadcast to (time=365, lat, lon): uniform across longitude
    I_1yr = np.repeat(I_lat_day[:, :, None], lon.size, axis=2).transpose(1, 0, 2).astype(np.float32)

    return I_1yr, lat.astype(np.float32), lon.astype(np.float32)

def write_potential_radiation_annual_files(
    out_dir: str | Path,
    start_year: int = 1901,
    end_year: int = 2023,
    overwrite: bool = False,
    lat_chunk: int = 1,
    lon_chunk: int = 1,
):
    """
    Create one NetCDF per year (1901..2023), each with 365 daily timesteps on
    r720x360. Writes 1901 once, then copies and updates only the 'time' variable.

    Time is stored as integer days since 1901-01-01 (noleap), so for year Y:
      time = 365*(Y-1901) + [0..364]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Build single-year template data (365, 360, 720) ---
    I_1yr, lat, lon = compute_potential_radiation_1yr()  # your function, returns float32 arrays

    # --- 2) Write 1901 once ---
    template_path = out_dir / f"potential_radiation_{start_year}.nc"
    if not template_path.exists() or overwrite:
        if template_path.exists():
            template_path.unlink()

        # numeric time values for 1901
        time_vals = np.arange(365, dtype="i4")  # 0..364

        da = xr.DataArray(
            I_1yr,
            dims=("time", "lat", "lon"),
            coords={"time": time_vals, "lat": lat, "lon": lon},
            name="potential_radiation",
            attrs={
                "units": "kWh m-2 day-1",
                "long_name": "Daily Mean Incoming Solar Radiation",
                "standard_name": "surface_downwelling_shortwave_flux_in_air",
            },
        )
        ds = da.to_dataset()
        ds["time"].attrs.update({
            "units": "days since 1901-01-01 00:00:00",
            "calendar": "noleap",
        })
        ds.attrs.update({
            "title": "Synthetic Incoming Solar Radiation (365-day year)",
            "source": "Simple solar geometry (noleap, uniform across longitude)",
            "grid": "r720x360",
            "note": "lon 0..359.5, lat -89.75..89.75; 365 days only (no leap day)",
            "year": start_year,
        })

        encoding = {
            "potential_radiation": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
                "chunksizes": (365, lat_chunk, lon_chunk),
            },
            "time": {"dtype": "i4"},
        }
        ds.to_netcdf(template_path, engine="netcdf4", format="NETCDF4", encoding=encoding)
        print(f"[OK] Wrote template: {template_path}")
    else:
        print(f"[SKIP] Using existing template: {template_path}")

    # --- 3) Copy and update 'time' for subsequent years ---
    for year in range(start_year + 1, end_year + 1):
        dst = out_dir / f"potential_radiation_{year}.nc"

        if dst.exists() and not overwrite:
            print(f"[SKIP] {dst} exists")
            continue

        shutil.copy2(template_path, dst)

        # new numeric time values for this year: 365*(Y-1901) + 0..364
        base = 365 * (year - 1901)
        new_time = (np.arange(365, dtype="i4") + base)

        # overwrite only the 'time' variable in-place
        with netCDF4.Dataset(dst, mode="r+") as nc:
            tvar = nc.variables["time"]
            if tvar.dtype.str not in (">i4", "<i4", "|i4"):  # be lenient about endianness
                raise TypeError(f"{dst}: 'time' dtype is {tvar.dtype}, expected int32")
            if tvar.shape != (365,):
                raise ValueError(f"{dst}: expected time shape (365,), got {tvar.shape}")
            tvar[:] = new_time
            # (optional) update the 'year' global attribute
            nc.setncattr("year", year)

        print(f"[OK] Wrote {dst}")
    
# Run the function
write_potential_radiation_annual_files(out_dir, overwrite=OVERWRITE)
finite_mask(out_dir / "potential_radiation_1901.nc", plot_dir / "finite_mask", overwrite = True)
first_timestep(out_dir / "potential_radiation_1901.nc", plot_dir / "first_timestep", overwrite=True)

