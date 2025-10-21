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

"""This requires:
1. Taking annual averages of the climate data.
2. Combining them into a single rolling mean.
3. Chunking them."""

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

OVERWRITE = False
clevel = 4 
LAT_CHUNK  = 5 
LON_CHUNK  = 5

# Stop Buffering
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
    
from src.utils.visualisation import finite_mask, first_timestep
from src.utils.preprocessing import (
    print_chunk_sizes,
    _open_ds,
    time_avg_cdo,
    extract_year_from_filename
)

# Paths
climate_dir = project_root / "data/preprocessed/historical/annual_files"
climate_vars = ["pre", 
                 "tmp", 
                 "spfh", 
                 "tmax", 
                 "tmin"]
dirs_to_process = []


# Set the output directory and make it
base_dir = project_root / "scripts/preprocessing/rolling_means"
data_dir = base_dir / "data"
out_dir_1 = data_dir / "1.time_avg"
out_dir_2 = data_dir /"2.combined_files"
historical_out_dir = project_root / "data/preprocessed/historical/full_time"
preindustrial_out_dir = project_root /"data/preprocessed/preindustrial/full_time"
plot_dir = base_dir / "val_plots"

# Make dirs
out_dir_1.mkdir(exist_ok=True, parents=True)
out_dir_2.mkdir(exist_ok=True, parents=True)
historical_out_dir.mkdir(exist_ok=True, parents=True)
preindustrial_out_dir.mkdir(exist_ok=True, parents=True)
plot_dir.mkdir(exist_ok=True, parents=True)

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

def combine_annual_files(files, out_path):
    files = [Path(f) for f in files]
    items = []
    ref_lat = ref_lon = None
    time_attrs = None
    vname_ref = None

    for f in files:
        year = extract_year_from_filename(f)
        if year is None:
            raise ValueError(f"Could not extract year from: {f}")

        ds, vname = _open_ds(f)
        if vname_ref is None:
            vname_ref = vname  # remember the data var name from the first file

        lat = ds["lat"].values; lon = ds["lon"].values
        if ref_lat is None:
            ref_lat, ref_lon = lat, lon
            time_attrs = dict(getattr(ds.get("time"), "attrs", {}))
        else:
            if (lat.shape != ref_lat.shape or not np.allclose(lat, ref_lat)) \
               or (lon.shape != ref_lon.shape or not np.allclose(lon, ref_lon)):
                ds.close()
                raise ValueError(f"Lat/Lon mismatch in {f}")

        da = ds[vname]
        if "time" not in da.dims or da.sizes["time"] != 1:
            ds.close()
            raise ValueError(f"{f}: expected time dim of length 1, got {da.sizes.get('time')}")
        da = da.isel(time=0).astype("float32")
        da.attrs = ds[vname].attrs
        items.append((year, da))
        ds.close()

    items.sort(key=lambda x: x[0])
    years = np.array([y for y, _ in items], dtype=np.int32)
    das = [da for _, da in items]
    combined = xr.concat(das, dim="time")

    time_vals = ((years - 1901) * 365).astype(np.int32)
    combined = combined.assign_coords(time=xr.DataArray(time_vals, dims=("time",), attrs=(time_attrs or {})))

    ds_out = combined.to_dataset(name=vname_ref)
    ds_out.to_netcdf(out_path, engine="netcdf4", format="NETCDF4")
    print(f"[OK] Wrote combined file with {combined.sizes['time']} years to {out_path}")
    
# Preindustrial rolling mean
def make_preindustrial_time_mean(files, out_path, clevel=clevel, LAT_CHUNK=LAT_CHUNK, LON_CHUNK=LON_CHUNK):
    """
    From a list of annual NetCDFs:
      1) accept variables shaped (lat, lon) OR (time, lat, lon) with time=1,
      2) sort by year (extracted from filename),
      3) take the mean of the first 19 years,
      4) replicate that 2D mean to 123 annual steps (1901..2023),
         with CF-compliant time (units 'days since 1901-01-01 00:00:00', calendar 'noleap'),
         time values 0, 365, 730, ...,
      5) write compressed/chunked NetCDF.
    """
    from pathlib import Path
    import numpy as np
    import xarray as xr

    files = [Path(f) for f in files]
    items = []
    ref_lat = ref_lon = None
    ref_vname = None

    for f in files:
        year = extract_year_from_filename(f)
        if year is None:
            raise ValueError(f"Could not extract year from: {f}")

        ds, vname = _open_ds(f)  # your helper that returns (dataset, single data var name)
        da = ds[vname]

        # Accept (lat, lon) or (time, lat, lon) with time=1
        if set(da.dims) == {"lat", "lon"}:
            da2 = da
        elif ("time" in da.dims) and (set(da.dims) == {"time", "lat", "lon"}):
            tlen = int(da.sizes["time"])
            if tlen != 1:
                ds.close()
                raise ValueError(f"{f}: time dimension must be length 1, got {tlen}")
            # squeeze to (lat, lon)
            da2 = da.isel(time=0, drop=True)
        else:
            ds.close()
            raise ValueError(f"{f}: variable {vname} must have dims (lat, lon) or (time=1, lat, lon); got {da.dims}")

        # grid consistency checks
        lat = ds["lat"].values
        lon = ds["lon"].values
        if ref_lat is None:
            ref_lat, ref_lon, ref_vname = lat, lon, vname
        else:
            if (lat.shape != ref_lat.shape or not np.allclose(lat, ref_lat)) \
               or (lon.shape != ref_lon.shape or not np.allclose(lon, ref_lon)) \
               or (vname != ref_vname):
                ds.close()
                raise ValueError(f"Grid/var mismatch in {f} vs first file.")

        items.append((year, da2.astype(np.float32)))
        ds.close()

    if not items:
        raise ValueError("No input files provided.")

    # Take mean of first 19 years
    items.sort(key=lambda x: x[0])
    first_19 = items[:19]
    if len(first_19) < 19:
        print(f"[WARN] Only {len(first_19)} files available; averaging all of them.")

    stack = xr.concat([da for _, da in first_19], dim="stack")
    mean2d = stack.mean(dim="stack", keep_attrs=True)  # (lat, lon)

    # Replicate to 123 annual steps (1901..2023)
    nyears = 123
    time_days = (np.arange(nyears, dtype=np.int32) * 365).astype(np.int32)
    mean3d = mean2d.expand_dims(time=nyears).copy()
    mean3d[:] = mean2d.values  # broadcast same 2D field across time

    mean3d = mean3d.assign_coords(
        time=xr.DataArray(
            time_days,
            dims=("time",),
            attrs={
                "units": "days since 1901-01-01 00:00:00",
                "calendar": "noleap",
                "standard_name": "time",
                "axis": "T",
            },
        )
    )

    out_vname = f"{ref_vname}_rolling_mean"
    ds_out = mean3d.to_dataset(name=out_vname)
    # (encoding stays the same except key)
    enc = {
        out_vname: {
            "zlib": True,
            "complevel": clevel,
            "dtype": "float32",
            "chunksizes": (tlen, LAT_CHUNK, LON_CHUNK),
        },
        "time": {"dtype": "int32", "chunksizes": (tlen,)},
        "lat": {"chunksizes": (LAT_CHUNK,)},
        "lon": {"chunksizes": (LON_CHUNK,)},
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=enc)
    ds_out.close()
    print(f"[OK] Wrote 123-year series to {out_path}")
    
# Take the rolling mean
def rolling_mean(in_path, out_path, window=30, clevel=clevel, lat_chunk=LAT_CHUNK, lon_chunk=LON_CHUNK):
    """
    Compute a backward-looking rolling mean over `window` years.
    - Uses only the *previous* `window` years (including current).
    - First `window-1` years are filled by repeating the first valid rolling mean
      so the output keeps the same length (123 years).

    Assumes the file has a single data variable with dims (time, lat, lon)
    and a 123-year annual time axis.
    """
    with xr.open_dataset(in_path, decode_times=False) as ds:
        vname = next(iter(ds.data_vars))
        da = ds[vname]
        da_roll = da.rolling(time=window, min_periods=window).mean()

        if da_roll.sizes["time"] < window:
            raise ValueError(f"Window ({window}) larger than time length ({da_roll.sizes['time']}).")

        first_valid = da_roll.isel(time=window - 1)
        if window > 1:
            leading = xr.concat([first_valid.expand_dims(time=1)] * (window - 1), dim="time")
            da_roll = xr.concat([leading, da_roll.isel(time=slice(window - 1, None))], dim="time")

        da_roll = da_roll.astype("float32")
        # carry attrs and adjust long_name if present
        da_roll.attrs = da.attrs.copy()
        out_vname = f"{vname}_rolling_mean"
        if "long_name" in da_roll.attrs:
            da_roll.attrs["long_name"] = f"{da_roll.attrs['long_name']} (rolling mean, {window}y)"

        ds_out = da_roll.to_dataset(name=out_vname)
        ds_out = ds_out.assign_coords(time=ds["time"])

        tlen = ds_out.dims["time"]
        enc = {
            out_vname: {
                "zlib": True,
                "complevel": clevel,
                "dtype": "float32",
                "chunksizes": (tlen, lat_chunk, lon_chunk),
            },
            "time": {"dtype": "int32", "chunksizes": (tlen,)},
            "lat": {"chunksizes": (lat_chunk,)},
            "lon": {"chunksizes": (lon_chunk,)},
        }

        ds_out.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=enc)
        print(f"[OK] Wrote {out_path} (backward {window}-yr rolling mean, padded front) as var='{out_vname}'")

# Take the time average of the annual files
time_avg_dict = {}
for var in vars_to_process:
    time_avg_list = []
    dir = climate_dir / var
    dirs_to_process.append(dir)
    annual_files = sorted(dir.glob("*.nc"))
    for file in annual_files:
        time_avg_path = out_dir_1 / file.name 
        time_avg_list.append(time_avg_path)
        if time_avg_path.exists() and not OVERWRITE:
            print(f"Skipping time avg of {file}, already exists") 
        else:
            time_avg_cdo(file, time_avg_path)
    time_avg_dict[var]=time_avg_list    

OVERWRITE=True
# Combine the annual files
combined_files = []
for var in vars_to_process:
    out_path = out_dir_2 / f"{var}.nc"
    combined_files.append(out_path)
    if out_path.exists() and not OVERWRITE:
        print(f"Skipping the combined file for {var} as file exists at {out_path}")
    else:
        combine_annual_files(time_avg_dict[var], out_path)

# Compute the rolling means
final_files = []
for file in combined_files:
    out_path = historical_out_dir / f"{file.stem}_rolling_mean.nc"
    final_files.append(out_path)
    if out_path.exists() and not OVERWRITE:
        print(f"Skipping the rolling mean for {var} as file exists at {out_path}")    
    else:
        rolling_mean(file, out_path)
    
# run the function
preindustrial_files = []
for var in vars_to_process:
    file_list = time_avg_dict[var]
    out_path = preindustrial_out_dir / f"{var}_rolling_mean.nc"
    preindustrial_files.append(out_path)
    if out_path.exists() and not OVERWRITE:
        print(f"Skipping the rolling mean for {var} as file exists at {out_path}")    
    else:
        make_preindustrial_time_mean(time_avg_dict[var], out_path)
    
# Plotting
for path in final_files:
    first_timestep(path, plot_dir / "historical/first_timestep", title=path.stem, overwrite=OVERWRITE)
    finite_mask(path,   plot_dir / "historical/finite_mask",   title=path.stem, overwrite=OVERWRITE)
    print_chunk_sizes(path)

for path in preindustrial_files:
    first_timestep(path, plot_dir / "preindustrial/first_timestep", title=path.stem, overwrite=OVERWRITE)
    finite_mask(path,   plot_dir / "preindustrial/finite_mask",   title=path.stem, overwrite=OVERWRITE)
    print_chunk_sizes(path)
    
