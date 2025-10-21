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
from typing import Dict, List, Sequence, Tuple, Union
import pandas as pd
import xarray as xr, subprocess
import re 

def _open_ds(path:str)-> Tuple[xr.Dataset, str]:
    """Opens a dataset and searches for a variable which is not time, lat, lon, bnds etc. 
    Returns the ds and the variable name"""
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    exclude = {"time", "lat", "lon", "time_bnds", "bnds", "time_bounds",
               "lat_bnds", "lon_bnds", "lat_bounds", "lon_bounds"}
    varnames = [v for v in ds.data_vars if v not in exclude and not v.endswith("_bnds")]
    if len(varnames) == 0:
        ds.close()
        raise ValueError(f"No plottable data variables found in {path}")
    if len(varnames) != 1:
        ds.close()
        raise ValueError(f"Expected exactly one data var, found {len(varnames)} in {path}: {varnames}")
    return ds, varnames[0]

def build_zlib_encoding(ds, default_level=4):
    enc = {}
    for name, var in ds.variables.items():
        if name in ds.coords:     # coords: keep simple/contiguous
            enc[name] = {}
            continue
        # start from existing encoding so we preserve chunksizes/_FillValue etc.
        e = dict(var.encoding) if hasattr(var, "encoding") else {}
        # set compression flags
        e["zlib"] = True
        e["shuffle"] = True
        e["complevel"] = default_level
        # keep existing chunksizes if present; don't invent chunks unless you intend to
        # don't set dtype unless you want to coerce types
        enc[name] = e
    return enc

def select_var(in_path: str, out_path: str, var: str, overwrite=False) ->None:
    """
    Open a dataset and select only one variable.
    Save the reduced dataset to out_path and return it with the variable name.
    """
    if out_path.exists() and not overwrite:
        print(f"Skipping selecting var for {var} as file exists at {out_path}")
        
    ds = xr.open_dataset(in_path, engine="netcdf4", decode_times=False)
    
    if var not in ds.variables:
        raise ValueError(f"Variable '{var}' not found in {in_path}. "
                         f"Available: {list(ds.variables)}")
    
    # keep only the selected variable
    ds_new = ds[[var]]
    
    # save
    ds_new.to_netcdf(out_path, engine="netcdf4")
    ds.close()
    
    print(f"[OK] Saved {var} to {out_path}")

def make_annual_time(in_path: str | Path, out_path: str | Path) -> str:
    """
    Add a time dimension with 123 annual steps (1901..2023) to a (lat, lon) NetCDF.
    Time is written CF-style with units 'days since 1901-01-01 00:00:00' and calendar 'noleap'.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    # Open input (assumed (lat, lon) variables)
    ds = xr.open_dataset(in_path)

    # Build CF time coordinate directly as cftime (avoids pandas freq deprecation & tz issues)
    time = xr.cftime_range(start="1901-01-01", periods=123, freq="YS", calendar="noleap")

    # Broadcast data along new time dimension
    expanded = ds.expand_dims(time=time)

    # IMPORTANT: remove any attrs on 'time' so xarray's encoder can set them from encoding
    expanded = expanded.copy()
    expanded["time"].attrs = {}

    # CF encoding for time
    enc = {
        "time": {
            "units": "days since 1901-01-01 00:00:00",
            "calendar": "noleap"
        }
    }

    # Write
    expanded.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=enc)

    ds.close()
    expanded.close()
    print(f"[OK] Added annual time (1901–2023) and saved to {out_path}")

def replace_time_axis(in_path: str | Path, out_path: str | Path, time_res: str) -> None:
    in_path = Path(in_path)
    out_path = Path(out_path)
    time_res = time_res.lower()

    if time_res not in {"daily", "monthly", "annual"}:
        raise ValueError("time_res must be 'daily', 'monthly', or 'annual'.")

    if time_res == "daily":
        nyears = 2023 - 1901 + 1
        tlen   = nyears * 365
        time_vals = np.arange(tlen, dtype=np.int32)
    elif time_res == "annual":
        nyears = 2023 - 1901 + 1
        tlen   = nyears
        time_vals = (np.arange(tlen, dtype=np.int64) * 365).astype(np.int32)
    else:
        nyears = 2023 - 1901 + 1
        tlen   = nyears * 12
        month_starts_1yr = np.array([0,31,59,90,120,151,181,212,243,273,304,334], dtype=np.int32)
        year_offsets = (np.arange(nyears, dtype=np.int64) * 365).astype(np.int32)
        time_vals = (month_starts_1yr[None, :] + year_offsets[:, None]).reshape(-1)

    with xr.open_dataset(in_path, engine="netcdf4", decode_times=False) as ds:
        if "time" not in ds.dims:
            raise KeyError(f"'time' dim not found in {in_path}")
        if int(ds.sizes["time"]) != tlen:
            raise ValueError(f"{in_path} has time length {int(ds.sizes['time'])}, expected {tlen} for '{time_res}'.")

        # Assign new numeric time with CF attrs
        ds2 = ds.assign_coords(
            time=xr.DataArray(
                time_vals,
                dims=("time",),
                attrs={"units": "days since 1901-01-01 00:00:00",
                       "calendar": "noleap",
                       "standard_name": "time",
                       "axis": "T"}
            )
        )

        # --- Preserve encodings for ALL variables (incl. data vars’ _FillValue) ---
        encoding = {}
        for name, v in ds2.variables.items():
            e = {}
            for k in ("zlib","shuffle","complevel","chunksizes","fletcher32","contiguous","dtype","_FillValue"):
                if k in v.encoding and v.encoding[k] is not None:
                    e[k] = v.encoding[k]
            # Do NOT set _FillValue on coords such as time/lat/lon
            if name in ds2.coords:
                e.pop("_FillValue", None)
            encoding[name] = e

        ds2.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=encoding)

    print(f"[OK] Replaced time axis ({time_res}) and wrote: {out_path}")

def repeat_last_timestamp(
    in_file: str | Path,
    out_file: str | Path,
    n_repeats: int = 1
) -> None:
    """
    Repeat the last timestep of a dataset `n_repeats` times
    and write the result to a new NetCDF file.

    Assumes time is numeric (decode_times=False).
    """

    ds, _ = _open_ds(in_file)  # opens with decode_times=False
    try:
        if n_repeats <= 0:
            ds.to_netcdf(out_file, engine="netcdf4", format="NETCDF4")
            print(f"[OK] No repeats requested; wrote {out_file}")
            return

        # --- Repeat last slice ---
        last_slice = ds.isel(time=-1)
        repeated_slices = [last_slice] * n_repeats
        ds_repeated = xr.concat([ds] + repeated_slices, dim="time")

        # --- Build new time coordinate ---
        time_values = ds["time"].values
        if time_values.size >= 2:
            step = time_values[-1] - time_values[-2]
        else:
            step = type(time_values[-1])(1)  # fallback: step = 1

        extra_times = time_values[-1] + step * np.arange(1, n_repeats + 1, dtype=time_values.dtype)
        new_time = np.concatenate([time_values, extra_times])

        ds_repeated = ds_repeated.assign_coords(time=("time", new_time))

        # --- Save ---
        ds_repeated.to_netcdf(out_file, engine="netcdf4", format="NETCDF4")
        print(f"[OK] Repeated last timestep {n_repeats} times and wrote {out_file}")

    finally:
        ds.close()
        try:
            ds_repeated.close()
        except NameError:
            pass

def daily_avg_cdo(in_path: Path, out_path: Path) -> None:
    cmd = ["cdo", "daymean", str(in_path), str(out_path)]
    print(f"[INFO] Taking Daily Average: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[OK] Saved daily average tmp file: {out_path}")

def yearmean_cdo(in_path, out_path, overwrite=True):
    in_path  = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        print(f"[INFO] Skipping yearmean (exists): {out_path}")
        return out_path

    tmp = out_path.with_suffix(out_path.suffix + ".__cdo__.nc")
    subprocess.run(["cdo", "yearmean", str(in_path), str(tmp)], check=True)
    _drop_bounds_inplace(tmp, out_path)
    try: tmp.unlink()
    except FileNotFoundError: pass
    print(f"[OK] Wrote annual means to: {out_path}")
    return out_path
    
def reassign_720_360_grid(in_path: str | Path, out_path: str | Path) -> str:
    """
    Reassign longitude and latitude coordinates so the file matches the
    standard r720x360 grid used in CDO remapbil:
      - lon: 0.0, 0.5, ..., 359.5  (720 points)
      - lat: -89.75, -89.25, ..., 89.75  (360 points)
    """
    ds, varname = _open_ds(in_path)

    # --- Fix longitude: wrap to [0, 360), sort ascending ---
    ds = ds.assign_coords(lon=(ds.lon % 360))
    ds = ds.sortby("lon")

    # --- Fix latitude: sort ascending (-90 -> 90) ---
    ds = ds.sortby("lat")

    # --- Overwrite with exact target coords to avoid float drift ---
    target_lon = np.arange(0.0, 360.0, 0.5, dtype=np.float64)   # 720 points
    target_lat = np.linspace(-89.75, 89.75, 360, dtype=np.float64)  # 360 points
    ds = ds.assign_coords(lon=target_lon, lat=target_lat)

    # --- Save ---
    ds.to_netcdf(out_path, engine="netcdf4", format="NETCDF4")
    ds.close()

    print(f"[OK] Reassigned lat/lon to r720x360 and saved {out_path}")
    return str(out_path)

def _drop_bounds_inplace(in_path: str | Path, out_path: str | Path) -> Path:
    """Drop any *bnds*/bounds variables and the orphan 'bnds' dim, preserve attrs/encoding."""
    in_path  = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(in_path, engine="netcdf4", decode_times=False) as ds:
        drop_vars = [v for v in ds.variables
                     if v in {"time_bnds","time_bounds","bnds","lat_bnds","lon_bnds","lat_bounds","lon_bounds"}
                     or v.endswith("_bnds") or v.endswith("_bounds")]
        if drop_vars:
            ds = ds.drop_vars(drop_vars)

        # Preserve encodings where possible
        encoding = {}
        for name, var in ds.variables.items():
            enc = {}
            for k in ("zlib","shuffle","complevel","chunksizes","fletcher32","contiguous","dtype","_FillValue"):
                if k in var.encoding:
                    enc[k] = var.encoding[k]
            encoding[name] = enc

        ds.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=encoding)

    return out_path

def cdo_add3(in1, in2, in3, out_path, var_name="cTotal", overwrite=False, cdo_bin="cdo"):
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        print(f"[INFO] Output exists, skipping: {out_path}")
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp1 = out_path.with_suffix(".tmp_add1.nc")
    tmp2 = out_path.with_suffix(".tmp_add2.nc")
    tmp3 = out_path.with_suffix(".tmp_add3.nc")  # final tmp from CDO before cleaning

    try:
        subprocess.run([cdo_bin, "add", str(in1), str(in2), str(tmp1)], check=True)
        subprocess.run([cdo_bin, "add", str(tmp1), str(in3), str(tmp2)], check=True)
        subprocess.run([cdo_bin, f"setname,{var_name}", str(tmp2), str(tmp3)], check=True)
        _drop_bounds_inplace(tmp3, out_path)
        print(f"[OK] Wrote {out_path} with variable name '{var_name}'")
        return out_path
    finally:
        for t in (tmp1, tmp2, tmp3):
            try: Path(t).unlink()
            except FileNotFoundError: pass

def cdo_subtract(in1, file_to_subtract, out_path, var_name="npp", overwrite=False):
    """
    Subtract two NetCDF files with CDO (in1 - in2) and write to out_path,
    renaming the resulting variable to `var_name`.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        print(f"[INFO] Output exists, skipping: {out_path}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["cdo", f"-setname,{var_name}", "-sub", str(in1), str(file_to_subtract), str(out_path)]
    print(f"[INFO] Running: {' '.join(cmd)}")  # optional but handy
    subprocess.run(cmd, check=True)

    print(f"[OK] Wrote {out_path} with variable name '{var_name}' (computed as {in1} - {file_to_subtract})")
    return out_path

def zero_out_netcdf(in_path, out_path, new_var_name=None, engine="netcdf4", overwrite=False):
    """
    Create a copy of a NetCDF file where all data variables are set to 0.
    Optionally, rename the variables to `new_var_name`.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        print(f"Skipping zeroing as file exists at {out_path}")
        return
    
    ds = xr.open_dataset(in_path)
    ds_zero = ds.copy(deep=True)

    for v in list(ds_zero.data_vars):
        ds_zero[v] = 0 * ds_zero[v]

    # Rename if requested
    if new_var_name is not None:
        if len(ds_zero.data_vars) == 1:
            old_name = list(ds_zero.data_vars.keys())[0]
            ds_zero = ds_zero.rename({old_name: new_var_name})
        else:
            # if multiple vars, rename all to <new_var_name>_<old_name>
            ds_zero = ds_zero.rename(
                {v: f"{new_var_name}_{v}" for v in ds_zero.data_vars}
            )

    ds_zero.to_netcdf(out_path, engine=engine)
    print(f"[OK] Wrote zeroed dataset to {out_path}")
    return out_path


def xr_ensmean(in_paths, out_path, overwrite=False, compress_level=3):
    """
    Strict-NaN ensemble mean with final NetCDF chunking: time full, lat=5, lon=5.
    Avoids misaligned-open warnings by opening with native ('auto') chunks
    and rechunking only at the end.
    """
    in_paths  = [Path(p) for p in in_paths]
    out_path  = Path(out_path)

    if not in_paths:
        raise ValueError("No input files provided for ensemble mean")

    if out_path.exists():
        if not overwrite:
            print(f"[INFO] Ensemble mean exists, skipping: {out_path}")
            return out_path
        out_path.unlink()

    # Open lazily with native file chunks -> no misaligned-chunk warnings
    dsets = [xr.open_dataset(p, chunks="auto") for p in in_paths]

    # Ensure exact coord alignment (raises if mismatch)
    dsets = xr.align(*dsets, join="exact")

    # Concatenate along a new member dimension
    dsets = [ds.assign_coords(member=i).expand_dims("member") for i, ds in enumerate(dsets)]
    ds_all = xr.concat(dsets, dim="member")

    # Strict NaN: any NaN in any member -> NaN in mean
    ds_mean = ds_all.mean(dim="member", skipna=False)

    # Rechunk ONLY now (after math), to desired output chunking
    # (full time, 5×5 spatial)
    # If some variables lack a dimension, xarray will ignore it per-variable.
    ds_mean = ds_mean.chunk({"time": -1, "lat": 5, "lon": 5})

    # Build per-variable encoding for NetCDF (compression + chunksizes)
    t = ds_mean.sizes.get("time", 1)
    encoding = {}
    for v in ds_mean.data_vars:
        dims = ds_mean[v].dims
        if set(("time", "lat", "lon")).issubset(dims):
            encoding[v] = {"zlib": True, "complevel": compress_level, "chunksizes": (t, 5, 5)}
        elif set(("lat", "lon")).issubset(dims):
            encoding[v] = {"zlib": True, "complevel": compress_level, "chunksizes": (5, 5)}
        else:
            encoding[v] = {"zlib": True, "complevel": compress_level}

    ds_mean.to_netcdf(out_path, engine="netcdf4", encoding=encoding)
    print(f"[OK] Wrote ensemble mean: {out_path}")
    return out_path

def cdo_ensmean(in_paths, out_path, overwrite=True):
    """
    Compute ensemble mean of a list of NetCDF files using CDO.
    If any member is NaN at a pixel, result will be NaN (strict).
    """
    out_path = Path(out_path)

    if not in_paths:
        raise ValueError("No input files provided for ensemble mean")

    if out_path.exists():
        if not overwrite:
            print(f"[INFO] Ensemble mean exists, skipping: {out_path}")
            return out_path
        else:
            out_path.unlink()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure strict NaN handling
    env = os.environ.copy()
    env["CDO_MISSVALS"] = "1"

    cmd = ["cdo", "ensmean"] + [str(p) for p in in_paths] + [str(out_path)]
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

    print(f"[OK] Wrote ensemble mean: {out_path}")
    return out_path

def xarray_chunk(in_path: Path, out_path: Path, complevel: int = 4, lat_chunk: int = 30, lon_chunk: int = 30) -> None:
    ds, var = _open_ds(in_path)

    tlen = ds.sizes.get("time", 1)
    ds = ds.chunk({"time": tlen, "lat": lat_chunk, "lon": lon_chunk})

    encoding = {}
    for var, da in ds.data_vars.items():
        if da.chunks:
            dim_to_first = {dim: chunks[0] for dim, chunks in zip(da.dims, da.chunks)}
            chunks = tuple(dim_to_first[d] for d in da.dims)
        else:
            chunks = tuple(da.sizes[d] for d in da.dims)
        encoding[var] = {"zlib": True, "complevel": complevel, "shuffle": True, "dtype": "float32", "chunksizes": chunks}

    for c in ("lat", "lon", "time"):
        if c in ds.coords:
            encoding[c] = {}  # contiguous

    ds.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=encoding)
    ds.close()
    print(f"[OK] Saved chunked/compressed file: {out_path}")
    
def nccopy_chunk(in_path: Path, out_path: Path, clevel: int = 4, lat_chunk: int = 30, lon_chunk: int = 30, overwrite=True) -> None:
    
    if out_path.exists():
        if not overwrite:
            print(f"skipping chunking {out_path}, exists.")
            return
        else:
            out_path.unlink()
            
    # Inspect sizes (and ensure we have a time dim if present)
    with xr.open_dataset(in_path, chunks={}, decode_times=False) as ds:
        tlen = int(ds.sizes.get("time", 0)) or 1
        nlat = int(ds.sizes.get("lat", ds.sizes.get("y", 1)))
        nlon = int(ds.sizes.get("lon", ds.sizes.get("x", 1)))

    chunkspec = f"time/{tlen},lat/{lat_chunk},lon/{lon_chunk}"

    # First try: preserve container, add shuffle+deflate+chunks
    cmd = [
        "nccopy", "-s", "-d", str(clevel),
        "-c", chunkspec,
        str(in_path), str(out_path),
    ]
    print("[RUN]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] Wrote {out_path}")
        return
    except subprocess.CalledProcessError as e:
        print("[WARN] Preserve-container nccopy failed; retrying with -k nc4classic:", e)

    # Fallback: force NetCDF-4 classic model (needed if input is netCDF-3)
    cmd = [
        "nccopy", "-k", "nc4classic", "-s", "-d", str(clevel),
        "-c", chunkspec,
        str(in_path), str(out_path),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[OK] Wrote {out_path}")
        
        
def floor_time(infile: str | Path, outfile: str | Path) -> None:
    """
    Round down the 'time' coordinate values using xarray/NumPy,
    preserving original NetCDF chunking/compression as much as possible.
    """
    infile = Path(infile)
    outfile = Path(outfile)

    # Open without decoding CF-time so we can floor numeric values directly.
    # engine='netcdf4' helps xarray populate .encoding with on-disk settings.
    ds = xr.open_dataset(infile, engine="netcdf4", decode_times=False, mask_and_scale=False)

    if "time" not in ds.coords:
        ds.close()
        raise KeyError(f"'time' coordinate not found in {infile}")

    t = ds["time"]
    orig_dtype = t.dtype
    # Floor numeric times; if already integer, this is a no-op after astype.
    t_floored = np.floor(t.values).astype(orig_dtype, copy=False)

    # Keep original attributes
    ds = ds.assign_coords(time=(t.dims, t_floored))
    ds["time"].attrs = t.attrs

    # Build an encoding dict that preserves per-variable on-disk settings
    # (chunksizes, zlib, complevel, shuffle, dtype, etc.) when available.
    encoding = {}
    for name, var in ds.variables.items():  # includes coords and data_vars
        enc = {}
        # xarray populates .encoding when opened with netcdf4 engine
        for key in ("zlib", "complevel", "shuffle", "chunksizes",
                    "fletcher32", "contiguous", "dtype"):
            if key in var.encoding:
                enc[key] = var.encoding[key]
        # Ensure chunk tuple order matches current dims if present
        if "chunksizes" in enc and enc["chunksizes"] is not None:
            # Some files store chunks per dim; trust existing values.
            pass
        encoding[name] = enc

    # Write out preserving encodings
    ds.to_netcdf(Path(outfile), engine="netcdf4", format="NETCDF4", encoding=encoding)
    ds.close()
    print(f"[SUCCESS] Floored time axis written to {outfile}")

def drop_time_bnds(in_path, out_path):
    ds = xr.open_dataset(in_path, decode_times=False)
    for v in ["time_bnds", "bnds", "time_bounds", "lat_bounds", "lat_bnds", "lon_bnds", "lon_bounds"]:
        if v in ds:
            ds = ds.drop_vars(v)
    # write dataset to out_path (tmp path)
    ds.to_netcdf(out_path, engine="netcdf4", format="NETCDF4")
    ds.close()
    print(f"[OK] Dropped bounds and wrote: {out_path}")

def _is_fill(da, sentinels):
    mask = False
    for s in sentinels:
        mask = mask | xr.apply_ufunc(np.isclose, da, s)
    return mask

from pathlib import Path
from typing import Union
import numpy as np
import xarray as xr

def set_negative_to_zero(in_path: Union[str, Path], out_path: Union[str, Path]) -> Path:
    """
    Open a NetCDF, set all real data values < 0 to 0, and save to out_path.
    - Preserves coord (e.g., 'time') attrs/encoding
    - Preserves the original data variable's _FillValue if present
    - Leaves NaNs and fill/missing sentinels untouched
    """
    in_path  = Path(in_path)
    out_path = Path(out_path)

    ds, var = _open_ds(in_path)  # opens with engine="netcdf4", decode_times=False
    try:
        da = ds[var]

        # --- Determine fill/missing sentinels to protect ---
        # Prefer encoding._FillValue, then attrs._FillValue / missing_value
        fv = da.encoding.get("_FillValue", da.attrs.get("_FillValue", da.attrs.get("missing_value", None)))
        # Build a small list of sentinels to guard against (only include defined values)
        sentinels = [fv] if fv is not None else []
        # (Do NOT add generic constants here; we only want to protect actual file's sentinel)

        # Build "is fill" mask robustly (float-safe)
        is_fill = False
        for s in sentinels:
            is_fill = is_fill | xr.apply_ufunc(np.isclose, da, s)

        # --- Replace negatives with 0, but NOT where it's fill/missing; NaNs pass through ---
        da_new = xr.where((da < 0) & (~is_fill), 0, da)

        # Ensure floating type if needed so NaN/fill are representable
        if not np.issubdtype(da_new.dtype, np.floating):
            da_new = da_new.astype("float32")

        # --- Build output dataset ---
        ds_new = ds.copy(deep=True)
        ds_new[var] = da_new

        # ---- Preserve coord attrs/encoding (CRITICAL for time) ----
        for cname in ds_new.coords:
            # copy attrs verbatim
            ds_new[cname].attrs = dict(ds[cname].attrs)
            # copy a safe subset of encoding keys if present
            enc_src = getattr(ds[cname], "encoding", {})
            enc_dst = {}
            for k in ("zlib", "complevel", "shuffle", "chunksizes", "dtype"):
                if k in enc_src and enc_src[k] is not None:
                    enc_dst[k] = enc_src[k]
            # never set _FillValue on coords
            ds_new[cname].encoding = enc_dst

        # ---- Data-var encoding: preserve what's there, don't invent a new _FillValue ----
        encoding = {}
        enc_src = getattr(ds[var], "encoding", {})
        e = {}
        for k in ("zlib", "complevel", "shuffle", "chunksizes", "dtype", "_FillValue"):
            if k in enc_src and enc_src[k] is not None:
                e[k] = enc_src[k]
        # If we casted to float but dtype was fixed in encoding (e.g., int), drop it
        if "dtype" in e and not np.issubdtype(da_new.dtype, np.dtype(e["dtype"]).type):
            e["dtype"] = str(da_new.dtype)
        encoding[var] = e

        # Clean conflicting attrs on the data var (let encoding control on-disk fill)
        for k in ("_FillValue", "missing_value"):
            ds_new[var].attrs.pop(k, None)
            ds_new[var].encoding.pop(k, None)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds_new.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=encoding)
        print(f"[OK] Negatives→0 (fill preserved) → {out_path}")

    finally:
        ds.close()

    return out_path

def standardise_vars(
    in_path,
    out_path,
    default_fill: float = -99999.0,
    normalize_coords: bool = True,
    overwrite=True
):
    in_path  = Path(in_path)
    out_path = Path(out_path)
    
    if out_path.exists() and not overwrite:
        print(f"skipping multiplying by -1, exists at {out_path}")
        return

    with xr.open_dataset(in_path, engine="netcdf4", decode_times=False) as ds:
        ds = ds.copy()

        # 1) Drop common bounds/bnds variables
        drop_candidates = [
            "time_bnds", "time_bounds", "bnds",
            "lat_bnds", "lat_bounds",
            "lon_bnds", "lon_bounds",
        ]
        keep = [v for v in ds.variables if v not in drop_candidates]
        ds = ds[keep]

        # 2) Normalize coord names
        if normalize_coords:
            rename_map = {}
            if "longitude" in ds and "lon" not in ds:
                rename_map["longitude"] = "lon"
            if "latitude" in ds and "lat" not in ds:
                rename_map["latitude"] = "lat"
            if rename_map:
                ds = ds.rename(rename_map)
            # ensure lon/lat are coords with sensible attrs
            for cname, std_name, units in [("lon", "longitude", "degrees_east"),
                                           ("lat", "latitude", "degrees_north")]:
                if cname in ds and cname not in ds.coords:
                    ds = ds.set_coords(cname)
                if cname in ds:
                    a = ds[cname].attrs
                    a.setdefault("standard_name", std_name)
                    a.setdefault("long_name", std_name)
                    a.setdefault("units", units)
                # strip fill attrs from coords
                for k in ("_FillValue", "missing_value"):
                    ds[cname].attrs.pop(k, None)
                    ds[cname].encoding.pop(k, None)

        # 3) Resolve fill for data variables
        encoding = {}
        for name, da in ds.data_vars.items():
            # Remove conflicting attrs & encodings first
            for k in ("_FillValue", "missing_value"):
                da.attrs.pop(k, None)
                da.encoding.pop(k, None)

            # detect any sentinel in data (historic -99999 etc.)
            # choose a single fill (prefer existing mv/fv if present; else default)
            # note: we already popped, so look for common sentinels directly
            fill = default_fill

            # If any of these typical sentinels are present, mask them to NaN
            # (extend list if you encounter others)
            sentinels = (-99999.0, -9.969209968386869e36, -1e20, 1e20)
            data = da.data  # lazy ok
            # mask only if the var is numeric
            try:
                # where() will load as needed; ok for typical sizes
                for s in sentinels:
                    da = da.where(da != s)
            except Exception:
                pass

            # ensure float dtype so NaN is representable (and fill encoding is valid)
            if not np.issubdtype(da.dtype, np.floating):
                da = da.astype("float32")

            ds[name] = da

            # Set a single, explicit _FillValue in encoding
            encoding[name] = {"_FillValue": np.float32(fill)}
            # Optional: also set matching missing_value attribute (NOT required)
            # ds[name].attrs["missing_value"] = np.float32(fill)

        ds.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=encoding)

    print(f"[OK] Dropped bounds, normalized coords, fixed fill values → {out_path}")

def regrid_file(
    in_path: Path,
    out_path: Path,
    overwrite: bool = True
) -> Path:
    """
    Bilinear regrid with CDO to `grid_spec`.
    """
    grid_spec = "r720x360"
    out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        print(f"[INFO] Skipping regridding (exists): {out_path}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["cdo", f"remapbil,{grid_spec}", str(in_path), str(out_path)]
    print(f"[INFO] Regridding: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[OK] Wrote regridded: {out_path}")

    return out_path
    
def trim_time_xarray(
    in_path: Path,
    out_path: Path,
    start_idx: int,
    end_idx: int,
    time_res: str
) -> None:
    in_path = Path(in_path)
    out_path = Path(out_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".__trim__.nc")

    ds, var = _open_ds(in_path)  # opens with engine="netcdf4", decode_times=False
    try:
        print(f"[INFO] Trimming {in_path.name}::{var} -> {tmp_path}")
        stop = None if end_idx in (None, -1) else end_idx
        ds_trim = ds.isel(time=slice(start_idx, stop))

        tlen = int(ds_trim.sizes.get("time", 0))
        if tlen <= 0:
            raise ValueError(f"No time steps after trim in {in_path}::{var}")

        # --- Build encoding that preserves per-variable settings (incl. _FillValue) ---
        enc = {}
        for name, v in ds_trim.variables.items():
            e = {}
            for k in ("zlib","shuffle","complevel","chunksizes","fletcher32","contiguous","dtype","_FillValue"):
                if k in v.encoding and v.encoding[k] is not None:
                    e[k] = v.encoding[k]
            # Never set _FillValue on coords
            if name in ds_trim.coords:
                e.pop("_FillValue", None)
            enc[name] = e

        # Write trimmed tmp preserving encodings
        ds_trim.to_netcdf(tmp_path, engine="netcdf4", format="NETCDF4", encoding=enc)
        print(f"[OK] Wrote trimmed tmp: {tmp_path}")

    finally:
        ds.close()

    try:
        replace_time_axis(tmp_path, out_path, time_res)  # now also preserves enc (see below)
        print(f"[OK] Wrote trimmed with canonical time: {out_path}")
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        
def print_chunk_sizes(path: str | Path):
    """Print chunk sizes of all variables in a NetCDF file."""
    with xr.open_dataset(path, engine="netcdf4", decode_times=False, chunks={}) as ds:
        print(f"[INFO] Chunk sizes in {path}:")
        for name, var in ds.variables.items():
            chunks = var.encoding.get("chunksizes", None)
            if chunks is None:
                print(f"  {name:20s}: not chunked")
            else:
                print(f"  {name:20s}: {chunks}")

def rename_to_var(in_path: Path, out_path: Path) -> Path:
    """
    Detect the single data variable name via _open_ds, then copy the file to
    <out_dir>/<var>.nc using rsync (preserves chunking, compression, attrs).
    """
    ds, var = _open_ds(in_path)
    try:
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{var}.nc"

    finally:
        ds.close()
        
    cmd = ["rsync", "-a", str(in_path), str(target)]
    subprocess.run(cmd, check=True)
    print(f"[OK] Renamed (copied) {in_path} -> {target}")
    return target
    
def run_function(list_files, out_dir: Path, function, overwrite=False, arg1=None, arg2=None, arg3=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    list_out = []
    for file in list_files:
        out_file = out_dir / file.name
        out_tmp  = out_dir / f"{file.stem}.__tmp__.nc"

        if out_tmp.exists():
            try:
                out_tmp.unlink()
                print(f"[CLEAN] Deleted tmp {out_tmp}")
            except Exception as e:
                print(f"[WARN] Could not delete tmp {out_tmp}: {e}")

        if out_file.exists() and not overwrite:
            print(f"[SKIP] File exists and overwrite=False: {out_file}")
            list_out.append(out_file)
            continue

        # build args list (exclude None)
        args = [a for a in (arg1, arg2, arg3) if a is not None]

        # run stage function -> it must write to out_tmp
        function(file, out_tmp, *args)

        shutil.move(out_tmp, out_file)
        print(f"[OK] Moved tmp to final: {out_file}")
        list_out.append(out_file)

    return list_out

def run_function_without_tmp(list_files, out_dir: Path, function, overwrite=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for file in list_files:
        out_file = out_dir / file.name  # placeholder; function may choose a different name

        if out_file.exists() and not overwrite:
            print(f"[SKIP] Exists and overwrite=False: {out_file}")
            results.append(out_file)
            continue

        ret = function(file, out_file)  # allow function to return the actual path
        results.append(Path(ret) if ret else out_file)

    return results

def extract_year_from_filename(path: str | Path) -> int:
    """
    Extract the first 4-digit year (1901–2023) from a filename,
    regardless of surrounding characters.
    Raises RuntimeError if no valid year is found.
    """
    fname = Path(path).name
    match = re.search(r"(\d{4})", fname)
    if match:
        year = int(match.group(1))
        if 1901 <= year <= 2023:
            return year
    raise RuntimeError(f"No valid year (1901–2023) found in filename: {path}")

def process_time_annual_file(in_path: Path, out_path: Path) -> None:
    year = extract_year_from_filename(in_path)
    if year is None:
        raise ValueError(f"Could not extract year from filename: {in_path}")

    # Compute daily values for that year
    start = (year - 1901) * 365
    end = start + 365
    new_time = np.arange(start, end, dtype=np.int32)

    with xr.open_dataset(in_path) as ds:
        ds = ds.assign_coords(time=("time", new_time))
        ds["time"].attrs = {
            "units": "days since 1901-01-01",
            "calendar": "noleap",
            "standard_name": "time",
            "axis": "T",
        }

        # don’t overwrite encoding → keep chunking as it was
        ds.to_netcdf(out_path, engine="netcdf4", format="NETCDF4")

    print(f"[OK] Wrote with preserved chunking: {out_path}")
    
    

def time_avg_cdo(in_path: str | Path, out_path: str | Path, overwrite: bool = True) -> Path:
    """
    Compute the time average of a NetCDF file using CDO.
    Saves result to out_path.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        print(f"[INFO] Skipping timmean (exists, overwrite=False): {out_path}")
        return out_path

    cmd = ["cdo", "timmean", str(in_path), str(out_path)]
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[OK] Saved time average to {out_path}")
    return out_path
    
def make_mask_from_paths_df(df, model=None, scenario=None, variable=None):
    """
    Create a boolean mask for df where model, scenario, and variable
    can each be a str or list of str. If None, that field is ignored.
    """
    mask = pd.Series(True, index=df.index)

    if model is not None:
        if isinstance(model, str):
            mask &= df["model"] == model
        else:
            mask &= df["model"].isin(model)

    if scenario is not None:
        if isinstance(scenario, str):
            mask &= df["scenario"] == scenario
        else:
            mask &= df["scenario"].isin(scenario)

    if variable is not None:
        if isinstance(variable, str):
            mask &= df["variable"] == variable
        else:
            mask &= df["variable"].isin(variable)

    return mask

def reconstruct_monthly_cTotal_jan_anchor(annual_cTotal_file, monthly_nbp_file, output_file, overwrite=True):
    """
    Reconstruct monthly cTotal from annual cTotal and monthly NBP.

    Parameters
    ----------
    annual_cTotal_file : str or Path
        NetCDF file with annual cTotal (time=years).
    monthly_nbp_file : str or Path
        NetCDF file with monthly NBP (time=months).
    output_file : str or Path
        Path to write the reconstructed monthly cTotal NetCDF.
    """
    if output_file.exists() and not overwrite:
        print(f"skipping reconstruction monthly ctotal {output_file.name}")
        return
    
    annual_ds = xr.open_dataset(annual_cTotal_file)
    nbp_ds    = xr.open_dataset(monthly_nbp_file)

    cTotal_annual = annual_ds["cTotal"]   # (year, lat, lon)
    nbp_monthly   = nbp_ds["nbp"]         # (month, lat, lon)

    # seconds in each month (fixed 365-day calendar, no leap years)
    days_per_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=np.float32)
    seconds_per_month = days_per_month * 86400

    n_years = cTotal_annual.sizes["time"]
    n_months = nbp_monthly.sizes["time"]
    assert n_months == n_years * 12, "Monthly NBP length must equal 12 × annual cTotal length"

    lat = cTotal_annual["lat"]
    lon = cTotal_annual["lon"]

    # Prepare output array
    monthly_cTotal = xr.full_like(nbp_monthly, np.nan, dtype=np.float32)
    monthly_cTotal.name = "cTotal"

    for y in range(n_years):
        start = y * 12
        end = start + 12

        # January anchor = annual cTotal for that year
        jan_val = cTotal_annual.isel(time=y)
        monthly_cTotal[start] = jan_val

        # Forward integrate through months
        for m in range(1, 12):
            prev_cTotal = monthly_cTotal[start + m - 1]
            prev_nbp    = nbp_monthly[start + m - 1]
            dt_seconds  = seconds_per_month[m - 1]

            monthly_cTotal[start + m] = prev_cTotal + prev_nbp * dt_seconds
    
    # Save
    out_ds = xr.Dataset({"cTotal_monthly": monthly_cTotal}, coords={"time": nbp_monthly["time"], "lat": lat, "lon": lon})
    out_ds["cTotal_monthly"].attrs["units"] = "kgC m-2"  # or whatever units cTotal has
    out_ds.to_netcdf(output_file, engine="netcdf4")
    print(f"[OK] Wrote monthly cTotal to {output_file}")

def reconstruct_monthly_cTotal_dec_anchor(
    annual_cTotal_file, monthly_nbp_file, output_file, overwrite=True
):
    """
    Reconstruct monthly cTotal given:
      - annual cTotal = END-OF-YEAR (December) state for each year
      - monthly NBP (flux; e.g., kgC m-2 s-1)
    We integrate *backwards* within each year:
        cTotal[m] = cTotal[m+1] - NBP[m] * seconds_in_month[m]
    so that December matches the annual cTotal, and we step back to January.
    """
    output_file = Path(output_file)
    if output_file.exists() and not overwrite:
        print(f"[SKIP] {output_file.name} exists and overwrite=False")
        return

    # Open inputs
    annual_ds = xr.open_dataset(annual_cTotal_file, decode_times=False)
    nbp_ds    = xr.open_dataset(monthly_nbp_file, decode_times=False)

    cTotal_annual = annual_ds["cTotal"]   # (year, lat, lon); DECEMBER (end-of-year)
    nbp_monthly   = nbp_ds["nbp"]         # (month, lat, lon)

    # Basic checks
    n_years  = cTotal_annual.sizes["time"]
    n_months = nbp_monthly.sizes["time"]
    if n_months != n_years * 12:
        raise ValueError("Monthly NBP length must equal 12 × annual cTotal length")

    lat = cTotal_annual["lat"]
    lon = cTotal_annual["lon"]

    # Seconds per month (365-day calendar, no leap years)
    days_per_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=np.float32)
    seconds_per_month = xr.DataArray(
        days_per_month * 86400.0, dims=("month",), coords={"month": np.arange(12)}
    )

    # Prepare the output array on the monthly time grid
    monthly_time = nbp_monthly["time"]
    monthly_cTotal = xr.zeros_like(nbp_monthly, dtype=np.float32) * np.nan
    monthly_cTotal.name = "cTotal"

    # Backward integrate within each year:
    # Set December (index 11) to the annual cTotal (end-of-year),
    # then for m=10..0: cTotal[m] = cTotal[m+1] - nbp[m] * seconds_in_month[m]
    for y in range(n_years):
        start = y * 12
        # December index within that year
        dec_idx = start + 11

        # Anchor December to the annual cTotal (end-of-year state)
        dec_val = cTotal_annual.isel(time=y)
        monthly_cTotal[dec_idx] = dec_val

        # Step backward through months 10..0
        for m in range(10, -1, -1):
            cur_idx  = start + m
            next_idx = start + m + 1

            # NBP for month m (same sign convention as forward integration;
            # we subtract its monthly integral to go backward in time)
            nbp_m = nbp_monthly[cur_idx]
            dt_m  = float(seconds_per_month.values[m])  # seconds in month m

            monthly_cTotal[cur_idx] = monthly_cTotal[next_idx] - nbp_m * dt_m

    # Package and write
    out = xr.Dataset(
        {"cTotal_monthly": monthly_cTotal},
        coords={"time": monthly_time, "lat": lat, "lon": lon},
    )
    out["cTotal_monthly"].attrs.update({
        "long_name": "Monthly terrestrial carbon total (backward integrated from Dec anchor)",
        "units": "kgC m-2",  # adjust if your cTotal uses different units
    })

    out.to_netcdf(output_file, engine="netcdf4")
    print(f"[OK] Wrote monthly cTotal to {output_file}")

    # Close inputs
    annual_ds.close()
    nbp_ds.close()


