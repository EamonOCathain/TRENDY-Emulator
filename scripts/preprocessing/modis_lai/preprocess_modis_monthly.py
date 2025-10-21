#!/usr/bin/env python3
# scripts/preprocessing/modis_lai/make_monthly_from_daily.py
import xarray as xr
from pathlib import Path
import os, sys, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.utils.visualisation import finite_mask, first_timestep
from src.utils.preprocessing import (
    nccopy_chunk,
    print_chunk_sizes,
    run_function,
    regrid_file,
)
from src.utils.tools import slurm_shard

# ---------------- Settings ---------------- #
OVERWRITE = False
clevel = 4
LAT_CHUNK = 5
LON_CHUNK = 5

# Directories
raw_dir   = Path("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d25_daily/glass/v60/Data/LAI")
base_dir  = project_root / "scripts/preprocessing/modis_lai"
data_dir  = base_dir / "data"
out_dir_1 = data_dir / "1.trimmed"
out_dir_2 = data_dir / "2.regrid"
out_dir_3 = data_dir / "3.chunk"
out_dir_4 = data_dir / "4.monthly"
out_dir_5 = data_dir / "5.rename_vars"
final_dir = project_root / "data/preprocessed/transfer_learning/modis_lai_monthly"
plot_dir  = base_dir / "val_plots"

for d in [out_dir_1, out_dir_2, out_dir_3, out_dir_4, out_dir_5, final_dir, plot_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Barrier markers
done_dir = base_dir / "done_markers_monthly"
done_dir.mkdir(parents=True, exist_ok=True)

# SLURM vars
task_id    = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
num_arrays = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

# ---------------- Helper funcs ---------------- #
def trim_lai(path, out_path=None, overwrite=False):
    """
    Trim a NetCDF file along the time axis:
      - If time length is 366, drop the last timestep (day 366).
      - Otherwise, leave unchanged.
    Writes to out_path (defaults to overwrite the input).
    """
    path = Path(path)
    out_path = Path(out_path) if out_path is not None else path

    if out_path.exists() and not overwrite:
        return out_path

    with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
        if "time" not in ds.dims:
            raise ValueError("Dataset has no 'time' dimension.")

        if ds.sizes["time"] == 366:
            ds_trim = ds.isel(time=slice(0, 365))
        else:
            ds_trim = ds

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds_trim.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", mode="w")
        return out_path

def monthly_mean_file(in_path: str | Path, out_path: str | Path | None = None, *, overwrite: bool = False) -> Path:
    """
    Convert a single **daily** LAI file to **monthly mean**.
    - Decodes CF times (noleap is fine).
    - Resamples to month starts ("MS") and averages.
    - Preserves dtype; applies zlib compression to data vars.
    - Writes to out_path (defaults beside input with _monthly suffix).
    """
    in_path  = Path(in_path)
    if out_path is None:
        out_path = in_path.with_name(in_path.stem + "_monthly.nc")
    out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        return out_path

    # decode_times=True -> cftime objects if noleap calendar
    ds = xr.open_dataset(in_path, engine="netcdf4", decode_times=True)

    try:
        # Identify LAI var (before rename stage it’s usually "LAI")
        lai_name = "LAI" if "LAI" in ds.data_vars else next(
            (v for v in ds.data_vars if v.lower() == "lai"), None
        )
        if lai_name is None:
            raise KeyError(f"Could not find LAI variable in {in_path}; data_vars={list(ds.data_vars)}")

        # Ensure time is present and sorted
        if "time" not in ds.dims:
            raise ValueError(f"No 'time' dimension in {in_path}")
        ds = ds.sortby("time")

        # Monthly mean at month-start labels
        m = ds.resample(time="MS").mean(keep_attrs=True)

        # Keep only LAI + coords (avoid accidental other vars)
        keep = [lai_name]
        m = m[keep]

        # Set encodings: compress data vars; keep coords uncompressed
        encoding = {"time": {}}
        for v in m.data_vars:
            enc = {"zlib": True, "shuffle": True, "complevel": clevel}
            # if a chunksizes exists on var, preserve it
            if hasattr(m[v].data, "chunks") and m[v].data.chunks is not None:
                # make sure chunksizes is a tuple of ints per dimension
                enc["chunksizes"] = tuple(int(c[0]) for c in m[v].data.chunks)
            encoding[v] = enc

        out_path.parent.mkdir(parents=True, exist_ok=True)
        m.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=encoding)
        return out_path
    finally:
        ds.close()

def rename_lai_vars(in_path: str | Path, out_path: str | Path | None = None, overwrite: bool = False) -> Path:
    """
    Rename dims/vars for LAI files while preserving original NetCDF
    chunking, dtype, and _FillValue. Adds compression to data variables
    without touching coordinate encodings.
    - Renames latitude/longitude → lat/lon
    - Renames LAI → modis_lai
    """
    in_path  = Path(in_path)
    out_path = Path(out_path) if out_path is not None else in_path

    if out_path.exists() and not overwrite:
        return out_path

    ds = xr.open_dataset(in_path, engine="netcdf4", decode_times=False)

    try:
        # rename dims
        ren = {}
        if "latitude"  in ds.dims: ren["latitude"]  = "lat"
        if "longitude" in ds.dims: ren["longitude"] = "lon"
        ds = ds.rename(ren)

        # rename var
        if "LAI" in ds.data_vars:
            ds = ds.rename({"LAI": "modis_lai"})
        elif "lai" in ds.data_vars:
            ds = ds.rename({"lai": "modis_lai"})

        # Build encodings
        encoding: dict[str, dict] = {}
        for name, var in ds.variables.items():
            e = {}
            enc_src = getattr(var, "encoding", {})

            for k in ("dtype", "chunksizes", "fletcher32", "contiguous", "_FillValue"):
                if k in enc_src and enc_src[k] is not None:
                    e[k] = enc_src[k]

            if name in ds.data_vars:
                e["zlib"] = True
                e["shuffle"] = True
                if "complevel" not in enc_src or enc_src.get("complevel") is None:
                    e["complevel"] = clevel
            else:
                e.pop("_FillValue", None)
                for k in ("zlib", "shuffle", "complevel"):
                    e.pop(k, None)

            encoding[name] = e

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=encoding)
        return out_path
    finally:
        ds.close()

def combine_lai_monthlies(
    files,
    out_path,
    start_date="2000-01-01",
    end_date="2021-12-01",
):
    """
    Concatenate per-year **monthly** files and force the final time axis to be
    monthly from start_date..end_date (month starts) on a noleap calendar.

    Expects each input to already be monthly (12 steps per year).
    Works whether var is LAI or already renamed modis_lai (renaming occurs after).
    """
    files = [Path(p) for p in files]
    if not files:
        raise ValueError("No files provided for monthly merge.")

    dsets = []
    for fp in sorted(files):
        ds = xr.open_dataset(fp, engine="netcdf4", decode_times=True)
        if "time" not in ds.dims:
            ds.close()
            raise ValueError(f"No 'time' in {fp}")
        dsets.append(ds)

    combined = xr.concat(
        dsets, dim="time", data_vars="minimal", coords="minimal", compat="override", join="override"
    )

    # Target monthly noleap axis
    tgt_time = xr.cftime_range(start=start_date, end=end_date, freq="MS", calendar="noleap")

    # Reindex to the desired continuous monthly axis
    combined = combined.sortby("time").reindex(time=tgt_time, method=None)

    # Set CF attrs for time
    combined = combined.assign_coords(time=("time", tgt_time))
    combined["time"].attrs.update({
        "units": "days since 1901-01-01 00:00",
        "calendar": "noleap",
    })
    combined["time"].encoding = {
        "units": "days since 1901-01-01 00:00",
        "calendar": "noleap",
    }

    # Compression for data vars
    encoding = {"time": {"units": "days since 1901-01-01 00:00", "calendar": "noleap"}}
    for v in combined.data_vars:
        enc = {"zlib": True, "shuffle": True, "complevel": clevel}
        if hasattr(combined[v].data, "chunks") and combined[v].data.chunks is not None:
            enc["chunksizes"] = tuple(int(cs[0]) for cs in combined[v].data.chunks)
        encoding[v] = enc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_netcdf(out_path, format="NETCDF4", engine="netcdf4", encoding=encoding)

    for ds in dsets:
        ds.close()
    return out_path

# ---------------- Shard work ---------------- #
list_files = sorted(raw_dir.glob("*.nc"))
to_process = slurm_shard(list_files)   # this shard's files only

# 1) trim → 2) regrid → 3) chunk → 4) monthly-mean → 5) rename
out_files_1 = run_function(to_process, out_dir_1, trim_lai)
out_files_2 = run_function(out_files_1, out_dir_2, regrid_file)
out_files_3 = run_function(out_files_2, out_dir_3, nccopy_chunk, arg1=clevel, arg2=LAT_CHUNK, arg3=LON_CHUNK)
out_files_4 = run_function(out_files_3, out_dir_4, monthly_mean_file)
out_files_5 = run_function(out_files_4, out_dir_5, rename_lai_vars)

# Mark this task done (after successful processing)
marker = done_dir / f"task_{task_id}.done"
marker.write_text("ok\n")

# ---------------- Last task merges ---------------- #
if task_id == (num_arrays - 1):
    print("[BARRIER] Waiting for all tasks to finish …")
    while True:
        done = list(done_dir.glob("task_*.done"))
        if len(done) >= num_arrays:
            break
        time.sleep(30)

    print("[BARRIER] All tasks done. Merging monthly files…")
    files_to_merge = sorted(out_dir_5.glob("*.nc"))
    out_path = final_dir / "modis_lai_monthly.nc"
    if out_path.exists() and not OVERWRITE:
        print(f"{out_path} already exists. Set OVERWRITE=True to overwrite.")
    else:
        combine_lai_monthlies(
            files_to_merge, out_path,
            start_date="2000-01-01",
            end_date="2021-12-01",
        )
        print(f"[OK] Monthly merge complete → {out_path}")

    # Validation plots (merged output, variable renamed to modis_lai)
    first_timestep(out_path, plot_dir / "preprocessed_monthly" / "first_timestep",
                   title=out_path.stem, varname="modis_lai")
    finite_mask(out_path, plot_dir / "preprocessed_monthly" / "finite_mask",
                title=out_path.stem, varname="modis_lai")
    print_chunk_sizes(out_path)

    # Raw sample plots (first input file), for sanity
    if list_files:
        first_timestep(list_files[0], plot_dir / "raw" / "first_timestep",
                       title=list_files[0].stem, varname="LAI")
        finite_mask(list_files[0], plot_dir / "raw" / "finite_mask",
                    title=list_files[0].stem, varname="LAI")

print("script finished successfully")