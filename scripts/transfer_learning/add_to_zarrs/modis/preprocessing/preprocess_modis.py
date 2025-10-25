import xarray as xr
from pathlib import Path
import os, sys, re, time

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
out_dir_4 = data_dir / "4.rename_vars"
final_dir = project_root / "data/preprocessed/transfer_learning/modis_lai"
plot_dir  = base_dir / "val_plots"

for d in [out_dir_1, out_dir_2, out_dir_3, out_dir_4, final_dir, plot_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Barrier markers
done_dir = base_dir / "done_markers"
done_dir.mkdir(parents=True, exist_ok=True)

# SLURM vars
task_id   = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
num_arrays = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

# ---------------- Helper funcs ---------------- #
def rename_lai_vars(in_path: str | Path, out_path: str | Path | None = None, overwrite: bool = False) -> Path:
    """
    Rename dims/vars for LAI files while preserving original NetCDF
    chunking, dtype, and _FillValue. Adds compression to data variables
    without touching coordinate encodings.
    """
    in_path  = Path(in_path)
    out_path = Path(out_path) if out_path is not None else in_path

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; set overwrite=True.")

    # Use netcdf4 engine + decode_times=False so xarray populates .encoding
    ds = xr.open_dataset(in_path, engine="netcdf4", decode_times=False)

    try:
        # --- rename dims/coords if present ---
        ren = {}
        if "latitude"  in ds.dims: ren["latitude"]  = "lat"
        if "longitude" in ds.dims: ren["longitude"] = "lon"
        ds = ds.rename(ren)

        # --- rename data var 'LAI' if present ---
        if "LAI" in ds.data_vars:
            ds = ds.rename({"LAI": "modis_lai"})

        # --- build per-variable encodings, preserving existing settings ---
        encoding: dict[str, dict] = {}
        for name, var in ds.variables.items():
            e = {}

            # Start from existing on-disk encoding if available
            enc_src = getattr(var, "encoding", {})

            # Preserve common keys if they exist
            for k in ("dtype", "chunksizes", "fletcher32", "contiguous", "_FillValue"):
                if k in enc_src and enc_src[k] is not None:
                    e[k] = enc_src[k]

            if name in ds.data_vars:
                # Apply compression only to data variables
                # (preserve chunksizes if already present; don't invent new ones)
                e["zlib"] = True
                e["shuffle"] = True
                # don't blindly overwrite an existing complevel
                if "complevel" not in enc_src or enc_src["complevel"] is None:
                    e["complevel"] = 4
            else:
                # Coordinates: never set _FillValue or compression flags
                e.pop("_FillValue", None)
                for k in ("zlib", "shuffle", "complevel"):
                    e.pop(k, None)

            encoding[name] = e

        # Write, preserving encodings
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=encoding)
        return out_path

    finally:
        ds.close()

def combine_lai_years(
    files,
    out_path,
    expected_days_per_file=365,
    decode_times=False,
    start_date="2000-01-01",
    end_date="2021-12-31",
):
    """
    Concatenate per-year daily files and force the final time axis to be daily
    from start_date..end_date on a noleap calendar, with units
    'days since 1901-01-01 00:00'.
    Assumes each yearly file has exactly expected_days_per_file timesteps (noleap).
    """
    files = [Path(p) for p in files]
    if not files:
        raise ValueError("No files provided.")

    # Open and check time length per file
    dsets = []
    for fp in sorted(files):
        ds = xr.open_dataset(fp, chunks="auto", decode_times=decode_times)
        if "time" not in ds.dims:
            raise ValueError(f"No 'time' in {fp}")
        if int(ds.dims["time"]) != expected_days_per_file:
            raise ValueError(f"{fp}: expected {expected_days_per_file} time steps (noleap).")
        dsets.append(ds)

    combined = xr.concat(
        dsets, dim="time", data_vars="minimal", coords="minimal", compat="override", join="override"
    )

    # Target noleap daily axis
    tgt_time = xr.cftime_range(start=start_date, end=end_date, freq="D", calendar="noleap")
    if combined.sizes["time"] != len(tgt_time):
        raise AssertionError(
            f"Concatenated length {combined.sizes['time']} != target length {len(tgt_time)} "
            f"for {start_date}..{end_date} (daily, noleap)."
        )

    # Assign new coordinate and encoding
    combined = combined.assign_coords(time=("time", tgt_time))
    combined["time"].attrs.update({
        "units": "days since 1901-01-01 00:00",
        "calendar": "noleap",
    })
    combined["time"].encoding = {
        "units": "days since 1901-01-01 00:00",
        "calendar": "noleap",
    }

    # Preserve compression/chunking for data variables
    encoding = {"time": {"units": "days since 1901-01-01 00:00", "calendar": "noleap"}}
    for v in combined.data_vars:
        enc = {"zlib": True, "complevel": 4}
        arr = combined[v]
        if hasattr(arr.data, "chunks") and arr.data.chunks is not None:
            enc["chunksizes"] = tuple(int(cs[0]) for cs in arr.data.chunks)
        encoding[v] = enc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_netcdf(out_path, format="NETCDF4", engine="netcdf4", encoding=encoding)

    for ds in dsets:
        ds.close()
    return out_path


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
        raise FileExistsError(f"{out_path} exists; set overwrite=True.")

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

# ---------------- Shard work ---------------- #
list_files = sorted(raw_dir.glob("*.nc"))
to_process = slurm_shard(list_files)   # this shard's files only

# 1) trim → 2) regrid → 3) chunk → 4) rename
out_files_1 = run_function(to_process, out_dir_1, trim_lai)
out_files_2 = run_function(out_files_1, out_dir_2, regrid_file)
out_files_3 = run_function(out_files_2, out_dir_3, nccopy_chunk, arg1=clevel, arg2=LAT_CHUNK, arg3=LON_CHUNK)
out_files_4 = run_function(out_files_3, out_dir_4, rename_lai_vars)

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

    print("[BARRIER] All tasks done. Merging…")
    files_to_merge = sorted((out_dir_4).glob("*.nc"))
    out_path = final_dir / "modis_lai.nc"
    if out_path.exists() and not OVERWRITE:
        print(f"{out_path} already exists. Use OVERWRITE=True to overwrite.")
    else:
        combine_lai_years(
            files_to_merge, out_path,
            expected_days_per_file=365,  # noleap daily files
            decode_times=False,
            start_date="2000-01-01",
            end_date="2021-12-31",
        )
        print(f"[OK] Merge complete → {out_path}")

    # Validation plots (only for merged output)
    first_timestep(out_path, plot_dir / "preprocessed" / "first_timestep", title=out_path.stem, varname = "modis_lai")
    finite_mask(out_path, plot_dir / "preprocessed" / "finite_mask", title=out_path.stem, varname = "modis_lai")
    print_chunk_sizes(out_path)
    
    # Raw
    first_timestep(list_files[0], plot_dir / "raw" / "first_timestep", title=list_files[0].stem, varname = "LAI")
    finite_mask(list_files[0], plot_dir / "raw" / "finite_mask", title=list_files[0].stem, varname = "LAI")

'''    # trim
    first_timestep(out_files_1[0], plot_dir / "trimmed" / "first_timestep", title=out_files_1[0].stem, varname= "LAI")
    finite_mask(out_files_1[0], plot_dir / "trimmed" / "finite_mask", title=out_files_1[0].stem, varname= "LAI")

    # regrid
    first_timestep(out_files_2[0], plot_dir / "regrid" / "first_timestep", title=out_files_2[0].stem, varname= "LAI")
    finite_mask(out_files_2[0], plot_dir / "regrid" / "finite_mask", title=out_files_2[0].stem, varname= "LAI")
    
    # chunk
    first_timestep(out_files_3[0], plot_dir / "chunk" / "first_timestep", title=out_files_3[0].stem, varname= "LAI")
    finite_mask(out_files_3[0], plot_dir / "chunk" / "finite_mask", title=out_files_3[0].stem, varname= "LAI")

    # rename
    first_timestep(out_files_4[0], plot_dir / "renamed" / "first_timestep", title=out_files_4[0].stem, varname="modis_lai")
    finite_mask(out_files_4[0], plot_dir / "renamed" / "finite_mask", title=out_files_4[0].stem, varname="modis_lai")'''

print("script finished successfully")