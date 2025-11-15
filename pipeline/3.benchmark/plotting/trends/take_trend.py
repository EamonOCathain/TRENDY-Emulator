#!/usr/bin/env python3
"""
Compute per-pixel trend slope for multiple NetCDFs.

Features:
 - Removes mean seasonal cycle before regression.
 - Detects annual data repeated monthly (cVeg, cLitter, cSoil) and collapses to yearly means.
 - Fits linear trend per pixel via xarray.polyfit.
 - Saves ONLY the slope directly in out_dir as <orig_stem>_slope.nc.
 - Preserves the original variable name in the output.
 - Adds a single time coordinate (2023-12-31).
 - Standardises slope units to per year, with correct physical units.
 - Distributes work across SLURM array tasks using slurm_shard.
"""

import argparse
from pathlib import Path
import re
import numpy as np
import xarray as xr
import sys
import warnings
warnings.simplefilter("ignore", np.RankWarning)

# ---------------- Project imports ---------------- #
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))
from src.utils.tools import slurm_shard

# Variable detection
VAR_NAMES = ("cVeg", "cLitter", "cTotal", "cSoil", "mrso", "lai")
REPEATED_ANNUAL_VARS = {"cVeg", "cLitter", "cSoil"}
VAR_UNITS = {
    "cVeg": "kg m-2 yr-1",
    "cSoil": "kg m-2 yr-1",
    "cLitter": "kg m-2 yr-1",
    "cTotal": "kg m-2 yr-1",
    "mrso": "kg m-2 yr-1",
    "lai": "m m-2 yr-1",
}

# ----------------------------------------------------------------------------- #
# Core helpers
# ----------------------------------------------------------------------------- #

def deseasonalise(da: xr.DataArray) -> xr.DataArray:
    """Remove mean seasonal cycle (monthly climatology) if available, else remove overall mean."""
    if "time" in da.dims and hasattr(da.time, "dt") and hasattr(da.time.dt, "month"):
        clim = da.groupby("time.month").mean("time", skipna=True)
        return da.groupby("time.month") - clim
    return da - da.mean("time", skipna=True)

def collapse_repeated_annual_if_needed(da: xr.DataArray, varname: str) -> tuple[xr.DataArray, bool]:
    """
    If var is in REPEATED_ANNUAL_VARS and time looks monthly (T % 12 == 0),
    collapse to annual by averaging each 12-month block.
    Returns (possibly-collapsed da, collapsed_flag).
    """
    if varname not in REPEATED_ANNUAL_VARS:
        return da, False

    T = int(da.sizes.get("time", 0))
    if T % 12 != 0 or T == 0:
        return da, False

    Y = T // 12
    arr = da.transpose("time", ...).values  # [T, ...]
    arr = arr.reshape(Y, 12, *arr.shape[1:])  # [Y, 12, ...]
    with np.errstate(all="ignore"):
        arr_y = np.nanmean(arr, axis=1)        # [Y, ...]

    coords = {k: v for k, v in da.coords.items() if k != "time"}
    da_y = xr.DataArray(
        arr_y,
        dims=("time",) + tuple(d for d in da.dims if d != "time"),
        coords={"time": np.arange(Y), **coords},
        name=da.name,
        attrs=dict(da.attrs) if da.attrs else {},
    )
    da_y.attrs["note"] = "Collapsed monthly (repeated-annual) to annual means for trend fitting."
    return da_y, True

def polyfit_slope(da: xr.DataArray) -> xr.DataArray:
    """Return the slope from a 1st-degree polyfit along time (per-native-time-step)."""
    beta = da.polyfit(dim="time", deg=1, skipna=True)["polyfit_coefficients"]
    slope = beta.sel(degree=1).reset_coords("degree", drop=True)
    slope.attrs.update({
        "long_name": "Linear trend slope (per time step) after seasonal removal",
        "description": "Slope units follow the native 'time' step (year if annual, month if monthly).",
    })
    return slope

def infer_var_from_name(path: Path) -> str | None:
    """Return first matching variable name from VAR_NAMES anywhere in the filename."""
    lower = path.name.lower()
    for v in VAR_NAMES:
        if v.lower() in lower:
            return v
    return None

def process_one(in_path: Path, out_dir: Path, varname: str, overwrite: bool) -> None:
    ds = xr.open_dataset(in_path, decode_times=True, use_cftime=True)
    if varname not in ds:
        print(f"[SKIP] {in_path.name}: variable '{varname}' not found.")
        return
    da = ds[varname]

    # Collapse annual-repeated vars if applicable
    da_proc, collapsed = collapse_repeated_annual_if_needed(da, varname)

    # Deseasonalise
    anom = deseasonalise(da_proc)

    # Slope per native time step
    slope_native = polyfit_slope(anom)

    # Standardise to per year (multiply by 12 if monthly-like)
    T = int(anom.sizes.get("time", 0))
    if collapsed or T == 123:
        slope_per_year = slope_native
        cadence = "annual"
    elif T % 12 == 0 and T > 0:
        slope_per_year = slope_native * 12.0
        cadence = "monthly (×12 rescaled)"
    else:
        slope_per_year = slope_native
        cadence = "unknown (not rescaled)"

    # Add single timestamp
    slope_3d = slope_per_year.expand_dims(time=[np.datetime64("2023-12-31")])
    slope_3d = slope_3d.rename(varname)

    # Assign correct units
    slope_3d.attrs["units"] = VAR_UNITS.get(varname, "avg. delta / year")
    slope_3d.attrs["conversion_note"] = f"Standardised to per year from {cadence} cadence."
    slope_3d.attrs["derived_from"] = varname
    slope_3d.attrs["long_name"] = f"Linear trend slope of deseasonalised {varname}"

    # Save
    stem = in_path.stem
    out_file = out_dir / f"{stem}_slope.nc"
    if not overwrite and out_file.exists():
        print(f"[SKIP] {in_path.name}: output already exists.")
        return

    slope_3d.to_dataset(name=varname).to_netcdf(out_file)
    print(f"[OK] {in_path.name} → {out_file.name}")

# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Compute per-pixel linear trend slope after deseasonalisation. "
                    "Handles annual variables repeated monthly and uses slurm_shard."
    )
    ap.add_argument("--in_dir", type=Path, required=True, help="Directory containing input NetCDFs")
    ap.add_argument("--out_dir", type=Path, required=True, help="Directory to write outputs")
    ap.add_argument("--glob", default="*.nc", help="Filename glob pattern (default: *.nc)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    overwrite: bool = args.overwrite
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(in_dir.rglob(args.glob))
    tasks: list[tuple[Path, str]] = []
    for f in all_files:
        varname = infer_var_from_name(f)
        if varname:
            tasks.append((f, varname))

    if not tasks:
        print(f"[WARN] No matching files found in {in_dir} for vars {VAR_NAMES}.")
        return

    tasks_this = slurm_shard(tasks)
    print(f"[INFO] Processing {len(tasks_this)}/{len(tasks)} files in this shard.")

    for f, varname in tasks_this:
        try:
            process_one(f, out_dir, varname, overwrite)
        except Exception as e:
            print(f"[ERR] {f.name}: {e}")

if __name__ == "__main__":
    main()