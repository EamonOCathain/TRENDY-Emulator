#!/usr/bin/env python3
"""
mask_with_tvt.py
----------------
Recursively apply tvt_mask to selected NetCDF files in a directory.

For every *.nc file under --root:
  - optionally skip unless filename contains one of {cVeg,cSoil,cLitter}
  - open dataset
  - for each data variable with lat/lon dims, mask values where tvt_mask ∉ {0,1,2}
  - save back in-place with standard NetCDF encoding (no special preservation)

Usage:
  python mask_with_tvt.py --root /path/to/dir [--only-carbon] [--dry-run]
"""

import argparse
from pathlib import Path
import shutil

import xarray as xr
import numpy as np  # (currently unused, but fine to keep)


CARBON_KEYWORDS = ("cVeg", "cSoil", "cLitter")


def load_tvt_mask():
    """Load tvt_mask and build a boolean land mask (lat, lon)."""
    tvt_path = (
        "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/"
        "NewModel/data/masks/tvt_mask.nc"
    )
    tvt = xr.open_dataset(tvt_path)["tvt_mask"]
    land_mask = tvt.isin([0, 1, 2])
    return land_mask.load()


def apply_mask_to_dataset(ds: xr.Dataset, land_mask: xr.DataArray) -> xr.Dataset:
    """Return new dataset with land_mask applied to vars with lat/lon dims."""
    ds_masked = ds.copy()
    for name, da in ds.data_vars.items():
        if {"lat", "lon"}.issubset(da.dims):
            ds_masked[name] = da.where(land_mask)
    return ds_masked


def file_is_carbon_related(path: Path) -> bool:
    """Return True if filename contains cVeg, cSoil, or cLitter."""
    return any(k in path.name for k in CARBON_KEYWORDS)


def process_file(nc_path: Path, land_mask: xr.DataArray, dry_run: bool = False):
    print(f"Processing: {nc_path}")
    ds = xr.open_dataset(nc_path)
    ds_masked = apply_mask_to_dataset(ds, land_mask)

    if dry_run:
        print("  [dry-run] Skipping write")
        ds.close()
        ds_masked.close()
        return

    # Write to a temporary file with default NetCDF encodings,
    # then atomically replace the original file.
    tmp_path = nc_path.with_suffix(nc_path.suffix + ".tmp")
    ds_masked.to_netcdf(tmp_path)

    ds.close()
    ds_masked.close()

    shutil.move(tmp_path, nc_path)
    print(f"  Saved masked file in-place: {nc_path}")


def main():
    ap = argparse.ArgumentParser(description="Apply tvt_mask to selected NetCDF files.")
    ap.add_argument("--root", required=True, help="Root directory to search for .nc files.")
    ap.add_argument("--dry-run", action="store_true", help="Only print actions, do not write.")
    ap.add_argument(
        "--only-carbon",
        action="store_true",
        help="Only process files containing {cVeg,cSoil,cLitter} in filename.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    print(f"Scanning for .nc files under: {root}")
    nc_files = sorted(root.rglob("*.nc"))
    print(f"Found {len(nc_files)} NetCDF files.")

    if args.only_carbon:
        nc_files = [p for p in nc_files if file_is_carbon_related(p)]
        print(f"Filtered to {len(nc_files)} carbon-related files (cVeg/cSoil/cLitter).")

    if not nc_files:
        print("No files to process.")
        return

    land_mask = load_tvt_mask()

    for nc_path in nc_files:
        process_file(nc_path, land_mask, dry_run=args.dry_run)


if __name__ == "__main__":
    main()