#!/usr/bin/env python3
"""
mask_with_tvt.py
----------------
Recursively apply tvt_mask to all NetCDF files in a directory.

For every *.nc file under --root:
  - open dataset
  - for each data variable that has lat/lon dimensions,
    set values to NaN where tvt_mask ∉ {0, 1, 2}
  - save back in-place, preserving encodings/metadata

Usage:
  python mask_with_tvt.py --root /path/to/root_dir [--dry-run]

Options:
  --dry-run   : only print which files *would* be modified, do not write.
"""

import argparse
from pathlib import Path
import shutil

import xarray as xr
import numpy as np


def load_tvt_mask():
    """Load tvt_mask and build a boolean land mask (lat, lon)."""
    tvt_path = (
        "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/"
        "NewModel/data/masks/tvt_mask.nc"
    )
    tvt = xr.open_dataset(tvt_path)["tvt_mask"]
    # True where in {0,1,2}, False otherwise
    land_mask = tvt.isin([0, 1, 2])
    # For speed, bring into memory once
    return land_mask.load()


def apply_mask_to_dataset(ds: xr.Dataset, land_mask: xr.DataArray) -> xr.Dataset:
    """
    Return a new Dataset with land_mask applied to all data_vars
    that have both 'lat' and 'lon' in their dims.
    """
    ds_masked = ds.copy()

    for var_name, da in ds.data_vars.items():
        dims = set(da.dims)
        if {"lat", "lon"}.issubset(dims):
            # xarray will broadcast land_mask (lat, lon) over time/other dims
            ds_masked[var_name] = da.where(land_mask)

    return ds_masked


def process_file(nc_path: Path, land_mask: xr.DataArray, dry_run: bool = False):
    print(f"Processing: {nc_path}")

    # Open dataset normally (decode CF etc.)
    ds = xr.open_dataset(nc_path)

    # Apply mask
    ds_masked = apply_mask_to_dataset(ds, land_mask)

    if dry_run:
        print("  [dry-run] Skipping write")
        ds.close()
        ds_masked.close()
        return

    # Preserve encodings as much as possible
    encoding = {}
    for name in ds.variables:
        enc = ds[name].encoding.copy()
        if enc:
            encoding[name] = enc

    # Write to a temporary file then atomically replace
    tmp_path = nc_path.with_suffix(nc_path.suffix + ".tmp")

    ds_masked.to_netcdf(tmp_path, encoding=encoding)
    ds.close()
    ds_masked.close()

    # Replace original file
    shutil.move(tmp_path, nc_path)
    print(f"  Saved masked file in-place: {nc_path}")


def main():
    ap = argparse.ArgumentParser(description="Apply tvt_mask to all .nc files under a root directory.")
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory to search recursively for *.nc files.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be processed, but do not modify anything.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root path is not a directory: {root}")

    print(f"Scanning for .nc files under: {root}")
    nc_files = sorted(root.rglob("*.nc"))
    print(f"Found {len(nc_files)} NetCDF files.")

    if not nc_files:
        return

    # Load mask once
    land_mask = load_tvt_mask()

    for nc_path in nc_files:
        process_file(nc_path, land_mask, dry_run=args.dry_run)


if __name__ == "__main__":
    main()