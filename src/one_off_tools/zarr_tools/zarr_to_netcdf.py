#!/usr/bin/env python3
"""
zarr_to_netcdf.py
-----------------
Convert a Zarr dataset to NetCDF.

python /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/src/one_off_tools/zarr_tools/zarr_to_netcdf.py \
  --in-zarr lai_avh15c1_filled_30x30.zarr \
  --out-nc  lai_avh15c1_filled_30x30.nc \
  --consolidated
"""

import argparse
from pathlib import Path
import xarray as xr

def main():
    ap = argparse.ArgumentParser(description="Convert Zarr dataset to NetCDF.")
    ap.add_argument("--in-zarr", required=True, help="Path to input Zarr store.")
    ap.add_argument("--out-nc", required=True, help="Path to output NetCDF file.")
    ap.add_argument(
        "--consolidated",
        action="store_true",
        help="Open Zarr with consolidated metadata (if .zmetadata exists).",
    )
    args = ap.parse_args()

    in_path = Path(args.in_zarr)
    out_path = Path(args.out_nc)

    if not in_path.exists():
        raise FileNotFoundError(f"Input Zarr not found: {in_path}")

    print(f"Opening Zarr: {in_path}")
    ds = xr.open_zarr(in_path, consolidated=args.consolidated)

    # Optional: print a quick summary
    print(ds)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    print(f"Writing NetCDF: {out_path}")
    ds.to_netcdf(out_path)
    print("Done.")

if __name__ == "__main__":
    main()