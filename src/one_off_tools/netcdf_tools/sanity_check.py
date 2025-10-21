#!/usr/bin/env python3
import xarray as xr
from pathlib import Path
import numpy as np
import argparse
import sys

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))
from src.utils.tools import sanity_check

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Recursively check all .nc files for correct time length, variables, grid, units, and calendar."
    )
    p.add_argument("base_dir", type=Path, help="Directory to scan")
    args = p.parse_args()

    # Recursively collect all .nc files under base_dir
    nc_files = list(args.base_dir.rglob("*.nc"))
    if not nc_files:
        print(f"[INFO] No .nc files found under {args.base_dir}")
        sys.exit(0)

    print(f"[INFO] Found {len(nc_files)} NetCDF files under {args.base_dir}")

    # Run sanity check on all files
    sanity_check(nc_files)