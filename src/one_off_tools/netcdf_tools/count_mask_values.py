#!/usr/bin/env python3
import sys
import numpy as np
import xarray as xr

def count_mask_values(mask: np.ndarray):
    # Ensure integer values
    mask = mask.astype(int)

    # Count unique values
    unique, counts = np.unique(mask, return_counts=True)
    return dict(zip(unique, counts))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python count_mask.py <mask_file.nc> <varname>")
        sys.exit(1)

    mask_file = sys.argv[1]
    varname = sys.argv[2]

    # Load mask from NetCDF file
    ds = xr.open_dataset(mask_file)
    if varname not in ds:
        print(f"Variable '{varname}' not found in {mask_file}")
        sys.exit(1)

    mask = ds[varname].values
    ds.close()

    counts = count_mask_values(mask)
    for val in [0, 1, 2]:
        print(f"Value {val}: {counts.get(val, 0)}")