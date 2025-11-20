#!/usr/bin/env python3
"""
Collapse Wang & Mao multilayer soil moisture to a single time-lat-lon field.

Usage:
  python collapse_mrsol_layers.py \
      --in /path/to/mrsol_olc.nc \
      --out /path/to/mrsol_olc_collapsed.nc
"""

import argparse
from pathlib import Path
import xarray as xr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_nc",  required=True,
                    help="Input layered mrsol file")
    ap.add_argument("--out", dest="out_nc", required=True,
                    help="Output collapsed file")
    args = ap.parse_args()

    in_nc  = Path(args.in_nc)
    out_nc = Path(args.out_nc)

    ds = xr.open_dataset(in_nc)

    # Original variables
    da          = ds["mrsol"]          # (time, depth, lat, lon)
    depth_bnds  = ds["depth_bnds"]     # (depth, bnds)

    # Layer thickness = upper_bound - lower_bound
    # convention: depth_bnds(depth, 0) = upper, (depth, 1) = lower
    thickness = depth_bnds.isel(bnds=1) - depth_bnds.isel(bnds=0)  # (depth)

    # Thickness-weighted vertical mean (keeps units m3 m-3)
    mrsol_weighted = da.weighted(thickness).mean(dim="depth")      # (time, lat, lon)

    # Build output dataset, preserving coordinates and global attrs
    out = xr.Dataset(
        {
            "mrsol": mrsol_weighted
        },
        coords={
            "time": ds["time"],
            "lat":  ds["lat"],
            "lon":  ds["lon"],
        },
        attrs=ds.attrs,  # copy global attrs
    )

    # Copy time_bnds if present so ILAMB can find it
    if "time_bnds" in ds.variables:
        out["time_bnds"] = ds["time_bnds"]
        # Ensure time coord points to the bounds variable
        time_attrs = dict(ds["time"].attrs)
        time_attrs.setdefault("bounds", "time_bnds")
        out["time"].attrs.update(time_attrs)

    # Update variable attributes
    out["mrsol"].attrs.update({
        "long_name": "total column soil moisture (thickness-weighted mean)",
        "units":     da.attrs.get("units", "m3 m-3"),
        "note":      "Depth-collapsed from multilayer Wang & Mao product "
                     "using layer thickness from depth_bnds.",
    })

    out.to_netcdf(out_nc)
    print(f"Saved collapsed file to {out_nc}")

if __name__ == "__main__":
    main()