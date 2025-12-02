#!/usr/bin/env python3
"""
make_latband_mask.py

Create a 2D lat-lon mask on the same 360x720 grid as a template NetCDF file,
with zones:

  - Tropics     (-15,  15)          -> code 1
  - Subtropics  (±15..±35)          -> code 2
  - Temperate   (±35..±60)          -> code 3
  - Boreal      (60..90, NH only)   -> code 4

All longitudes -180..180 (global) are included for each band.

Usage:
  python make_latband_mask.py \
      --template /path/to/cLitter.nc \
      --output   /path/to/latband_mask.nc
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def build_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Build a (lat, lon) mask array with integer codes for climate zones.

    Input:
      lat: 1D array, shape (nlat,)
      lon: 1D array, shape (nlon,)  in degrees, 0..360 (like your file)

    Output:
      mask: 2D array, shape (nlat, nlon), dtype=int8
    """

    # Broadcast to 2D grids: LAT[lat,lon], LON[lat,lon]
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")  # LAT: (nlat,nlon), LON: (nlat,nlon)

    # Convert LON from [0,360) to [-180,180) so the zone definitions match
    LON_wrapped = ((LON + 180.0) % 360.0) - 180.0

    # Start with all zeros (0 = outside defined zones, e.g. Antarctic)
    mask = np.zeros_like(LAT, dtype=np.int8)

    # Define zones: (code, label, lat_min, lat_max, lon_min, lon_max)
    # We use half-open intervals [lat_min, lat_max) to avoid overlaps.
    ZONES = [
        # Tropics: -15 to 15
        (1, "tropics",    -15.0,  15.0, -180.0,  180.0),
        # Subtropics: 15..35 (NH) and -35..-15 (SH)
        (2, "subtropics",  15.0,  35.0, -180.0,  180.0),
        (2, "subtropics", -35.0, -15.0, -180.0,  180.0),
        # Temperate: 35..60 (NH) and -60..-35 (SH)
        (3, "temperate",   35.0,  60.0, -180.0,  180.0),
        (3, "temperate",  -60.0, -35.0, -180.0,  180.0),
        # Boreal: 60..90 (NH only)
        (4, "boreal",      60.0,  90.0, -180.0,  180.0),
    ]

    for code, label, lat_min, lat_max, lon_min, lon_max in ZONES:
        # Build boolean mask for this zone
        lat_cond = (LAT >= lat_min) & (LAT < lat_max)
        lon_cond = (LON_wrapped >= lon_min) & (LON_wrapped < lon_max)
        zone_mask = lat_cond & lon_cond

        # Assign zone code (later zones overwrite earlier ones if they overlap)
        mask[zone_mask] = code

    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, required=True,
                    help="Template NetCDF file with lat/lon (e.g. cLitter.nc)")
    ap.add_argument("--output", default="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/zones_mask.nc", type=Path, required=True,
                    help="Path for output mask NetCDF")
    ap.add_argument("--varname", type=str, default="zones_mask",
                    help="Name of mask variable in output file")
    args = ap.parse_args()

    # Open template and grab lat/lon
    ds = xr.open_dataset(args.template)
    if "lat" not in ds or "lon" not in ds:
        raise RuntimeError("Template file must contain 'lat' and 'lon' coordinates.")

    lat = ds["lat"].values  # shape (360,)
    lon = ds["lon"].values  # shape (720,)

    mask = build_mask(lat, lon)  # shape (360, 720)

    # Build output dataset
    ds_out = xr.Dataset(
        {
            args.varname: (("lat", "lon"), mask.astype("int8")),
        },
        coords={
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )

    # Add some metadata
    ds_out[args.varname].attrs.update(
        {
            "long_name": "Latitudinal climate band mask",
            "description": (
                "Integer mask: 0=other/polar, 1=Tropics, 2=Subtropics, "
                "3=Temperate, 4=Boreal"
            ),
            "bands": "1=tropics, 2=subtropics, 3=temperate, 4=boreal",
        }
    )
    ds_out["lat"].attrs.update(ds["lat"].attrs)
    ds_out["lon"].attrs.update(ds["lon"].attrs)

    # Write NetCDF
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(args.output)
    print(f"Wrote mask to {args.output}")


if __name__ == "__main__":
    main()