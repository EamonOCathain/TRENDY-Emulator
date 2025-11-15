#!/usr/bin/env python3
"""
nee_from_gpp_rh_ra.py

Compute ecosystem NEE = GPP - RH - RA from three NetCDF files
with identical time/lat/lon.

Sign convention:
    - GPP, RH, RA are all positive-valued fluxes
    - Positive NEE means a NET CARBON SINK (ecosystem gains carbon)
    - Negative NEE means a NET CARBON SOURCE (ecosystem loses carbon)

Usage:
  python nee_from_gpp_rh_ra.py gpp.nc rh.nc ra.nc out.nc

Assumptions:
- Variables are named 'gpp', 'rh', and 'ra'
- Dims/coords match exactly across all three files
- Output is uncompressed
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import xarray as xr
import argparse
import re

# Only allow encoding keys netCDF4 backend understands
NC4_ALLOWED = {
    "_FillValue", "contiguous", "shuffle", "complevel", "szip_coding",
    "compression", "quantize_mode", "significant_digits", "endian",
    "fletcher32", "zlib", "dtype", "least_significant_digit",
    "szip_pixels_per_block", "chunksizes", "blosc_shuffle"
}

def filter_nc4_encoding(enc: dict | None) -> dict:
    if not enc:
        return {}
    return {k: v for k, v in enc.items() if k in NC4_ALLOWED}

def check_same_coords(a: xr.DataArray, b: xr.DataArray, name_a: str, name_b: str):
    """Raise if dims/sizes/coords of two DataArrays differ."""
    if a.dims != b.dims:
        raise ValueError(f"Dim order mismatch between {name_a} and {name_b}: {a.dims} vs {b.dims}")
    for d in a.dims:
        if a.sizes[d] != b.sizes[d]:
            raise ValueError(
                f"Size mismatch in dim '{d}' between {name_a} and {name_b}: "
                f"{a.sizes[d]} vs {b.sizes[d]}"
            )
        if not np.array_equal(a.coords[d].values, b.coords[d].values):
            raise ValueError(
                f"Coordinate values differ on dim '{d}' between {name_a} and {name_b}."
            )

def find_var_file(directory: Path, pattern: str) -> Path:
    """Return the first file in directory whose name contains pattern (case-insensitive)."""
    for p in directory.iterdir():
        if p.is_file() and re.search(pattern, p.name, re.IGNORECASE):
            return p
    raise FileNotFoundError(f"No file containing '{pattern}' found in {directory}")

def main():

    ap = argparse.ArgumentParser(
        description="Compute ecosystem NEE = GPP - RH - RA (positive = net carbon sink)."
    )
    ap.add_argument("--in-dir",  type=Path, required=True,
                    help="Directory containing ENSMEAN NetCDF files")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory to write NEE NetCDF file into")
    ap.add_argument("--scenario", type=str, required=True,
                    help="Scenario name, e.g. S0, S1, S2, S3")
    ap.add_argument("--out-name", type=str, default=None,
                    help="Output filename (default: ENSMEAN_<scenario>_nee.nc)")
    args = ap.parse_args()

    scenario = args.scenario.upper()

    in_dir  = args.in_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build expected filenames
    gpp_path = in_dir / f"ENSMEAN_{scenario}_gpp.nc"
    rh_path  = in_dir / f"ENSMEAN_{scenario}_rh.nc"
    ra_path  = in_dir / f"ENSMEAN_{scenario}_ra.nc"

    # Default output name
    if args.out_name is None:
        out_path = out_dir / f"ENSMEAN_{scenario}_nee.nc"
    else:
        out_path = out_dir / args.out_name

    # Existence checks (nice errors)
    for p, name in [(gpp_path, "GPP"), (rh_path, "RH"), (ra_path, "RA")]:
        if not p.exists():
            raise FileNotFoundError(
                f"Expected {name} file not found: {p}\n"
                f"(Looked for ENSMEAN_{scenario}_{name.lower()}.nc)"
            )

    print(f"Using GPP: {gpp_path}")
    print(f"Using RH : {rh_path}")
    print(f"Using RA : {ra_path}")
    print(f"Output   : {out_path}")

    ds_gpp = xr.open_dataset(gpp_path, decode_times=True, use_cftime=True)
    ds_rh  = xr.open_dataset(rh_path,  decode_times=True, use_cftime=True)
    ds_ra  = xr.open_dataset(ra_path,  decode_times=True, use_cftime=True)

    if "gpp" not in ds_gpp.data_vars:
        raise ValueError(f"'gpp' not found in {gpp_path}. Available: {list(ds_gpp.data_vars)}")
    if "rh" not in ds_rh.data_vars:
        raise ValueError(f"'rh' not found in {rh_path}. Available: {list(ds_rh.data_vars)}")
    if "ra" not in ds_ra.data_vars:
        raise ValueError(f"'ra' not found in {ra_path}. Available: {list(ds_ra.data_vars)}")

    gpp = ds_gpp["gpp"]
    rh  = ds_rh["rh"]
    ra  = ds_ra["ra"]

    # Strict coord checks (use gpp as reference)
    check_same_coords(gpp, rh, "gpp", "rh")
    check_same_coords(gpp, ra, "gpp", "ra")

    # Compute NEE = GPP - RH - RA (positive = net carbon sink)
    gpp_f = gpp.where(np.isfinite(gpp)).astype("float64")
    rh_f  = rh.where(np.isfinite(rh)).astype("float64")
    ra_f  = ra.where(np.isfinite(ra)).astype("float64")

    nee = (gpp_f - rh_f - ra_f).rename("nee")
    nee = nee.where(np.isfinite(gpp) & np.isfinite(rh) & np.isfinite(ra))

    # Attributes
    units = gpp.attrs.get("units", rh.attrs.get("units", ra.attrs.get("units", "")))
    now_utc = datetime.now(timezone.utc).isoformat()
    nee.attrs.update({
        "long_name": "net ecosystem exchange (positive = net carbon sink)",
        # Do NOT claim the CF standard_name with opposite sign convention
        # "standard_name": "net_ecosystem_exchange",
        "units": units,
        "history": f"{now_utc}: computed nee = gpp - rh - ra (positive = sink)",
        "source": "derived",
        "sign_convention": (
            "positive = net carbon sink (ecosystem gains carbon, atmosphere loses carbon); "
            "negative = net carbon source"
        ),
    })

    out_ds = nee.to_dataset()
    out_ds.attrs.update({
        "title": "Ecosystem NEE (positive = net carbon sink) derived from GPP, RH, and RA",
        "definition": "nee = gpp - rh - ra, positive = land carbon sink",
        "input_gpp_file": str(gpp_path),
        "input_rh_file": str(rh_path),
        "input_ra_file": str(ra_path),
        "created": now_utc,
        "Conventions": "CF-1.7",
    })

    # Prepare encodings: write variable uncompressed; avoid illegal keys on coords
    encoding = {
        "nee": {"zlib": False, "complevel": 0, "contiguous": True}
    }
    for coord in ["time", "lat", "lon"]:
        if coord in out_ds.coords:
            if coord in ds_gpp.coords:
                src_enc = ds_gpp.coords[coord].encoding
            else:
                src_enc = {}
            encoding[coord] = filter_nc4_encoding(src_enc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(out_path, encoding=encoding)
    print(f"Wrote 'nee' (positive = net carbon sink) to {out_path}")

if __name__ == "__main__":
    main()