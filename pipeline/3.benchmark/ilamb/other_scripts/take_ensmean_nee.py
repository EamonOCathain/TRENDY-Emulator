#!/usr/bin/env python3
"""
nee_from_npp_rh.py
Compute NEE = NPP - RH from two NetCDF files with identical time/lat/lon.

Usage:
  python nee_from_npp_rh.py npp.nc rh.nc out.nc

Assumptions:
- Variables are named 'npp' and 'rh'
- Dims/coords match exactly
- Output is uncompressed
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import xarray as xr

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

def main():
    if len(sys.argv) != 4:
        print("Usage: python nee_from_npp_rh.py <npp.nc> <rh.nc> <out.nc>", file=sys.stderr)
        sys.exit(2)

    npp_path = Path(sys.argv[1]); rh_path = Path(sys.argv[2]); out_path = Path(sys.argv[3])

    ds_npp = xr.open_dataset(npp_path, decode_times=True, use_cftime=True)
    ds_rh  = xr.open_dataset(rh_path,  decode_times=True, use_cftime=True)

    if "npp" not in ds_npp.data_vars:
        raise ValueError(f"'npp' not found in {npp_path}. Available: {list(ds_npp.data_vars)}")
    if "rh" not in ds_rh.data_vars:
        raise ValueError(f"'rh' not found in {rh_path}. Available: {list(ds_rh.data_vars)}")

    npp = ds_npp["npp"]; rh = ds_rh["rh"]

    # Strict coord checks
    if npp.dims != rh.dims:
        raise ValueError(f"Dim order mismatch: {npp.dims} vs {rh.dims}")
    for d in npp.dims:
        if npp.sizes[d] != rh.sizes[d]:
            raise ValueError(f"Size mismatch in dim '{d}': {npp.sizes[d]} vs {rh.sizes[d]}")
        if not np.array_equal(npp.coords[d].values, rh.coords[d].values):
            raise ValueError(f"Coordinate values differ on dim '{d}'.")

    # Compute NEE = NPP - RH (mask non-finite)
    npp_f = npp.where(np.isfinite(npp))
    rh_f  = rh.where(np.isfinite(rh))
    nee = (npp_f.astype("float64") - rh_f.astype("float64")).rename("nee")
    nee = nee.where(np.isfinite(npp_f) & np.isfinite(rh_f))

    # Attributes
    units = npp.attrs.get("units", rh.attrs.get("units", ""))
    now_utc = datetime.now(timezone.utc).isoformat()
    nee.attrs.update({
        "long_name": "net ecosystem exchange (NEE) = NPP - RH",
        "standard_name": "net_ecosystem_exchange",
        "units": units,
        "history": f"{now_utc}: computed nee = npp - rh",
        "source": "derived",
    })

    out_ds = nee.to_dataset()
    out_ds.attrs.update({
        "title": "NEE derived from NPP and RH",
        "definition": "nee = npp - rh",
        "input_npp_file": str(npp_path),
        "input_rh_file": str(rh_path),
        "created": now_utc,
        "Conventions": "CF-1.7",
    })

    # Prepare encodings: write variable uncompressed; avoid illegal keys on coords
    encoding = {
        "nee": {"zlib": False, "complevel": 0, "contiguous": True}
    }
    for coord in ["time", "lat", "lon"]:
        if coord in out_ds.coords:
            src_enc = ds_npp.coords.get(coord, xr.DataArray()).encoding if coord in ds_npp.coords else {}
            encoding[coord] = filter_nc4_encoding(src_enc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(out_path, encoding=encoding)
    print(f"Wrote 'nee' to {out_path}")

if __name__ == "__main__":
    main()