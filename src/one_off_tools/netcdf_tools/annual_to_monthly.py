#!/usr/bin/env python3
import argparse
import numpy as np
import xarray as xr
import cftime

ALLOWED_TIME_KEYS = {
    "units", "calendar", "dtype", "chunksizes", "endian",
    "contiguous", "zlib", "shuffle", "complevel", "fletcher32",
    "least_significant_digit", "significant_digits", "quantize_mode",
    "_FillValue",
}

def main(in_nc, out_nc):
    # Keep cftime (noleap) intact
    ds = xr.open_dataset(in_nc, decode_times=True, use_cftime=True)

    if "time" not in ds.dims:
        raise ValueError("No 'time' dimension found.")

    n_years = ds.sizes["time"]

    # Repeat each annual step 12×
    rep_idx = xr.DataArray(np.repeat(np.arange(n_years), 12), dims="time")
    ds_m = ds.isel(time=rep_idx)

    # Build monthly timestamps (1st of each month, noleap)
    years = ds["time"].dt.year.values.astype(int)  # length = n_years
    new_time = [cftime.DatetimeNoLeap(int(y), m, 1) for y in years for m in range(1, 13)]
    ds_m = ds_m.assign_coords(time=("time", new_time))

    # Preserve only safe time-encoding keys (units+calendar are the important ones)
    src_time_enc = getattr(ds["time"], "encoding", {}) or {}
    time_enc = {k: v for k, v in src_time_enc.items() if k in ALLOWED_TIME_KEYS}
    time_enc.setdefault("units", "days since 1901-01-01 00:00:00")
    time_enc.setdefault("calendar", "noleap")

    # (Optional) ensure no accidental var-level invalid encodings are carried over
    enc = {"time": time_enc}

    ds_m.to_netcdf(out_nc, encoding=enc)
    print(f"Wrote monthly file → {out_nc}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Repeat annual values to monthly (noleap, 1st-of-month).")
    ap.add_argument("--in_nc", required=True, help="Input annual NetCDF")
    ap.add_argument("--out_nc", required=True, help="Output monthly NetCDF")
    args = ap.parse_args()
    main(args.in_nc, args.out_nc)