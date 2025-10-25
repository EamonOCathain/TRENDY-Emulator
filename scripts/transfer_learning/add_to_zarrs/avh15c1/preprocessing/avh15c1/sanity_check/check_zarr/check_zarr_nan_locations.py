#!/usr/bin/env python3
import sys
from pathlib import Path
import xarray as xr
import numpy as np

def check_store(zarr_path: Path):
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True, decode_times=False)
    except Exception as e:
        print(f"[SKIP] {zarr_path} (open failed: {e})")
        return

    var = "lai_avh15c1"
    if var not in ds.data_vars:
        print(f"[SKIP] {zarr_path} (missing '{var}')")
        return

    da = ds[var]

    # Expect (time, scenario, location) — pick scenario=3 if available, else 0
    if "scenario" in da.dims:
        s = 3 if int(da.sizes["scenario"]) > 3 else 0
        da = da.isel(scenario=s)
    else:
        print(f"[WARN] {zarr_path} has no 'scenario' dim; using as-is")

    if "time" not in da.dims or "location" not in da.dims:
        print(f"[SKIP] {zarr_path} (missing 'time' or 'location' dims)")
        return

    # Identify locations where the entire time series is NaN or 0
    # All-NaN:
    all_nan = da.isnull().all(dim="time")

    # All-zero: at least one non-NaN exists but all non-NaNs are zero
    # (avoid counting the all-NaN case as all-zero)
    all_zero = (~all_nan) & (da.fillna(0) == 0).all(dim="time")

    nan_locs = np.where(all_nan.values)[0]
    zero_locs = np.where(all_zero.values)[0]

    # Try to pull lat/lon per-location if available
    lat = ds.get("lat")
    lon = ds.get("lon")
    have_latlon = (lat is not None and lon is not None and "location" in lat.dims and "location" in lon.dims)

    print(f"\n[STORE] {zarr_path}")
    print(f"  time={da.sizes['time']} locations={da.sizes['location']}")
    print(f"  all-NaN locations: {len(nan_locs)}")
    if len(nan_locs) > 0:
        if have_latlon:
            for i in nan_locs[:10]:
                print(f"    loc={i}  lat={float(lat.isel(location=i)):.3f}  lon={float(lon.isel(location=i)):.3f}")
        else:
            print(f"    first 10 indices: {nan_locs[:10].tolist()}")

    print(f"  all-zero locations: {len(zero_locs)}")
    if len(zero_locs) > 0:
        if have_latlon:
            for i in zero_locs[:10]:
                print(f"    loc={i}  lat={float(lat.isel(location=i)):.3f}  lon={float(lon.isel(location=i)):.3f}")
        else:
            print(f"    first 10 indices: {zero_locs[:10].tolist()}")

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    if not root.exists():
        sys.exit(f"Path not found: {root}")

    zarr_paths = sorted(p for p in root.rglob("monthly.zarr") if p.is_dir())
    if not zarr_paths:
        print(f"[INFO] No monthly.zarr found under {root}")
        return

    for zp in zarr_paths:
        check_store(zp)

if __name__ == "__main__":
    main()