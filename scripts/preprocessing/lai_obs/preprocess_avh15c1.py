#!/usr/bin/env python3
# convert_avh15c1_to_1901_noleap.py

import xarray as xr
from pathlib import Path
import numpy as np
import sys
import shutil

# --- project helpers you provided ---
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.utils.preprocessing import (
    _open_ds,
    _drop_bounds_inplace,
    reassign_720_360_grid,
    nccopy_chunk,
)

def convert_time_1850_to_1901(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert numeric CF time coord from 'days since 1850-01-01' (noleap)
    to 'days since 1901-01-01 00:00' (noleap) by subtracting 51*365.
    """
    if "time" not in ds.coords:
        return ds
    # days between 1850-01-01 and 1901-01-01 in a noleap (365-day) calendar
    delta_days = 51 * 365  # 1850..1900 inclusive
    t = ds["time"].values
    # keep dtype (time is float with .5 mid-month centers)
    t_new = (t - delta_days).astype(t.dtype, copy=False)
    ds = ds.assign_coords(time=("time", t_new))
    ds["time"].attrs = {
        "units": "days since 1901-01-01 00:00:00",
        "calendar": "noleap",
        "standard_name": "time",
        "axis": "T",
    }
    return ds

def main(in_path: str,
         out_path: str,
         tmp_dir: str = "./__tmp_avh15c1__",
         clevel: int = 4,
         lat_chunk: int = 5,
         lon_chunk: int = 5):
    in_path  = Path(in_path)
    out_path = Path(out_path)
    tmp_dir  = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp1 = tmp_dir / "step1_keep_and_rename.nc"
    tmp2 = tmp_dir / "step2_drop_bounds.nc"
    tmp3 = tmp_dir / "step3_time_1901.nc"
    tmp4 = tmp_dir / "step4_reassign_grid.nc"

    # --- Step 1: keep only {time, lat, lon, lai} and rename to avh15c1_lai ---
    with xr.open_dataset(in_path, engine="netcdf4", decode_times=False) as ds:
        # Ensure expected variable present
        if "lai" not in ds.variables:
            raise ValueError(f"'lai' not found in {in_path}. Vars: {list(ds.variables)}")

        # Keep only essentials + data var
        keep = ["time", "lat", "lon", "lai"]
        keep = [v for v in keep if v in ds]  # guard
        ds = ds[keep]

        # Rename data var
        ds = ds.rename({"lai": "avh15c1_lai"})

        # Write out (let bounds be removed in next step)
        ds.to_netcdf(tmp1, engine="netcdf4", format="NETCDF4")

    # --- Step 2: drop any *bounds/*bnds vars cleanly (helper preserves encodings) ---
    _drop_bounds_inplace(tmp1, tmp2)

    # --- Step 3: convert time units to days since 1901-01-01 noleap ---
    with xr.open_dataset(tmp2, engine="netcdf4", decode_times=False) as ds:
        ds = convert_time_1850_to_1901(ds)
        # Strip any stray fill attrs from coords (safety)
        for c in ("time", "lat", "lon"):
            if c in ds.coords:
                ds[c].attrs.pop("_FillValue", None)
                ds[c].encoding.pop("_FillValue", None)
        ds.to_netcdf(tmp3, engine="netcdf4", format="NETCDF4")

    # --- Step 4: reassign grid to exactly lon 0..359.5 and lat -89.75..89.75 ---
    reassign_720_360_grid(tmp3, tmp4)

    # --- Step 5: final chunking: time=full, lat=5, lon=5 + compression ---
    # uses nccopy so we don't mess with data encodings via re-encode in xarray
    nccopy_chunk(
        in_path=tmp4,
        out_path=out_path,
        clevel=clevel,
        lat_chunk=lat_chunk,
        lon_chunk=lon_chunk,
        overwrite=True
    )

    # Cleanup tmp
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    print(f"[OK] Wrote: {out_path}")
    # quick sanity print
    with xr.open_dataset(out_path, engine="netcdf4", decode_times=False, chunks={}) as ds:
        print(f"time units: {ds['time'].attrs.get('units')}, calendar: {ds['time'].attrs.get('calendar')}")
        print(f"coords: lat({ds.dims.get('lat')}), lon({ds.dims.get('lon')}), time({ds.dims.get('time')})")
        # show chunking of data var
        v = ds["avh15c1_lai"]
        print("avh15c1_lai chunksizes:", v.encoding.get("chunksizes"))

if __name__ == "__main__":
    # Example usage:
    # python convert_avh15c1_to_1901_noleap.py \
    #   /path/to/AVH15C1/lai.nc \
    #   /path/to/output/avh15c1_lai_1901_noleap_r720x360.nc
    if len(sys.argv) < 3:
        print("Usage: convert_avh15c1_to_1901_noleap.py <in.nc> <out.nc>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])