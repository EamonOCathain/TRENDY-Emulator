#!/usr/bin/env python3
"""
Rewrite NetCDF time axes to a noleap calendar with a fixed range/resolution.

- If BASE_PATH is a file, process that file.
- If BASE_PATH is a directory, scan it (non-recursive by default) for *.nc files.
- Replaces the 'time' coordinate with a freshly built noleap cftime index
  spanning [START_DATE, END_DATE] at TIME_RES resolution.
- Writes numeric time with units 'days since 1901-01-01' and calendar 'noleap'.
- Saves in place via a temporary file then atomic replace.

This script is STRICT about length: it will raise if old_len != new_len.
"""

from pathlib import Path
import shutil
import warnings

import xarray as xr

# ============================= CONFIG (edit me) ============================= #
BASE_PATH   = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/avh15c1_lai.nc")
RECURSIVE  = False                                # True = walk subfolders when BASE_PATH is a directory
START_DATE = "1981-01-01"
END_DATE   = "2019-12-31"
TIME_RES   = "monthly"                            # "daily", "monthly", or "annual"
TIME_VAR   = "time"
TIME_UNITS = "days since 1901-01-01"
CALENDAR   = "noleap"
# =========================================================================== #

def _freq_from_time_res(time_res: str) -> str:
    """
    Map desired time resolution to an xarray.cftime_range frequency string.
      - daily   -> 'D'  (day start)
      - monthly -> 'MS' (month start)
      - annual  -> 'YS' (year start)
    """
    t = str(time_res).lower()
    if t == "daily":
        return "D"
    if t == "monthly":
        return "MS"
    if t == "annual":
        return "YS"
    raise ValueError(f"Unsupported TIME_RES: {time_res!r}. Use 'daily', 'monthly', or 'annual'.")

def _build_noleap_index(start: str, end: str, time_res: str):
    """Build a noleap cftime index using xarray.cftime_range."""
    freq = _freq_from_time_res(time_res)
    return xr.cftime_range(start=start, end=end, freq=freq, calendar=CALENDAR)

def process_nc_file(nc_path: Path):
    print(f"[INFO] Processing: {nc_path}")
    # Open without decoding original times (we're going to replace them)
    ds = xr.open_dataset(nc_path, decode_times=False)

    try:
        if TIME_VAR not in ds.coords and TIME_VAR not in ds:
            warnings.warn(f"[WARN] {nc_path.name}: no '{TIME_VAR}' coordinate/variable; skipping.")
            return

        # Build new noleap time index
        new_time = _build_noleap_index(START_DATE, END_DATE, TIME_RES)

        # Determine current time length
        if TIME_VAR in ds.dims:
            old_len = ds.sizes[TIME_VAR]
        else:
            # best-effort fallback if time exists but is not currently a dim
            tv = ds[TIME_VAR]
            if TIME_VAR in tv.sizes:
                old_len = tv.sizes[TIME_VAR]
            else:
                # fall back to the first size of that variable
                old_len = next(iter(tv.sizes.values()))

        new_len = len(new_time)

        # Strict check: never auto-trim/pad
        if new_len != old_len:
            raise RuntimeError(
                f"{nc_path.name}: time length mismatch (old={old_len}, new={new_len}). "
                "Refusing to auto-trim/pad. Adjust START/END/TIME_RES or preprocess first."
            )

        # Replace time coordinate with cftime noleap index
        ds = ds.assign_coords({TIME_VAR: new_time})

        # Ensure numeric time is written with desired units/calendar
        ds[TIME_VAR].encoding = {
            "units": TIME_UNITS,
            "calendar": CALENDAR,
            "dtype": "float64",
        }

        # Save in place safely (tmp -> replace)
        tmp_path = nc_path.with_suffix(".nc.tmp")
        ds.to_netcdf(tmp_path)
        # Atomic replace
        shutil.move(str(tmp_path), str(nc_path))
        print(f"[OK] Wrote: {nc_path} (calendar='{CALENDAR}', units='{TIME_UNITS}', res='{TIME_RES}')")

    finally:
        ds.close()

def main():
    if not BASE_PATH.exists():
        raise SystemExit(f"BASE_PATH does not exist: {BASE_PATH}")

    if BASE_PATH.is_file():
        nc_files = [BASE_PATH]
    else:
        if RECURSIVE:
            nc_files = [p for p in BASE_PATH.rglob("*.nc") if p.is_file()]
        else:
            nc_files = [p for p in BASE_PATH.glob("*.nc") if p.is_file()]

    if not nc_files:
        print(f"[INFO] No .nc files found in {BASE_PATH} (RECURSIVE={RECURSIVE}).")
        return

    print(f"[INFO] Found {len(nc_files)} .nc file(s) in {BASE_PATH} (RECURSIVE={RECURSIVE}).")
    print(f"[INFO] Target calendar={CALENDAR}, units='{TIME_UNITS}', range={START_DATE}..{END_DATE}, res={TIME_RES}")

    for nc in sorted(nc_files):
        try:
            process_nc_file(nc)
        except Exception as e:
            warnings.warn(f"[ERR] Failed on {nc.name}: {e}")

if __name__ == "__main__":
    main()