#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import xarray as xr

try:
    import cftime
except Exception:
    print("This script requires the 'cftime' package (conda/pip install cftime).", file=sys.stderr)
    raise

# Set project root (for slurm_shard)
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))
from src.utils.tools import slurm_shard  # splits work across SLURM array tasks

# ---------- Constants ----------
UNITS = "days since 1901-01-01"
CAL = "noleap"
EARLY_START = cftime.DatetimeNoLeap(1901, 1, 1)
EARLY_END   = cftime.DatetimeNoLeap(1918, 12, 31)
LATE_START  = cftime.DatetimeNoLeap(2018, 1, 1)
LATE_END    = cftime.DatetimeNoLeap(2023, 12, 31)


def is_annual_time(ds: xr.Dataset) -> bool:
    """Annual series are identified by time length == 123 (1901..2023 inclusive)."""
    if "time" not in ds.coords:
        return False
    return int(ds.sizes.get("time", 0)) == 123


def annual_to_monthly_repeat(ds: xr.Dataset) -> xr.Dataset:
    """
    Repeat each annual value 12× along time, and replace the time coord with
    monthly noleap timestamps (YYYY-01 .. YYYY-12 for each year).
    """
    if "time" not in ds.coords:
        raise ValueError("Dataset has no 'time' coordinate.")

    n = ds.sizes["time"]

    # Decode just the coordinates to obtain years robustly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        ds_dec = xr.decode_cf(ds, use_cftime=True)

    years = np.asarray([int(t.year) for t in ds_dec.time.values])
    if years.size != n:
        raise ValueError("Unexpected time decoding for annual dataset; year extraction failed.")

    # New monthly time coordinate (cftime noleap)
    new_times = [cftime.DatetimeNoLeap(int(y), m, 1) for y in years for m in range(1, 13)]

    # Repeat values by fancy indexing (lazy)
    idx = np.repeat(np.arange(n), 12)
    ds_monthly = ds.isel(time=idx).copy()
    ds_monthly = ds_monthly.assign_coords(time=("time", np.array(new_times, dtype=object)))

    # Ensure time encoding is CF-compliant noleap
    ds_monthly["time"].encoding.update({"units": UNITS, "calendar": CAL})
    return ds_monthly


def slice_window(ds: xr.Dataset, start: cftime.DatetimeNoLeap, end: cftime.DatetimeNoLeap) -> xr.Dataset:
    """
    Slice a dataset with a cftime noleap time axis between [start, end].
    Works whether the file had been decoded or not.
    """
    if "time" not in ds.coords:
        return ds

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        ds_dec = xr.decode_cf(ds, use_cftime=True)

    ds_win = ds_dec.sel(time=slice(start, end))

    if "time" in ds_win.coords:
        ds_win["time"].encoding.update({"units": UNITS, "calendar": CAL})
    return ds_win


def save_nc(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if "time" in ds.coords:
        ds["time"].encoding.update({"units": UNITS, "calendar": CAL})
    ds.to_netcdf(path, engine="netcdf4")


def process_file(in_path: Path, in_root: Path, out_root: Path, overwrite: bool = False) -> None:
    """
    Process one file and write three outputs (early/late/combined) into the mirrored
    subdirectory under out_root.
    """
    rel = in_path.relative_to(in_root)
    out_dir = (out_root / rel.parent)

    print(f"[INFO] Processing {rel}")

    ds = xr.open_dataset(in_path, decode_cf=False, mask_and_scale=True)

    if "time" not in ds.coords:
        print(f"[WARN] {rel}: no 'time' coord found; skipping.")
        ds.close()
        return

    # Annual → monthly (repeat), else decode to cftime for slicing
    if is_annual_time(ds):
        print(f"[INFO]  -> identified as ANNUAL (len=123); repeating each time step 12×")
        ds = annual_to_monthly_repeat(ds)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            ds = xr.decode_cf(ds, use_cftime=True)
        if "time" in ds.coords:
            ds["time"].encoding.update({"units": UNITS, "calendar": CAL})

    # Slice to windows
    ds_early = slice_window(ds, EARLY_START, EARLY_END)
    ds_late  = slice_window(ds, LATE_START,  LATE_END)

    # Concatenate along time, preserving original (gapped) time values
    ds_combined = xr.concat([ds_early, ds_late], dim="time")

    # Outputs (preserve input filename stem, write beside mirrored dir)
    stem = in_path.stem
    out_early = out_dir.parent / "seperate" / f"{out_dir.stem}_early".lower() / f"{stem}_early.nc".lower()
    out_late  = out_dir.parent / "seperate" / f"{out_dir.stem}_late".lower() / f"{stem}_late.nc".lower()
    out_comb  = out_dir.parent / "combined" / f"{out_dir.stem}_test".lower() / f"{stem}.nc".lower()

    # Save obeying overwrite
    if out_early.exists() and not overwrite:
        print(f"[SKIP] {out_early.relative_to(out_root)} exists (use --overwrite to replace).")
    else:
        print(f"[WRITE] {out_early.relative_to(out_root)}")
        save_nc(ds_early, out_early)

    if out_late.exists() and not overwrite:
        print(f"[SKIP] {out_late.relative_to(out_root)} exists (use --overwrite to replace).")
    else:
        print(f"[WRITE] {out_late.relative_to(out_root)}")
        save_nc(ds_late, out_late)

    if out_comb.exists() and not overwrite:
        print(f"[SKIP] {out_comb.relative_to(out_root)} exists (use --overwrite to replace).")
    else:
        print(f"[WRITE] {out_comb.relative_to(out_root)}")
        save_nc(ds_combined, out_comb)

    print(f"[OK] {rel} → early/late/combined written.")
    ds.close()


def main():
    ap = argparse.ArgumentParser(
        description="Recursively process .nc files: repeat annual (len=123) to monthly, "
                    "trim to test windows (1901-01–1918-12, 2018-01–2023-12), and write "
                    "early/late/combined outputs mirroring the input folder structure."
    )
    ap.add_argument("--in_dir",  type=Path, required=True, help="Root directory to search recursively for .nc files.")
    ap.add_argument("--out_dir", type=Path, required=True, help="Parent output directory; input substructure is mirrored here.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = ap.parse_args()

    in_root: Path = args.in_dir
    out_root: Path = args.out_dir

    if not in_root.is_dir():
        raise SystemExit(f"Input directory not found: {in_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    # Recursively find .nc files under in_root
    all_files = sorted([p for p in in_root.rglob("*.nc") if p.is_file()])
    if not all_files:
        raise SystemExit(f"No .nc files found under {in_root}")

    # Shard across SLURM array tasks using your helper
    files = slurm_shard(all_files)  # expected to return the sublist for this task
    print(f"[INFO] Total files discovered: {len(all_files)}; this shard will process: {len(files)}")

    for p in files:
        try:
            process_file(p, in_root, out_root, overwrite=args.overwrite)
        except Exception as e:
            rel = p.relative_to(in_root)
            print(f"[ERROR] {rel}: {e}", file=sys.stderr)

    print("[DONE]")


if __name__ == "__main__":
    main()