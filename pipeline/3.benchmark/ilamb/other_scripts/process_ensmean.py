#!/usr/bin/env python3
"""
convert_or_copy_nc_slurm.py
---------------------------
Batch-convert NetCDFs in a directory.

- If filename contains {cVeg, cSoil, cLitter}, converts annual→monthly.
- Otherwise, copies unchanged.
- Output files are grouped under out_dir/S0..S3 (or misc if none found).
- When run under SLURM array, each task processes one file.
"""

import argparse
import os
import shutil
from pathlib import Path
import numpy as np
import xarray as xr
import cftime

CONVERT_VARS = ("cVeg", "cSoil", "cLitter")
SCENARIOS = ("S0", "S1", "S2", "S3")

def detect_scenario(name: str) -> str:
    for s in SCENARIOS:
        if s in name:
            return s
    return "misc"

def should_convert(name: str) -> bool:
    return any(tok in name for tok in CONVERT_VARS)

def convert_to_monthly(in_nc: Path, out_nc: Path):
    ds = xr.open_dataset(in_nc, decode_times=True, use_cftime=True)
    if "time" not in ds.dims:
        raise ValueError(f"No 'time' dim in {in_nc}")
    n_years = ds.sizes["time"]
    rep_idx = np.repeat(np.arange(n_years), 12)
    ds_m = ds.isel(time=rep_idx)
    years = ds["time"].dt.year.values.astype(int)
    new_time = [cftime.DatetimeNoLeap(int(y), m, 1) for y in years for m in range(1, 13)]
    ds_m = ds_m.assign_coords(time=("time", new_time))
    enc = {"time": {"units": "days since 1901-01-01 00:00:00", "calendar": "noleap"}}
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds_m.to_netcdf(out_nc, encoding=enc)
    print(f"[convert] {in_nc.name} → {out_nc}")

def copy_nc(in_nc: Path, out_nc: Path):
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(in_nc, out_nc)
    print(f"[copy] {in_nc.name} → {out_nc}")

def main(in_dir: Path, out_dir: Path):
    files = sorted(in_dir.glob("*.nc"))
    if not files:
        print(f"No .nc files found in {in_dir}")
        return

    # detect slurm array index (0-based)
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        i = int(task_id)
        if i >= len(files):
            print(f"Task {i} exceeds file count {len(files)}")
            return
        files = [files[i]]
        print(f"[SLURM] Task {i}/{len(files)} processing {files[0].name}")

    for f in files:
        scen = detect_scenario(f.name)
        out_nc = out_dir / scen / f.name
        if should_convert(f.name):
            convert_to_monthly(f, out_nc)
        else:
            copy_nc(f, out_nc)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert cVeg/cSoil/cLitter NetCDFs to monthly; copy others.")
    ap.add_argument("--in_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()
    main(args.in_dir, args.out_dir)