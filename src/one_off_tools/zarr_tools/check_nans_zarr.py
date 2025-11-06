#!/usr/bin/env python3
"""
scan_zarr_nans.py — log Zarr stores that contain NaNs (SLURM-array friendly).

Args:
  root (positional): directory to search for *.zarr
  --verbose        : print progress + OK stores; otherwise only NaN stores + summary

Hard-coded rechunk:
  time = full length, location = 70, scenario = 1
"""
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import xarray as xr

def find_zarrs(root: Path) -> List[Path]:
    zarrs: List[Path] = []
    for r, ds, _ in os.walk(root):
        for d in ds:
            if d.endswith(".zarr"):
                p = Path(r) / d
                if (p / ".zgroup").exists() or (p / ".zarray").exists():
                    zarrs.append(p.resolve())
    return sorted(zarrs)

def slurm_shard(paths: List[Path]) -> List[Path]:
    """Strided sharding via SLURM_ARRAY_TASK_ID/COUNT; falls back to all paths."""
    try:
        tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        tct = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    except ValueError:
        tid, tct = 0, 1
    if tct <= 1:
        return paths
    return [p for i, p in enumerate(paths) if (i % tct) == (tid % tct)]

def open_zarr_safely(p: Path) -> xr.Dataset:
    """Try consolidated first; fall back to non-consolidated; don’t decode times."""
    try:
        return xr.open_zarr(p, consolidated=True, decode_times=False)
    except Exception:
        return xr.open_zarr(p, consolidated=False, decode_times=False)

def count_nans_in_ds(ds: xr.Dataset, path_str: str) -> Tuple[int, Dict[str, int]]:
    """
    Return (total_nan_count, per_var_nan_counts) for data variables in the dataset.
    Uses dask-backed computation when available.
    """
    per_var: Dict[str, int] = {}
    total = 0
    for name, da in ds.data_vars.items():
        try:
            if hasattr(da.data, "chunks"):  # dask-backed
                n_nan = int(da.isnull().sum().compute())
            else:
                n_nan = int(np.isnan(da.values).sum())
        except Exception as e:
            n_nan = -1
            print(f"[WARN] Failed NaN count for {name} in {path_str}: {e}", file=sys.stderr)
        per_var[name] = n_nan
        if n_nan > 0:
            total += n_nan
    return total, per_var

def hardcoded_rechunk(ds: xr.Dataset) -> xr.Dataset:
    """
    Apply hard-coded rechunk:
      - time: full length (single chunk)
      - location: 70
      - scenario: 1
    Only apply for dims that exist.
    """
    chunks: Dict[str, int] = {}
    if "time" in ds.dims:
        chunks["time"] = int(ds.sizes["time"])  # full time in one chunk
    if "location" in ds.dims:
        chunks["location"] = 70
    if "scenario" in ds.dims:
        chunks["scenario"] = 1
    if chunks:
        ds = ds.chunk(chunks)
    return ds

def main():
    ap = argparse.ArgumentParser(description="Scan .zarr trees and LOG stores that contain NaNs.")
    ap.add_argument("root", type=Path, help="Root directory to search recursively for *.zarr")
    ap.add_argument("--verbose", action="store_true", help="Print progress and OK stores")
    args = ap.parse_args()

    root = args.root.resolve()
    paths = find_zarrs(root)
    shard = slurm_shard(paths)

    if args.verbose:
        tid = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        tct = os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")
        print(f"[INFO] Found {len(paths)} stores; shard({tid}/{tct}) -> {len(shard)} to process")

    scanned = 0
    affected = 0

    for p in shard:
        p_str = str(p)
        if args.verbose:
            print(f"[SCAN] {p_str}")
        try:
            ds = open_zarr_safely(p)
            ds = hardcoded_rechunk(ds)

            total_nans, per_var = count_nans_in_ds(ds, p_str)
            scanned += 1

            if total_nans > 0:
                affected += 1
                vars_with = [f"{v}({n})" for v, n in sorted(per_var.items()) if n > 0]
                vars_str = ", ".join(vars_with) if vars_with else "<none>"
                print(f"[NaN] {p_str} | total_nans={total_nans} | vars_with_nans=[{vars_str}]")
            elif args.verbose:
                print(f"[OK]  {p_str} | total_nans=0")

            ds.close()
        except Exception as e:
            print(f"[ERROR] Failed to open/scan {p_str}: {e}", file=sys.stderr)

    print(f"[SUMMARY] scanned={scanned} | affected={affected} | root={root}")

if __name__ == "__main__":
    main()