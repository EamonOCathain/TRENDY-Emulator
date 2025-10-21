#!/usr/bin/env python3
"""
scan_zarr_nans.py — count NaNs in many Zarr stores (recursive), SLURM-array friendly.

Usage (local):
  python scan_zarr_nans.py /path/to/root --out nans.csv

Usage (SLURM array):
  #!/bin/bash
  #SBATCH -J zarr-nans
  #SBATCH -A your_account
  #SBATCH -t 01:00:00
  #SBATCH -p your_partition
  #SBATCH --array=0-15        # shard across 16 tasks
  #SBATCH -c 2
  #SBATCH --mem=8G
  module load python  # if needed
  srun python scan_zarr_nans.py /path/to/root --out nans_${SLURM_ARRAY_TASK_ID}.csv

You can later combine CSVs with:  cat nans_*.csv > nans_all.csv  (keeping header once).
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import xarray as xr
import zarr
import pandas as pd

def find_zarrs(root: Path) -> List[Path]:
    zarrs: List[Path] = []
    for r, ds, _ in os.walk(root):
        for d in ds:
            if d.endswith(".zarr"):
                p = Path(r) / d
                # accept group or array zarr dirs
                if (p / ".zgroup").exists() or (p / ".zarray").exists():
                    zarrs.append(p.resolve())
    return sorted(zarrs)

def slurm_shard(paths: List[Path]) -> List[Path]:
    """Simple strided sharding using SLURM_ARRAY_TASK_ID/COUNT; falls back to all paths."""
    try:
        tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        tct = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    except ValueError:
        tid, tct = 0, 1
    if tct <= 1:
        return paths
    return [p for i, p in enumerate(paths) if (i % tct) == (tid % tct)]

def open_zarr_safely(p: Path) -> xr.Dataset:
    """
    Try consolidated first; fall back to non-consolidated if .zmetadata is missing.
    Disable time decoding to avoid CF/calendar quirks.
    """
    try:
        return xr.open_zarr(p, consolidated=True, decode_times=False)
    except Exception as e:
        # If the failure is due to missing .zmetadata, fall back
        try:
            return xr.open_zarr(p, consolidated=False, decode_times=False)
        except Exception as e2:
            raise RuntimeError(f"Failed to open {p} (cons={e!r}; non-cons={e2!r})")

def count_nans_in_ds(ds: xr.Dataset, rechunk: Dict[str, int] | None = None) -> List[Dict[str, object]]:
    """
    Returns a list of dict rows with NaN stats per variable and a store summary row.
    Uses dask-backed computation if arrays are chunked; otherwise loads in manageable chunks.
    """
    # Optional lightweight rechunk to keep memory bounded; only apply for dims that exist
    if rechunk:
        present = {k: v for k, v in rechunk.items() if k in ds.dims}
        if present:
            ds = ds.chunk(present)

    rows: List[Dict[str, object]] = []
    total_vals = 0
    total_nans = 0

    for name, da in ds.data_vars.items():
        try:
            # Use xarray/dask primitives; this stays lazy if dask-backed
            n_vals = int(np.prod([ds.sizes[d] for d in da.dims], dtype=np.int64))
            # .isnull() works for NaN (and masked) values; sum() on boolean -> int
            n_nan = int(da.isnull().sum().compute() if hasattr(da.data, "chunks") else int(np.isnan(da.values).sum()))
            frac = (n_nan / n_vals) if n_vals > 0 else np.nan
        except Exception as e:
            n_vals, n_nan, frac = 0, -1, np.nan  # mark failure
            rows.append({
                "path": str(getattr(ds, "_source_path", "")),
                "var": name,
                "n_values": n_vals,
                "n_nans": n_nan,
                "nan_frac": frac,
                "error": repr(e),
            })
            continue

        rows.append({
            "path": str(getattr(ds, "_source_path", "")),
            "var": name,
            "n_values": n_vals,
            "n_nans": n_nan,
            "nan_frac": frac,
            "error": "",
        })
        total_vals += n_vals
        total_nans += n_nan

    # Add a summary row for the dataset
    rows.append({
        "path": str(getattr(ds, "_source_path", "")),
        "var": "__STORE_SUMMARY__",
        "n_values": total_vals,
        "n_nans": total_nans,
        "nan_frac": (total_nans / total_vals) if total_vals > 0 else np.nan,
        "error": "",
    })
    return rows

def main():
    ap = argparse.ArgumentParser(description="Scan a directory tree for .zarr stores and count NaNs per variable.")
    ap.add_argument("root", type=Path, help="Root directory to search recursively for *.zarr")
    ap.add_argument("--out", type=Path, default=Path("zarr_nan_report.csv"), help="Output CSV path")
    ap.add_argument("--time-chunk", type=int, default=128, help="Optional time rechunk (if 'time' dim exists)")
    ap.add_argument("--loc-chunk", type=int, default=512, help="Optional location rechunk (if 'location' dim exists)")
    ap.add_argument("--scen-chunk", type=int, default=1, help="Optional scenario rechunk (if 'scenario' dim exists)")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N stores from this shard (0=all)")
    ap.add_argument("--verbose", action="store_true", help="Print progress")
    args = ap.parse_args()

    root = args.root.resolve()
    paths = find_zarrs(root)
    shard = slurm_shard(paths)
    if args.limit > 0:
        shard = shard[: args.limit]

    if args.verbose:
        tid = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        tct = os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")
        print(f"[INFO] Found {len(paths)} stores; shard({tid}/{tct}) -> {len(shard)} to process")

    all_rows: List[Dict[str, object]] = []

    for p in shard:
        if args.verbose:
            print(f"[SCAN] {p}")
        try:
            ds = open_zarr_safely(p)
            # record path on the object for later
            setattr(ds, "_source_path", str(p))
            rows = count_nans_in_ds(
                ds,
                rechunk={"time": args.time_chunk, "location": args.loc_chunk, "scenario": args.scen_chunk},
            )
            all_rows.extend(rows)
            ds.close()  # polite close
        except Exception as e:
            # record a failed-open summary row
            all_rows.append({
                "path": str(p),
                "var": "__OPEN_FAILED__",
                "n_values": 0,
                "n_nans": -1,
                "nan_frac": np.nan,
                "error": repr(e),
            })
            if args.verbose:
                print(f"[ERROR] {p}: {e}", file=sys.stderr)

    # Write CSV (one header per file)
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows, columns=["path", "var", "n_values", "n_nans", "nan_frac", "error"])
    df.to_csv(out_path, index=False)
    if args.verbose:
        print(f"[DONE] Wrote {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()