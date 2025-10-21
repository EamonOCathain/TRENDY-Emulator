#!/usr/bin/env python3
"""
Merge per-var temp stores into final inference store.

Example:
  python merge_inference_stores.py --scenario S0 --time-res annual
"""
from __future__ import annotations
from pathlib import Path
import sys
import argparse
import xarray as xr
import zarr

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))
from src.paths.paths import zarr_dir

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=["S0","S1","S2","S3"])
    p.add_argument("--time-res", required=True, choices=["annual","monthly","daily"])
    p.add_argument("--tmp-root", default="inference_tmp")
    p.add_argument("--final-root", default="inference_new")  # or "inference"
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    tmp_dir = zarr_dir / f"{args.tmp_root}/{args.scenario}/{args.time_res}"
    out_store = zarr_dir / f"{args.final-root}/{args.scenario}/{args.time_res}.zarr"  # noqa

    # fix dash access bug:
    final_root = getattr(args, "final_root")
    out_store = zarr_dir / f"{final_root}/{args.scenario}/{args.time_res}.zarr"

    var_stores = sorted(tmp_dir.glob("*.zarr"))
    if not var_stores:
        raise FileNotFoundError(f"No temp stores found at {tmp_dir}")

    out_store.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if (args.overwrite or not out_store.exists()) else "a"

    # Initialise with the first var
    ds0 = xr.open_zarr(var_stores[0], consolidated=False)
    ds0.to_zarr(out_store, mode=mode, compute=True, consolidated=False)

    # Append the rest
    for vs in var_stores[1:]:
        dsv = xr.open_zarr(vs, consolidated=False)
        dsv.to_zarr(out_store, mode="a", append_dim=None, compute=True, consolidated=False)

    # One consolidation at the end
    zarr.consolidate_metadata(str(out_store))
    print(f"[DONE] merged {len(var_stores)} vars -> {out_store}")

if __name__ == "__main__":
    main()