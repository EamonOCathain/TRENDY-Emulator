#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import xarray as xr
import zarr

# --- project root & slurm_shard helper ---
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.utils.tools import slurm_shard
from src.dataset.variables import luh2_states

# config
zarr_dir = project_root / "data/zarrs/training_new"
target_vars = list(luh2_states)  # variables to process


def find_annual_stores(root: Path) -> list[Path]:
    """Return all paths like .../annual.zarr under root (recursive)."""
    return sorted(p for p in root.rglob("annual.zarr") if p.is_dir())


def main():
    ap = argparse.ArgumentParser(description="Create *_delta vars (year-to-year differences) in annual.zarr stores.")
    ap.add_argument("--overwrite", action="store_true",
                    help="If set, overwrite existing *_delta variables (otherwise skip).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List actions without writing.")
    args = ap.parse_args()

    stores = find_annual_stores(zarr_dir)
    if not stores:
        print(f"[INFO] No annual.zarr stores found under: {zarr_dir}")
        return

    # Optionally shard the work if running in a SLURM array
    stores = slurm_shard(stores)

    print(f"[INFO] Found {len(stores)} annual.zarr store(s) to process.")

    for store in stores:
        print(f"\n[STORE] {store}")
        # open lazily; keep chunking/coords as-is; we don't need decoded times
        ds = xr.open_zarr(store, consolidated=True, decode_times=False, chunks={})

        # figure out which vars we can process (must exist and have 'time' dim)
        todo = []
        for v in target_vars:
            if v in ds.data_vars:
                if "time" in ds[v].dims:
                    todo.append(v)
                else:
                    print(f"  - skip {v}: no 'time' dimension")
            else:
                print(f"  - missing {v} in store; skip")

        if not todo:
            print("  (no matching variables with 'time' dim)")
            continue

        for v in todo:
            out_name = f"{v}_delta"
            exists_already = out_name in ds.data_vars
            if exists_already and not args.overwrite:
                print(f"  - {out_name} already exists → skip (use --overwrite to replace)")
                continue

            da = ds[v]

            # year-to-year delta along time; first year becomes 0 (same dtype)
            delta = da - da.shift(time=1, fill_value=0)
            delta.name = out_name
            delta = delta.transpose("time", ...)

            # carry over some attrs
            attrs = dict(da.attrs)
            attrs["long_name"] = attrs.get("long_name", v) + " (yearly delta)"
            attrs["description"] = f"Year-to-year difference of {v}; first year is 0."
            delta = delta.assign_attrs(attrs)

            # preserve chunks if available (match existing var’s chunks)
            enc = {}
            try:
                # dask-chunked arrays expose chunk sizes via .chunksizes
                if hasattr(da.data, "chunks") and isinstance(da.chunks, dict):
                    # xarray expects a tuple in the dim order
                    enc[out_name] = {"chunks": tuple(da.chunks[d][0] for d in da.dims)}
            except Exception:
                pass

            if args.dry_run:
                print(f"  - would write {out_name} with dims {delta.dims} and shape {tuple(delta.shape)}")
                continue

            # Overwrite behavior: remove the array if it exists and --overwrite
            if exists_already and args.overwrite:
                try:
                    zg = zarr.open_group(str(store), mode="a")
                    if out_name in zg:
                        del zg[out_name]
                        print(f"  - deleted existing {out_name}")
                except Exception as e:
                    print(f"  ! warn: failed to delete existing {out_name}: {e}")

            # append just this variable
            print(f"  - writing {out_name}")
            xr.Dataset({out_name: delta}).to_zarr(
                store=str(store),
                mode="a",
                consolidated=False,  # consolidate after all writes
                encoding=enc,
            )

        # consolidate metadata so future opens with consolidated=True work cleanly
        if not args.dry_run:
            try:
                zarr.consolidate_metadata(str(store))
                print("  - consolidated metadata")
            except Exception as e:
                print(f"  ! warn: consolidate_metadata failed: {e}")

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()