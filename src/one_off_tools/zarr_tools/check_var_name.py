#!/usr/bin/env python3
from pathlib import Path
import os
import json

import zarr  # pip install zarr

'''Takes directory as target, searches for Zarr stores within it, finds target vars and renames metadata and files, reconsolidates'''

ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/training_new")   # <-- change this
OLD = "avh15c1_lai"
NEW = "lai_avh15c1"

def is_zarr_store(p: Path) -> bool:
    # Heuristic: a zarr store usually contains .zgroup (root) or .zattrs
    return p.is_dir() and (p / ".zgroup").exists()

def rename_var_dirs_in_store(store_path: Path, old: str, new: str) -> int:
    """
    Find any subdirectory whose *leaf* name == old and that contains '.zarray',
    rename it to 'new'. Returns the number of renames performed.
    """
    count = 0
    # Search for candidate variable dirs (leaf name match)
    for var_dir in store_path.rglob(old):
        if not var_dir.is_dir():
            continue
        # It's a Zarr array if it has a .zarray file inside
        if not (var_dir / ".zarray").exists():
            continue

        new_dir = var_dir.with_name(new)
        if new_dir.exists():
            print(f"[SKIP] Target already exists: {new_dir}")
            continue

        print(f"[RENAME] {var_dir}  ->  {new_dir}")
        os.rename(var_dir, new_dir)
        count += 1

    return count

def maybe_update_root_zattrs(store_path: Path, old: str, new: str) -> None:
    """
    If the root .zattrs has a 'variables' list (not guaranteed), replace old with new.
    """
    zattrs_path = store_path / ".zattrs"
    if not zattrs_path.exists():
        return
    try:
        with open(zattrs_path, "r") as f:
            attrs = json.load(f)
        changed = False
        if isinstance(attrs, dict) and isinstance(attrs.get("variables"), list):
            vars_list = attrs["variables"]
            new_list = [new if v == old else v for v in vars_list]
            if new_list != vars_list:
                attrs["variables"] = new_list
                changed = True
        if changed:
            with open(zattrs_path, "w") as f:
                json.dump(attrs, f)
            print(f"[UPDATE] Root .zattrs variables list updated in {store_path}")
    except Exception as e:
        print(f"[WARN] Could not update root .zattrs in {store_path}: {e}")

def reconsolidate(store_path: Path) -> None:
    """
    Rebuild .zmetadata so consolidated metadata matches the on-disk structure.
    """
    try:
        store = zarr.DirectoryStore(str(store_path))
        zarr.consolidate_metadata(store)  # writes/overwrites .zmetadata
        print(f"[OK] Re-consolidated metadata: {store_path / '.zmetadata'}")
    except Exception as e:
        print(f"[WARN] Could not consolidate {store_path}: {e}")

def main():
    # Iterate over all *.zarr directories (depth-first)
    for z in ROOT.rglob("*.zarr"):
        if not is_zarr_store(z):
            continue
        print(f"\n=== Scanning store: {z} ===")
        n = rename_var_dirs_in_store(z, OLD, NEW)
        if n == 0:
            print("[INFO] No matching variables found.")
            continue
        # Optional: update root-level variables list if present
        maybe_update_root_zattrs(z, OLD, NEW)
        # Always re-consolidate so .zmetadata reflects the new names
        reconsolidate(z)

if __name__ == "__main__":
    main()