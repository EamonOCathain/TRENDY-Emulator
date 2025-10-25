#!/usr/bin/env python3
from pathlib import Path
import zarr

# --- Config ---
BASE_DIR = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference")
VAR_NAME = "lai_avh15c1"

def remove_var_from_zarr(zarr_path: Path, var_name: str):
    """Remove a variable from a single Zarr store and reconsolidate metadata."""
    try:
        root = zarr.open_group(str(zarr_path), mode="a")
    except Exception as e:
        print(f"[WARN] Could not open {zarr_path}: {e}")
        return

    if var_name not in root:
        return  # skip silently if var not present

    print(f"[INFO] Removing '{var_name}' from {zarr_path}")
    del root[var_name]

    try:
        zarr.consolidate_metadata(str(zarr_path))
        print(f"[OK] Removed '{var_name}' and reconsolidated metadata in {zarr_path}")
    except Exception as e:
        print(f"[WARN] consolidate_metadata failed for {zarr_path}: {e}")


def main():
    print(f"[INFO] Searching for Zarr stores under {BASE_DIR}")
    for subdir in BASE_DIR.rglob("*.zarr"):
        remove_var_from_zarr(subdir, VAR_NAME)

    print("[DONE] Variable removal complete across all Zarrs.")


if __name__ == "__main__":
    main()