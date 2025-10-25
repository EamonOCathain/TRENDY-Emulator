#!/usr/bin/env python3
from pathlib import Path
import zarr

# --- Config ---
zarr_path = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/modis_lai/modis_lai_monthly_filled.zarr")
old_name = "modis_lai"
new_name = "lai_modis"

# --- Open group ---
root = zarr.open_group(str(zarr_path), mode="a")

if old_name not in root:
    raise KeyError(f"Variable '{old_name}' not found. Available: {list(root.array_keys())}")

# Rename key (Zarr 2.x doesn't have rename(), so we manually move it)
print(f"[INFO] Renaming variable '{old_name}' → '{new_name}' in {zarr_path}")
root.move(old_name, new_name)  # works since Zarr v2.16+

# Re-consolidate metadata so xarray.open_zarr(consolidated=True) still works
print("[INFO] Reconsolidating metadata")
zarr.consolidate_metadata(str(zarr_path))

print("[OK] Done.")