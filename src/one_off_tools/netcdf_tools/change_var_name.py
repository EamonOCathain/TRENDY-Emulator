#!/usr/bin/env python3
import xarray as xr
from pathlib import Path
import tempfile
import os

# --- config ---
path = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/lai_avh15c1.nc")
old_var = "avh15c1_lai"
new_var = "lai_avh15c1"

print(f"[INFO] Opening {path}")
ds = xr.open_dataset(path)

if old_var not in ds:
    raise KeyError(f"{old_var} not found in {path}. Available: {list(ds.data_vars)}")

print(f"[INFO] Renaming '{old_var}' → '{new_var}'")
ds_renamed = ds.rename({old_var: new_var})

# Write to temporary file first for safety
tmp_path = Path(tempfile.mktemp(suffix=".nc", dir=path.parent))
ds_renamed.to_netcdf(tmp_path, format="NETCDF4")

# Replace the original file atomically
os.replace(tmp_path, path)
print(f"[OK] Renamed variable in place → {path}")