#!/usr/bin/env python3
"""
Make a NaN mask NetCDF and plots (first timestep of data + mask)
for a single hardcoded NetCDF file using the same utilities
as in the main masking pipeline.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Project bootstrap (so imports work when run standalone)
# ---------------------------------------------------------------------
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

# Project imports
from src.paths.paths import preprocessing_dir
from src.utils.tools import finite_mask
from src.utils.visualisation import first_timestep

# ---------------------------------------------------------------------
# Hardcoded configuration
# ---------------------------------------------------------------------
# Input NetCDF
in_path = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/avh15c1_lai.nc")

# Output directories
current_dir = preprocessing_dir / "lai_obs" / "avh15c1"
sanity_dir = current_dir / "sanity_check"
mask_dir = sanity_dir / "nan_mask"
plots_dir = sanity_dir / "plots"
plots_orig_dir = plots_dir / "original"
plots_mask_dir = plots_dir / "mask"

# Create directories
mask_dir.mkdir(parents=True, exist_ok=True)
plots_orig_dir.mkdir(parents=True, exist_ok=True)
plots_mask_dir.mkdir(parents=True, exist_ok=True)

# Overwrite behaviour
OVERWRITE = False

# Optional timesteps override (set to None unless you need 365)
N_TIMESTEPS = 365 if "potential_radiation" in in_path.stem else None

# ---------------------------------------------------------------------
# Derived output paths
# ---------------------------------------------------------------------
mask_nc = mask_dir / in_path.name
plot_orig = plots_orig_dir / f"{in_path.stem}.png"
plot_mask = plots_mask_dir / f"{in_path.stem}.png"

# ---------------------------------------------------------------------
# Run mask generation + plotting
# ---------------------------------------------------------------------
print(f"[INFO] Input file: {in_path}")
print(f"[INFO] Output mask: {mask_nc}")
print(f"[INFO] Plot outputs:")
print(f"        - Original: {plot_orig}")
print(f"        - Mask:     {plot_mask}")

# 1. Create NaN mask NetCDF
if mask_nc.exists() and not OVERWRITE:
    print(f"[SKIP] Mask already exists: {mask_nc}")
else:
    print(f"[RUN] Generating NaN mask for {in_path.name}")
    if N_TIMESTEPS:
        finite_mask(in_path, mask_nc, overwrite=OVERWRITE, n_timesteps=N_TIMESTEPS)
    else:
        finite_mask(in_path, mask_nc, overwrite=OVERWRITE)

# 2. Plot original first timestep
if plot_orig.exists() and not OVERWRITE:
    print(f"[SKIP] Plot already exists: {plot_orig}")
else:
    print(f"[RUN] Plotting first timestep of original data")
    first_timestep(in_path, output_path=plot_orig, overwrite=OVERWRITE)

# 3. Plot mask first timestep
if plot_mask.exists() and not OVERWRITE:
    print(f"[SKIP] Plot already exists: {plot_mask}")
else:
    print(f"[RUN] Plotting first timestep of NaN mask")
    first_timestep(mask_nc, output_path=plot_mask, overwrite=OVERWRITE)

print("\n[DONE] NaN mask and plots written successfully.")