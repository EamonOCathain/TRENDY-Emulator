import xarray as xr
import os
from pathlib import Path
import sys
import pandas as pd

OVERWRITE = True

# Some Paths
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.paths.paths import (
    masks_dir,
    masking_dir
)

from src.utils.tools import combine_masks
from src.utils.visualisation import first_timestep

current_dir = masking_dir / "nan_mask"
single_nan_masks_dir = current_dir / "data"
forcing_nan = single_nan_masks_dir / "forcing"
model_nan = single_nan_masks_dir / "model_outputs"
ensmean_nan = single_nan_masks_dir / "ensmean"

# Collect files
all_forcing = sorted(forcing_nan.rglob("*.nc"))
all_models  = sorted(model_nan.rglob("*.nc"))
all_ensmean = sorted(ensmean_nan.rglob("*.nc"))

forcing_and_models = all_models + all_forcing + all_ensmean

# Ensure output dir exists
masks_dir.mkdir(parents=True, exist_ok=True)

forcing_out_path  = masks_dir / "forcing_nan_mask.nc"
combined_nan_path = masks_dir / "training_nan_mask.nc"

# Combine masks (guard against empties)
if all_forcing:
    combine_masks(all_forcing, forcing_out_path, overwrite=OVERWRITE, new_var_name="finite_mask")
else:
    print("No forcing mask files found in:", forcing_nan)

if forcing_and_models:
    combine_masks(forcing_and_models, combined_nan_path, overwrite=OVERWRITE, new_var_name="finite_mask")
else:
    print("No forcing+model mask files found in:", model_nan, "and", forcing_nan)

# Plots
plot_dir = current_dir / "val_plot/combined"
plot_dir.mkdir(exist_ok=True, parents=True)

plot_path_forcing  = plot_dir / "finite_mask_forcing.png"
first_timestep(forcing_out_path,  output_path=plot_path_forcing,  overwrite=OVERWRITE)

plot_path_combined = plot_dir / "finite_mask_combined.png"
first_timestep(combined_nan_path, output_path=plot_path_combined, overwrite=OVERWRITE)