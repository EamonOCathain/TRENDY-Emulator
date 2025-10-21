import xarray as xr
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd

OVERWRITE = False

# Some Paths
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.paths.paths import (
    scripts_dir, 
    preprocessing_dir,
    raw_data_dir,
    historical_dir,
    raw_outputs_dir,
    model_outputs_dir,
    masking_dir,
    preprocessed_dir
)

from src.utils.tools import slurm_shard, finite_mask
from src.utils.visualisation import first_timestep

# Paths
current_dir = masking_dir / "nan_mask"
misc_dir = preprocessing_dir / "model_outputs/data/4.misc"
model_outputs_dir = preprocessed_dir / "1x1/model_outputs"
# Collect all files in preprocessed_dir
forcing_files = list(historical_dir.rglob("*.nc"))
model_output_files = list(misc_dir.rglob('*.nc'))
ensmean_files = list(model_outputs_dir.rglob('*.nc'))
all_files = forcing_files + model_output_files + ensmean_files
subset_files = slurm_shard(ensmean_files)

out_dir = current_dir / "data"
out_dir.mkdir(parents=True, exist_ok=True)

out_dir = current_dir / "data"
(out_dir / "forcing").mkdir(parents=True, exist_ok=True)
(out_dir / "model_outputs").mkdir(parents=True, exist_ok=True)
(out_dir / "ensmean").mkdir(parents=True, exist_ok=True)

# Forcing nan masks 
for file in forcing_files:
    if file not in subset_files:
        continue
    out_path = out_dir / "forcing" / file.name
    if out_path.exists() and not OVERWRITE:
        print(f"Skipping {out_path}", flush=True)
        continue
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if 'nfer' in file.stem:
        print(f"skipping nfer file {file.stem}", flush=True)
        continue
    elif 'potential_radiation' in file.stem:
        finite_mask(file, out_path, overwrite=OVERWRITE, n_timesteps=365)
    else:
        finite_mask(file, out_path, overwrite=OVERWRITE)

# Nan masks for individual model outputs
model_output_nan_masks = []
for file in model_output_files:
    if file not in subset_files:
        continue
    out_path = out_dir / "model_outputs" / file.name
    if out_path.exists() and not OVERWRITE:
        print(f"Skipping {out_path}", flush=True)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        finite_mask(file, out_path, overwrite=OVERWRITE)
    model_output_nan_masks.append(out_path)

# Ensmean masks
ens_plot_dir = current_dir / "val_plot/ensmean"
ens_plot_dir.mkdir(parents=True, exist_ok=True)

for file in ensmean_files:
    if file not in subset_files:
        continue

    out_path = out_dir / "ensmean" / file.name   # mask path (.nc)
    if out_path.exists() and not OVERWRITE:
        print(f"[SKIP] {out_path} exists", flush=True)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        finite_mask(file, out_path, overwrite=OVERWRITE)

    # Plot to PNG (use plot_path, not out_path)
    plot_path = ens_plot_dir / f"{file.stem}.png"
    first_timestep(file, output_path=plot_path, overwrite=OVERWRITE)

# Plots
plot_dir = current_dir / "val_plot/model_outputs"
plot_dir.mkdir(parents=True, exist_ok=True)
for file in model_output_nan_masks:
    out_path = plot_dir / f"{file.stem}.png"
    first_timestep(file, output_path=out_path, overwrite=OVERWRITE)
