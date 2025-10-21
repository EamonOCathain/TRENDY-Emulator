
from __future__ import annotations

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import sys

# ---------------- Project imports ---------------- #
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.analysis.vis_modular import *
from src.analysis.process_arrays import *
from src.paths.paths import visualisation_dir, masks_dir

path_pred = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/inversion_check/S3/zarr/annual.zarr")
path_lab = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference/S3/annual.zarr")

var = "cVeg"

current_dir = visualisation_dir / "vis_modular"
out_dir = current_dir / "test_plots"
out_dir.mkdir(parents=True, exist_ok=True)

da_pred = open_and_standardise(path_pred, var)
da_lab = open_and_standardise(path_lab, var)

t = 1

da_pred_2d = da_pred[t, :, :].squeeze()
da_pred_1d = da_pred.sel(lat = 51, lon = 11, method="nearest").squeeze()
da_lab_2d = da_lab[t, :, :].squeeze()
da_lab_1d = da_lab.sel(lat = 51, lon=11.75, method="nearest").squeeze()

'''# 1) Single plot — auto-inferred kind
plot_one(da_pred_2d, spec=PlotSpec(title="Preds Map"), out_path = out_dir /"preds_map.png")
plot_one(da_lab_2d, spec=PlotSpec(title="Labs Map"), out_path = out_dir /"labs_map.png")

# 2) Force a specific kind (line)
plot_one(da_pred_1d, spec=PlotSpec(kind="line", title="Timeseries"), out_path = out_dir /"1d_test.png")

# 3) Scatter (pass a pair)
plot_one((da_pred_2d, da_lab_2d), spec=PlotSpec(kind="scatter", title="Obs vs Pred"), out_path =out_dir / "scatter_test.png")'''

# 4) Grid: pass the SAME data items you would pass to plot_one, not Axes.
items = [da_pred_2d,
         da_lab_2d     
]

specs = [
    PlotSpec(title="Preds Map"),
    PlotSpec(title="Labs Map")
]

# Stack and save
stack_maps_vertical(items, specs=specs, suptitle="Overview", out_path= out_dir / "stacked_maps.png")



