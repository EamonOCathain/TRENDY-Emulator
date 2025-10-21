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
import argparse
# ---------------- Project imports ---------------- #
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.analysis.vis_modular import *
from src.analysis.process_arrays import *
from src.paths.paths import visualisation_dir, masks_dir
from src.dataset.variables import var_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", default="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/nudge_array_check/z_adaptive", type=Path)
    ap.add_argument("--labs_dir",  default="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference", type=Path)
    ap.add_argument("--scenario", default="S3")
    ap.add_argument("--tile_index", default = 217, help="Add this if you want to run for a single tile.")
    ap.add_argument("--analysis_cache_dir", default=None)
    ap.add_argument("--out_dir", default="nudge_array_check/z_adaptive", help = "A folder name which gets added under the main plot path.")
    ap.add_argument("--spatial_avg_timeseries", default=True, action="store_true" )
    
    args = ap.parse_args()

    monthly_vars = var_names['monthly_outputs']
    annual_vars  = var_names['annual_outputs']
    all_vars = monthly_vars + annual_vars

    # List the arrays to work through
    MIN = 0.00
    MAX = 0.05
    STEP = 0.01
    DECIMALS = 2

    # safer: use linspace with calculated number of steps
    num_steps = int(round((MAX - MIN) / STEP)) + 1
    values = np.linspace(MIN, MAX, num_steps)

    arrays = [f"nudge_{val:.{DECIMALS}f}" for val in values]
    #arrays.append("nudge_0.00")

    print(arrays)
    
    # Analysis cache dir
    if args.analysis_cache_dir:
        ana_cache_dir = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/analysis/visualise/analysis_cache") / args.analysis_cache_dir
    else:
        ana_cache_dir = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/analysis/visualise/analysis_cache") / Path(args.preds_dir).stem
    # Plots out dir
    plot_root_dir = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/plots")
    if not args.out_dir:
            out_dir = plot_root_dir / Path(args.preds_dir).stem
    else:
        out_dir = plot_root_dir / args.out_dir

    # precompute tile bounds if provided
    if args.tile_index is not None:
        lat_min, lat_max, lon_min, lon_max, *_ = tile_bounds_and_indices(args.tile_index)

    for var in all_vars:
        if var not in var_names['states']:
            continue

        # reset per-var containers
        space_avg_timeseries_items = []
        space_avg_timeseries_labels = []

        # labels (ground truth)
        labels_path = args.labs_dir / args.scenario / ("monthly.zarr" if var in monthly_vars else "annual.zarr")
        labs_ds = open_and_standardise(labels_path, var)
        
        if args.tile_index is not None:
            labs_ds = labs_ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            
        if args.spatial_avg_timeseries:
            labs_space_avg = space_avg(labs_ds)
            space_avg_timeseries_items.append(labs_space_avg) 
            space_avg_timeseries_labels.append(f"Labels - {var}")  

        # predictions for each nudge level
        for array in arrays:
            # Paths and open ds
            preds_path = args.preds_dir / args.scenario / array / ("monthly.zarr" if var in monthly_vars else "annual.zarr")
            print(preds_path)
            preds_ds = open_and_standardise(preds_path, var)
            # Cut to region
            if args.tile_index is not None:
                preds_ds = preds_ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            # Take space avg
            if args.spatial_avg_timeseries:
                preds_space_avg = space_avg(preds_ds)
                space_avg_timeseries_items.append(preds_space_avg)
                space_avg_timeseries_labels.append(f"Preds - {array}")

        # Plot multi line
        if args.spatial_avg_timeseries and space_avg_timeseries_items:
            plot_one(
                space_avg_timeseries_items,
                spec=PlotSpec(kind="line_multi",
                              extras={"series_labels": space_avg_timeseries_labels},
                              title=f"{args.scenario} - {var} - spatial mean"),
                out_path=out_dir / "spatial_avg_timeseries" / f"{var}_timeseries_spatial_avg.png",
            )
    

if __name__ == "__main__":
    try:
        print("[BOOT] starting plot.py")
        main()
        print("[BOOT] finished plot.py")
    except Exception as e:
        import traceback
        print("[FATAL] Unhandled exception in predict.py:", repr(e))
        traceback.print_exc()
        sys.exit(1)