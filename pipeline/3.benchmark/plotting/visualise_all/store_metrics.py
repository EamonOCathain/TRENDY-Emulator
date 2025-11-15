#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xarray as xr
import numpy as np
import cftime
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import re
import os, dask
import pandas

dask.config.set(scheduler="threads")

# ---------------- Project imports ---------------- #
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.dataset.variables import var_names
from src.utils.tools import slurm_shard
from src.paths.paths import masks_dir
from src.analysis.process_arrays import *
from src.analysis.vis_modular import *
from src.analysis.metrics import *
    
# Plotting
'''
ds  (base dataset with preds & labels)
│
├─ Early branch
    ├─ first_timestep(ds)
    ├─ last_timestep(ds)
│   ├─ scenario_avg(ds)
│   ├─ space_avg(ds)
│   ├─ r2_global(ds)
│   ├─ r2_spatial(ds)
│   └─ rmse_spatial(ds)
│
└─ Bias-rooted branch
    └─ bias(ds)
        ├─ time_avg(ds, bias)
        ├─ first/last timestep plots
        └─ trend(ds, bias)
            ├─ plot: trend slopes
            └─ detrend(ds, trend_ds, bias)
                ├─ Monthly variables:
                │   └─ seasonality(detrended_ds)
                │       └─ iav(detrended_ds, seasonality_ds)
                │           ├─ time_avg(iav_ds)
                │           └─ space_avg(iav_ds)
                │
                └─ Annual variables:
                    └─ iav_ds := detrended_ds
                        ├─ time_avg(iav_ds)
                        └─ space_avg(iav_ds)
'''


# Run main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_name", required=True, help="Name to store plots and caches under.")
    ap.add_argument("--preds_dir", type=Path, default="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/first_full_run")
    ap.add_argument("--labels_dir", type=Path, default="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference")
    ap.add_argument("--plot_dir", type=Path, default="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/plots/first_full_run")
    ap.add_argument("--cache_dir", type=Path, default="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/analysis/metric_arrays")
    ap.add_argument("--scenarios", nargs="+", choices=['S0', 'S1', 'S2', 'S3', 'all'], required=True, help="Scenarios to include. Pass 'all' or any combination, e.g. --scenarios S0 S2.")
    ap.add_argument("--overwrite_cache", action="store_true", help = "overwrite the cached zarrs of the metric arrays")

    # Mutually exclusive: cannot set both test_subset and time_slice
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--test_subset", action="store_true", help="Use predefined test subset.")
    group.add_argument("--time_slice", type=parse_time_slice_arg, default=None, help="Time slice as two YYYY-MM-DD dates, e.g. '1902-02-02,1905-02-02'.")

    args = ap.parse_args()
    
    # First paths
    preds_dir = Path(args.preds_dir)
    labels_dir = Path(args.labels_dir)
    # Paths
    job_suffix = "test" if args.test_subset else ""
    plot_dir  = Path(args.plot_dir)  / f"{args.job_name}{'_' + job_suffix if job_suffix else ''}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) / f"{args.job_name}{'_' + job_suffix if job_suffix else ''}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Tasks for slurm
    metrics = ["first_timestep", "last_timestep", "space_avg", "time_avg", "r2_global", "r2_spatial", "rmse_spatial", "trends"]
    
    variables = (
        list(var_names.get("monthly_outputs", [])) +
        list(var_names.get("annual_outputs", []))
    )

    # Build all (var, metric) tasks
    all_tasks = [(var, metric) for var in variables for metric in metrics]
    print(len(all_tasks))

    # Shard across SLURM
    tasks = slurm_shard(all_tasks)
    
    if "all" in args.scenarios:
        scenarios = ["S0", "S1", "S2", "S3"]
    else:
        scenarios = args.scenarios
    
    tvt_mask = xr.open_dataarray(masks_dir / "tvt_mask.nc")
    
    # Main Loop
    for var, metric in tasks:
        
        print(f"Processing {var}.")
        pred_da_list = []
        label_da_list = []
        
        # Plotting Paths
        first_step_plot_dir = plot_dir / "timesteps" / "first" / var
        last_step_plot_dir = plot_dir / "timesteps" / "last" / var
        time_avg_plot_dir = plot_dir / "time_avg" / var
        spatial_avg_plot_dir = plot_dir / "spatial_avg" / var
        trend_plot_dir = plot_dir / "trends" / var
        seasonality_plot_dir = plot_dir / "seasonality" / var
        iav_plot_dir = plot_dir / "iav" / var
        iav_time_avg_plot_dir = iav_plot_dir / "time_avg" / var
        iav_space_avg_plot_dir = iav_plot_dir / "space_avg" / var
        r2_global_plot_dir = plot_dir / "r2" / "global" / var
        r2_spatial_plot_dir = plot_dir / "r2" / "spatial" / var
        rmse_spatial_plot_dir = plot_dir / "rmse" / var

        # create them all
        dirs = [
            first_step_plot_dir,
            last_step_plot_dir,
            time_avg_plot_dir,
            spatial_avg_plot_dir,
            trend_plot_dir,
            seasonality_plot_dir,
            iav_plot_dir,
            iav_time_avg_plot_dir,
            iav_space_avg_plot_dir,
            r2_global_plot_dir,
            r2_spatial_plot_dir,
            rmse_spatial_plot_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        # Cache outputs
        bias_cache_out = cache_dir / "bias" / f"{var}_bias.zarr"
        space_avg_cache_out = cache_dir / "space_avg" / f"{var}_space_avg.zarr"
        time_avg_cache_out = cache_dir / "time_avg" / f"{var}_time_avg.zarr"
        r2_global_cache_out = cache_dir / "r2_global" / f"{var}_r2_global.zarr"
        r2_spatial_out = cache_dir / "r2_spatial" / f"{var}_r2_spatial.zarr"
        rmse_spatial_out = cache_dir / "rmse_spatial" / f"{var}_rmse_spatial.zarr"
        trend_cache_out = cache_dir / "trend" / f"{var}_trend.zarr"
        seasonality_cache_out = cache_dir / "seasonality" / f"{var}_seasonality.zarr"
        seasonality_one_year_sa_cache_out = cache_dir / "seasonality" / f"{var}_seasonality_one_year_sa.zarr"
        iav_cache_out = cache_dir / "iav" / f"{var}_iav.zarr"
        iav_space_avg_cache_out = cache_dir / "iav" / f"{var}_iav_space_avg.zarr"
        iav_time_avg_cache_out = cache_dir / "iav" / f"{var}_iav_time_avg.zarr"
        detrended_out = cache_dir / "detrended" / f"{var}_detrended.zarr"
        
        # Loop scenarios and store the preds and labs path
        for scenario in scenarios:
            if var in var_names['annual_outputs']:
                preds_path = preds_dir / scenario / "zarr" / "annual.zarr"
                labels_path = labels_dir / scenario / "annual.zarr"
                
            elif var in var_names['monthly_outputs']:
                preds_path = preds_dir / scenario / "zarr" / "monthly.zarr"
                labels_path = labels_dir / scenario / "monthly.zarr"

            # open and standardise the das
            preds_da = open_and_standardise(preds_path, var)
            labels_da = open_and_standardise(labels_path, var)

            # Mask to the test if arg'ed
            if args.test_subset:
               # location subset
                mask_b = tvt_mask
                if "time" in preds_da.coords:
                    mask_b = tvt_mask.broadcast_like(preds_da, exclude=("time",))

                preds_da_test_loc  = preds_da.where(mask_b == 2).rename(f"{preds_da.name}_test_location")
                labels_da_test_loc = labels_da.where(mask_b == 2).rename(f"{labels_da.name}_test_location")

                # time subsets (two periods)
                preds_da_test_time_early, preds_da_test_time_late = subset_time(preds_da, test_subset=args.test_subset, time_slice=args.time_slice)
                labels_da_test_time_early, labels_da_test_time_late = subset_time(labels_da, test_subset=args.test_subset, time_slice=args.time_slice)
                
                # Create dict of subsets
                subsets = {
                    "test_loc":   (preds_da_test_loc,   labels_da_test_loc),
                    "test_early": (preds_da_test_time_early, labels_da_test_time_early),
                    "test_late":  (preds_da_test_time_late,  labels_da_test_time_late),
                    "full":       (preds_da,           labels_da)
                }
                for sub, (p,l) in subsets.items():
                    pred_da_list.append((f"{scenario}", p.expand_dims(subset=[sub])))
                    label_da_list.append((f"{scenario}", l.expand_dims(subset=[sub])))
                
            else:
                pred_da_list.append((scenario, preds_da))
                label_da_list.append((scenario, labels_da))
        
        # Combine to single ds 
        ds = combine_to_dataset(pred_da_list, label_da_list)
        
        # Slice time (this slices either to test subset or time slice or doesnt slice)
        ds = subset_time(ds, test_subset=False, time_slice=args.time_slice)
        
        # Add scenario averages and bias
        if len(scenarios) > 1:
           ds = scenario_avg(ds)
           
        # First Timestep
        if metric == 'first_timestep':
            first_ds = ds.isel(time=0).squeeze()
            stack_map_pairs(first_ds, suptitle=f"{var.upper()} First Timestep", out_path= first_step_plot_dir / f"{var}_first_timestep.png")
            continue
        
        # Last Timestep    
        if metric == 'last_timestep':
            last_ds = ds.isel(time=-1).squeeze()
            stack_map_pairs(last_ds, suptitle=f"{var.upper()} Last Timestep", out_path= last_step_plot_dir / f"{var}_last_timestep.png")
            continue
        
        # Space avg branch
        if metric == "space_avg":
            space_avg_ds = compute_cache_and_open(func=space_avg, ds = ds, store_path = space_avg_cache_out, overwrite = args.overwrite_cache)
            # Plot Spatial Average Timeseries
            plot_timeseries_pairs_grid(space_avg_ds, suptitle=f"Spatial Average of {var}", out_path=spatial_avg_plot_dir / f"spatial_avg_{var}.png")
            continue
        
        if metric == 'r2_global':
            # R2 
            stack_global_r2_scatter(ds=ds, suptitle = f"Global r2 of {var.upper()}", out_path = r2_global_plot_dir / f"global_r2_{var}.png")
            continue
            
        if metric == 'r2_spatial':
            ds_chunked = ds.chunk({'time': -1, 'lat': 30, 'lon': 30})
            r2_spatial_ds = compute_cache_and_open(func=spatial_r2_rmse, ds = ds_chunked, store_path = r2_spatial_out, overwrite = args.overwrite_cache, metric = 'r2')
            stack_map_pairs(r2_spatial_ds, suptitle=f"{var.upper()} R² (spatial)", out_path=r2_spatial_plot_dir / f"{var}_r2_spatial.png")
            continue
            
        if metric == 'rmse_spatial':
            ds_chunked = ds.chunk({'time': -1, 'lat': 30, 'lon': 30})
            rmse_spatial_ds = compute_cache_and_open(func=spatial_r2_rmse, ds = ds_chunked, store_path = rmse_spatial_out, overwrite = args.overwrite_cache, metric = 'rmse')
            stack_map_pairs(rmse_spatial_ds, suptitle=f"{var.upper()} RMSE (spatial)", out_path=rmse_spatial_plot_dir / f"{var}_rmse_spatial.png")
            continue
        
        # Bias branch
        if metric in ['trends', 'time_avg']:
            bias_ds = bias(ds)
            
            # Time avg branch
            if metric == 'time_avg':
                time_avg_ds = compute_cache_and_open(func=time_avg, ds = bias_ds, store_path = time_avg_cache_out, overwrite = args.overwrite_cache)
                stack_map_pairs(time_avg_ds, suptitle=f"Time Average of {var}", out_path=  time_avg_plot_dir / f"time_avg_{var}.png")
                continue
        
            bias_ds_chunked = bias_ds.chunk({'time': 123, 'lat': 30, 'lon': 30})
        
            # Trends branch
            if metric == 'trends':  
                trend_ds = compute_cache_and_open(func=trend, ds = bias_ds_chunked, store_path = trend_cache_out, overwrite = args.overwrite_cache)
                slope_ds = trend_ds[[v for v in trend_ds.data_vars if v.endswith("_slope")]]
                # Plot trends
                stack_map_pairs(slope_ds, suptitle=f"Slope of Long Term Trend of {var.upper()}", out_path=  trend_plot_dir / f"{var}_trend_slope.png")
        
                # Detrended
                detrended_ds = compute_cache_and_open(func=detrend, ds=bias_ds_chunked, store_path=detrended_out, overwrite=args.overwrite_cache, trend_ds=trend_ds).persist()
                
                # Seasonality and IAV
                if var in var_names['monthly_outputs']:
                    seasonality_ds = compute_cache_and_open(func=seasonality, ds = detrended_ds, store_path = seasonality_cache_out, overwrite = args.overwrite_cache).persist()
                    seasonality_one_year_ds = seasonality_ds.isel(time=slice(0, 12))
                    seasonality_one_year_sa = compute_cache_and_open(func=space_avg, ds = seasonality_one_year_ds, store_path = seasonality_one_year_sa_cache_out, overwrite = args.overwrite_cache).persist()

                    # Seasonality
                    plot_timeseries_pairs_grid(seasonality_one_year_sa,  suptitle=f"Seasonality of {var.upper()}", out_path=seasonality_plot_dir / f"{var}_seasonality.png")
            
                    # IAV
                    iav_ds = compute_cache_and_open(func=iav, ds = detrended_ds, store_path = iav_cache_out, overwrite = args.overwrite_cache, seasonality_ds = seasonality_ds).persist()
    
                else:
                    iav_ds = detrended_ds
            
                iav_time_avg_ds = compute_cache_and_open(func=time_avg, ds = iav_ds, store_path = iav_time_avg_cache_out, overwrite = args.overwrite_cache)
                iav_space_avg_ds = compute_cache_and_open(func=space_avg, ds = iav_ds, store_path = iav_space_avg_cache_out, overwrite = args.overwrite_cache)
                
                # Plot IAV
                plot_timeseries_pairs_grid(iav_space_avg_ds,  suptitle=f"Inter-Annual Variability of {var.upper()}", out_path=iav_space_avg_plot_dir / f"{var}_iav_space_avg.png")
                plot_timeseries_pairs_grid(iav_time_avg_ds,  suptitle=f"Inter-Annual Variability of {var.upper()}", out_path=iav_time_avg_plot_dir / f"{var}_iav_time_avg.png")
    
if __name__ == "__main__":
    main()