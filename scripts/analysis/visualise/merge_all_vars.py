#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xarray as xr
import numpy as np
import dask
from typing import List

dask.config.set(scheduler="threads")

# ---------------- Project imports ---------------- #
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.dataset.variables import var_names
from src.utils.tools import slurm_shard
from src.analysis.process_arrays import open_zarr
from src.analysis.vis_modular import (
    stack_map_pairs,
    stack_maps,
    plot_timeseries_pairs_grid,
    stack_global_r2_scatter,
)

# ---------------------------------------------------------------------
# This script reads metric caches for ALL variables
# and produces ONE figure per metric (subplots = variables).
# Only metrics that are cached by the first script are included here.
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_name", required=True, help="Name used to locate plots and caches.")
    ap.add_argument("--plot_dir", type=Path, required=True, help="Base plot directory (same root as writer script).")
    ap.add_argument("--cache_dir", type=Path, required=True, help="Base cache directory (same root as writer script).")
    
    # Optional: allow a subset of variables
    ap.add_argument("--vars", nargs="+", default=None, help="Optional subset of variables to include (default: all outputs).")

    args = ap.parse_args()

    # Paths
    plot_dir  = Path(args.plot_dir)  / args.job_name
    cache_dir = Path(args.cache_dir) / args.job_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Metrics we know are cached by the first script
    metrics = ["space_avg", "time_avg", "r2_global", "r2_spatial", "rmse_spatial", "trends", "iav"]

    # Shard only by metric
    tasks = slurm_shard(metrics)

    # Which variables to include
    variables = args.vars if args.vars is not None else var_names["outputs"]

    # Per-metric plot directories (group-all-variables)
    by_metric_plot_dirs = {
        "space_avg":   plot_dir / "spatial_avg" / "all_vars",
        "time_avg":    plot_dir / "time_avg"     / "all_vars",
        "r2_global":   plot_dir / "r2" / "global" / "all_vars",  
        "r2_spatial":  plot_dir / "r2" / "spatial" / "all_vars",
        "rmse_spatial":plot_dir / "rmse" / "all_vars",
        "trends":      plot_dir / "trends" / "all_vars",
    }
    for p in by_metric_plot_dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    # Helper to open many Zarrs if present
    def _open_if_exists(path: Path) -> xr.Dataset | None:
        try:
            if path.exists():
                return open_zarr(path)
        except Exception as e:
            print(f"[WARN] Failed to open {path}: {e}", flush=True)
        return None

    for metric in tasks:
        print(f"[ALL-VARS] Processing metric: {metric}", flush=True)

        if metric == "space_avg":
            # Each var has: cache_dir/space_avg/{var}_space_avg.zarr
            ds_list: List[xr.Dataset] = []
            for var in variables:
                z = cache_dir / "space_avg" / f"{var}_space_avg.zarr"
                ds = _open_if_exists(z)
                if ds is not None:
                    ds_list.append(ds)
            if not ds_list:
                print("[SKIP] No space_avg caches found.", flush=True)
                continue

            # Merge all variables (Label_*, Predicted_* across vars/scenarios) into one ds
            ds_all = xr.merge(ds_list, compat="override")
            out_path = by_metric_plot_dirs["space_avg"] / "spatial_avg_ALL.png"
            plot_timeseries_pairs_grid(
                ds_all,
                suptitle="Spatial Average – All variables",
                out_path=out_path,
            )
            print(f"[OK] Wrote {out_path}", flush=True)

        elif metric == "time_avg":
            # Each var has: cache_dir/time_avg/{var}_time_avg.zarr (from the bias branch)
            ds_list: List[xr.Dataset] = []
            for var in variables:
                z = cache_dir / "time_avg" / f"{var}_time_avg.zarr"
                ds = _open_if_exists(z)
                if ds is not None:
                    ds_list.append(ds)
            if not ds_list:
                print("[SKIP] No time_avg caches found.", flush=True)
                continue

            ds_all = xr.merge(ds_list, compat="override")
            out_path = by_metric_plot_dirs["time_avg"] / "time_avg_ALL.png"
            # This Dataset contains Label_/Predicted_/Bias_ variables → pairs are auto-detected
            stack_map_pairs(
                ds_all,
                suptitle="Time Average – All variables",
                out_path=out_path,
            )
            print(f"[OK] Wrote {out_path}", flush=True)

        elif metric == "r2_spatial":
            # Each var has: cache_dir/r2_spatial/{var}_r2_spatial.zarr
            # These datasets contain 2D fields named like {suffix}_r2 (no Label/Pred prefixes)
            ds_list: List[xr.Dataset] = []
            for var in variables:
                z = cache_dir / "r2_spatial" / f"{var}_r2_spatial.zarr"
                ds = _open_if_exists(z)
                if ds is not None:
                    ds_list.append(ds)
            if not ds_list:
                print("[SKIP] No r2_spatial caches found.", flush=True)
                continue

            ds_all = xr.merge(ds_list, compat="override")
            out_path = by_metric_plot_dirs["r2_spatial"] / "r2_spatial_ALL.png"
            # No (Label, Predicted) naming → stack_map_pairs will fall back and plot all 2D vars
            stack_map_pairs(
                ds_all,
                suptitle="Spatial R² – All variables",
                out_path=out_path,
            )
            print(f"[OK] Wrote {out_path}", flush=True)

        elif metric == "rmse_spatial":
            # Each var has: cache_dir/rmse_spatial/{var}_rmse_spatial.zarr
            ds_list: List[xr.Dataset] = []
            for var in variables:
                z = cache_dir / "rmse_spatial" / f"{var}_rmse_spatial.zarr"
                ds = _open_if_exists(z)
                if ds is not None:
                    ds_list.append(ds)
            if not ds_list:
                print("[SKIP] No rmse_spatial caches found.", flush=True)
                continue

            ds_all = xr.merge(ds_list, compat="override")
            out_path = by_metric_plot_dirs["rmse_spatial"] / "rmse_spatial_ALL.png"
            stack_map_pairs(
                ds_all,
                suptitle="Spatial RMSE – All variables",
                out_path=out_path,
            )
            print(f"[OK] Wrote {out_path}", flush=True)

        elif metric == "trends":
            # Each var has: cache_dir/trend/{var}_trend.zarr with *_slope and *_intercept
            # We’ll collect ALL *_slope across variables and plot them together
            slope_ds_list: List[xr.Dataset] = []
            for var in variables:
                z = cache_dir / "trend" / f"{var}_trend.zarr"
                ds = _open_if_exists(z)
                if ds is None:
                    continue
                slope_vars = [v for v in ds.data_vars if v.endswith("_slope")]
                if slope_vars:
                    slope_ds_list.append(ds[slope_vars])
            if not slope_ds_list:
                print("[SKIP] No trend slope fields found.", flush=True)
                continue

            ds_all_slopes = xr.merge(slope_ds_list, compat="override")
            out_path = by_metric_plot_dirs["trends"] / "trend_slopes_ALL.png"
            # Names do not match Label/Pred → fallback will plot all 2D maps
            stack_map_pairs(
                ds_all_slopes,
                suptitle="Trend slopes – All variables",
                out_path=out_path,
            )
            print(f"[OK] Wrote {out_path}", flush=True)
        
        elif metric == "r2_global":
            # Each var has: cache_dir/r2_global/{var}_r2_global.zarr
            ds_list: List[xr.Dataset] = []
            for var in variables:
                z = cache_dir / "r2_global" / f"{var}_r2_global.zarr"
                ds = _open_if_exists(z)
                if ds is not None:
                    ds_list.append(ds)
            if not ds_list:
                print("[SKIP] No r2_global caches found.", flush=True)
                continue

            # Merge all variables into a single Dataset. The plotting util detects pairs internally.
            ds_all = xr.merge(ds_list, compat="override")
            out_path = by_metric_plot_dirs["r2_global"] / "global_r2_ALL.png"
            stack_global_r2_scatter(
                ds=ds_all,
                suptitle="Global R² – All variables",
                out_path=out_path,
            )
            print(f"[OK] Wrote {out_path}", flush=True)

        else:
            print(f"[SKIP] Metric '{metric}' is not handled here.", flush=True)


if __name__ == "__main__":
    main()