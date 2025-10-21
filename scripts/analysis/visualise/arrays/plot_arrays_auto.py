from __future__ import annotations

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Optional, Sequence
import sys
import argparse

# ---------------- Project imports ---------------- #
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.analysis.vis_modular import *
from src.analysis.process_arrays import *
from src.paths.paths import visualisation_dir, masks_dir
from src.dataset.variables import var_names


def _make_values_in_segment(a: float, b: float, step: float, decimals: int) -> list[float]:
    """Return a numerically-stable [a..b] sequence (inclusive) with given step, rounded."""
    # how many steps fit in [a, b]
    n = int(round((b - a) / step)) + 1
    vals = np.linspace(a, b, n)
    # guard against 1e-16 drift with rounding
    return [round(float(v), decimals) for v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", default="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/nudge_array_check/z_adaptive", type=Path)
    ap.add_argument("--labs_dir",  default="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference", type=Path)
    ap.add_argument("--scenario", default="S3")
    ap.add_argument("--tile_index", type=int, default=217, help="Run for a single tile (lat/lon subsetting).")
    ap.add_argument("--analysis_cache_dir", default=None)
    ap.add_argument("--out_dir", default="nudge_array_check/z_adaptive", help="Root folder under plots/. Segment label will be appended.")
    ap.add_argument("--spatial_avg_timeseries", action="store_true", default=True)

    # sweep/segment controls
    ap.add_argument("--min_lambda", type=float, default=0.25)
    ap.add_argument("--max_lambda", type=float, default=0.50)
    ap.add_argument("--step",       type=float, default=0.01, help="Spacing of nudge folders inside each segment.")
    ap.add_argument("--segments",   type=int,   default=5,    help="Number of equal-width segments in [min,max].")
    ap.add_argument("--decimals",   type=int,   default=2,    help="Label formatting/rounding for folder names.")

    args = ap.parse_args()

    monthly_vars = var_names['monthly_outputs']
    annual_vars  = var_names['annual_outputs']
    all_vars = monthly_vars + annual_vars

    # figure out plot root
    plot_root_dir = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/plots")

    # analysis cache dir
    if args.analysis_cache_dir:
        ana_cache_dir = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/analysis/visualise/analysis_cache") / args.analysis_cache_dir
    else:
        ana_cache_dir = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/analysis/visualise/analysis_cache") / Path(args.preds_dir).stem

    # precompute tile bounds if provided
    if args.tile_index is not None:
        lat_min, lat_max, lon_min, lon_max, *_ = tile_bounds_and_indices(args.tile_index)

    # ---- build equal-width segments on [min_lambda, max_lambda]
    lam0 = float(args.min_lambda)
    lam1 = float(args.max_lambda)
    nseg = max(1, int(args.segments))
    boundaries = np.linspace(lam0, lam1, nseg + 1)

    # process each segment independently
    for si in range(nseg):
        a = round(float(boundaries[si]),     args.decimals)
        b = round(float(boundaries[si + 1]), args.decimals)

        # build the nudge array folder names that live in this [a,b] segment
        vals = _make_values_in_segment(a, b, args.step, args.decimals)
        arrays = [f"nudge_{v:.{args.decimals}f}" for v in vals]

        seg_label = f"{a:.{args.decimals}f}-{b:.{args.decimals}f}"
        seg_out_dir = plot_root_dir / args.out_dir / seg_label

        print(f"[SEG {si+1}/{nseg}] {seg_label} -> arrays: {arrays}")

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

            # predictions for each nudge level in this segment
            for array in arrays:
                preds_path = args.preds_dir / args.scenario / array / "zarr" / ("monthly.zarr" if var in monthly_vars else "annual.zarr")
                print(f"  [VAR {var}] {preds_path}")
                preds_ds = open_and_standardise(preds_path, var)

                if args.tile_index is not None:
                    preds_ds = preds_ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

                if args.spatial_avg_timeseries:
                    preds_space_avg = space_avg(preds_ds)
                    space_avg_timeseries_items.append(preds_space_avg)
                    space_avg_timeseries_labels.append(f"Preds - {array}")

            # Plot multi-line timeseries for this segment
            if args.spatial_avg_timeseries and space_avg_timeseries_items:
                plot_one(
                    space_avg_timeseries_items,
                    spec=PlotSpec(
                        kind="line_multi",
                        extras={"series_labels": space_avg_timeseries_labels,
                                "legend_title": "Series",
                                "legend_ncol": 1,
                                "width_factor": 1.6},
                        title=f"{args.scenario} - {var} - spatial mean [{seg_label}]"
                    ),
                    out_path=seg_out_dir / "spatial_avg_timeseries" / f"{var}_timeseries_spatial_avg.png",
                )


if __name__ == "__main__":
    try:
        print("[BOOT] starting plot.py")
        main()
        print("[BOOT] finished plot.py")
    except Exception as e:
        import traceback
        print("[FATAL] Unhandled exception in plot.py:", repr(e))
        traceback.print_exc()
        sys.exit(1)