#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import dask

dask.config.set(scheduler="threads")

# ---------------- Project imports ---------------- #
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.dataset.variables import var_names
from src.paths.paths import masks_dir
from src.utils.tools import slurm_shard
from src.analysis.process_arrays import open_and_standardise, subset_time


# ---------- helpers ----------
def gather_points(preds: xr.DataArray, labs: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten, drop NaNs, return (x=labels, y=preds)."""
    x = labs.values.reshape(-1)
    y = preds.values.reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def combined_subsets_points(
    preds_da: xr.DataArray,
    labels_da: xr.DataArray,
    *,
    tvt_mask: xr.DataArray | None,
    test_subset: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return concatenated (x, y) points for a variable, either full data
    or combined test subsets (test locations + early + late).
    """
    xs, ys = [], []

    if test_subset:
        # -- test locations
        preds_loc = preds_da.where(tvt_mask == 2)
        labs_loc  = labels_da.where(tvt_mask == 2)
        x, y = gather_points(preds_loc, labs_loc)
        if x.size:
            xs.append(x); ys.append(y)

        # -- test periods (early / late)
        preds_early, preds_late = subset_time(preds_da, test_subset=True), None
        labs_early,  labs_late  = subset_time(labels_da, test_subset=True), None
        # NOTE: your subset_time (latest version) returns *two* arrays when test_subset=True.
        # If your current function returns a single concatenated array, just adapt the two lines above accordingly.

        # handle if subset_time returns a tuple (early, late); if not, fall back to whole obj
        if isinstance(preds_early, tuple):
            preds_early, preds_late = preds_early
        if isinstance(labs_early, tuple):
            labs_early, labs_late = labs_early

        if preds_early is not None and labs_early is not None:
            x, y = gather_points(preds_early, labs_early)
            if x.size:
                xs.append(x); ys.append(y)

        if preds_late is not None and labs_late is not None:
            x, y = gather_points(preds_late, labs_late)
            if x.size:
                xs.append(x); ys.append(y)
    else:
        x, y = gather_points(preds_da, labels_da)
        if x.size:
            xs.append(x); ys.append(y)

    if xs:
        return np.concatenate(xs), np.concatenate(ys)
    else:
        return np.array([]), np.array([])


def r2_from_xy(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def plot_scenario(
    *,
    scenario: str,
    variables: list[str],
    preds_dir: Path,
    labels_dir: Path,
    out_path: Path,
    test_subset: bool,
    tvt_mask: xr.DataArray | None,
    ncols: int = 3,
):
    """
    One figure per scenario; subplots = variables (3 columns).
    Each subplot: single-color scatter, alpha=0.1, combined subsets if test_subset=True.
    """
    nvars = len(variables)
    nrows = math.ceil(nvars / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes = axes.ravel()

    for i, var in enumerate(variables):
        ax = axes[i]

        # open preds & labels
        if var in var_names["annual_outputs"]:
            preds_path = preds_dir / scenario / "zarr" / "annual.zarr"
            labels_path = labels_dir / scenario / "annual.zarr"
        else:
            preds_path = preds_dir / scenario / "zarr" / "monthly.zarr"
            labels_path = labels_dir / scenario / "zarr" / "monthly.zarr"

        preds_da = open_and_standardise(preds_path, var)
        labels_da = open_and_standardise(labels_path, var)

        # gather points (combined subsets if requested)
        x, y = combined_subsets_points(preds_da, labels_da, tvt_mask=tvt_mask, test_subset=test_subset)

        if x.size == 0:
            ax.set_title(f"{var.upper()} — no data", fontsize=10)
            ax.axis("off")
            continue

        # scatter (single color, alpha=0.1)
        ax.scatter(x, y, s=2, alpha=0.1, color="C0")

        # 1:1 line
        lo = np.nanmin([x.min(), y.min()])
        hi = np.nanmax([x.max(), y.max()])
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)

        # R²
        r2 = r2_from_xy(x, y)

        ax.set_title(f"{var.upper()}  (R²={r2:.3f})", fontsize=10)
        ax.set_xlabel("Labels")
        ax.set_ylabel("Predictions")

    # turn off any unused axes
    for j in range(nvars, len(axes)):
        axes[j].axis("off")

    sufix = "TEST" if test_subset else "FULL"
    fig.suptitle(f"{scenario} — R² Scatter (combined subsets: {sufix})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_name", required=True)
    ap.add_argument("--preds_dir", type=Path, required=True)
    ap.add_argument("--labels_dir", type=Path, required=True)
    ap.add_argument("--plot_dir", type=Path, required=True)
    ap.add_argument("--scenarios", nargs="+", choices=["S0", "S1", "S2", "S3", "all"], required=True)
    ap.add_argument("--test_subset", action="store_true",
                    help="If set, combine test locations and early/late test periods in each subplot.")
    ap.add_argument("--ncols", type=int, default=3, help="Number of subplot columns (default: 3).")
    args = ap.parse_args()

    # scenarios to run
    scenarios = ["S0", "S1", "S2", "S3"] if "all" in args.scenarios else args.scenarios
    variables = var_names["outputs"]

    # base out directory
    plot_root = Path(args.plot_dir) / args.job_name / "global" / "r2_scatter_combined"
    plot_root.mkdir(parents=True, exist_ok=True)

    # shard by scenarios so each task produces its own figure
    tasks = slurm_shard(scenarios)
    print(f"[SLURM] This shard will process scenarios: {tasks}", flush=True)

    # load mask only if needed
    tvt_mask = xr.open_dataarray(masks_dir / "tvt_mask.nc") if args.test_subset else None

    for scenario in tasks:
        out_path = plot_root / f"{scenario}_r2_scatter_all_vars.png"
        plot_scenario(
            scenario=scenario,
            variables=variables,
            preds_dir=Path(args.preds_dir),
            labels_dir=Path(args.labels_dir),
            out_path=out_path,
            test_subset=args.test_subset,
            tvt_mask=tvt_mask,
            ncols=args.ncols,
        )


if __name__ == "__main__":
    main()