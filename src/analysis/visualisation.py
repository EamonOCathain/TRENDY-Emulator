#!/usr/bin/env python3
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
from sklearn.metrics import r2_score
# ---------------- Project imports ---------------- #
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))
from src.analysis.process_arrays import _find_var_groups

# --------------------- Applied to a DataArray ---------------------
def robust_limits_da(
    arr,
    *,
    mode="range",
    pct_low=1.0,
    pct_high=99.0,
    pct=99.0,
    zero_floor_if_nonneg=True,
    default=1.0,
):
    """
    Compute robust plotting limits from a single array.

    Parameters
    ----------
    arr : array-like
        Input array (any shape). NaNs are ignored.
    mode : {"range", "symmetric"}
        - "range": return (vmin, vmax) using percentiles.
        - "symmetric": return (-v, +v) using abs(values) percentile.
    pct_low : float
        Lower percentile for range mode.
    pct_high : float
        Upper percentile for range mode.
    pct : float
        Percentile for symmetric mode.
    zero_floor_if_nonneg : bool
        If True and data min >= 0, vmin forced to 0 in range mode.
    default : float
        Value to return if no finite data are present.

    Returns
    -------
    (vmin, vmax) : tuple of floats
    """
    arr = np.asarray(arr)
    if arr.size == 0 or not np.isfinite(arr).any():
        return (-default, default) if mode == "symmetric" else (None, None)

    x = arr[np.isfinite(arr)].ravel()

    if mode == "range":
        data_min = float(np.nanmin(x))
        data_max = float(np.nanmax(x))
        vmin = float(np.nanpercentile(x, pct_low))
        vmax = float(np.nanpercentile(x, pct_high))
        vmin = max(vmin, data_min)
        vmax = min(vmax, data_max)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return data_min, data_max
        if zero_floor_if_nonneg and data_min >= 0:
            vmin = 0.0
        return vmin, vmax

    elif mode == "symmetric":
        absp = float(np.nanpercentile(np.abs(x), pct))
        if not np.isfinite(absp) or absp == 0:
            absp = default
        return -absp, absp

    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
def plot_2d_map(da: xr.DataArray, ax=None, title=None, bias=False):
    """
    Plot a 2D DataArray with dims (lat, lon). Accepts either normal (cividis) or bias (red-blue).
    """
    vals = da.values
    finite_vals = vals[np.isfinite(vals)]

    if finite_vals.size > 0:
        if bias:
            vmax = np.nanpercentile(np.abs(finite_vals), 99)
            vmin = -vmax
        else:
            vmin = np.nanpercentile(finite_vals, 1)
            vmax = np.nanpercentile(finite_vals, 99)
    else:
        vmin, vmax = None, None

    cmap = "RdBu_r" if bias else "cividis"

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    im = ax.pcolormesh(da["lon"], da["lat"], vals, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=da.name if da.name else "")
    
    if title:
        ax.set_title(title.replace("_", " "))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    return ax

def plot_2d_map_cartopy(
    ax: plt.Axes,
    da2d: xr.DataArray,
    *,
    title: Optional[str] = None,
    units: Optional[str] = None,
    cbar_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
):
    """
    Plot a single 2D lat/lon DataArray on a cartopy PlateCarree axis.
    Returns the QuadMesh (mappable) so colorbars can be added by the caller if desired.
    If vmin/vmax are not provided, uses _robust_limits in 'range' mode (1–99th pct).
    """
    if "lat" not in da2d.dims or "lon" not in da2d.dims:
        raise ValueError("plot_timestep expects a 2D DataArray with dims ('lat','lon').")

    arr = da2d.astype("float32")

    # Auto limits if not provided
    if vmin is None or vmax is None:
        vmin_auto, vmax_auto = _robust_limits(arr.values, mode="range", pct_low=1.0, pct_high=99.0, zero_floor_if_nonneg=True)
        if vmin is None:
            vmin = vmin_auto
        if vmax is None:
            vmax = vmax_auto

    # Pick label priority: explicit cbar_label > units > arr.name
    label = cbar_label if cbar_label is not None else (units if units else (arr.name or ""))

    mappable = arr.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        add_colorbar=True,
        cbar_kwargs={"label": label},
    )
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    if title:
        ax.set_title(title)
    return mappable

def plot_timeseries(labels: xr.DataArray, preds: xr.DataArray, ax=None, title=None):
    """
    Plot a simple time series comparing labels and predictions on the given ax.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(labels["time"], labels, color="blue", label="Labels")
    ax.plot(preds["time"], preds, color="darkorange", label="Predictions")

    ax.set_xlabel("Time")
    ax.set_ylabel(labels.name if labels.name else "")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax

# --------------------- Applied to a Multiple DataArrays ---------------------

def plot_three_maps(
    label2d: xr.DataArray,
    pred2d: xr.DataArray,
    bias2d: xr.DataArray,
    *,
    titles: Sequence[str],          # e.g. ["<top title>", "<middle title>", "<bottom title>"]
    cbar_labels: Sequence[str],     # e.g. ["<top units>", "<middle units>", "<bottom units>"]
    out_path: Path,
    # scaling shared with first two panels + symmetric bias for bottom
    data_vmin: Optional[float] = None,
    data_vmax: Optional[float] = None,
    bias_vmax: Optional[float] = None,
    data_cmap: Optional[str] = None,
    bias_cmap: str = "RdBu_r",
    figsize=(12, 12),
    dpi: int = 300,
    overwrite: bool = True,
):
    """
    Stack three maps vertically (top/middle/bottom) and save to out_path.
    All inputs must be 2D lat/lon DataArrays on the same grid.

    Parameters
    ----------
    titles : [top, middle, bottom]
        Titles for each subplot.
    cbar_labels : [top, middle, bottom]
        Colorbar labels for each subplot.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        print(f"[SKIP] File exists and overwrite=False: {out_path}")
        return out_path

    # Determine consistent color scales for data (Label & Pred) if not provided
    if data_vmin is None or data_vmax is None:
        dvmin, dvmax = _robust_limits(
            [label2d.values, pred2d.values],
            mode="range",
            pct_low=1.0,
            pct_high=99.0,
            zero_floor_if_nonneg=True
        )
        if data_vmin is None:
            data_vmin = dvmin
        if data_vmax is None:
            data_vmax = dvmax

    # Symmetric scale for bias if not provided
    if bias_vmax is None:
        bmin, bmax = _robust_limits(bias2d.values, mode="symmetric", pct=99.0, default=1.0)
        bias_vmax = max(abs(bmin), abs(bmax)) if np.isfinite([bmin, bmax]).all() else 1.0

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, hspace=0.12)

    # Top
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    plot_2d_map(
        label2d,
        ax=ax1,
        title=titles[0],
        bias=False,
    )
    # Re-enforce consistent limits/cmap for the first panel
    for c in ax1.collections:
        c.set_clim(data_vmin, data_vmax)
        c.set_cmap(data_cmap or "cividis")

    # Middle
    ax2 = fig.add_subplot(gs[1, 0], projection=proj)
    plot_2d_map(
        pred2d,
        ax=ax2,
        title=titles[1],
        bias=False,
    )
    # Re-enforce consistent limits/cmap for the second panel
    for c in ax2.collections:
        c.set_clim(data_vmin, data_vmax)
        c.set_cmap(data_cmap or "cividis")

    # Bottom (Bias)
    ax3 = fig.add_subplot(gs[2, 0], projection=proj)
    plot_2d_map(
        bias2d,
        ax=ax3,
        title=titles[2],
        bias=True,
    )
    # Re-enforce symmetric limits/cmap for bias panel
    for c in ax3.collections:
        c.set_clim(-bias_vmax, bias_vmax)
        c.set_cmap(bias_cmap)

    # Update colorbar labels if provided
    for ax, lbl in zip((ax1, ax2, ax3), cbar_labels):
        # Try to find the colorbar attached to this axes
        # (xarray/matplotlib attach cbar via figure; safest is to add a label to the last colorbar)
        # If needed, re-create a colorbar:
        pass  # keep your existing behavior; labels were set in plot_2d_map via da.name

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")
    return out_path

def r2_scatter(da1: xr.DataArray, da2: xr.DataArray, title: str = "Scatter with R²", ax=None):
    """
    Plot scatter of two DataArrays and compute R².
    Arrays must have identical shape, any dimensions allowed.
    If `ax` is provided, draw on it; otherwise create a new figure.
    """
    if da1.shape != da2.shape:
        raise ValueError("DataArrays must have the same shape")

    # Flatten and mask NaNs
    x = da1.values.ravel()
    y = da2.values.ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # Compute R²
    r2 = r2_score(y, x) if x.size > 1 else np.nan

    # Prepare axes
    created_fig = False
    if ax is None:
        created_fig = True
        plt.figure(figsize=(5, 5))
        ax = plt.gca()

    # Scatterplot + 1:1
    ax.scatter(x, y, alpha=0.3, s=5)
    if x.size and y.size:
        mn = float(min(x.min(), y.min()))
        mx = float(max(x.max(), y.max()))
        ax.plot([mn, mx], [mn, mx], "r--", lw=2)
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title}\nR² = {r2:.3f}")

    if created_fig:
        plt.tight_layout()
        plt.show()

    return r2

# --------------------- Applied to a DataSet ---------------------
def _robust_limits(
    arrays,
    *,
    mode="range",             
    pct_low=1.0,
    pct_high=99.0,
    pct=99.0,
    zero_floor_if_nonneg=True,
    default=1.0
):
    """
    Compute robust plotting limits from one or more arrays.

    Parameters
    ----------
    arrays : array-like or list of array-like
        1D/2D/ND arrays to consider. NaNs are ignored.
    mode : {"range", "symmetric"}
        - "range": return (vmin, vmax) using percentiles.
        - "symmetric": return (-v, +v) using abs(values) percentile.
    pct_low : float
        Lower percentile for range mode.
    pct_high : float
        Upper percentile for range mode.
    pct : float
        Percentile for symmetric mode.
    zero_floor_if_nonneg : bool
        If True and data min >= 0, vmin forced to 0 in range mode.
    default : float
        Value to return if no finite data are present.

    Returns
    -------
    (vmin, vmax) : tuple of floats
    """
    # Normalize input
    if not isinstance(arrays, (list, tuple)):
        arrays = [arrays]

    vals = []
    for a in arrays:
        a = np.asarray(a)
        if a.size == 0:
            continue
        m = np.isfinite(a)
        if m.any():
            vals.append(a[m].ravel())
    if not vals:
        return (-default, default) if mode == "symmetric" else (None, None)

    x = np.concatenate(vals, axis=0)

    if mode == "range":
        data_min = float(np.nanmin(x))
        data_max = float(np.nanmax(x))
        vmin = float(np.nanpercentile(x, pct_low))
        vmax = float(np.nanpercentile(x, pct_high))
        vmin = max(vmin, data_min)
        vmax = min(vmax, data_max)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return data_min, data_max
        if zero_floor_if_nonneg and data_min >= 0:
            vmin = 0.0
        return vmin, vmax

    elif mode == "symmetric":
        absp = float(np.nanpercentile(np.abs(x), pct))
        if not np.isfinite(absp) or absp == 0:
            absp = default
        return -absp, absp

    else:
        raise ValueError(f"Unsupported mode: {mode}")

def plot_maps(ds: xr.Dataset, fig_title=None, save_path=None, avgs_only=False):
    """
    Plot maps of all variables in the dataset.
    Assemble prediction:label:bias pairs (or triplets).
    If avgs_only it only plots the scenario average arrays.
    """
    groups = _find_var_groups(ds)  # list of tuples (Label, Predicted, [Bias])

    # filter for averages only if requested
    if avgs_only:
        groups = [g for g in groups if any(name.endswith("_avg") for name in g)]

    # Fallback: if no groups (e.g. slope-only dataset), just plot every var in a single column
    if not groups:
        names = list(ds.data_vars)
        nrows = len(names)
        fig, axes = plt.subplots(nrows, 1, figsize=(6, 4 * nrows), squeeze=False)
        for ax, name in zip(axes.ravel(), names):
            plot_2d_map(ds[name], ax=ax, title=name.replace("_", " "))
        if fig_title:
            fig.suptitle(fig_title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
        return fig, axes

    # determine layout
    nrows = len(groups)
    ncols = max(len(g) for g in groups)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
    )

    for row, group in enumerate(groups):
        for col, name in enumerate(group):
            is_bias = name.startswith("Bias_")
            plot_2d_map(
                ds[name],
                ax=axes[row, col],
                title=name.replace("_", " "),
                bias=is_bias,
            )

    if fig_title:
        fig.suptitle(fig_title, fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    print("Plotted maps.")
    return fig, axes

def plot_timeseries_grid(ds: xr.Dataset, fig_title=None, save_path=None, avgs_only=False):
    """
    Assemble time series subplots from a dataset with variables named:
      Label_{var}_{scenario}, Predicted_{var}_{scenario}.
    Groups them into pairs and plots them together in a grid.
    """
    groups = _find_var_groups(ds)  # list of (Label, Predicted) or (Label, Predicted, Bias)

    # filter to Label/Predicted only (ignore Bias here)
    pairs = [(g[0], g[1]) for g in groups if len(g) >= 2]

    # filter to averages only if requested
    if avgs_only:
        pairs = [p for p in pairs if any(name.endswith("_avg") for name in p)]

    if not pairs:
        raise ValueError("No Label/Predicted pairs found to plot.")

    # --- layout ---
    nrows = len(pairs)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 4 * nrows), squeeze=False)

    # --- plot ---
    for ax, (lab_name, pred_name) in zip(axes.ravel(), pairs):
        lab = ds[lab_name]
        pre = ds[pred_name]
        title = f"{lab_name.replace('_', ' ')} vs {pred_name.replace('_', ' ')}"
        plot_timeseries(lab, pre, ax=ax, title=title)

    if fig_title:
        fig.suptitle(fig_title, fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        
    print("Plotted timeseries.")

    return fig, axes

def plot_r2_grid(ds: xr.Dataset, fig_title: str | None = None, save_path: str | None = None):
    """
    Plot an R² scatter for each (Label_*, Predicted_*) pair found via _find_var_groups(ds)
    in a 2-column grid.
    """
    groups = _find_var_groups(ds)
    if not groups:
        raise ValueError("No (Label_*, Predicted_*) pairs found in dataset.")

    n = len(groups)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), squeeze=False)

    for ax, grp in zip(axes.ravel(), groups):
        label_name, pred_name = grp[0], grp[1]
        suffix = label_name.replace("Label_", "")
        title = suffix
        r2_scatter(ds[label_name], ds[pred_name], title=title, ax=ax)

    # Hide any unused axes
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    if fig_title:
        fig.suptitle(fig_title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    print("Plotted R² scatter grid.")
    return fig, axes

