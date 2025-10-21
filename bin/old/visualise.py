#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xarray as xr
import numpy as np
import cftime
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import os, dask
from numcodecs import Blosc
import gc
import traceback

dask.config.set(scheduler="threads")

# ---------------- Project imports ---------------- #
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.dataset.variables import var_names
from src.utils.tools import slurm_shard

# open the dataset (all vars)
def open_zarr(path: Path) -> xr.Dataset:
    """Open a Zarr store lazily; fall back if not consolidated."""
    p = str(path)
    try:
        return xr.open_zarr(p, consolidated=True, decode_times=False, chunks="auto")
    except Exception:
        print(f"[WARN] consolidated open failed for {p}; trying non-consolidated")
        return xr.open_zarr(p, consolidated=False, decode_times=False, chunks="auto")

def open_zarr_ds(path: Path) -> xr.Dataset:
    """Convenience opener for cached Zarr datasets with fallback."""
    p = str(path)
    try:
        return xr.open_zarr(p, consolidated=True, decode_times=False, chunks="auto")
    except Exception:
        print(f"[WARN] consolidated open failed for {p}; trying non-consolidated")
        return xr.open_zarr(p, consolidated=False, decode_times=False, chunks="auto")

# open a variable from the dataset
def get_var(ds: xr.Dataset, var: str) -> xr.DataArray:
    """Get a variable from a dataset, erroring if not found."""
    if var not in ds.data_vars:
        raise SystemExit(f"variable '{var}' not found. available: {list(ds.data_vars)}")
    return ds[var]

def save_to_zarr(ds: xr.Dataset, store_path: Path, overwrite: bool = True, compressor=None, chunks=None):
    """
    Save an xarray Dataset to a standalone Zarr store, safely handling chunk dims.
    """
    store_path = Path(store_path)

    # Overwrite handling
    if overwrite and store_path.exists():
        import shutil
        shutil.rmtree(store_path)

    # Compression
    if compressor is None:
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

    # Choose sensible default chunks, but only for dims that exist
    if chunks is None:
        desired = {'time': -1, 'lat': 30, 'lon': 30}
        chunks = {k: v for k, v in desired.items() if k in ds.dims}
    else:
        # If a custom chunks dict was passed, intersect with existing dims
        chunks = {k: v for k, v in dict(chunks).items() if k in ds.dims}

    # Rechunk only if we actually have dims to chunk
    if chunks:
        ds = ds.chunk(chunks)

    # Apply compressor to all data variables
    encoding = {var: {"compressor": compressor} for var in ds.data_vars}

    # Write
    ds.to_zarr(str(store_path), mode="w", encoding=encoding, compute=True)
    print(f"[INFO] Saved dataset with {len(ds.data_vars)} variables and dims {dict(ds.dims)} to {store_path}")

# construct a standard time axis for given length
def construct_time_axis(length: int) -> np.ndarray:
    ref = cftime.DatetimeNoLeap(1901, 1, 1)

    if length == 44895:
        # 123 years * 365 noleap days
        time_vals = np.arange(0, 123 * 365, dtype="int32")
    elif length == 1476:
        # first day of each month 1901-01..2023-12
        vals = []
        for y in range(1901, 2024):
            for m in range(1, 13):
                dt = cftime.DatetimeNoLeap(y, m, 1)
                vals.append((dt - ref).days)
        time_vals = np.asarray(vals, dtype="int32")
    elif length == 123:
        # Jan 1 each year 1901..2023
        vals = []
        for y in range(1901, 2024):
            dt = cftime.DatetimeNoLeap(y, 1, 1)
            vals.append((dt - ref).days)
        time_vals = np.asarray(vals, dtype="int32")
    else:
        raise ValueError(
            f"Unsupported time length {length}. "
            "Expected one of {44895 (daily), 1476 (monthly), 123 (annual)}."
        )
        
    return time_vals

# normalise the coordinates of a DataArray
# Define lat and lon arrays
std_lat = np.arange(-89.75, 90.0, 0.5, dtype="float32")
std_lon = np.arange(0.0, 360.0, 0.5, dtype="float32")

def normalise_coords(da: xr.DataArray) -> xr.DataArray:
    """
    Replace coords with standard values of time, lat and lon
    """
    da = da.copy()

    # time
    if "time" in da.dims:
        ntime = int(da.sizes["time"])
        da = da.assign_coords(time=("time", construct_time_axis(ntime)))
        da["time"].attrs.update({
            "units": "days since 1901-01-01 00:00:00",
            "calendar": "noleap",
        })

    # lat (only load two scalars to check order)
    if "lat" in da.dims:
        lat0 = float(da["lat"].isel(lat=0))
        latN = float(da["lat"].isel(lat=-1))
        if lat0 > latN:
            da = da.sortby("lat")
        da = da.assign_coords(lat=("lat", std_lat))

    # lon
    if "lon" in da.dims:
        lon0 = float(da["lon"].isel(lon=0))
        lonN = float(da["lon"].isel(lon=-1))
        if lon0 > lonN:
            da = da.sortby("lon")
        da = da.assign_coords(lon=("lon", std_lon))

    return da

# Create date string from time integer
def time_label(day_val: int) -> str:
    """Convert a time integer into a date string"""
    units = "days since 1901-01-01 00:00:00"
    cal = "noleap"
    dt = cftime.num2date(int(day_val), units=units, calendar=cal)
    return dt.strftime("%Y-%m-%d")

# Put it all together
def open_and_standardise(path: Path, var: str) -> xr.DataArray:
    ds = open_zarr(path)
    da = get_var(ds, var)
    da = normalise_coords(da)
    return da

def combine_to_dataset(preds: List[Tuple[str, xr.DataArray]], labels: List[Tuple[str, xr.DataArray]]) -> xr.Dataset:
    """Combine a list of (scenario, DataArray) pairs into a single Dataset, renaming variables."""
    all_das = []
    for scenario, da in preds:
        var = da.name
        new_var = f"Predicted_{var}_{scenario}"
        all_das.append(da.rename(new_var))
    for scenario, da in labels:
        var = da.name
        new_var = f"Label_{var}_{scenario}"
        all_das.append(da.rename(new_var))
    
    ds = xr.merge(all_das)
    return ds

def _find_var_groups(ds: xr.Dataset) -> List[Tuple[str, ...]]:
    """
    Find matching variable groups in a dataset.
    - (Label, Predicted)
    - (Label, Predicted, Bias) if a Bias variable is present.
    """
    grouped: Dict[str, Dict[str, str]] = {}

    for name in ds.data_vars:
        m = re.match(r"(Label|Predicted|Bias)_(.+)", name)
        if not m:
            continue
        prefix, suffix = m.groups()
        grouped.setdefault(suffix, {})[prefix] = name

    groups: List[Tuple[str, ...]] = []
    for suffix, g in grouped.items():
        if "Label" in g and "Predicted" in g:
            if "Bias" in g:
                groups.append((g["Label"], g["Predicted"], g["Bias"]))
            else:
                groups.append((g["Label"], g["Predicted"]))

    return groups

# Averaging
def time_avg(ds: xr.Dataset) -> xr.Dataset:
    """
    Take average over time and return 2D maps.
    """
    out = {}
    for name, da in ds.data_vars.items():
        out[name] = da.mean(dim="time", skipna=True)
    ds = xr.Dataset(out, coords={k: v for k, v in ds.coords.items() if k != "time"})
    print("Took time avg")
    return ds

def space_avg(ds: xr.Dataset) -> xr.Dataset:
    """
    Take an average across space to produce a 1D array (dim = time).
    """
    out = {}
    for name, da in ds.data_vars.items():
        out[name] = da.mean(dim=("lat", "lon"), skipna=True)
    ds = xr.Dataset(out, coords={"time": ds["time"]})
    print("Took space average.")
    return ds

def scenario_avg(ds: xr.Dataset) -> xr.Dataset:
    """
    Take an average across scenarios and store as a new array.
    """
    out = ds.copy()

    # Group by variable type
    predicted_vars = [v for v in ds.data_vars if "Predicted" in v]
    label_vars     = [v for v in ds.data_vars if "Label" in v]

    # Compute averages
    if predicted_vars:
        pred_mean = ds[predicted_vars].to_array().mean("variable")
        for v in predicted_vars:
            base = v.replace("Predicted_", "").split("_")[0]
        out[f"Predicted_{base}_avg"] = pred_mean

    if label_vars:
        lab_mean = ds[label_vars].to_array().mean("variable")
        for v in label_vars:
            base = v.replace("Label_", "").split("_")[0]
        out[f"Label_{base}_avg"] = lab_mean
        
    print("Took scenario average and added to ds")
    
    return out

# Bias
def bias(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute the bias and add them to the dataset.
    """
    out = {}
    groups = _find_var_groups(ds)

    for group in groups:
        if len(group) >= 2:  # always (Label, Predicted, [Bias?])
            label_name, pred_name = group[0], group[1]
            suffix = pred_name.replace("Predicted_", "")
            bias_name = f"Bias_{suffix}"
            out[bias_name] = (ds[pred_name] - ds[label_name]).rename(bias_name)
            
    print("Calculate bias and added to ds")
    return ds.assign(**out)

# Trends, Seasonality and IAV
def trend(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute pixel-wise linear trend (slope + intercept) along 'time' for each 
    data variable in the dataset. Returns a new dataset without the time 
    dimension, where each variable has two arrays: slope and intercept.
    
    Slopes are expressed in 'units per day'.
    """
    # Keep time as one chunk for stable reductions; leave lat/lon as-is or set your preferred sizes
    ds = ds.unify_chunks().chunk({'time': -1})

    # Time as float64 for numerical stability
    t = ds['time'].astype('float64')

    out = {}
    for name, da in ds.data_vars.items():
        x = da.astype('float64')

        # Valid mask per time step/pixel
        valid = x.notnull()

        # Only use times where x is valid
        t_masked  = t.where(valid)
        x_masked  = x.where(valid)

        # Per-pixel counts and sums
        n   = valid.sum(dim='time')                                
        Sx  = x_masked.sum(dim='time')
        St  = t_masked.sum(dim='time')
        St2 = (t_masked * t_masked).sum(dim='time')
        Stx = (x_masked * t_masked).sum(dim='time')

        # OLS closed-form
        denom = n * St2 - St * St
        slope = xr.where(denom != 0, (n * Stx - St * Sx) / denom, np.nan)
        intercept = xr.where(n != 0, (Sx - slope * St) / n, np.nan)

        # Optional: carry simple attrs
        slope.attrs.update({'long_name': f"{name} trend (per day)", 'units': f"{da.attrs.get('units','')}/day".strip('/')})
        intercept.attrs.update({'long_name': f"{name} intercept", 'units': da.attrs.get('units', '')})

        out[f"{name}_slope"] = slope.rename(f"{name}_slope")
        out[f"{name}_intercept"] = intercept.rename(f"{name}_intercept")

    return xr.Dataset(out)

def detrend(orig_ds: xr.Dataset, trend_ds: xr.Dataset) -> xr.Dataset:
    """
    Subtract the linear trend from each pixel in the array`.
    """
    out = {}
    time = orig_ds["time"]  
    for name, da in orig_ds.data_vars.items():
        slope_name = f"{name}_slope"
        intercept_name = f"{name}_intercept"
        a = trend_ds[slope_name]       
        b = trend_ds[intercept_name]
        # Broadcast to (time, lat, lon)
        fitted = a * time + b
        out[name] = (da - fitted).rename(name)
    
    print("Detrended.")
        
    return xr.Dataset(out, coords=orig_ds.coords)

def seasonality(detrended_ds: xr.Dataset) -> xr.Dataset:
    """
    Monthly averages expanded out to full length of the array.
    """
    print("starting to calculate seasonality")
    n_time = detrended_ds.sizes["time"]
    if n_time % 12 != 0:
        raise ValueError("seasonality() expects monthly data with length multiple of 12.")

    # keep time as one chunk so reductions don't multiply chunks
    detrended_ds = detrended_ds.chunk({"time": -1})

    # 12-month climatology by simple slicing (avoids groupby overhead)
    month_means = []
    for m in range(12):
        month_means.append(detrended_ds.isel(time=slice(m, None, 12)).mean("time", skipna=True))
    means12 = xr.concat(month_means, dim="time")  # time=12 here

    # tile back to full length
    reps = n_time // 12
    repeated = xr.concat([means12] * reps, dim="time")
    repeated = repeated.assign_coords(time=detrended_ds["time"])

    # preserve units/calendar if present
    for k in ("units", "calendar"):
        if k in detrended_ds["time"].attrs:
            repeated["time"].attrs[k] = detrended_ds["time"].attrs[k]

    return repeated
    
def iav(detrended_ds: xr.Dataset, seasonality_ds: xr.Dataset) -> xr.Dataset:
    """
    Inter-annual variability = detrended - repeating seasonal cycle.
    """
    # time sanity check
    if detrended_ds.sizes.get("time") != seasonality_ds.sizes.get("time"):
        raise ValueError("time lengths differ between detrended and seasonality datasets")

    out = {}
    for name, da in detrended_ds.data_vars.items():
        if name not in seasonality_ds:
            raise KeyError(f"seasonality dataset missing variable '{name}'")
        # align to be safe (ensures identical coords before subtraction)
        a, s = xr.align(da, seasonality_ds[name], join="exact")
        out[name] = (a - s).rename(name)
        
    print("Calculated inter-annual variability.")

    return xr.Dataset(out, coords=detrended_ds.coords)

# Regression/Skill
def _r2_da(y_true: xr.DataArray, y_pred: xr.DataArray) -> float:
    """
    Compute coefficient of determination R2 between two DataArrays,
    flattening across all dimensions. NaNs are ignored.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    yt = y_true.values.ravel()
    yp = y_pred.values.ravel()

    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return np.nan

    yt = yt[mask]
    yp = yp[mask]

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

def r2(ds: xr.Dataset) -> xr.Dataset:
    """
    For each Label/Predicted pair in the dataset, compute R² across all
    points and return as a Dataset with scalar values.
    """
    groups = _find_var_groups(ds)
    out: Dict[str, xr.DataArray] = {}

    for lab_name, pred_name, *_ in groups:  # ignore bias
        lab = ds[lab_name]
        pred = ds[pred_name]
        r2_val = _r2_da(lab, pred)
        varname = lab_name.replace("Label_", "R2_")
        out[varname] = xr.DataArray(r2_val)

    print("Took R2.")
    return xr.Dataset(out)
    
# Plotting

# Tunables for plot size/quality
_CELL_W = 3.0   # inches per subplot (width)
_CELL_H = 2.4   # inches per subplot (height)
_SAVE_DPI = 150 # lower DPI for faster/smaller PNGs
_SAVE_KW = dict(dpi=_SAVE_DPI, pil_kwargs={"compress_level": 1})  # PNG compression level 0..9

def _percentile_limits(*arrays, lo=1, hi=99, symmetric=False):
    """Compute percentile-based vmin/vmax over multiple arrays."""
    vals = []
    for a in arrays:
        v = np.asarray(a.values if hasattr(a, "values") else a)
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    if not vals:
        return None, None
    allv = np.concatenate(vals)
    if symmetric:
        vmax = np.nanpercentile(np.abs(allv), hi)
        return -vmax, vmax
    vmin = np.nanpercentile(allv, lo)
    vmax = np.nanpercentile(allv, hi)
    return vmin, vmax

def plot_2d_map(da: xr.DataArray, ax=None, title=None, cmap="cividis", vmin=None, vmax=None):
    """
    Lightweight 2D plot helper. No colorbar here (handled by parent).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(_CELL_W, _CELL_H), constrained_layout=True)

    im = ax.pcolormesh(da["lon"], da["lat"], da.values, cmap=cmap, vmin=vmin, vmax=vmax)
    # Make rasterization explicit for speed when exporting vector formats
    im.set_rasterized(True)

    if title:
        ax.set_title(title.replace("_", " "), fontsize=10)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.tick_params(labelsize=8)
    return im, ax

def plot_maps(ds: xr.Dataset, fig_title=None, save_path=None, avgs_only=False, overwrite=False):
    """
    Plot maps for all Label/Predicted pairs (and Bias if present).
    - Label & Predicted share identical vmin/vmax and a single shared colorbar.
    - Bias (if present) uses symmetric RdBu_r with its own colorbar.

    If save_path exists and overwrite=False, no-op.
    """
    if save_path is not None and (not overwrite) and Path(save_path).exists():
        print(f"[PLOT] Exists, skipping: {save_path}")
        return None, None

    groups = _find_var_groups(ds)  # tuples like (Label_..., Predicted_..., [Bias_...])

    if avgs_only:
        groups = [g for g in groups if any(name.endswith("_avg") for name in g)]

    # Fallback: no pairs (e.g., dataset of slopes only) -> plot each var alone
    if not groups:
        names = list(ds.data_vars)
        nrows, ncols = len(names), 1
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(_CELL_W*ncols, _CELL_H*nrows),
            squeeze=False,
            constrained_layout=True
        )
        for ax, name in zip(axes.ravel(), names):
            vmin, vmax = _percentile_limits(ds[name])
            im, _ = plot_2d_map(ds[name], ax=ax, title=name.replace("_"," "), vmin=vmin, vmax=vmax, cmap="cividis")
            # Per-axis colorbar (no pairing here)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        if fig_title:
            fig.suptitle(fig_title, fontsize=12)
        if save_path:
            fig.savefig(save_path, **_SAVE_KW)
            plt.close(fig)
        else:
            plt.show()
        print("Plotted maps.")
        return fig, axes

    # Layout: each group sits on a row, with 2 or 3 columns (Label, Predicted, [Bias])
    nrows = len(groups)
    ncols = max(len(g) for g in groups)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(_CELL_W*ncols, _CELL_H*nrows),
        squeeze=False,
        constrained_layout=True
    )

    for row, group in enumerate(groups):
        # Identify components
        label_name     = next((n for n in group if n.startswith("Label_")), None)
        predicted_name = next((n for n in group if n.startswith("Predicted_")), None)
        bias_name      = next((n for n in group if n.startswith("Bias_")), None)

        # --- Shared scale for Label/Predicted ---
        if label_name and predicted_name:
            vmin, vmax = _percentile_limits(ds[label_name], ds[predicted_name])
        else:
            vmin, vmax = None, None

        # Column 0: Label (if present)
        col = 0
        if label_name:
            imL, _ = plot_2d_map(
                ds[label_name], ax=axes[row, col],
                title=label_name, cmap="cividis", vmin=vmin, vmax=vmax
            )
            col += 1
        else:
            imL = None

        # Column 1: Predicted (if present)
        if predicted_name:
            imP, _ = plot_2d_map(
                ds[predicted_name], ax=axes[row, col],
                title=predicted_name, cmap="cividis", vmin=vmin, vmax=vmax
            )
            # Shared colorbar for label+predicted (use one of the mappables; scales are identical)
            # Attach colorbar to both axes so it spans the pair
            pair_axes = []
            # Label axis is at col-1 if label exists
            if label_name:
                pair_axes.append(axes[row, col-1])
            pair_axes.append(axes[row, col])
            fig.colorbar(imP, ax=pair_axes, fraction=0.02, pad=0.02)
            col += 1

        # Column 2: Bias (if present) with symmetric diverging cmap
        if bias_name:
            bmin, bmax = _percentile_limits(ds[bias_name], symmetric=True)
            imB, _ = plot_2d_map(
                ds[bias_name], ax=axes[row, col],
                title=bias_name, cmap="RdBu_r", vmin=bmin, vmax=bmax
            )
            fig.colorbar(imB, ax=axes[row, col], fraction=0.02, pad=0.02)

        # Hide any unused cells on this row
        for extra_col in range(len(group), ncols):
            axes[row, extra_col].axis("off")

    if fig_title:
        fig.suptitle(fig_title, fontsize=12)

    if save_path:
        fig.savefig(save_path, **_SAVE_KW)
        plt.close(fig)
    else:
        plt.show()

    print("Plotted maps.")
    return fig, axes

def plot_timeseries(labels: xr.DataArray, preds: xr.DataArray, ax=None, title=None):
    """
    Compact time series comparison for a single pair (Labels vs Predictions).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(_CELL_W*1.5, _CELL_H), constrained_layout=True)

    ax.plot(labels["time"], labels, label="Labels")
    ax.plot(preds["time"], preds, label="Predictions")

    ax.set_xlabel("Time", fontsize=9)
    ax.set_ylabel(labels.name if labels.name else "", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    return ax

def plot_timeseries_grid(ds: xr.Dataset, fig_title=None, save_path=None, avgs_only=False, overwrite=False):
    """
    Plot a vertical stack of time series for all Label/Predicted pairs.

    If save_path exists and overwrite=False, no-op.
    """
    if save_path is not None and (not overwrite) and Path(save_path).exists():
        print(f"[PLOT] Exists, skipping: {save_path}")
        return None, None

    groups = _find_var_groups(ds)
    pairs = [(g[0], g[1]) for g in groups if len(g) >= 2]  # label, predicted

    if avgs_only:
        pairs = [p for p in pairs if any(name.endswith("_avg") for name in p)]

    if not pairs:
        raise ValueError("No Label/Predicted pairs found to plot.")

    nrows = len(pairs)
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(_CELL_W*2.2, _CELL_H*nrows),
        squeeze=False,
        constrained_layout=True
    )

    for ax, (lab_name, pred_name) in zip(axes.ravel(), pairs):
        lab = ds[lab_name]
        pre = ds[pred_name]
        title = f"{lab_name.replace('_', ' ')} vs {pred_name.replace('_', ' ')}"
        plot_timeseries(lab, pre, ax=ax, title=title)

    if fig_title:
        fig.suptitle(fig_title, fontsize=12)

    if save_path:
        fig.savefig(save_path, **_SAVE_KW)
        plt.close(fig)
    else:
        plt.show()

    print("Plotted timeseries.")
    return fig, axes

def plot_r2_single(ax, name: str, value: float):
    """Compact single R² bar."""
    ax.bar([0], [value])
    ax.set_ylim(0, 1)
    ax.set_xticks([0])
    ax.set_xticklabels([name.replace("_", " ")], fontsize=8)
    ax.set_ylabel("R²", fontsize=9)
    ax.set_title(name.replace("_", " "), fontsize=10)

def plot_r2_grid(r2_ds: xr.Dataset, fig_title=None, save_path=None, overwrite=False):
    """
    Bar grid for R² values.

    If save_path exists and overwrite=False, no-op.
    """
    if save_path is not None and (not overwrite) and Path(save_path).exists():
        print(f"[PLOT] Exists, skipping: {save_path}")
        return None, None

    names = list(r2_ds.data_vars)
    values = [float(r2_ds[n].values) for n in names]

    n = len(names)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(_CELL_W*ncols, _CELL_H*nrows),
        squeeze=False,
        constrained_layout=True
    )

    for ax, name, val in zip(axes.ravel(), names, values):
        plot_r2_single(ax, name, val)

    # hide any unused axes if odd number of items
    for ax in axes.ravel()[len(values):]:
        ax.axis("off")

    if fig_title:
        fig.suptitle(fig_title, fontsize=12)

    if save_path:
        fig.savefig(save_path, **_SAVE_KW)
        plt.close(fig)
    else:
        plt.show()

    print("Plotted r2.")
    return fig, axes

# Run main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", default="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/first_full_run", type=Path)
    ap.add_argument("--labels_dir", default="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference", type=Path)
    ap.add_argument("--output_dir", default="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/plots/first_full_run", type=Path)
    ap.add_argument("--overwrite_cache", action="store_true", help="Force recomputation and overwrite existing Zarr caches")
    ap.add_argument("--overwrite_plots", action="store_true", help="Overwrite existing plot image files")
    args = ap.parse_args()

    preds_dir = Path(args.preds_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = slurm_shard(var_names['outputs'])
    scenarios = ["S0", "S1", "S2", "S3"]

    for var in tasks:
        try:
            print(f"Processing {var}.")
            pred_das = []
            label_das = []

            # Plot dirs
            plot_dir = output_dir
            time_avg_dir = plot_dir / "time_avg" / var
            spatial_avg_dir = plot_dir / "spatial_avg" / var
            trend_dir = plot_dir / "trends" / var
            seasonality_dir = plot_dir / "seasonality" / var
            iav_dir = plot_dir / "iav" / var
            iav_time_avg_dir = iav_dir / "time_avg"
            iav_space_avg_dir = iav_dir / "space_avg"
            r2_dir = plot_dir / "r2" / var

            for d in [time_avg_dir, spatial_avg_dir, trend_dir, seasonality_dir, r2_dir, iav_time_avg_dir, iav_space_avg_dir]:
                d.mkdir(parents=True, exist_ok=True)

            # Cache paths
            cache_dir = (Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/analysis/visualise/analysis_cache") / "first_full_train" / var)
            cache_dir.mkdir(parents=True, exist_ok=True)

            p_space_avg   = cache_dir / "space_avg.zarr"
            p_time_avg    = cache_dir / "time_avg.zarr"
            p_r2          = cache_dir / "r2.zarr"
            p_trend       = cache_dir / "trend.zarr"
            p_slope       = cache_dir / "slope.zarr"
            p_detrended   = cache_dir / "detrended.zarr"
            p_seasonality = cache_dir / "seasonality.zarr"
            p_iav         = cache_dir / "iav.zarr"

            # Read preds/labels for all scenarios
            for scenario in scenarios:
                if var in var_names['annual_outputs']:
                    preds_path = preds_dir / scenario / "zarr" / "annual.zarr"
                    labels_path = labels_dir / scenario / "annual.zarr"
                elif var in var_names['monthly_outputs']:
                    preds_path = preds_dir / scenario / "zarr" / "monthly.zarr"
                    labels_path = labels_dir / scenario / "monthly.zarr"

                if not preds_path.exists():
                    print(f"[FATAL] Missing preds path: {preds_path}")
                if not labels_path.exists():
                    print(f"[FATAL] Missing labels path: {labels_path}")

                preds_da = open_and_standardise(preds_path, var)
                labels_da = open_and_standardise(labels_path, var)

                pred_das.append((scenario, preds_da))
                label_das.append((scenario, labels_da))

            # Combine and augment
            ds = combine_to_dataset(pred_das, label_das)
            ds = scenario_avg(ds)
            ds = bias(ds)
            ds = ds.chunk({'time': -1, 'lat': 30, 'lon': 30})

            # --------- Compute & cache datasets (skip if cached unless overwrite) ---------
            if p_space_avg.exists() and not args.overwrite_cache:
                print("[CACHE] space_avg exists, skipping compute")
            else:
                space_avg_ds = space_avg(ds)
                save_to_zarr(space_avg_ds, p_space_avg, overwrite=True)
                del space_avg_ds; gc.collect()

            if p_time_avg.exists() and not args.overwrite_cache:
                print("[CACHE] time_avg exists, skipping compute")
            else:
                time_avg_ds = time_avg(ds)
                save_to_zarr(time_avg_ds, p_time_avg, overwrite=True)
                del time_avg_ds; gc.collect()

            if p_r2.exists() and not args.overwrite_cache:
                print("[CACHE] r2 exists, skipping compute")
            else:
                r2_ds_per_var = r2(ds)
                save_to_zarr(r2_ds_per_var, p_r2, overwrite=True)
                del r2_ds_per_var; gc.collect()

            if p_trend.exists() and not args.overwrite_cache:
                print("[CACHE] trend exists, skipping compute")
            else:
                trend_ds = trend(ds)
                save_to_zarr(trend_ds, p_trend, overwrite=True)
                slope_ds = trend_ds[[v for v in trend_ds.data_vars if v.endswith("_slope")]]
                save_to_zarr(slope_ds, p_slope, overwrite=True)
                del slope_ds, trend_ds; gc.collect()

            if p_detrended.exists() and not args.overwrite_cache:
                print("[CACHE] detrended exists, skipping compute")
            else:
                detrended_ds = detrend(ds, open_zarr_ds(p_trend)).persist()
                save_to_zarr(detrended_ds, p_detrended, overwrite=True)
                del detrended_ds; gc.collect()

            if var in var_names['monthly_outputs']:
                if p_seasonality.exists() and not args.overwrite_cache:
                    print("[CACHE] seasonality exists, skipping compute")
                else:
                    seasonality_ds = seasonality(open_zarr_ds(p_detrended)).persist()
                    save_to_zarr(seasonality_ds, p_seasonality, overwrite=True)
                    del seasonality_ds; gc.collect()

                if p_iav.exists() and not args.overwrite_cache:
                    print("[CACHE] iav exists, skipping compute")
                else:
                    iav_ds = iav(open_zarr_ds(p_detrended), open_zarr_ds(p_seasonality))
                    save_to_zarr(iav_ds, p_iav, overwrite=True)
                    del iav_ds; gc.collect()
            else:
                if p_iav.exists() and not args.overwrite_cache:
                    print("[CACHE] iav exists (annual), skipping compute")
                else:
                    save_to_zarr(open_zarr_ds(p_detrended), p_iav, overwrite=True)

            # -------------------- Plotting -------------------- #
            plot_timeseries_grid(
                open_zarr_ds(p_space_avg),
                f"Spatial Average of {var}",
                spatial_avg_dir / f"spatial_avg_{var}_all.png",
                overwrite=args.overwrite_plots,
            )
            plot_timeseries_grid(
                open_zarr_ds(p_space_avg),
                f"Spatial Average of {var}",
                spatial_avg_dir / f"spatial_avg_{var}_scen_avg.png",
                avgs_only=True,
                overwrite=args.overwrite_plots,
            )

            plot_maps(
                open_zarr_ds(p_time_avg),
                f"Time Average of {var}",
                time_avg_dir / f"time_avg_{var}_all.png",
                overwrite=args.overwrite_plots,
            )
            plot_maps(
                open_zarr_ds(p_time_avg),
                f"Time Average of {var}",
                time_avg_dir / f"time_avg_{var}_all_scen_avg.png",
                avgs_only=True,
                overwrite=args.overwrite_plots,
            )

            plot_r2_grid(
                open_zarr_ds(p_r2),
                f"R² for {var}",
                r2_dir / f"r2_{var}.png",
                overwrite=args.overwrite_plots,
            )

            plot_maps(
                open_zarr_ds(p_slope),
                fig_title=f"Slopes of {var}",
                save_path=trend_dir / f"{var}_trend_slope.png",
                overwrite=args.overwrite_plots,
            )

            if var in var_names['monthly_outputs']:
                seasonality_ds = open_zarr_ds(p_seasonality)
                seasonality_one_year_ds = seasonality_ds.isel(time=slice(0, 12))
                seasonality_one_year_sa = space_avg(seasonality_one_year_ds)
                plot_timeseries_grid(
                    seasonality_one_year_sa,
                    f"Seasonality of {var} - Spatially Averaged",
                    seasonality_dir / f"seasonality_{var}_all.png",
                    overwrite=args.overwrite_plots,
                )
                plot_timeseries_grid(
                    seasonality_one_year_sa,
                    f"Seasonality of {var} - Spatially Averaged",
                    seasonality_dir / f"seasonality_{var}_scen_avg.png",
                    avgs_only=True,
                    overwrite=args.overwrite_plots,
                )
                del seasonality_ds, seasonality_one_year_ds, seasonality_one_year_sa; gc.collect()

            iav_ds = open_zarr_ds(p_iav)
            iav_time = time_avg(iav_ds)
            plot_maps(
                iav_time,
                f"IAV time-avg {var}",
                iav_time_avg_dir / f"iav_{var}_time_avg.png",
                overwrite=args.overwrite_plots,
            )
            del iav_time; gc.collect()

            iav_space = space_avg(iav_ds)
            plot_timeseries_grid(
                iav_space,
                f"IAV time-avg {var}",
                iav_space_avg_dir / f"iav_{var}_time_avg.png",
                overwrite=args.overwrite_plots,
            )
            del iav_space, iav_ds; gc.collect()

        except Exception as e:
            print(f"[ERROR] while processing var={var}: {e}")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()