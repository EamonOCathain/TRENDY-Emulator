import sys
from pathlib import Path  
import numpy as np        
import xarray as xr       
from typing import Dict 
import re

# ---------------- Project imports ---------------- #
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.analysis.process_arrays import find_var_groups

# Bias
def bias(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute the bias and add them to the dataset.
    """
    out = {}
    groups = find_var_groups(ds)

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

def _suffix_from_name(name: str) -> str:
    m = re.match(r"^(Label|Predicted|Bias)_(.+)$", name)
    return m.group(2) if m else name

def timeseries_r2(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """
    R² for one time series at a pixel (1D arrays). Returns NaN if:
      - no valid overlapping samples, or
      - variance of y is zero.
    """
    m = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return np.float32(np.nan)
    y = y[m]; yhat = yhat[m]
    sst = np.sum((y - np.mean(y))**2)
    if sst <= 0:
        return np.float32(np.nan)
    sse = np.sum((yhat - y)**2)
    return np.float32(1.0 - sse / sst)

def timeseries_rmse(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """RMSE for one time series at a pixel (1D arrays). NaN if no valid overlap."""
    m = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return np.float32(np.nan)
    y = y[m]; yhat = yhat[m]
    return np.float32(np.sqrt(np.mean((yhat - y)**2)))

def spatial_r2_rmse(ds: xr.Dataset, *, metric: str = "r2") -> xr.Dataset:
    """
    For each (Label_*, Predicted_*) pair in `ds`, compute a pixel-wise R² or RMSE
    across 'time' and return a Dataset with one 2D (lat, lon) DataArray per pair.

    Output variable names are `{suffix}_{metric}`, where `suffix` is the part after
    the Label_/Predicted_ prefix (i.e., what `find_var_groups` grouped by).
    """
    if "time" not in ds.dims:
        raise ValueError("Dataset must have a 'time' dimension.")
    if metric not in {"r2", "rmse"}:
        raise ValueError("metric must be 'r2' or 'rmse'.")

    pairs = find_var_groups(ds)  # returns tuples like (Label_foo, Predicted_foo[, Bias_foo])
    if not pairs:
        raise ValueError("No (Label_*, Predicted_*) pairs found in dataset.")

    func = timeseries_r2 if metric == "r2" else timeseries_rmse
    out_vars: Dict[str, xr.DataArray] = {}

    for grp in pairs:
        label_name, pred_name = grp[0], grp[1]
        y = ds[label_name]
        yhat = ds[pred_name]

        # align on time (and any shared coords)
        y, yhat = xr.align(y, yhat, join="inner")

        # apply_ufunc over the 'time' core dim → scalar per (lat,lon)
        da_metric = xr.apply_ufunc(
            func,
            y, yhat,
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32],
        )

        suffix = _suffix_from_name(label_name)
        var_name = f"{suffix}_{metric}"
        da_metric.name = var_name
        out_vars[var_name] = da_metric

    return xr.Dataset(out_vars)

