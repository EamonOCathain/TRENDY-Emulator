#!/usr/bin/env python3
import xarray as xr
import numpy as np
from pathlib import Path
from numcodecs import Blosc
import shutil

# ==== Config ====
ENABLE_TEMPORAL_FILL = True   # set False to skip the temporal neighbor-month pass
SPATIAL_KERNEL = 3            # 3x3 spatial mean

# ------------------------------ Paths ------------------------------
src_nc    = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/lai_avh15c1.nc")
tvt_nc    = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/tvt_mask.nc")
dst_zarr  = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/lai_avh15c1_seasonality.zarr")

var_in    = "lai_avh15c1"    # variable in the LAI NetCDF
var_out   = "lai_avh15c1"    # variable name in the output Zarr

print(f"[INFO] Opening LAI: {src_nc}")
ds = xr.open_dataset(src_nc, use_cftime=True)

if var_in not in ds:
    raise KeyError(f"Variable '{var_in}' not found in {src_nc}. Have: {list(ds.data_vars)}")

da = ds[var_in]

# Normalize fill values to NaN (0 is VALID)
fillv = da.attrs.get("_FillValue", None)
if fillv is not None:
    da = da.where(da != fillv)
da = da.where(da != -999)

# ------------------------------ TVT mask ------------------------------
print(f"[INFO] Opening TVT mask: {tvt_nc}")
m = xr.open_dataset(tvt_nc)
mask_var = "tvt_mask" if "tvt_mask" in m.data_vars else list(m.data_vars)[0]
tvt = m[mask_var]

# Harmonize lat/lon names
lat_dim = next(d for d in ("lat", "latitude", "y") if d in tvt.dims)
lon_dim = next(d for d in ("lon", "longitude", "x") if d in tvt.dims)
tvt = tvt.rename({lat_dim: "lat", lon_dim: "lon"})

# Align mask to data grid if needed
if not (tvt.lat.identical(da.lat) and tvt.lon.identical(da.lon)):
    tvt = tvt.sel(lat=da.lat, lon=da.lon, method="nearest")

# Eligible land: values 0/1/2 (ocean or missing = ineligible)
elig = tvt.isin([0, 1, 2])
land_bool = elig.fillna(False)  # boolean land mask for safe boolean ops

# Restrict data to land only before climatology (oceans ignored and stay NaN)
da_land = da.where(elig)

# ----------------------- Compute base seasonality (12-month climatology) -----------------------
# Keep pixels that ever have any finite value (over entire record)
valid_any = da_land.notnull().any(dim="time")

# 12-month climatology (NaNs ignored)
clim = da_land.groupby("time.month").mean(dim="time", skipna=True)
# Drop pixels that never have data
clim = clim.where(valid_any)

# Rename 'month' -> 'time' with 1..12
clim = clim.rename({"month": "time"}).assign_coords(time=np.arange(1, 13))
clim.attrs["description"] = (
    "Monthly climatology (1..12) for LAI over TVT land (0/1/2). "
    "NaN where pixel never has any valid data. 0 is treated as valid."
)

# Final land mask on the climatology (keeps oceans as NaN by construction)
clim = clim.where(elig)

# ------------------------------ Reporting helper ------------------------------
def report_nan_stats(title: str, arr: xr.DataArray, land_mask: xr.DataArray) -> None:
    """Report land-only NaN stats over 12 months."""
    months = arr.sizes.get("time", 0)
    land_pixels = int(land_mask.sum().values)
    total_cells = land_pixels * months if land_pixels else 0

    isnan = xr.apply_ufunc(np.isnan, arr, dask="parallelized")
    nan_over_land = (isnan & land_mask)

    nan_cells = int(nan_over_land.sum().values) if total_cells else 0
    overall_nan_frac = (nan_cells / total_cells) if total_cells else np.nan

    # fully-NaN pixels across the 12-month seasonality
    has_any_data_pixel = ((~isnan) & land_mask).any(dim="time")
    fully_nan_pixels = int((~has_any_data_pixel & land_mask).sum().values)

    # fully-NaN months (all land pixels NaN in that month)
    nan_count_per_month = nan_over_land.sum(dim=("lat", "lon"))
    land_count = land_mask.sum(dim=("lat", "lon"))
    fully_nan_months = int((nan_count_per_month == land_count).sum().values)

    print(f"\n[REPORT — {title} — TVT land only]")
    print(f"  • Land pixels: {land_pixels:,}")
    print(f"  • Overall NaN fraction (months × land): {overall_nan_frac:.3%}")
    print(f"  • Fully-NaN pixels: {fully_nan_pixels:,} / {land_pixels:,} "
          f"({(fully_nan_pixels/land_pixels) if land_pixels else 0:.3%})")
    print(f"  • Fully-NaN months: {fully_nan_months} / {months} "
          f"({(fully_nan_months/months if months else 0):.3%})")

# ------------------------------ Report (before fills) ------------------------------
report_nan_stats("seasonality (raw climatology)", clim, land_bool)

# ------------------------------ Spatial 3×3 fill (land-only, month-by-month) ------------------
from scipy.ndimage import uniform_filter

def _spatial_fill_2d(arr2d: np.ndarray, elig2d: np.ndarray, k: int) -> np.ndarray:
    """
    NaN-aware k×k local-mean fill for a single 2D field, restricted to eligible land.
    - Only fills where elig2d=True and arr2d is NaN.
    - Donors are only from elig2d=True pixels.
    """
    arr = arr2d.copy()
    # Only donors on land
    arr[~elig2d] = np.nan

    vals = np.nan_to_num(arr, copy=False)
    cnts = np.isfinite(arr).astype(np.float32)

    mean_vals = uniform_filter(vals, size=k, mode="nearest")
    mean_cnts = uniform_filter(cnts, size=k, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        neigh_mean = mean_vals / mean_cnts

    out = arr2d.copy()
    fillable = elig2d & np.isnan(out) & (mean_cnts > 0)
    out[fillable] = neigh_mean[fillable]
    return out

# Apply per month
clim_spatial = xr.apply_ufunc(
    _spatial_fill_2d,
    clim,
    land_bool,
    xr.DataArray(SPATIAL_KERNEL),
    input_core_dims=[["lat", "lon"], ["lat", "lon"], []],
    output_core_dims=[["lat", "lon"]],
    vectorize=True,        # loops across time
    dask="parallelized",
    output_dtypes=[clim.dtype],
)

report_nan_stats("after spatial 3×3 fill", clim_spatial, land_bool)

# ------------------------------ Temporal neighbor-month fill (optional) -----------------------
if ENABLE_TEMPORAL_FILL:
    # Previous and next month with wrap-around; only consider land pixels
    prev_m = clim_spatial.roll(time=1, roll_coords=False)
    next_m = clim_spatial.roll(time=-1, roll_coords=False)

    neighbor_mean = xr.concat([prev_m.where(land_bool), next_m.where(land_bool)], dim="neighbor").mean(
        dim="neighbor", skipna=True
    )

    # Fill only where it's land AND still NaN after spatial fill
    to_fill = xr.apply_ufunc(np.isnan, clim_spatial, dask="parallelized") & land_bool
    clim_filled = clim_spatial.where(~to_fill, neighbor_mean)
    report_nan_stats("after temporal neighbor-month fill", clim_filled, land_bool)
else:
    clim_filled = clim_spatial

# ----------------------------- Write Zarr ------------------------------
# Use modest spatial chunks; time=12 fits in one chunk
out = clim_filled.to_dataset(name=var_out).chunk({"time": 12, "lat": 5, "lon": 5})

encoding = {
    var_out: {
        "compressor": Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE),
    }
}

if dst_zarr.exists():
    print(f"[INFO] Removing existing {dst_zarr} to write fresh store")
    shutil.rmtree(dst_zarr)

print(f"[INFO] Writing Zarr to {dst_zarr} with chunks (12,5,5)")
out.to_zarr(
    store=str(dst_zarr),
    mode="w",
    encoding=encoding,
    consolidated=True,
)

print("[OK] Done.")