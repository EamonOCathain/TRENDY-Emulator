#!/usr/bin/env python3
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr
from numcodecs import Blosc
import cftime

# ------------------ Paths ------------------
OUT_DIR = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/modis_lai")
IN_NC   = OUT_DIR / "modis_lai.nc"               # daily (or subdaily) input
MON_NC  = OUT_DIR / "modis_lai_monthly_not_filled.nc"       # monthly mean (xarray output)
CLIM_Z  = OUT_DIR / "modis_lai_seasonality.zarr"  # 12-step seasonal mask (land-only)
FILLED_Z = OUT_DIR / "modis_lai_monthly_filled.zarr" # filled monthly series (Zarr)

TVT_MASK = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/tvt_mask.nc")
VAR_NAME = "modis_lai"   # main variable name in IN_NC

# Helper: force 'time' coord to first-of-month cftime noleap and encode as days since 1901-01-01
def _to_month_start_noleap(da: xr.DataArray) -> xr.DataArray:
    # Try to get year/month; if numeric time, decode with CF first (best effort)
    t = da["time"]
    if np.issubdtype(t.dtype, np.number) and "units" in t.attrs:
        try:
            # decode on a tiny dataset to get dt fields
            _decoded = xr.decode_cf(da.to_dataset(name="__tmp__"))["time"]
            years = _decoded.dt.year.values
            months = _decoded.dt.month.values
        except Exception:
            # fall back to assuming it's already month-start points
            return da
    else:
        years = t.dt.year.values
        months = t.dt.month.values

    cf = np.array([cftime.DatetimeNoLeap(int(y), int(m), 1) for y, m in zip(years, months)], dtype=object)
    return da.assign_coords(time=("time", cf))

# ------------------ 1) Monthly mean via xarray ------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Computing monthly mean via xarray → {MON_NC}")

ds_in = xr.open_dataset(IN_NC, decode_times=True)  # decode if possible
if VAR_NAME not in ds_in:
    raise KeyError(f"Variable '{VAR_NAME}' not found. Have: {list(ds_in.data_vars)}")
da_in = ds_in[VAR_NAME]

# normalise fill values
fillv = da_in.attrs.get("_FillValue", None)
if fillv is not None:
    da_in = da_in.where(da_in != fillv)
da_in = da_in.where(~np.isnan(da_in))

# Monthly mean, labeled at month start (first day)
# Using 'MS' ensures bins are month-start to next month-start
da_mon = da_in.resample(time="MS").mean(skipna=True)

# Force time to first-of-month in noleap calendar and write with CF encoding
da_mon = _to_month_start_noleap(da_mon)

print(f"[INFO] Writing monthly mean → {MON_NC}")
da_mon.to_dataset(name=VAR_NAME).to_netcdf(
    MON_NC,
    encoding={
        "time": {"units": "days since 1901-01-01", "calendar": "noleap"},
        VAR_NAME: {"_FillValue": np.float32(np.nan)},
    },
)

# ------------------ Open data & mask ------------------
print(f"[INFO] Opening monthly mean: {MON_NC}")
# Open without decoding so we keep integer days since 1901 on the coord
ds = xr.open_dataset(MON_NC, decode_times=False)
if VAR_NAME not in ds:
    raise KeyError(f"Variable '{VAR_NAME}' not found. Have: {list(ds.data_vars)}")
da = ds[VAR_NAME]

# Normalize fill values if present
fillv = da.attrs.get("_FillValue", None)
if fillv is not None:
    da = da.where(da != fillv)
da = da.where(~np.isnan(da))  # ensure NaN is NaN

print(f"[INFO] Opening TVT mask: {TVT_MASK}")
m = xr.open_dataset(TVT_MASK)
mask_var = list(m.data_vars)[0]  # use first var if not named specifically
tvt = m[mask_var]

# align mask to data grid if needed
lat_dim_t = next(d for d in ("lat", "latitude", "y") if d in tvt.dims)
lon_dim_t = next(d for d in ("lon", "longitude", "x") if d in tvt.dims)
tvt = tvt.rename({lat_dim_t: "lat", lon_dim_t: "lon"})
if not (tvt.lat.identical(da.lat) and tvt.lon.identical(da.lon)):
    tvt = tvt.sel(lat=da.lat, lon=da.lon, method="nearest")

# Land-only (eligible): tvt in {0,1,2}
elig = tvt.isin([0, 1, 2]).astype(bool)

# ------------------ 2) Seasonal (12-step) climatology, land-only ------------------
print("[INFO] Computing monthly climatology (12 steps, land-only)")
da_land = da.where(elig)

clim12 = da_land.groupby("time.month").mean(dim="time", skipna=True)  # (month, lat, lon)
clim12 = clim12.rename({"month": "time"}).assign_coords(time=np.arange(1, 13))
clim12.attrs["description"] = "Monthly climatology (1..12) over land-only (TVT∈{0,1,2}); zeros treated as valid."

# Write seasonal mask to Zarr (12,1,1) chunks
if CLIM_Z.exists():
    import shutil; shutil.rmtree(CLIM_Z)
print(f"[INFO] Writing seasonal mask Zarr → {CLIM_Z}")
clim_ds = clim12.to_dataset(name=VAR_NAME).chunk({"time": 12, "lat": 256, "lon": 256})
clim_ds.to_zarr(
    str(CLIM_Z),
    mode="w",
    encoding={VAR_NAME: {"compressor": Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)}},
    consolidated=True,
)

# ------------------ 3) NaN stats over land (monthly-mean file) ------------------
print("[INFO] Computing NaN stats (land-only)")
da_stat_land = da.where(elig)

stacked = da_stat_land.stack(space=("lat", "lon"))
sel_idx = np.flatnonzero(elig.stack(space=("lat", "lon")).values)
if sel_idx.size == 0:
    raise RuntimeError("No land pixels found in TVT mask after alignment.")
stacked_land = stacked.isel(space=sel_idx)

total_cells = stacked_land.sizes["time"] * stacked_land.sizes["space"]
nan_cells = int(np.isnan(stacked_land).sum().values)
overall_nan_frac = nan_cells / total_cells if total_cells else np.nan

fully_nan_pixels = int(np.isnan(stacked_land).all(dim="time").sum().values)
fully_nan_timesteps = int(np.isnan(stacked_land).all(dim="space").sum().values)

print("\n[REPORT — land-only]")
print(f"  • Overall NaN fraction: {overall_nan_frac:.3%}")
print(f"  • Fully-NaN pixels:     {fully_nan_pixels} / {stacked_land.sizes['space']}")
print(f"  • Fully-NaN timesteps:  {fully_nan_timesteps} / {stacked_land.sizes['time']}\n")

# ------------------ 4) Fill monthly NaNs with seasonal value ------------------
print("[INFO] Filling monthly NaNs with per-month climatology (land-only)")
filled = da_land.groupby("time.month").fillna(clim12)
filled = filled.where(elig)

# ------------------ 5) Write filled monthly means to Zarr (time: days since 1901-01-01) ------------------
if FILLED_Z.exists():
    import shutil; shutil.rmtree(FILLED_Z)

print(f"[INFO] Writing filled monthly Zarr → {FILLED_Z}")
filled_ds = filled.to_dataset(name=VAR_NAME).chunk({"time": -1, "lat": 1, "lon": 1})

# Preserve the CF time encoding in Zarr too
filled_ds["time"].attrs.update({"units": "days since 1901-01-01", "calendar": "noleap"})

filled_ds.to_zarr(
    str(FILLED_Z),
    mode="w",
    encoding={
        VAR_NAME: {"compressor": Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)},
        "time":   {"units": "days since 1901-01-01", "calendar": "noleap"},
    },
    consolidated=True,
)

print("[OK] All done.")