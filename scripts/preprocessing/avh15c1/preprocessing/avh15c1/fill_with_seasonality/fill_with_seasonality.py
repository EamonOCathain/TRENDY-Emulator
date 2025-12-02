#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

# --- Inputs (you provided) ---
seasonality_zarr = "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/lai_avh15c1_seasonality.zarr"
src_nc          = "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/lai_avh15c1.nc"
tvt_path        = "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/tvt_mask.nc"
out_path        = "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/lai_avh15c1_filled_30x30.zarr"

# --- Helper: pick first data variable if the name isn't known ---
def first_var_name(ds: xr.Dataset) -> str:
    if not ds.data_vars:
        raise RuntimeError("Dataset has no data variables.")
    return list(ds.data_vars)[0]

# --- Load datasets ---
print("[INFO] Opening source NC, seasonality Zarr, and tvt mask …")
ds_src = xr.open_dataset(src_nc)                          # expects monthly time series
ds_seas = xr.open_zarr(seasonality_zarr, consolidated=True)  # expects 12-month climatology
ds_mask = xr.open_dataset(tvt_path)

# Detect variable names
var_src = first_var_name(ds_src)      # e.g., "lai_avh15c1"
var_seas = first_var_name(ds_seas)    # e.g., "lai_avh15c1" climatology
var_mask = first_var_name(ds_mask)    # e.g., "tvt_mask" (0,1,2 on land)

print(f"[INFO] Variables: src={var_src}, seasonality={var_seas}, mask={var_mask}")

# Align/rename common coords if needed
def maybe_rename_latlon(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude" in ds.coords and "lat" not in ds.coords: rename["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords: rename["longitude"] = "lon"
    return ds.rename(rename) if rename else ds

ds_src  = maybe_rename_latlon(ds_src)
ds_seas = maybe_rename_latlon(ds_seas)
ds_mask = maybe_rename_latlon(ds_mask)

# Ensure we share lat/lon & reindex seasonality/mask onto src grid if necessary
if not (np.array_equal(ds_src.lat, ds_seas.lat) and np.array_equal(ds_src.lon, ds_seas.lon)):
    print("[WARN] Reindexing seasonality onto source lat/lon grid …")
    ds_seas = ds_seas.reindex(lat=ds_src.lat, lon=ds_src.lon, method=None)

if not (np.array_equal(ds_src.lat, ds_mask.lat) and np.array_equal(ds_src.lon, ds_mask.lon)):
    print("[WARN] Reindexing mask onto source lat/lon grid …")
    ds_mask = ds_mask.reindex(lat=ds_src.lat, lon=ds_src.lon, method=None)

# --- Build land mask: land is where mask in {0,1,2} ---
mask_vals = ds_mask[var_mask]
land_mask = mask_vals.isin([0, 1, 2])

# --- Apply land mask (keep only land; ocean -> NaN) ---
da_src = ds_src[var_src]
da_src = da_src.where(land_mask)

# --- Prepare seasonality to match time (month-wise fill) ---
# Seasonality could have a 'month' dim (1..12) or a 12-length 'time' dim
da_seas = ds_seas[var_seas]
if "month" in da_seas.dims:
    seas_by_month = da_seas
elif "time" in da_seas.dims and da_seas.sizes["time"] == 12:
    # try to derive month numbers; fallback to 1..12
    try:
        months = da_seas["time"].dt.month
    except Exception:
        months = xr.DataArray(np.arange(1, 13), dims=("time",))
    seas_by_month = (
        da_seas
        .assign_coords(month=("time", months.data))
        .swap_dims({"time": "month"})
        .drop_vars("time")
    )


else:
    raise RuntimeError(
        "Seasonality must have either 'month' dim (len=12) or 'time' dim (len=12)."
    )

# Broadcast seasonality to each timestamp in source by selecting month
month_index = da_src["time"].dt.month
# Vectorized selection using a DA indexer will return a result indexed by 'time' automatically.
seas_full = seas_by_month.sel(month=month_index)

# If a leftover 'month' coordinate is present (not a dimension anymore), drop it to avoid confusion.
if "month" in seas_full.coords:
    seas_full = seas_full.drop_vars("month")

# --- Fill gaps using per-month seasonality ---
da_filled = da_src.fillna(seas_full)

# --- Compute % NaNs over land in final product ---
# Build a time-broadcast land mask
land_mask_time = land_mask.broadcast_like(da_filled)

# Count NaNs over land across all dims (time, lat, lon)
# Use .compute() to materialize dask results before casting to Python ints.
n_nan = int(da_filled.isnull().where(land_mask_time).sum().compute())

# Total land cells over all time steps (land pixels * time length)
# Since land_mask_time is broadcast to time, summing it over all dims gives that directly.
n_total_land = int(land_mask_time.sum().compute())

pct_nan = (100.0 * n_nan / n_total_land) if n_total_land > 0 else np.nan

print(f"[REPORT] Final % NaNs over land pixels: {pct_nan:.4f}% "
      f"({n_nan} / {n_total_land})")

# --- Package back into a Dataset and write to Zarr ---
ds_out = xr.Dataset({var_src: da_filled})

# Chunking: full time, lat=30, lon=30
# If time is large, 'full time' chunk means use the entire len(time) in one chunk.
chunks = {"time": ds_out.sizes["time"], "lat": 30, "lon": 30}
ds_out = ds_out.chunk(chunks)

# Zarr compressor level 1 and float32 for compactness
compressor = zarr.Blosc(cname="zstd", clevel=1, shuffle=2)
encoding = {
    var_src: {
        "dtype": "float32",
        "compressor": compressor,
        "chunks": (chunks["time"], chunks["lat"], chunks["lon"]),
        "_FillValue": np.float32(np.nan),
    }
}

out_path_p = Path(out_path)
if out_path_p.exists():
    print(f"[WARN] Removing existing {out_path_p} …")
    import shutil
    shutil.rmtree(out_path_p)

print(f"[WRITE] Writing Zarr to {out_path_p} (consolidated metadata) …")
ds_out.to_zarr(out_path_p, mode="w", encoding=encoding)
xr.backends.zarr.consolidate_metadata(str(out_path_p))

print("[DONE] Finished fill + write.")