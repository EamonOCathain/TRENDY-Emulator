#!/usr/bin/env python3
import xarray as xr, numpy as np, dask
from pathlib import Path

dask.config.set(scheduler="threads", num_workers=8)

lai_path = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/lai_avh15c1_seasonality.zarr")
tvt_path = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/tvt_mask.nc")

# Open LAI climatology with sane chunks
ds = xr.open_zarr(lai_path, consolidated=True, chunks={"time": 12, "lat": 256, "lon": 256})
da = ds["lai_avh15c1"]

# Open TVT mask and align to LAI grid
tvt_ds = xr.open_dataset(tvt_path)
mask_var = next(iter(tvt_ds.data_vars))  # or "tvt_mask" if that's guaranteed
tvt = tvt_ds[mask_var]

# Rename dims if needed
ren = {}
for want, cands in {"lat": ("lat", "latitude", "y"), "lon": ("lon", "longitude", "x")}.items():
    for c in cands:
        if c in tvt.dims: ren[c] = want; break
if ren: tvt = tvt.rename(ren)

# Reindex to LAI grid if coords differ slightly
if not (tvt.lat.identical(da.lat) and tvt.lon.identical(da.lon)):
    tvt = tvt.sel(lat=da.lat, lon=da.lon, method="nearest")

# Eligibility: land pixels 0/1/2 are True; oceans (NaN or other) are False
elig = tvt.isin([0, 1, 2])

# Boolean missing mask
isnan = xr.apply_ufunc(np.isnan, da, dask="parallelized")

# >>> CRUCIAL: apply elig to the boolean mask, so oceans become NaN and are excluded
isnan_land = isnan.where(elig)

# Compute stats only over land pixels (skipna=True)
overall_nan_frac, n_any, n_all, n_partial, n_pixels = dask.compute(
    isnan_land.mean(),                              # mean over time,lat,lon; oceans dropped
    isnan_land.any("time").sum(),                   # count land pixels with any NaN month
    isnan_land.all("time").sum(),                   # count land pixels with all 12 months NaN
    (isnan_land.any("time") & ~isnan_land.all("time")).sum(),  # partially NaN
    elig.sum(),                                     # total land pixels
)

print("\n[SEASONALITY MASK NaN CHECK — land only]")
print(f"  • Overall NaN fraction (all months × land grid): {float(overall_nan_frac):.3%}")
print(f"  • Pixels all-NaN (12/12 months NaN): {int(n_all):,} / {int(n_pixels):,} ({int(n_all)/int(n_pixels):.3%})")
print(f"  • Pixels partially-NaN (1–11 months NaN): {int(n_partial):,} / {int(n_pixels):,} ({int(n_partial)/int(n_pixels):.3%})")

# Mean fraction of NaN months among partially-NaN pixels
frac_nan_per_pixel = isnan_land.mean("time")
mean_partial_frac = frac_nan_per_pixel.where((frac_nan_per_pixel > 0) & (frac_nan_per_pixel < 1)).mean().compute()
print(f"  • Mean fraction of NaN months among partially-NaN pixels: {float(mean_partial_frac):.3f}")