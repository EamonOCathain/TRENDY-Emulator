import xarray as xr, numpy as np, dask
from pathlib import Path

dask.config.set(scheduler="threads", num_workers=8)

lai_path = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/lai_avh15c1_seasonality.zarr")
tvt_path = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/tvt_mask.nc")

# Open LAI with sane chunks
ds = xr.open_zarr(lai_path, consolidated=True, chunks={"time": 12, "lat": 256, "lon": 256})
da = ds["lai_avh15c1"]

# Open TVT and align dims/coords
tvt_ds = xr.open_dataset(tvt_path)
mask_var = "tvt_mask" if "tvt_mask" in tvt_ds else next(iter(tvt_ds.data_vars))
tvt = tvt_ds[mask_var]

ren = {}
for want, cands in {"lat": ("lat", "latitude", "y"), "lon": ("lon", "longitude", "x")}.items():
    for c in cands:
        if c in tvt.dims: ren[c] = want; break
if ren: tvt = tvt.rename(ren)
if not (tvt.lat.identical(da.lat) and tvt.lon.identical(da.lon)):
    tvt = tvt.sel(lat=da.lat, lon=da.lon, method="nearest")

# Land eligibility (boolean, no NaNs)
elig = tvt.isin([0, 1, 2])
T = da.sizes["time"]

# Boolean missing mask
isnan = xr.apply_ufunc(np.isnan, da, dask="parallelized")

# CRUCIAL: keep boolean, broadcast elig across time, and AND it in.
# This avoids NaNs and ensures oceans are simply False everywhere.
elig3 = xr.ones_like(da, dtype=bool) & elig  # broadcast elig to (time, lat, lon)
isnan_land = isnan & elig3                   # boolean (time, lat, lon)

# Land pixel count (scalar)
n_pixels = elig.sum()                        # (lat,lon) -> scalar via reduction
n_pixels = dask.compute(n_pixels)[0].item()

# Overall NaN fraction over land×time
nan_cells = isnan_land.sum()                 # counts True across (time,lat,lon)
nan_cells = dask.compute(nan_cells)[0].item()
overall_nan_frac = nan_cells / (n_pixels * T)

# Per-pixel any/all within the year cycle
nan_any  = isnan_land.any("time").sum()      # number of land pixels with any NaN month
nan_all  = isnan_land.all("time").sum()      # number of land pixels with all 12 months NaN
nan_part = (isnan_land.any("time") & ~isnan_land.all("time")).sum()

n_any, n_all, n_partial = dask.compute(nan_any, nan_all, nan_part)
n_any, n_all, n_partial = int(n_any), int(n_all), int(n_partial)

print("\n[SEASONALITY MASK NaN CHECK — land only]")
print(f"  • Overall NaN fraction (months × land grid): {overall_nan_frac:.3%}")
print(f"  • Pixels all-NaN (12/12 months NaN): {n_all:,} / {n_pixels:,} ({n_all/n_pixels:.3%})")
print(f"  • Pixels partially-NaN (1–11 months NaN): {n_partial:,} / {n_pixels:,} ({n_partial/n_pixels:.3%})")

# Optional: mean fraction of NaN months among partially-NaN pixels
frac_nan_per_pixel = isnan_land.mean("time")
mean_partial = frac_nan_per_pixel.where((frac_nan_per_pixel > 0) & (frac_nan_per_pixel < 1)).mean()
mean_partial = float(dask.compute(mean_partial)[0])
print(f"  • Mean fraction of NaN months among partially-NaN pixels: {mean_partial:.3f}")