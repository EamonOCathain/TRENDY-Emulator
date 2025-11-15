import xarray as xr
import numpy as np
import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))
from src.utils.tools import slurm_shard

# ---- Parse CLI arguments ----
parser = argparse.ArgumentParser(description="Compute metrics from NetCDF predictions.")
parser.add_argument(
    "--dir",
    type=str,
    required=True,
    help="Input directory containing NetCDF files (root to search recursively).",
)
parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
    help="Directory where output metrics should be saved.",
)

args = parser.parse_args()

dir = Path(args.dir)
out_dir = Path(args.out_dir)

# ---- Find all NC Files and Build a list of tasks ----
nc_files = list(dir.rglob("*.nc"))
tasks = slurm_shard(nc_files)

# one record per file: (subdir, var, DataArray)
records = []

for f in tasks:
    ds = xr.open_dataset(f)
    var = list(ds.data_vars)[0]
    subdir = f.relative_to(dir).parent  # e.g. '.', 'S1', 'S2/foo', ...
    print(subdir, var)
    records.append((subdir, var, ds[var]))

annual_vars = ['cVeg', 'cSoil', 'cLitter']

# Helpers
def save_netcdf(da, path, overwrite=True):
    if overwrite and path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(path)
    print(f"Saved: {path}")
    
# Mask to valid land
tvt_mask = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/tvt_mask.nc")
tvt_ds = xr.open_dataset(tvt_mask)
land_mask = tvt_ds["tvt_mask"].isin([0, 1, 2])

masked_records = []
for subdir, var, da in records:
    masked_records.append((subdir, var, da.where(land_mask)))
records = masked_records

# ---- Means ----
print("Calculating means time means...")
time_means = {}
for subdir, var, da in records:
    mean = da.mean(dim="time")
    time_means.setdefault(subdir, {})[var] = mean
    print(subdir, var, mean.shape)
    out_path = out_dir / subdir / "mean" / "time_mean" / f"{var}_time_mean.nc"
    save_netcdf(mean, out_path)

print("Calculating space means...")
space_mean = {}
for subdir, var, da in records:
    da_space = da
    if var in annual_vars:
        da_space = da_space.resample(time="YE").mean()
    mean = da_space.mean(dim=["lat", "lon"])
    space_mean.setdefault(subdir, {})[var] = mean
    print(subdir, var, mean.shape)
    out_path = out_dir / subdir / "mean" / "space_mean" / f"{var}_space_mean.nc"
    save_netcdf(mean, out_path)

# ---- Seasonality ----
print("Calculating seasonality 3D...")
seasonality_3D = {}
for subdir, var, da in records:
    if var not in ['cVeg','cSoil','cLitter']:
        monthly_mean = da.groupby("time.month").mean(dim="time")

        month_index = xr.DataArray(
            da["time.month"].values,
            dims=("time",),
            coords={"time": da.time},
            name="month",
        )

        monthly_mean_repeated = monthly_mean.sel(month=month_index)

        seasonality_3D.setdefault(subdir, {})[var] = monthly_mean_repeated
        print(subdir, var, monthly_mean_repeated.shape)
        out_path = out_dir / subdir / "seasonality_3D" / f"{var}_seasonality_3D.nc"
        save_netcdf(monthly_mean_repeated, out_path)

seasonality_1D = {}
print("Calculating seasonality 1D...")
for subdir, var, da in records:
    if var not in annual_vars:
        monthly_mean = da.groupby("time.month").mean(dim=["time", "lat", "lon"])
        seasonality_1D.setdefault(subdir, {})[var] = monthly_mean
        print(subdir, var, monthly_mean.shape)
        out_path = out_dir / subdir / "seasonality_1D" / f"{var}_seasonality_1D.nc"
        save_netcdf(monthly_mean, out_path)

# ---- Trend ----
print("Calculating 2D Trend...")
trend_2D = {}
for subdir, var, da in records:
    annual_da = da.resample(time="YE").mean()
    n_years = annual_da.sizes["time"]
    annual_da = annual_da.assign_coords(time=np.arange(n_years))

    pf = annual_da.polyfit(dim="time", deg=1)
    coeff = pf["polyfit_coefficients"]

    orig_units = da.attrs.get("units", "")
    if orig_units:
        coeff.attrs["units"] = orig_units
        coeff.attrs["slope_units"] = f"{orig_units} yr-1"
    else:
        coeff.attrs["units"] = ""
        coeff.attrs["slope_units"] = "yr-1"

    coeff.attrs["time_units"] = "years since 1901-01-01"

    trend_2D.setdefault(subdir, {})[var] = coeff
    print(subdir, var, coeff.shape)

    out_path = out_dir / subdir / "trend_2D" / f"{var}_trend_2D.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_netcdf(coeff, out_path)

# ---- 1D Trend ----
print("Calculating 1D Trend...")
trend_1d = {}
for subdir, var_dict in trend_2D.items():
    for var, trend in var_dict.items():
        mean_trend = trend.mean(dim=["lat", "lon"])
        trend_1d.setdefault(subdir, {})[var] = mean_trend
        print(subdir, var, mean_trend.shape)
        out_path = out_dir / subdir / "trend_1D" / f"{var}_trend_1D.nc"
        save_netcdf(mean_trend, out_path)

# ---- Inter-Annual Variability ----
print("Calculating IAV 3D...")
iav_3D = {}
for subdir, var, da in records:
    annual_da = da.resample(time="YE").mean()
    n_years = annual_da.sizes["time"]
    annual_da = annual_da.assign_coords(time=np.arange(n_years))

    trend = trend_2D[subdir][var]
    intercept = trend.sel(degree=0)
    slope = trend.sel(degree=1)

    t = annual_da["time"]
    fitted = intercept + slope * t

    iav_da = annual_da - fitted

    # Correct metadata assignment
    iav_da.attrs["long_name"] = f"Interannual variability of {var}"
    iav_da.attrs["time_units"] = "years since 1901-01-01"

    iav_3D.setdefault(subdir, {})[var] = iav_da
    print(subdir, var, iav_da.shape)

    out_path = out_dir / subdir / "iav_3D" / f"{var}_iav_3D.nc"
    save_netcdf(iav_da, out_path)
    
print("Calculating IAV 1D...")
iav_1D = {}
for subdir, var, da in records:
    annual_da = da.resample(time="YE").mean()
    n_years = annual_da.sizes["time"]
    annual_da = annual_da.assign_coords(time=np.arange(n_years))

    trend = trend_2D[subdir][var]
    intercept = trend.sel(degree=0)
    slope = trend.sel(degree=1)

    t = annual_da["time"]
    fitted = intercept + slope * t

    iav_da = annual_da - fitted

    mean_iav = iav_da.mean(dim=["lat", "lon"])
    iav_1D.setdefault(subdir, {})[var] = mean_iav
    print(subdir, var, mean_iav.shape)
    
    mean_iav.attrs["long_name"] = f"Spatial mean interannual variability of {var}"
    mean_iav.attrs["time_units"] = "years since 1901-01-01"

    out_path = out_dir / subdir / "iav_1D" / f"{var}_iav_1D.nc"
    save_netcdf(mean_iav, out_path)

print("Finished all metrics.")