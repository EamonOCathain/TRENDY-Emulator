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
parser.add_argument(
    "--start_year",
    type=int,
    default=1901,
    help="First calendar year to include (e.g. 1982).",
)
parser.add_argument(
    "--end_year",
    type=int,
    default=2023,
    help="Last calendar year to include (e.g. 2018).",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing output files (default: skip existing files).",
)

args = parser.parse_args()

dir = Path(args.dir)
out_dir = Path(args.out_dir)
start_year = args.start_year
end_year = args.end_year

# ---- Find all NC Files and Build a list of tasks ----
nc_files = list(dir.rglob("*.nc"))
tasks = slurm_shard(nc_files)

# one record per file: (subdir, var, DataArray)
records = []
for f in tasks:
    ds = xr.open_dataset(f)
    var = list(ds.data_vars)[0]
    da = ds[var]

    # Restrict to requested time window if time dimension exists
    if "time" in da.dims:
        da = da.sel(
            time=slice(
                f"{start_year}-01-01",
                f"{end_year}-12-31",
            )
        )

    subdir = f.relative_to(dir).parent  # e.g. '.', 'S1', 'S2/foo', ...
    print(subdir, var)
    records.append((subdir, var, da))

annual_vars = ["cVeg", "cSoil", "cLitter"]

# Helpers
def save_netcdf(da, path, *, overwrite):
    if path.exists() and not overwrite:
        return  # silently skip
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    da.to_netcdf(path)
    print(f"Saved: {path}")


# ---- Masks ----
tvt_mask_path = Path(
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/tvt_mask.nc"
)
tvt_ds = xr.open_dataset(tvt_mask_path)

# tvt_mask has dims ('lat', 'lon')
land_mask_2d = tvt_ds["tvt_mask"].isin([0, 1, 2])
test_mask_2d = tvt_ds["tvt_mask"].isin([2])

global_masked_records = []
for subdir, var, da in records:
    global_masked_records.append((subdir, var, da.where(land_mask_2d)))

test_masked_records = []
for subdir, var, da in records:
    test_masked_records.append((subdir, var, da.where(test_mask_2d)))

records = [global_masked_records, test_masked_records]

for record in records:
    if record is global_masked_records:
        out_dir = Path(args.out_dir) / "global"
    else:
        out_dir = Path(args.out_dir) / "test_locations"

    # ---- Means ----
    print("Calculating means time means...")
    time_means = {}
    for subdir, var, da in record:
        mean = da.mean(dim="time")
        time_means.setdefault(subdir, {})[var] = mean
        print(subdir, var, mean.shape)
        out_path = out_dir / subdir / "mean" / "time_mean" / f"{var}_time_mean.nc"
        save_netcdf(mean, out_path, overwrite=args.overwrite)

    print("Calculating space means...")
    space_mean = {}
    for subdir, var, da in record:
        out_path = out_dir / subdir / "mean" / "space_mean" / f"{var}_space_mean.nc"
        if out_path.exists() and not args.overwrite:
            print("Skipping existing file:", out_path)
            continue
        da_space = da
        if var in annual_vars:
            da_space = da_space.resample(time="YE").mean()
        mean = da_space.mean(dim=["lat", "lon"])
        space_mean.setdefault(subdir, {})[var] = mean
        print(subdir, var, mean.shape)
        save_netcdf(mean, out_path, overwrite=args.overwrite)

    # ---- Seasonality ----
    print("Calculating seasonality 3D...")
    seasonality_3D = {}
    for subdir, var, da in record:
        if var not in annual_vars:

            out_path = out_dir / subdir / "seasonality_3D" / f"{var}_seasonality_3D.nc"
            if out_path.exists() and not args.overwrite:
                print("Skipping existing file:", out_path)
                continue

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

            save_netcdf(monthly_mean_repeated, out_path, overwrite=args.overwrite)

    seasonality_1D = {}
    print("Calculating seasonality 1D...")
    for subdir, var, da in record:
        if var not in annual_vars:

            out_path = out_dir / subdir / "seasonality_1D" / f"{var}_seasonality_1D.nc"
            if out_path.exists() and not args.overwrite:
                print("Skipping existing file:", out_path)
                continue

            monthly_mean = da.groupby("time.month").mean(dim=["time", "lat", "lon"])
            seasonality_1D.setdefault(subdir, {})[var] = monthly_mean
            print(subdir, var, monthly_mean.shape)
            save_netcdf(monthly_mean, out_path, overwrite=args.overwrite)

    # ---- Annual means (3D + 1D) ----
    print("Calculating annual means (3D + 1D)...")
    annual_means_3D = {}
    annual_means_1D = {}
    for subdir, var, da in record:
        # Resample to annual means (year-end)
        annual_da_3d = da.resample(time="YE").mean()

        annual_means_3D.setdefault(subdir, {})[var] = annual_da_3d
        print(subdir, var, annual_da_3d.shape)

        # Spatial mean → 1D (time)
        annual_da_1d = annual_da_3d.mean(dim=["lat", "lon"])
        annual_means_1D.setdefault(subdir, {})[var] = annual_da_1d
        print(subdir, var, annual_da_1d.shape)

        # Save 3D
        out_path_3d = out_dir / subdir / "annual_mean_3D" / f"{var}_annual_mean_3D.nc"
        save_netcdf(annual_da_3d, out_path_3d, overwrite=args.overwrite)

        # Save 1D
        out_path_1d = out_dir / subdir / "annual_mean_1D" / f"{var}_annual_mean_1D.nc"
        save_netcdf(annual_da_1d, out_path_1d, overwrite=args.overwrite)

    # ---- Trend ----
    print("Calculating 2D Trend...")
    trend_2D = {}
    for subdir, var, da in record:

        out_path = out_dir / subdir / "trend_2D" / f"{var}_trend_2D.nc"
        if out_path.exists() and not args.overwrite:
            print("Skipping existing file:", out_path)
            continue

        annual_da = annual_means_3D[subdir][var]
        n_years = annual_da.sizes["time"]

        if n_years < 2:
            print("Skipping trend calculation due to insufficient years:", n_years, "for", subdir, var)
            continue
        else:
            print("n_years:", n_years, "for", subdir, var)

        # Use a simple year index 0..n_years-1 for polyfit
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

        save_netcdf(coeff, out_path, overwrite=args.overwrite)

    # ---- 1D Trend ----
    print("Calculating 1D Trend...")
    trend_1d = {}
    for subdir, var_dict in trend_2D.items():
        for var, trend in var_dict.items():

            out_path = out_dir / subdir / "trend_1D" / f"{var}_trend_1D.nc"
            if out_path.exists() and not args.overwrite:
                print("Skipping existing file:", out_path)
                continue

            mean_trend = trend.mean(dim=["lat", "lon"])
            trend_1d.setdefault(subdir, {})[var] = mean_trend
            print(subdir, var, mean_trend.shape)
            save_netcdf(mean_trend, out_path, overwrite=args.overwrite)

    # ---- Inter-Annual Variability (1D) ----
    print("Calculating IAV 1D...")
    iav_1D = {}
    for subdir, var_dict in trend_2D.items():
        for var, trend in var_dict.items():
            out_path = out_dir / subdir / "iav_1D" / f"{var}_iav_1D.nc"
            if out_path.exists() and not args.overwrite:
                print("Skipping existing file:", out_path)
                continue

            annual_da = annual_means_3D[subdir][var]
            n_years = annual_da.sizes["time"]
            annual_da = annual_da.assign_coords(time=np.arange(n_years))

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

            save_netcdf(mean_iav, out_path, overwrite=args.overwrite)

    print("Finished all metrics.")