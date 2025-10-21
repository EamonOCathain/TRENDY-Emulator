#!/usr/bin/env python3
"""python qc_plot_netcdf.py /path/to/file.nc              # auto-detect var
python qc_plot_netcdf.py /path/to/file.nc -v cld """

import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False

def guess_var(ds):
    # Prefer a 3D+ var with time,lat,lon (order can vary)
    for v in ds.data_vars:
        dims = set(ds[v].dims)
        if {"time"}.issubset(dims) and ({"lat","lon"}.issubset(dims) or {"latitude","longitude"}.issubset(dims)):
            return v
    # fallback: first data var
    return next(iter(ds.data_vars))

def get_latlon_names(da):
    lat_names = ["lat","latitude","y"]
    lon_names = ["lon","longitude","x"]
    lat = next((n for n in lat_names if n in da.dims), None)
    lon = next((n for n in lon_names if n in da.dims), None)
    if lat is None or lon is None:
        raise ValueError(f"Couldn't find lat/lon dims in {da.dims}")
    return lat, lon

def wrap_lon_to_data(lon_data, lon_query):
    # If grid is 0..360 and query is -180..180 (or vice-versa), convert.
    minlon = float(lon_data.min())
    maxlon = float(lon_data.max())
    if maxlon > 180.0 and lon_query < 0:
        lon_query = lon_query % 360.0
    if maxlon <= 180.0 and lon_query > 180:
        lon_query = ((lon_query + 180) % 360) - 180
    return lon_query

def nearest_point(da, lat_name, lon_name, plat, plon):
    lats = da[lat_name].values
    lons = da[lon_name].values
    plon = wrap_lon_to_data(lons, plon)

    # 1D lat/lon assumed (regular grid)
    iy = int(np.argmin(np.abs(lats - plat)))
    ix = int(np.argmin(np.abs(lons - plon)))
    return iy, ix, float(lats[iy]), float(lons[ix])

def main():
    p = argparse.ArgumentParser(description="Quick NetCDF sanity check: map@t0 + timeseries in central Germany.")
    p.add_argument("file", help="Path to NetCDF")
    p.add_argument("-v","--var", help="Variable name (if omitted, guessed)")
    p.add_argument("--engine", default=None, help="xarray engine (e.g., netcdf4, h5netcdf)")
    p.add_argument("--lat", type=float, default=51.0, help="Probe latitude (default 51N)")
    p.add_argument("--lon", type=float, default=10.0, help="Probe longitude (default 10E)")
    args = p.parse_args()

    fp = Path(args.file)
    out_dir = fp.parent

    # Open
    ds = xr.open_dataset(fp, decode_times=True, engine=args.engine)

    var = args.var or guess_var(ds)
    if var not in ds:
        raise ValueError(f"Variable '{var}' not found. Available: {list(ds.data_vars)}")
    da = ds[var]

    # Basic stats
    print(f"[INFO] File: {fp}")
    print(f"[INFO] Var: {var} dims={da.dims} shape={tuple(da.shape)} dtype={da.dtype}")
    nans = int(np.isnan(da.values).sum()) if da.size < 5_000_000 else int(np.isnan(da.isel(time=0)).sum())
    print(f"[INFO] NaNs (quick check): {nans} {'(full array)' if da.size < 5_000_000 else '(first timestep)'}")

    # Dim names
    lat_name, lon_name = get_latlon_names(da)

    # Map at first time
    if "time" not in da.dims:
        raise ValueError("Variable has no 'time' dimension; cannot plot first timestamp.")
    da0 = da.isel(time=0)

    # Timeseries point (central Germany by default)
    iy, ix, plat, plon = nearest_point(da, lat_name, lon_name, args.lat, args.lon)
    ts = da.isel({lat_name: iy, lon_name: ix})

    # -------- Plot map --------
    map_png = out_dir / f"{fp.stem}_{var}_t0_map.png"
    plt.figure(figsize=(9,4.5))
    if HAVE_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        mesh = da0.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=True)
        ax.coastlines()
        ax.set_title(f"{var} @ first time • {fp.name}")
    else:
        # fallback without cartopy: imshow with lon/lat extents
        lats = da0[lat_name].values
        lons = da0[lon_name].values
        plt.imshow(da0.values, origin="lower",
                   extent=[float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())],
                   aspect="auto")
        plt.colorbar(label=var)
        plt.title(f"{var} @ first time • {fp.name}")
        plt.xlabel("lon"); plt.ylabel("lat")
    plt.tight_layout()
    plt.savefig(map_png, dpi=150)
    plt.close()
    print(f"[OK] Saved map @ t0 -> {map_png}")

    # -------- Plot time series --------
    ts_png = out_dir / f"{fp.stem}_{var}_timeseries_{plat:.2f}N_{plon:.2f}E.png"
    plt.figure(figsize=(9,4))
    ts.plot()
    plt.title(f"{var} • time series at ({plat:.2f}, {plon:.2f}) • nearest idx=({iy},{ix})")
    plt.tight_layout()
    plt.savefig(ts_png, dpi=150)
    plt.close()
    print(f"[OK] Saved time series -> {ts_png}")

    # Print a couple of values
    print(f"[SAMPLE] t0 mean={float(da0.mean().values):.4g}, std={float(da0.std().values):.4g}")
    print(f"[SAMPLE] TS first/last = {float(ts.isel(time=0).values):.4g} / {float(ts.isel(time=-1).values):.4g}")

if __name__ == "__main__":
    main()