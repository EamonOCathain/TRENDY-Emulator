#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import cftime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))
from src.dataset.variables import var_names
from src.analysis.visualisation import plot_timestep, robust_limits  # noqa: E402

def _time_label(da_time: xr.DataArray, ti: int) -> str:
    try:
        units = da_time.attrs.get("units")
        cal = da_time.attrs.get("calendar", "standard")
        val = da_time.isel(time=ti).values.item()
        try:
            dt = cftime.num2date(val, units=units, calendar=cal)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return str(np.array(val))
    except Exception:
        return f"t={ti}"

def _units_for(da: xr.DataArray) -> str:
    return da.attrs.get("units") or da.attrs.get("unit") or ""

def _strictly_increasing_slice(da: xr.DataArray, dim: str) -> xr.DataArray:
    vals = np.asarray(da[dim].values)
    order = np.argsort(vals)
    vals_sorted = vals[order]
    keep = np.concatenate(([True], np.diff(vals_sorted) > 0))
    return da.isel({dim: order[keep]})

def main():
    if len(sys.argv) < 2:
        print(f"usage: {Path(sys.argv[0]).name} <dataset.zarr> [tindex]", file=sys.stderr)
        sys.exit(2)

    zarr_dir = Path(sys.argv[1])
    tindex = int(sys.argv[2]) if len(sys.argv) >= 3 else 10

    # Open once
    ds = xr.open_zarr(zarr_dir, consolidated=True, decode_times=False)

    for var in var_names['monthly_outputs']:
        if var not in ds.data_vars:
            print(f"[SKIP] {var} not found.")
            continue

        da = ds[var]
        if "time" not in da.dims or not {"lat","lon"}.issubset(da.dims):
            print(f"[SKIP] {var} missing time/lat/lon dims.")
            continue

        ntime = int(da.sizes["time"])
        if not (0 <= tindex < ntime):
            print(f"[SKIP] {var}: tindex {tindex} out of bounds (0..{ntime-1}).")
            continue

        da2d = da.isel(time=tindex).transpose("lat", "lon", ...)
        try:
            if float(da2d.lon.max()) > 180:
                lon_wrapped = ((da2d.lon + 180) % 360) - 180
                da2d = da2d.assign_coords(lon=lon_wrapped)
        except Exception:
            pass

        da2d = _strictly_increasing_slice(da2d, "lat")
        da2d = _strictly_increasing_slice(da2d, "lon")

        vmin, vmax = robust_limits([da2d.values], mode="range", pct_low=1.0, pct_high=99.0, zero_floor_if_nonneg=True)
        
        if vmin is None or vmax is None or vmin == vmax:
            vmin = float(np.nanmin(da2d.values))
            vmax = float(np.nanmax(da2d.values))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, 1.0

        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        title = f"{var} · {_time_label(ds['time'], tindex)}"
        units = _units_for(da)

        plot_timestep(
            ax=ax,
            da2d=da2d,
            title=title,
            units=units,
            cbar_label=units or var,
            vmin=vmin,
            vmax=vmax,
            cmap=None,
            add_coastlines=True,
            add_borders=True,
        )

        fig.tight_layout()
        out_path = Path.cwd() / f"{var}_t{tindex:05d}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved {out_path}")

if __name__ == "__main__":
    main()