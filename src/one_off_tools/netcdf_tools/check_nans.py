#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import dask

"""
python check_nans.py --in /path/to/data.nc --var avh15c1_lai --mask /path/to/tvt_mask.nc

reports NaN coverage for a 3-D geospatial variable (time, lat, lon). 
It optionally uses TVT mask (values 0/1/2 = land; NaN/other = ignore) to restrict the analysis to land pixels. 
It will also work with a mask which is all 1 and -1, but not with a mask that is 0 and 1.
"""

# ---------- Helpers ----------
def parse_chunks(spec: str | None) -> dict | None:
    """
    Parse chunk spec like "time:auto,lat:256,lon:256" -> {'time': None, 'lat': 256, 'lon': 256}
    Returns None to let xarray decide (use file/native chunks).
    """
    if spec is None:
        return None
    spec = spec.strip()
    if not spec or spec.lower() == "auto":
        return None
    out = {}
    for kv in spec.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split(":")
        v = v.strip().lower()
        out[k.strip()] = None if v == "auto" else int(v)
    return out

def pick_var(ds: xr.Dataset, prefer: str | None = None) -> str:
    """Pick a 3D (time, lat, lon) variable. If `prefer` is given, validate it exists."""
    if prefer:
        if prefer in ds.data_vars:
            return prefer
        raise KeyError(f"Variable '{prefer}' not found. Available: {list(ds.data_vars)}")
    # otherwise pick first var that looks like (time,lat,lon)
    for vname, da in ds.data_vars.items():
        dims = set(da.dims)
        lat = any(d in dims for d in ("lat", "latitude", "y"))
        lon = any(d in dims for d in ("lon", "longitude", "x"))
        tim = any("time" == d or "time" in d for d in da.dims)
        if lat and lon and tim and da.ndim == 3:
            return vname
    # fallback: first data var
    return next(iter(ds.data_vars))

def rename_latlon(da: xr.DataArray) -> xr.DataArray:
    """Rename latitude/longitude dims to 'lat'/'lon' if needed."""
    ren = {}
    for want, candidates in {"lat": ("lat", "latitude", "y"), "lon": ("lon", "longitude", "x")}.items():
        if want in da.dims:
            continue
        for c in candidates:
            if c in da.dims:
                ren[c] = want
                break
    return da.rename(ren) if ren else da

def align_mask(mask_path: Path, target: xr.DataArray) -> xr.DataArray:
    """Load TVT mask, align to target grid, and return boolean eligibility (0/1/2 -> True)."""
    mds = xr.open_dataset(mask_path)
    # pick a variable
    mvar = "tvt_mask" if "tvt_mask" in mds else next(iter(mds.data_vars))
    m = mds[mvar]
    # rename dims and align to target
    ren = {}
    for want, cands in {"lat": ("lat", "latitude", "y"), "lon": ("lon", "longitude", "x")}.items():
        if want in m.dims:
            continue
        for c in cands:
            if c in m.dims:
                ren[c] = want
                break
    if ren:
        m = m.rename(ren)
    if not (m.lat.identical(target.lat) and m.lon.identical(target.lon)):
        # nearest is fine for masks at same resolution; if different, consider regridding upstream
        m = m.sel(lat=target.lat, lon=target.lon, method="nearest")
    # eligible: exactly 0,1,2 (land)
    elig = m.isin([0, 1, 2])
    return elig

def safe_time_label(tval) -> str:
    """Format a time coordinate nicely whether it's datetime64, cftime, or numeric."""
    try:
        return np.datetime_as_string(np.datetime64(tval), unit="D")
    except Exception:
        try:
            # cftime or pandas Timestamp
            return str(getattr(tval, "isoformat", lambda: str(tval))())
        except Exception:
            return str(tval)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Report NaN coverage for a (time, lat, lon) variable, optionally land-only via TVT mask."
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Input NetCDF/GRIB/Zarr path")
    ap.add_argument("--var", default=None, help="Variable name to analyze (default: first 3D time/lat/lon var)")
    ap.add_argument("--mask", default=None, help="TVT mask NetCDF (0/1/2 = land; NaN/other = ignore)")
    ap.add_argument("--chunks", default=None, help="Read chunks, e.g. 'time:auto,lat:256,lon:256' (default: file/native)")
    ap.add_argument("--workers", type=int, default=4, help="Dask worker threads (default: 4)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    chunks = parse_chunks(args.chunks)

    # modest parallelism
    dask.config.set(scheduler="threads", num_workers=args.workers)

    # Open input (xarray detects engine from suffix)
    ds = xr.open_dataset(in_path, chunks=chunks) if in_path.suffix != ".zarr" else xr.open_zarr(in_path, chunks=chunks)
    var = pick_var(ds, args.var)
    da = ds[var]
    da = rename_latlon(da)

    # normalize fill values to NaN
    fv = da.attrs.get("_FillValue", None)
    if fv is not None:
        da = da.where(da != fv)
    da = da.where(da != -999)  # common sentinel

    # Identify time dim name (default 'time')
    tdim = "time"
    if tdim not in da.dims:
        # pick any dim whose name contains 'time'
        tdim = next((d for d in da.dims if "time" in d), da.dims[0])

    # Build boolean missing mask
    isnan = xr.apply_ufunc(np.isnan, da, dask="parallelized")

    # Eligibility (land) mask, broadcast to (time,lat,lon) as boolean
    if args.mask:
        elig2d = align_mask(Path(args.mask), da)
    else:
        # all pixels eligible
        elig2d = xr.ones_like(da.isel({tdim: 0}), dtype=bool)  # (lat,lon)

    # Broadcast to 3D and AND (stay boolean; avoid .where() which introduces NaNs)
    elig3 = xr.ones_like(da, dtype=bool) & elig2d
    isnan_land = isnan & elig3

    # Counts
    # number of eligible pixels (scalar)
    n_pixels = dask.compute(elig2d.sum())[0].item()
    if n_pixels == 0:
        print("[ERROR] No eligible land pixels found (mask empty?).")
        return
    T = da.sizes[tdim]
    total_cells = n_pixels * T

    # Overall NaN fraction (over land × time)
    nan_cells = dask.compute(isnan_land.sum())[0].item()
    overall_nan_frac = nan_cells / total_cells

    # Fully-NaN timesteps (all land pixels NaN)
    fully_nan_time_bool = isnan_land.all(dim=("lat", "lon"))
    fully_nan_time_idx = dask.compute(fully_nan_time_bool)[0]
    n_full_timesteps = int(fully_nan_time_idx.sum().item())

    # Collect labels for fully-NaN timesteps (limit output)
    full_time_labels = []
    if n_full_timesteps > 0:
        tcoords = da[tdim].values
        where_idx = np.nonzero(fully_nan_time_idx.values)[0]
        # show up to first 10
        for ii in where_idx[:10]:
            full_time_labels.append(safe_time_label(tcoords[ii]))
        if n_full_timesteps > 10:
            full_time_labels.append(f"... (+{n_full_timesteps - 10} more)")

    # Fully-NaN pixels (NaN for all timesteps in land pixels)
    fully_nan_pix = isnan_land.all(dim=tdim).sum()
    n_full_pixels = int(dask.compute(fully_nan_pix)[0].item())

    # ---- Report ----
    print("\n[NaN Report]")
    print(f"File: {in_path}")
    print(f"Variable: {var}  |  shape: {tuple(da.sizes[d] for d in da.dims)}  |  dims: {da.dims}")
    print(f"Eligible land pixels: {n_pixels:,}  |  Timesteps: {T:,}")
    print(f"Overall NaN fraction (land × time): {overall_nan_frac:.3%}")
    print(f"Fully-NaN timesteps: {n_full_timesteps} / {T} ({n_full_timesteps / T:.2%})")
    if n_full_timesteps:
        print("  Examples:", ", ".join(full_time_labels))
    print(f"Fully-NaN pixels: {n_full_pixels} / {n_pixels} ({n_full_pixels / n_pixels:.2%})\n")

if __name__ == "__main__":
    main()