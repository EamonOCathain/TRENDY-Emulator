#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import xarray as xr

"""
Seasonally-aware LAI gap-fill (two-nearest years per month at pixel, then 3x3 spatial mean),
restricted to TVT land mask (0/1/2). Oceans (NaN in mask) are never filled or used as donors.

Reads NetCDF, writes Zarr. Parallelizable with Dask threads.

Reporting printed for the chosen time window over land:
- overall NaN fraction
- fully-NaN months
- fully-NaN pixels
"""

# ----------------- Helpers -----------------
def years_from_time(time: xr.DataArray) -> xr.DataArray:
    """Return int16 years for a (noleap) time axis that might be numeric or cftime."""
    try:
        return time.dt.year.astype("int16")
    except Exception:
        from cftime import num2date
        if np.issubdtype(time.dtype, np.number):
            dt = num2date(time.values, units="days since 1901-01-01", calendar="noleap")
        else:
            dt = time.values
        years = np.array([d.year for d in dt], dtype=np.int16)
        return xr.DataArray(years, dims=("time",))

def _fill_two_nearest_1d(ts: np.ndarray, years: np.ndarray) -> np.ndarray:
    """
    Fill a 1D timeseries (single pixel for a single calendar month) using
    the mean of up to two nearest non-NaN years (by |year - target_year|).
    """
    out = ts.copy()
    T = out.shape[0]
    for i in range(T):
        if np.isnan(out[i]):
            y0 = years[i]
            order = np.argsort(np.abs(years - y0))
            order = order[order != i]  # exclude self
            cnt = 0
            acc = 0.0
            for j in order:
                v = out[j]
                if not np.isnan(v):
                    acc += v
                    cnt += 1
                    if cnt == 2:
                        break
            if cnt > 0:
                out[i] = acc / cnt
    return out

def temporal_two_nearest_fill(da: xr.DataArray) -> xr.DataArray:
    """Per-month, per-pixel two-nearest-years fill. Dask-parallel over space."""
    years_all = years_from_time(da["time"]).compute()  # small (1D)
    da_stacked = da.stack(space=("lat", "lon"))
    filled_parts = []
    for m in range(1, 13):
        sub = da_stacked.where(da_stacked["time.month"] == m, drop=True)  # (Tm, space)
        if sub.sizes.get("time", 0) == 0:
            continue
        yrs_m = years_all.where(da["time.month"] == m, drop=True)
        sub_filled = xr.apply_ufunc(
            _fill_two_nearest_1d,
            sub,
            yrs_m,
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[sub.dtype],
        )
        filled_parts.append(sub_filled)
    if not filled_parts:
        return da
    filled_stacked = xr.concat(filled_parts, dim="time").sortby("time")
    filled = filled_stacked.unstack("space")
    return filled.sel(time=da.time)

def _spatial_fill_2d(arr2d: np.ndarray, elig2d: np.ndarray, k: int) -> np.ndarray:
    """NaN-aware k×k local-mean fill for a single 2D field, restricted to eligible land."""
    from scipy.ndimage import uniform_filter
    arr = arr2d.copy()
    arr[~elig2d] = np.nan
    vals = np.nan_to_num(arr, copy=False)         # NaNs->0, we'll divide by counts
    cnts = np.isfinite(arr).astype(np.float32)
    mean_vals = uniform_filter(vals, size=k, mode="nearest")
    mean_cnts = uniform_filter(cnts, size=k, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        neigh_mean = mean_vals / mean_cnts
    out = arr2d.copy()
    fillable = elig2d & np.isnan(out) & (mean_cnts > 0)
    out[fillable] = neigh_mean[fillable]
    return out

def spatial_fill(da: xr.DataArray, elig: xr.DataArray, k: int = 3) -> xr.DataArray:
    """Apply _spatial_fill_2d per time slice."""
    return xr.apply_ufunc(
        _spatial_fill_2d,
        da,
        elig,
        xr.DataArray(k),
        input_core_dims=[["lat", "lon"], ["lat", "lon"], []],
        output_core_dims=[["lat", "lon"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
    )

def parse_chunks(spec: str):
    """
    Parse 'time:240,lat:256,lon:256' -> ({'time': 240, 'lat': 256, 'lon': 256}, auto_flag)
    Use 'auto' for a dim to request automatic chunking on READ; if any dim is 'auto',
    we will pass chunks='auto' into xarray.open_dataset.
    """
    out = {}
    auto = False
    for kv in spec.split(","):
        if not kv.strip():
            continue
        kdim, kval = kv.split(":")
        kdim = kdim.strip()
        kval = kval.strip().lower()
        if kval == "auto":
            auto = True
        else:
            out[kdim] = int(kval)
    return out, auto

# ----------------- Main script -----------------
def main():
    p = argparse.ArgumentParser(description="Seasonally-aware LAI gap-fill with TVT mask, write Zarr.")
    p.add_argument("--in", dest="in_path", required=True,
                   help="Input NetCDF with LAI (e.g., .../lai_avh15c1.nc)")
    p.add_argument("--out", dest="out_path", required=True,
                   help="Output Zarr store (directory)")
    p.add_argument("--var", default="avh15c1_lai", help="Variable name in input (default: avh15c1_lai)")
    p.add_argument("--mask", required=True,
                   help="TVT mask NetCDF (values 0/1/2 => eligible; NaN => ocean)")
    p.add_argument("--in-chunks", default="time:240,lat:256,lon:256",
                   help="Dask read chunks, e.g. 'time:240,lat:256,lon:256' (use 'auto' for a dim to auto-chunk)")
    p.add_argument("--out-chunks", default="time:240,lat:256,lon:256",
                   help="Zarr write chunks for the output variable, e.g. 'time:-1,lat:1,lon:1'")
    p.add_argument("--kernel", type=int, default=3, help="Spatial kernel size (odd, default 3)")
    p.add_argument("--year-min", type=int, default=None, help="Optional minimum year to fill (inclusive)")
    p.add_argument("--year-max", type=int, default=None, help="Optional maximum year to fill (inclusive)")
    p.add_argument("--workers", type=int, default=4, help="Dask worker threads (LocalCluster not needed)")
    p.add_argument("--compressor", default="zstd",
                   choices=["zstd", "lz4", "blosclz", "lz4hc", "snappy", "zlib", "none"],
                   help="Zarr compressor (default: zstd)")
    p.add_argument("--clevel", type=int, default=1, help="Compressor level (default: 1)")
    args = p.parse_args()

    in_path  = Path(args.in_path)
    out_path = Path(args.out_path)
    mask_path = Path(args.mask)
    var = args.var
    k = args.kernel

    in_chunks_dict, in_chunks_auto   = parse_chunks(args.in_chunks)
    out_chunks_dict, _out_chunks_auto = parse_chunks(args.out_chunks)  # only ints matter for write

    # Dask config (threaded)
    import dask
    dask.config.set(scheduler="threads", num_workers=args.workers)

    print(f"[INFO] Opening LAI: {in_path}")
    if in_chunks_auto:
        ds = xr.open_dataset(in_path, chunks="auto")
    else:
        ds = xr.open_dataset(in_path, chunks=in_chunks_dict)

    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in {in_path}. Have: {list(ds.data_vars)}")
    da = ds[var]
    # Normalize fill values to NaN if present
    fillv = da.attrs.get("_FillValue", None)
    if fillv is not None:
        da = da.where(da != fillv)
    da = da.where(da != -999)

    # Load mask
    print(f"[INFO] Opening TVT mask: {mask_path}")
    m = xr.open_dataset(mask_path).to_array()[0]
    lat_dim = next(d for d in ("lat", "latitude", "y") if d in m.dims)
    lon_dim = next(d for d in ("lon", "longitude", "x") if d in m.dims)
    m = m.rename({lat_dim: "lat", lon_dim: "lon"})
    if not (m.lat.identical(da.lat) and m.lon.identical(da.lon)):
        m = m.sel(lat=da.lat, lon=da.lon, method="nearest")

    # Eligible land: values 0,1,2
    elig = m.isin([0, 1, 2])

    # Apply eligibility: oceans remain NaN and do not contribute in later passes
    da_masked = da.where(elig)

    # Optional time window restriction
    years_all = years_from_time(da_masked.time)
    if args.year_min is not None or args.year_max is not None:
        y0 = args.year_min if args.year_min is not None else int(years_all.min())
        y1 = args.year_max if args.year_max is not None else int(years_all.max())
        time_fill = (years_all >= y0) & (years_all <= y1)
        window_label = f"{y0}–{y1}"
    else:
        time_fill = xr.full_like(years_all, True, dtype=bool)
        window_label = f"{int(years_all.min())}–{int(years_all.max())}"

    da_inside  = da_masked.where(time_fill, drop=True)
    da_outside = da_masked.where(~time_fill, drop=True)

    # -------- Pass 1: temporal two-nearest (per month, per pixel) --------
    print("[INFO] Temporal pass (two nearest years per month)...")
    da_temp_filled = temporal_two_nearest_fill(da_inside)

    # Make lat/lon single chunks for the spatial pass (time stays chunked)
    da_temp_filled = da_temp_filled.chunk({"lat": -1, "lon": -1})
    elig = elig.chunk({"lat": -1, "lon": -1})

    # -------- Pass 2: spatial 3x3 per timestep (eligible land only) -------
    print("[INFO] Spatial pass (3x3 neighbor mean within same timestep)...")
    da_spatial_filled = spatial_fill(da_temp_filled, elig, k=k)

    # Re-assemble full time axis
    parts = []
    if da_outside.sizes.get("time", 0) > 0:
        parts.append(da_outside)
    parts.append(da_spatial_filled)

    da_filled_all = xr.concat(parts, dim="time").sortby("time")
    da_filled_all = da_filled_all.where(elig)  # keep oceans NaN

    # Clip to specified year range if requested (so the Zarr is actually trimmed)
    if args.year_min is not None or args.year_max is not None:
        da_filled_all = da_filled_all.sel(time=time_fill)

    # ----------------- Reporting (land-only, within window) -----------------
    years_all2 = years_from_time(da_filled_all.time)
    if args.year_min is not None or args.year_max is not None:
        y0 = args.year_min if args.year_min is not None else int(years_all2.min())
        y1 = args.year_max if args.year_max is not None else int(years_all2.max())
        time_sel = (years_all2 >= y0) & (years_all2 <= y1)
    else:
        time_sel = xr.full_like(years_all2, True, dtype=bool)

    da_win = da_filled_all.where(time_sel, drop=True).where(elig)
    months_total = int(da_win.sizes.get("time", 0))
    land_pixels_total = int(elig.sum().values)
    total_cells = months_total * land_pixels_total if months_total and land_pixels_total else 0

    if total_cells > 0:
        nan_cells = int(np.isnan(da_win).sum().compute() if hasattr(da_win.data, "compute") else np.isnan(da_win).sum().values)
        overall_nan_frac = nan_cells / total_cells
    else:
        nan_cells = 0
        overall_nan_frac = np.nan

    allnan_month = np.isnan(da_win).all(dim=("lat", "lon"))
    fully_nan_months = int(allnan_month.sum().compute() if hasattr(allnan_month.data, "compute") else allnan_month.sum().values)

    stacked = da_win.stack(space=("lat", "lon"))
    land_mask_stacked = elig.stack(space=("lat", "lon"))
    if land_pixels_total > 0 and months_total > 0:
        sel_idx = np.flatnonzero(land_mask_stacked.values)
        if sel_idx.size > 0:
            stacked_land = stacked.isel(space=sel_idx)
            allnan_pixel = np.isnan(stacked_land).all(dim="time")
            fully_nan_pixels = int(allnan_pixel.sum().compute() if hasattr(allnan_pixel.data, "compute") else allnan_pixel.sum().values)
        else:
            fully_nan_pixels = 0
    else:
        fully_nan_pixels = 0

    print(f"\n[REPORT] Window: {window_label}  |  Land pixels: {land_pixels_total:,}  |  Months: {months_total}")
    print(f"         Overall NaN fraction: {overall_nan_frac:.3%}")
    print(f"         Fully-NaN months:     {fully_nan_months} / {months_total} ({(fully_nan_months / months_total) if months_total else 0:.2%})")
    print(f"         Fully-NaN pixels:     {fully_nan_pixels} / {land_pixels_total} ({(fully_nan_pixels / land_pixels_total) if land_pixels_total else 0:.2%})\n")

    # ----------------- Write Zarr -----------------
    # Prepare compression (optional)
    compressor = None
    if args.compressor != "none":
        from numcodecs import Blosc
        compressor = Blosc(cname=args.compressor, clevel=args.clevel, shuffle=Blosc.SHUFFLE)

    # Ensure output chunking on the variable; only keep integer chunk sizes
    out_chunks_clean = {k: v for k, v in out_chunks_dict.items() if isinstance(v, int)}
    out_ds = da_filled_all.to_dataset(name=var)
    if out_chunks_clean:
        out_ds = out_ds.chunk(out_chunks_clean)

    encoding = {var: {"compressor": compressor}}

    print(f"[INFO] Writing Zarr (consolidated) → {out_path}")
    out_ds.to_zarr(
        store=str(out_path),
        mode="w",
        encoding=encoding,
        consolidated=True,
        compute=True,
    )
    print("[OK] Done.")

if __name__ == "__main__":
    main()