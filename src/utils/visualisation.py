from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import math
from typing import List

# ---------- Helpers ----------

def open_single_data_var(path: str | Path) -> tuple[xr.Dataset, str]:
    """
    Open dataset (decode_times=False) and return (ds, varname) where varname is the
    single non-coordinate data variable. Caller must ds.close().
    """
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    exclude = {"time", "lat", "lon", "time_bnds", "bnds", "time_bounds", "lat_bnds", "lon_bnds"}
    vars_ok = [v for v in ds.data_vars if v not in exclude and not v.endswith("_bnds")]
    if len(vars_ok) == 0:
        ds.close()
        raise ValueError(f"No plottable data variables found in {path}")
    if len(vars_ok) != 1:
        ds.close()
        raise ValueError(f"Expected exactly one data var, found {len(vars_ok)} in {path}: {vars_ok}")
    return ds, vars_ok[0]

def resolve_output_file(
    output_dir: str | Path | None,
    output_path: str | Path | None,
    default_name: str,
    overwrite: bool = True,
) -> Path:
    """
    Decide final output file path from (output_path or output_dir/default_name).
    Enforces: exactly one of output_path or output_dir must be provided.
    Creates parent dir if needed. Respects overwrite flag.
    """
    if (output_dir is None and output_path is None) or (output_dir is not None and output_path is not None):
        raise ValueError("Specify exactly one of output_dir or output_path.")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / default_name

    if out.exists() and not overwrite:
        # Returning the existing path allows caller to log/skip cleanly
        return out
    return out

def normalize_target_lon(lon_da, lon=10.0):
    """Return lon in the dataset's convention (0..360 or -180..180)."""
    lon_min = float(lon_da.min())
    lon_max = float(lon_da.max())
    if lon_max > 180:      # dataset likely 0..360
        return lon % 360.0
    if lon > 180:          # dataset likely -180..180
        return lon - 360.0
    return lon

# ---------- Plotting Functions ----------

def first_timestep(
    path: str | Path,
    output_dir: str | Path = None,
    output_path: str | Path = None,
    title: str = None,
    overwrite: bool = True,
    axis_label: str = None,
    varname: str = None
) -> Path:
    """Plots the first timestep of a netcdf. If out_path is provided it saves to that, or if out_dir is provided
    it saves the variable name in the directory. You must provide one or the other, but not both."""
    if varname is not None:
        ds = xr.open_dataset(path)
        da = ds[varname]
    else:
        ds, varname = open_single_data_var(path)
    try:
        arr = ds[varname]
        arr2d = arr.isel(time=0) if "time" in arr.dims else arr

        fname = resolve_output_file(output_dir, output_path, f"{varname}.png", overwrite=overwrite)
        if fname.exists() and not overwrite:
            print(f"[SKIP] File exists and overwrite=False: {fname}")
            return fname

        plt.figure(figsize=(12, 5))
        ax = plt.axes(projection=ccrs.PlateCarree()); ax.set_global()
        arr2d.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cbar_kwargs={"label": axis_label if axis_label else f"{varname}"}
        )
        ax.coastlines(); ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.set_title((title + ": ") if title else "" + ("First Timestep" if "time" in arr.dims else "Static Field"))
        plt.tight_layout(); plt.savefig(fname, dpi=300); plt.close()
        print(f"[OK] Saved plot: {fname}")
        return fname
    finally:
        ds.close()

def finite_mask(
    path: str | Path,
    output_dir: str | Path = None,
    output_path: str | Path = None,
    title: str = None,
    overwrite: bool = True,
    ntimesteps: int | None = None,
    varname: str = None
) -> Path:
    """Plots the finite mask of netcdf. If out_path is provided it saves to that, or if out_dir is provided
    it saves the variable name in the directory. You must provide one or the other, but not both.
    If n_timesteps is provided it only plots the firs n timesteps."""
    if varname is not None:
        ds = xr.open_dataset(path)
        da = ds[varname]
    else:
        ds, varname = open_single_data_var(path)
    try:
        arr = ds[varname]
        if "time" in arr.dims:
            if ntimesteps is not None:
                arr = arr.isel(time=slice(0, ntimesteps))
                print(f"[INFO] Using first {ntimesteps} timesteps for finite mask")
            mask = xr.where(np.isfinite(arr).all(dim="time"), 1, 0)
        else:
            mask = xr.where(np.isfinite(arr), 1, 0)

        fname = resolve_output_file(output_dir, output_path, f"{varname}.png", overwrite=overwrite)
        if fname.exists() and not overwrite:
            print(f"[SKIP] Exists: {fname}")
            return fname

        plt.figure(figsize=(12, 5))
        ax = plt.axes(projection=ccrs.PlateCarree()); ax.set_global()
        mask.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="binary", vmin=0, vmax=1,
                  cbar_kwargs={"label": "Finite Mask (1=finite, 0=NaN)"})
        ax.coastlines(); ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.set_title((title + ": ") if title else "" + "Finite Mask")
        plt.tight_layout(); plt.savefig(fname, dpi=300); plt.close()
        print(f"[OK] Saved finite mask plot: {fname}")
        return fname
    finally:
        ds.close()

def plot_timeseries(
    path: str | Path,
    output_dir: str | Path = None,
    output_path: str | Path = None,
    title: str = None,
    lat: float = 51.0,
    lon: float = 10.0,
    overwrite: bool = True,
) -> Path | None:
    """Plots a timeseries of a netcdf (standard in central germany). If out_path is provided it saves to that, or if out_dir is provided
    it saves the variable name in the directory. You must provide one or the other, but not both."""
    # Find var name without decoding; then reopen WITHOUT decoding to avoid cftime on the x-axis
    _, varname = open_single_data_var(path)  # this opens a ds—let it be GC'd quickly
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)  # <-- changed
    try:
        lon_tgt = normalize_target_lon(ds["lon"], lon)
        da = ds[varname].sel(lat=lat, lon=lon_tgt, method="nearest")

        if "time" not in da.dims:
            print(f"[WARN] Variable '{varname}' has no time dimension in {path}, skipping timeseries.")
            return None

        fname = resolve_output_file(output_dir, output_path, f"{varname}_timeseries.png", overwrite=overwrite)
        if fname.exists() and not overwrite:
            print(f"[SKIP] Exists: {fname}")
            return fname

        # convert numeric "days since 1901-01-01" to datetime64 for a nice axis
        t_num = ds["time"].values  # numeric days (noleap)
        t_dt64 = (np.datetime64("1901-01-01") + t_num.astype("timedelta64[D]"))

        plt.figure(figsize=(12, 4))
        plt.plot(t_dt64, da.values)  # <-- now plotting datetime64, not cftime
        plt.xlabel("Time"); plt.ylabel(varname)
        plt.title(title or f"{varname} time series @ (lat≈{lat}, lon≈{lon})")
        plt.tight_layout(); plt.savefig(fname, dpi=300); plt.close()
        print(f"[OK] Saved time series: {fname}")
        return fname
    finally:
        ds.close()

def plot_mean_seasonal_cycle(
    path: str | Path,
    output_dir: str | Path = None,
    output_path: str | Path = None,
    time_res: str = "monthly",
    title: str = None,
    lat: float = 51.0,
    lon: float = 10.0,
    overwrite: bool = True,
) -> Path | None:
    """Plots the mean seasonal cycle of a netcdf (at standard location in central germany). If out_path is provided it saves to that, or if out_dir is provided
    it saves the variable name in the directory. You must provide one or the other, but not both."""
    if time_res not in {"monthly", "daily"}:
        raise ValueError("time_res must be 'monthly' or 'daily' (not 'annual').")

    ds, varname = open_single_data_var(path)  # open_single_data_var should already open with decode_times=False
    try:
        lon_tgt = normalize_target_lon(ds["lon"], lon)
        da = ds[varname].sel(lat=lat, lon=lon_tgt, method="nearest")

        if "time" not in da.dims:
            print(f"[WARN] Variable '{varname}' has no time dimension in {path}, skipping seasonal cycle.")
            return None

        tlen = int(da.sizes["time"])
        if time_res == "monthly":
            month_index = xr.DataArray(((np.arange(tlen) % 12) + 1).astype(np.int16), dims=("time",), name="month")
            clim = da.groupby(month_index).mean("time")
            x = np.arange(1, 13); y = clim.values; xlab = "Month"
            suffix = "monthly"
        else:
            doy_index = xr.DataArray(((np.arange(tlen) % 365) + 1).astype(np.int16), dims=("time",), name="doy")
            clim = da.groupby(doy_index).mean("time")
            x = np.arange(1, 366); y = clim.values; xlab = "Day of Year"
            suffix = "daily"

        fname = resolve_output_file(output_dir, output_path, f"{varname}_seasonal_cycle_{suffix}.png", overwrite=overwrite)
        if fname.exists() and not overwrite:
            print(f"[SKIP] Exists: {fname}")
            return fname

        plt.figure(figsize=(12, 4))
        plt.plot(x, y); plt.xlabel(xlab); plt.ylabel(varname)
        default_title = f"{varname} mean seasonal cycle ({suffix}) @ (lat≈{lat}, lon≈{lon})"
        plt.title(title or default_title)
        plt.tight_layout(); plt.savefig(fname, dpi=300); plt.close()
        print(f"[OK] Saved seasonal cycle: {fname}")
        return fname
    finally:
        ds.close()

def plot_mean_seasonal_cycle_grid(
    paths: list[str | Path],
    output_path: str | Path,
    time_res: str = "monthly",
    lat: float = 51.0,
    lon: float = 10.0,
    titles: list[str] | None = None,  # kept for API compatibility; not used
    overwrite: bool = True,
) -> Path | None:
    """
    Plot mean seasonal cycles for multiple NetCDF files in a single figure with 2 columns.
    Subplot titles are '<MODEL> – True' if model in models_with_monthly_states, else
    '<MODEL> – Reconstructed'. Model is parsed from file name 'model_scenario_var.nc'.
    """
    import math
    from pathlib import Path

    if time_res not in {"monthly", "daily"}:
        raise ValueError("time_res must be 'monthly' or 'daily'.")

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"[SKIP] Exists: {output_path}")
        return output_path

    n = len(paths)
    if n == 0:
        print("[WARN] No paths provided.")
        return None

    models_with_monthly_states = {"ELM", "VISIT", "VISIT-UT"}

    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)

    for i, p in enumerate(paths):
        ax = axes[i // ncols][i % ncols]
        p = Path(p)

        # derive model from filename safely: model_scenario_var.nc
        # allow extra underscores in 'var' by limiting splits to 2
        try:
            model, scenario, _ = p.stem.split("_", 2)
        except ValueError:
            model = p.stem.split("_", 1)[0]  # fallback

        # open without decoding times to keep consistency with your helpers
        ds = xr.open_dataset(p, engine="netcdf4", decode_times=False)
        try:
            _, varname = open_single_data_var(p)
            lon_tgt = normalize_target_lon(ds["lon"], lon)
            da = ds[varname].sel(lat=lat, lon=lon_tgt, method="nearest")

            if "time" not in da.dims:
                ax.set_visible(False)
                print(f"[WARN] '{varname}' has no time dim in {p.name}, skipping panel.")
                continue

            tlen = int(da.sizes["time"])
            if time_res == "monthly":
                month_index = xr.DataArray(((np.arange(tlen) % 12) + 1).astype(np.int16),
                                           dims=("time",), name="month")
                clim = da.groupby(month_index).mean("time")
                x = np.arange(1, 13)
                y = clim.values
                xlab = "Month"
            else:
                doy_index = xr.DataArray(((np.arange(tlen) % 365) + 1).astype(np.int16),
                                         dims=("time",), name="doy")
                clim = da.groupby(doy_index).mean("time")
                x = np.arange(1, 366)
                y = clim.values
                xlab = "Day of Year"

            ax.plot(x, y)
            ax.set_xlabel(xlab)
            ax.set_ylabel(varname)

            # Title: "<MODEL> – True" or "<MODEL> – Reconstructed"
            label = "True" if model in models_with_monthly_states else "Reconstructed"
            ax.set_title(f"{model} – {label}")
        finally:
            ds.close()

    # hide any leftover empty axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved seasonal-cycle grid: {output_path}")
    return output_path