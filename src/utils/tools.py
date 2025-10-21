import os
from pathlib import Path
from typing import Dict, List, Union, Sequence, Union, Tuple
import numpy as np
import xarray as xr
import re
import numpy as np
import pandas as pd

def _open_ds(path:str)-> Tuple[xr.Dataset, str]:
    """Opens a dataset and searches for a variable which is not time, lat, lon, bnds etc. 
    Returns the ds and the variable name"""
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    exclude = {"time", "lat", "lon", "time_bnds", "bnds", "time_bounds",
               "lat_bnds", "lon_bnds", "lat_bounds", "lon_bounds"}
    varnames = [v for v in ds.data_vars if v not in exclude and not v.endswith("_bnds")]
    if len(varnames) == 0:
        ds.close()
        raise ValueError(f"No plottable data variables found in {path}")
    if len(varnames) != 1:
        ds.close()
        raise ValueError(f"Expected exactly one data var, found {len(varnames)} in {path}: {varnames}")
    return ds, varnames[0]

def slurm_shard(items):
    """
    Return the slice of `items` this SLURM array task should process.

    Defaults:
      - Round-robin slicing: items[tid::n_tasks]
      - If not running under SLURM, returns the full list.

    Env overrides:
      - ONLY_INDEX: if set (int), return just that single item (or row) and exit.
      - SHARD_COUNT: force the total number of shards (useful when submitting a single array index).
      - SHARD_ID: force the shard index (0-based), overriding SLURM_ARRAY_TASK_ID.
      - SHARD_STRATEGY: "rr" (round-robin, default) or "block" (contiguous chunk).
    """
    import os
    try:
        import pandas as pd  # optional; only needed if items is a DataFrame
    except Exception:
        pd = None

    # --- ONLY_INDEX: take exactly one item and exit ---
    only = os.getenv("ONLY_INDEX")
    if only is not None:
        idx = int(only)
        if isinstance(items, list) or isinstance(items, tuple):
            shard = items[idx:idx+1]
        elif hasattr(items, "iloc") and pd is not None:
            shard = items.iloc[[idx]]
        else:
            shard = items[idx:idx+1]
        print(f"[INFO] ONLY_INDEX={idx} -> {len(shard)} item(s)")
        return shard

    # --- Read SLURM vars ---
    tid_str  = os.getenv("SLURM_ARRAY_TASK_ID")
    tmin_str = os.getenv("SLURM_ARRAY_TASK_MIN")
    tmax_str = os.getenv("SLURM_ARRAY_TASK_MAX")

    # If not under SLURM and no overrides, process all
    if tid_str is None and os.getenv("SHARD_ID") is None:
        print("[INFO] No SLURM array vars; processing all items.")
        return items

    # Determine shard count
    if os.getenv("SHARD_COUNT") is not None:
        n_tasks = int(os.getenv("SHARD_COUNT"))
    elif tmin_str is not None and tmax_str is not None:
        n_tasks = int(tmax_str) - int(tmin_str) + 1
    else:
        # SLURM sets this to the number of elements in the --array spec.
        n_tasks = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

    # Determine shard id (0-based)
    if os.getenv("SHARD_ID") is not None:
        tid = int(os.getenv("SHARD_ID"))
    else:
        tid = int(tid_str or "0")

    if not (0 <= tid < max(1, n_tasks)):
        raise IndexError(f"Shard id {tid} out of range for n_tasks={n_tasks}")

    strategy = os.getenv("SHARD_STRATEGY", "rr").lower()  # "rr" or "block"

    # Slice
    if strategy in ("block", "contiguous"):
        # contiguous block partitioning
        import math
        n = len(items) if not hasattr(items, "shape") else (items.shape[0] if len(items.shape) > 0 else len(items))
        per = math.ceil(n / n_tasks)
        start = tid * per
        end = min(start + per, n)
        if hasattr(items, "iloc") and pd is not None:
            shard = items.iloc[start:end]
        else:
            shard = items[start:end]
    else:
        # round-robin (default)
        if hasattr(items, "iloc") and pd is not None:
            shard = items.iloc[tid::n_tasks]
        else:
            shard = items[tid::n_tasks]

    # Friendly log
    size = len(shard) if not hasattr(shard, "shape") else shard.shape[0]
    print(f"[INFO] SLURM shard: task {tid}/{n_tasks} ({strategy}) -> {size} items")
    return shard

def sanity_check(paths: Sequence[Union[str, Path]]) -> None:
    """
    For each NetCDF in `paths`, assert:
      - ONLY dimensions == {'time','lat','lon'}
      - ONLY coordinates == {'time','lat','lon'}
      - time length in {123, 1476, 44895}
      - time units like 'days since 1901-01-01[ <time>]' (00, 00:00, 00:00:00 allowed)
      - calendar in {'noleap', '365_day'}
      - 720 longitudes ending .0 or .5
      - 360 latitudes  ending .25 or .75

    Prints a neat report ONLY for files that fail at least one check.
    """
    expected_set = {"time", "lat", "lon"}
    expected_time = {123, 1476, 44895}
    any_printed = False

    # Accept:
    #   'days since 1901-01-01'
    #   'days since 1901-01-01 00'
    #   'days since 1901-01-01 00:00'
    #   'days since 1901-01-01 00:00:00'
    #   (also allow 'T' between date and time)
    _units_re = re.compile(
        r"""^\s*days\s+since\s+1901-01-01(?:[ T]00(?::00(?::00)?)?)?\s*$""",
        re.IGNORECASE,
    )
    _ok_calendars = {"noleap", "365_day"}

    def _frac(v):
        v = np.asarray(v, dtype=float)
        return np.mod(v, 1.0)

    for p in map(Path, paths):
        issues = []
        try:
            with xr.open_dataset(p, engine="netcdf4", decode_times=False) as ds:
                # --- dims & coords
                dims = set(ds.dims)
                coords = set(ds.coords)

                if dims != expected_set:
                    issues.append(f"• Dims mismatch: found {sorted(dims)}; expected {sorted(expected_set)}")
                if coords != expected_set:
                    issues.append(f"• Coords mismatch: found {sorted(coords)}; expected {sorted(expected_set)}")

                # --- time length
                tlen = int(ds.sizes.get("time", 0))
                if tlen not in expected_time:
                    issues.append(f"• Time length {tlen} not in {sorted(expected_time)}")

                # --- time units / calendar
                if "time" not in ds.coords:
                    issues.append("• Missing 'time' coordinate")
                else:
                    tunits = str(ds["time"].attrs.get("units", "")).strip()
                    tcal   = str(ds["time"].attrs.get("calendar", "")).strip().lower()

                    if not _units_re.match(tunits):
                        issues.append(f"• Bad time units: {tunits!r} (expected like 'days since 1901-01-01[ 00[:00[:00]]]')")

                    if tcal not in _ok_calendars:
                        issues.append(f"• Bad calendar: {tcal!r} (expected one of {sorted(_ok_calendars)})")

                # --- grid checks
                nlon = int(ds.sizes.get("lon", 0))
                nlat = int(ds.sizes.get("lat", 0))

                # lon length
                if nlon != 720:
                    issues.append(f"• lon size {nlon} (expected 720)")
                # lon endings
                if "lon" in ds:
                    lon_frac = _frac(ds["lon"].values)
                    lon_ok = np.all(
                        np.isclose(lon_frac, 0.0, atol=1e-9) |
                        np.isclose(lon_frac, 0.5, atol=1e-9)
                    )
                    if not lon_ok:
                        bad_idx = np.where(~(
                            np.isclose(lon_frac, 0.0, atol=1e-9) |
                            np.isclose(lon_frac, 0.5, atol=1e-9)
                        ))[0]
                        sample = ", ".join([f"{float(ds['lon'].values[i]):.3f}" for i in bad_idx[:5]])
                        more = "" if len(bad_idx) <= 5 else f" (+{len(bad_idx)-5} more)"
                        issues.append(f"• lon values not ending .0/.5 (e.g., {sample}){more}")
                else:
                    issues.append("• Missing 'lon' coordinate")

                # lat length
                if nlat != 360:
                    issues.append(f"• lat size {nlat} (expected 360)")
                # lat endings
                if "lat" in ds:
                    lat_frac = _frac(ds["lat"].values)
                    lat_ok = np.all(
                        np.isclose(lat_frac, 0.25, atol=1e-9) |
                        np.isclose(lat_frac, 0.75, atol=1e-9)
                    )
                    if not lat_ok:
                        bad_idx = np.where(~(
                            np.isclose(lat_frac, 0.25, atol=1e-9) |
                            np.isclose(lat_frac, 0.75, atol=1e-9)
                        ))[0]
                        sample = ", ".join([f"{float(ds['lat'].values[i]):.3f}" for i in bad_idx[:5]])
                        more = "" if len(bad_idx) <= 5 else f" (+{len(bad_idx)-5} more)"
                        issues.append(f"• lat values not ending .25/.75 (e.g., {sample}){more}")
                else:
                    issues.append("• Missing 'lat' coordinate")

                # data variables snapshot (helpful context if anything failed)
                if issues:
                    issues.append(f"• Data variables: {list(ds.data_vars)}")

        except Exception as e:
            issues = [f"• Failed to open/inspect file: {e!r}"]

        if issues:
            any_printed = True
            print(f"\n[FAIL] {p.name}")
            for msg in issues:
                print(f"    {msg}")

def finite_mask(
    in_path: str | Path,
    out_path: str | Path,
    overwrite: bool = False,
    n_timesteps: int | None = None
) -> Path:
    """
    Create a binary mask from a NetCDF file where:
      - 1 = all values finite (across time if present)
      - 0 = otherwise
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        print(f"[INFO] Mask exists, skipping: {out_path}")
        return out_path

    ds = xr.open_dataset(in_path, decode_times=False)

    # Pick the first data variable
    var = list(ds.data_vars)[0]
    arr = ds[var]

    # Optionally restrict to first n_timesteps
    if "time" in arr.dims:
        if n_timesteps is not None:
            arr = arr.isel(time=slice(0, n_timesteps))
            print(f"[INFO] Using first {n_timesteps} timesteps for mask")
        mask = xr.where(np.isfinite(arr).all(dim="time"), 1, 0)
    else:
        mask = xr.where(np.isfinite(arr), 1, 0)

    mask.name = f"{var}_finite_mask"

    # Save
    mask.to_netcdf(out_path, engine="netcdf4", format="NETCDF4")
    ds.close()
    print(f"[OK] Wrote finite mask to {out_path}")
    return out_path

def threshold_mask(
    input_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.1,
    overwrite: bool = True,
) -> Path:
    """
    Create a binary mask from a static NetCDF file:
      - 0 where values < threshold
      - 1 where values >= threshold

    Uses `_open_ds` to auto-detect the single valid data variable.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        print(f"[SKIP] Threshold mask exists: {output_path}")
        return output_path

    ds, var = _open_ds(input_path)  # auto-detects variable
    try:
        data = ds[var]
        mask = (data >= threshold).astype("int32")
        ds_mask = xr.Dataset({f"{var}_mask": mask}, coords={"lat": ds["lat"], "lon": ds["lon"]})
        ds_mask.to_netcdf(output_path, engine="netcdf4", format="NETCDF4")
    finally:
        ds.close()

    print(f"[OK] Saved threshold mask to {output_path}")
    return output_path

def combine_masks(
    mask_files: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    overwrite: bool = True,
    new_var_name: str = "combined_mask",
) -> Path:
    """
    Combine multiple mask NetCDFs into a single (lat, lon) mask where:
      - Output pixel = 1 only if ALL input masks are 1 at that pixel
      - Otherwise 0 (i.e., if ANY mask has 0 or NaN at that pixel)
    The output grid (lat/lon coordinates) is taken from the first file.

    Assumptions:
      - Each input mask has exactly one data variable (detected with _open_ds).
      - If a time dimension exists, require ALL timesteps = 1.
      - NaNs are treated as 0.
      - All input masks must already share exactly the same lat/lon grid.
    """
    mask_files = [Path(p) for p in mask_files]
    if not mask_files:
        raise ValueError("No mask files provided.")

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"[SKIP] Combined mask exists: {output_path}")
        return output_path

    # --- Template from first file ---
    ds0, var0 = _open_ds(mask_files[0])
    try:
        arr0 = ds0[var0]
        if "time" in arr0.dims:
            m0 = (arr0.fillna(0) >= 0.5).all("time")
        else:
            m0 = (arr0.fillna(0) >= 0.5)
        template = m0.squeeze(drop=True)
        lat0 = ds0["lat"].values
        lon0 = ds0["lon"].values
    finally:
        ds0.close()

    combined = template

    # --- Loop over the rest ---
    for f in mask_files[1:]:
        ds, var = _open_ds(f)
        try:
            arr = ds[var]
            if "time" in arr.dims:
                m = (arr.fillna(0) >= 0.5).all("time")
            else:
                m = (arr.fillna(0) >= 0.5)
            m = m.squeeze(drop=True)

            # --- Grid consistency check ---
            if not (np.array_equal(ds["lat"].values, lat0) and np.array_equal(ds["lon"].values, lon0)):
                raise ValueError(f"Grid mismatch in {f}: lat/lon do not match template grid")

            combined = combined & m
        finally:
            ds.close()

    # --- Wrap up ---
    final_mask = combined.astype("int32").rename(new_var_name)
    final_ds = xr.Dataset(
        {new_var_name: final_mask},
        coords={"lat": final_mask["lat"], "lon": final_mask["lon"]},
    )
    final_ds.to_netcdf(output_path, engine="netcdf4", format="NETCDF4")
    print(f"[OK] Wrote combined mask to {output_path} with variable '{new_var_name}'")
    return output_path

