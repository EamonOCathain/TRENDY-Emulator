from __future__ import annotations
import zarr
from cftime import DatetimeNoLeap
import numpy as np
import xarray as xr
from numcodecs import Blosc
from typing import List, Sequence, Tuple
from pathlib import Path
import pandas as pd


def create_and_add_variable_to_zarr(path: str, n_time, n_location, variable, n_scenario=4):
    root = zarr.open_group(path, mode='a')
    shape = (n_time, n_scenario, n_location)
    chunks = (n_time, 1, 70)
    root.create_dataset(
        variable,
        shape=shape,
        chunks=chunks,
        dtype='f4',
        fill_value=np.nan,
        compressor=zarr.Blosc(cname='zstd', clevel=8)
    )
    root[variable].attrs['_ARRAY_DIMENSIONS'] = ['time', 'scenario', 'location']

def make_zarr_skeleton(
    out_path: str | Path,
    *,
    time_res: str,
    start: str,
    end: str,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    overwrite: bool = False,
    lat_chunk: int = 6,
    lon_chunk: int = 12,
):
    """
    Create a skeleton Zarr store with only coordinates (time, lat, lon).
    Later you can append variables with .to_zarr(mode='a').
    """
    out_path = Path(out_path)
    if out_path.exists():
        if overwrite:
            import shutil
            shutil.rmtree(out_path)
        else:
            raise FileExistsError(f"{out_path} exists (use overwrite=True)")

    # Defaults
    if lat is None:
        lat = np.arange(-89.75, 90.0, 0.5, dtype="float32")
    if lon is None:
        lon = np.arange(0.0, 360.0, 0.5, dtype="float32")

    # Build time axis
    time = _make_time_axis(time_res, start, end)

    # Coords-only Dataset
    ds = xr.Dataset(
        coords=dict(
            time=("time", time),
            lat=("lat", lat),
            lon=("lon", lon),
        )
    )
    ds["time"].attrs.update({
        "units": "days since 1901-01-01 00:00:00",
        "calendar": "noleap",
    })
    ds["lat"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})
    ds["lon"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})

    # Ensure parent exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Chunk encoding for coords (full time, tiled space)
    compressor = Blosc(cname="zstd", clevel=8, shuffle=Blosc.BITSHUFFLE)
    encoding = {
        "time": {"chunks": (-1,), "compressor": compressor},
        "lat": {"chunks": (lat_chunk,), "compressor": compressor},
        "lon": {"chunks": (lon_chunk,), "compressor": compressor},
    }

    ds.to_zarr(out_path, mode="w", encoding=encoding)
    # Consolidate once at the end
    zarr.consolidate_metadata(str(out_path))
    print(f"[OK] Created skeleton Zarr at {out_path} "
          f"(time={len(time)}, lat={len(lat)}, lon={len(lon)})")
    return out_path


def create_synthetic_time(time_res, start: str, end: str):
    ref_date = DatetimeNoLeap(1901, 1, 1)
    start_dt = DatetimeNoLeap(*map(int, start.split('-')))
    end_dt = DatetimeNoLeap(*map(int, end.split('-')))
    if time_res == 'annual':
        time = xr.cftime_range(start=start_dt, end=end_dt, freq='YS', calendar='noleap')
    elif time_res == 'monthly':
        time = xr.cftime_range(start=start_dt, end=end_dt, freq='MS', calendar='noleap')
    elif time_res == 'daily':
        time = xr.cftime_range(start=start_dt, end=end_dt, freq='D', calendar='noleap')
    else:
        raise ValueError(f"Unsupported time resolution: {time_res}")
    time_numeric = np.array([(t - ref_date).days for t in time], dtype='i4')
    return time_numeric

def split_locations_tvt(path, shuffle=False, seed=None):
    """
    Given a DataArray with values:
        0 = train, 1 = val, 2 = test, 3 = unused
    return (train_list, val_list, test_list),
    where each list contains (lat, lon) tuples.
    """
    rng = np.random.default_rng(seed)
    results = []
    
    mask_da  = xr.open_dataset(path, decode_times=False)["tvt_mask"]

    for code in [0, 1, 2]:
        da_sel = mask_da.where(mask_da == code, drop=True)
        lat, lon = xr.broadcast(da_sel["lat"], da_sel["lon"])
        coords = list(zip(lat.values.ravel(), lon.values.ravel()))
        if shuffle:
            rng.shuffle(coords)
        results.append(coords)

    return tuple(results)

import xarray as xr
from pathlib import Path
import zarr
import numpy as np

def netcdf_to_zarr_var(
    nc_path: str | Path,
    zarr_store: str | Path,
    var_name: str,
    overwrite: bool = False,
    time_chunk: int = -1,         # NEW
    lat_chunk: int = 6,
    lon_chunk: int = 12,
    consolidate: bool = True,
) -> None:
    nc_path = Path(nc_path)
    zarr_store = Path(zarr_store)

    root = zarr.open_group(str(zarr_store), mode="a")
    var_exists = (var_name in root)  # simpler & robust

    if var_exists and not overwrite:
        print(f"[SKIP] {var_name} already exists in {zarr_store}")
        return
    if var_exists and overwrite:
        try:
            del root[var_name]
            print(f"[INFO] Removed existing '{var_name}' from {zarr_store}")
        except Exception as e:
            print(f"[WARN] Could not remove existing '{var_name}': {e}")

    with xr.open_dataset(nc_path, decode_times=False) as ds_in:
        if var_name not in ds_in:
            raise KeyError(f"'{var_name}' not found in {nc_path}. "
                           f"Available: {list(ds_in.data_vars)}")
        da = ds_in[var_name].astype("float32")

    # Apply desired dask chunks: time = time_chunk (e.g., 1, 12, 365)
    chunks = {}
    if "time" in da.dims and time_chunk > 0:
        chunks["time"] = time_chunk
    if "lat" in da.dims:
        chunks["lat"] = lat_chunk
    if "lon" in da.dims:
        chunks["lon"] = lon_chunk
    if chunks:
        da = da.chunk(chunks)

    ds_var = da.to_dataset(name=var_name)
    ds_var.to_zarr(str(zarr_store), mode="a")

    if consolidate:
        try:
            zarr.consolidate_metadata(str(zarr_store))
        except Exception as e:
            print(f"[WARN] consolidate_metadata failed: {e}")

    print(f"[OK] Wrote {var_name} from {nc_path} into {zarr_store} "
          f"(chunks={chunks if chunks else 'unchunked'})")

def _make_time_axis(time_res: str, start: str, end: str) -> np.ndarray:
    """
    Build a noleap time axis as integer days since 1901-01-01.
    start/end as 'YYYY-MM-DD'. Matches your pipeline semantics.
    """
    import cftime
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    if time_res == "annual":
        dates = xr.cftime_range(start=start, end=end, freq="YS", calendar="noleap")
    elif time_res == "monthly":
        dates = xr.cftime_range(start=start, end=end, freq="MS", calendar="noleap")
    elif time_res == "daily":
        dates = xr.cftime_range(start=start, end=end, freq="D", calendar="noleap")
    else:
        raise ValueError("time_res must be 'annual', 'monthly', or 'daily'")
    return np.array([(d - ref).days for d in dates], dtype="i4")


def initialise_var_in_store(
    store: str | Path,
    var_name: str,
    dtype: str = "float32",
    time_chunk: int = -1,         # NEW
    lat_chunks: int = 6,
    lon_chunks: int = 12,
    consolidated: bool = True,
    overwrite: bool = False,
):
    store = Path(store)

    ds = xr.open_zarr(store, consolidated=consolidated)
    T = ds.sizes["time"]; Y = ds.sizes["lat"]; X = ds.sizes["lon"]
    ds.close()

    root = zarr.open_group(str(store), mode="a")
    dims_attr = ["time", "lat", "lon"]

    need_create = False
    if var_name in root:
        arr = root[var_name]
        if overwrite or arr.shape != (T, Y, X):
            try:
                del root[var_name]
            except Exception:
                pass
            need_create = True
        else:
            if arr.attrs.get("_ARRAY_DIMENSIONS") != dims_attr:
                arr.attrs["_ARRAY_DIMENSIONS"] = dims_attr
    else:
        need_create = True

    if need_create:
        compressor = Blosc(cname="zstd", clevel=8, shuffle=Blosc.BITSHUFFLE)
        t_chunk = time_chunk if (time_chunk and time_chunk > 0) else min(T, 365)
        arr = root.create_dataset(
            name=var_name,
            shape=(T, Y, X),
            chunks=(t_chunk, lat_chunks, lon_chunks),
            dtype=dtype,
            fill_value=np.nan,
            compressor=compressor,
            overwrite=False,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = dims_attr

    return T, Y, X

def _to_days_since_1901(time_vals):
    """
    Convert various time coordinate types to integer days since 1901-01-01.
    Supports: ints, numpy.datetime64, cftime.DatetimeNoLeap, and object arrays.
    """
    import numpy as np
    import cftime

    # Already integer-like (e.g., your per-file daily 0..364)
    if np.issubdtype(np.asarray(time_vals).dtype, np.integer):
        return np.asarray(time_vals, dtype=np.int64)

    # numpy datetime64
    if np.issubdtype(np.asarray(time_vals).dtype, np.datetime64):
        base = np.datetime64("1901-01-01")
        return (np.asarray(time_vals) - base).astype("timedelta64[D]").astype(np.int64)

    # cftime array (DatetimeNoLeap etc.) or generic Python datetimes in object dtype
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    return np.array([(t - ref).days for t in time_vals], dtype=np.int64)


def annual_netcdf_to_zarr(
    year_files,
    store,
    var_name,
    time_chunk: int = -1,        # NEW
    lat_chunks: int = 6,
    lon_chunks: int = 12,
    dtype: str = "float32",
    consolidated: bool = True,
    overwrite: bool = False,
):
    store = Path(store)
    year_files = sorted(map(Path, year_files))

    # Ensure/repair target var with desired chunking
    T, Y, X = initialise_var_in_store(
        store=store,
        var_name=var_name,
        dtype=dtype,
        time_chunk=time_chunk,          # NEW
        lat_chunks=lat_chunks,
        lon_chunks=lon_chunks,
        consolidated=consolidated,
        overwrite=overwrite,
    )

    # Target time axis map
    tgt = xr.open_zarr(store, consolidated=consolidated)
    time_target_days = _to_days_since_1901(tgt["time"].values)
    tgt.close()
    if not np.all(np.diff(time_target_days) > 0):
        raise ValueError("Target Zarr time is not strictly increasing.")
    value_to_idx = {int(v): i for i, v in enumerate(time_target_days)}

    for fp in year_files:
        ds = xr.open_dataset(fp, decode_times=False, chunks="auto")
        if var_name not in ds:
            ds.close(); raise KeyError(f"'{var_name}' not in {fp}. Available: {list(ds.data_vars)}")
        if "time" not in ds.dims:
            ds.close(); raise ValueError(f"{fp} has no 'time' dimension.")

        da = ds[var_name].astype(dtype)

        # Use the same chunks here so the write aligns (not mandatory, but efficient)
        chunks = {"time": -1}  # write whole year at once
        if "lat" in da.dims: chunks["lat"] = lat_chunks
        if "lon" in da.dims: chunks["lon"] = lon_chunks
        da = da.chunk(chunks)

        # Payload without coords
        ds_payload = xr.Dataset({var_name: (da.dims, da.data)})

        tvals = _to_days_since_1901(ds["time"].values)
        if tvals.ndim != 1 or tvals.size == 0:
            ds.close(); raise ValueError(f"{fp}: unexpected time axis shape {tvals.shape}")
        # We *expect* a contiguous year in the file; if not true, you can relax this
        if not np.all(np.diff(tvals) == 1):
            ds.close(); raise ValueError(f"{fp}: daily time not consecutive by 1 day.")

        try:
            i0 = value_to_idx[int(tvals[0])]
            i1 = value_to_idx[int(tvals[-1])] + 1
        except KeyError as e:
            ds.close()
            raise ValueError(
                f"{fp}: time value {int(e.args[0])} not present in target Zarr. "
                "Units/calendar mismatch or wrong skeleton time axis."
            )
        if not np.array_equal(tvals, time_target_days[i0:i1]):
            ds.close()
            raise ValueError(
                f"{fp}: file time values don’t match target slice [{i0}:{i1}). "
                "Rebuild skeleton with correct daily noleap axis if needed."
            )

        ds_payload.to_zarr(store, mode="a", region={"time": slice(i0, i1)})
        ds.close()
        print(f"[OK] wrote {fp.name} -> [{i0}:{i1})")

def make_training_skeleton(
    out_path: Path,
    *,
    time: np.ndarray,             
    scenario_labels=(0,1,2,3),
    lat: np.ndarray,
    lon: np.ndarray,
    overwrite=False,
):
    """
    Create coords-only skeleton for training tensors:
      coords: time, scenario, location, lat(location), lon(location)
    No data variables are created here; call create_and_add_variable_to_zarr afterwards.
    """
    out_path = Path(out_path)
    if out_path.exists():
        if not overwrite:
            print(f"[SKIP skeleton] {out_path} exists")
            return out_path
        shutil.rmtree(out_path)

    nloc = len(lat)
    assert nloc == len(lon), "lat/lon length mismatch"

    ds_coords = xr.Dataset(
        coords=dict(
            time=("time", np.asarray(time)),
            scenario=("scenario", np.asarray(scenario_labels, dtype="i4")),
            location=("location", np.arange(nloc, dtype="i4")),
            lat=("location", np.asarray(lat, dtype="float32")),
            lon=("location", np.asarray(lon, dtype="float32")),
        )
    )
    ds_coords["time"].attrs.update({"units": "days since 1901-01-01 00:00:00", "calendar": "noleap"})

    # light compression on coords; no need to chunk scenario/location unless huge
    enc = {
        "time":     {"compressor": Blosc(cname="zstd", clevel=8)},
        "scenario": {"compressor": Blosc(cname="zstd", clevel=8)},
        "location": {"compressor": Blosc(cname="zstd", clevel=8)},
        "lat":      {"compressor": Blosc(cname="zstd", clevel=8)},
        "lon":      {"compressor": Blosc(cname="zstd", clevel=8)},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_coords.to_zarr(out_path, mode="w", encoding=enc)
    zarr.consolidate_metadata(str(out_path))
    print(f"[OK] training skeleton at {out_path}")
    return out_path


def _detect_data_var_name(store: Path) -> str:
    """
    For a per-variable store, return the data variable name.
    Assumes coords are named 'time','lat','lon'.
    """
    root = zarr.open_group(str(store), mode="r")
    # Prefer anything that's not a coord (works for per-var stores)
    for k in root.array_keys():
        if k not in ("time", "lat", "lon"):
            return k
    # Fallback: if only one array exists, use it
    keys = root.array_keys()
    if len(keys) == 1:
        return keys[0]
    raise RuntimeError(f"Could not detect data var name in {store}; arrays={list(keys)}")


def repeat_first_year_to_end(
    store: str | Path,
    var_name: str | None = None,
    *,
    days_per_year: int = 365,
    consolidated: bool = False,
) -> None:
    """
    After the first year is written, copy it forward to fill the rest of the time axis.
    Works in-place; copies one year at a time to keep memory bounded.
    Assumes noleap calendar (365-day years) by default.
    """
    store = Path(store)

    with xr.open_zarr(store, consolidated=consolidated) as ds:
        T = int(ds.sizes["time"])

    root = zarr.open_group(str(store), mode="a")
    if var_name is None:
        var_name = _detect_data_var_name(store)
    arr = root[var_name]

    if arr.shape[0] < days_per_year:
        raise RuntimeError(
            f"First year not fully written in {store} ({var_name}): "
            f"time={arr.shape[0]} < {days_per_year}"
        )

    total_years = (T + days_per_year - 1) // days_per_year  # ceil
    for year_idx in range(1, total_years):  # 0 == first year already present
        dst0 = year_idx * days_per_year
        dst1 = min(dst0 + days_per_year, T)
        n = dst1 - dst0
        if n <= 0:
            continue
        # copy from first year [0:n] into [dst0:dst1]
        arr[dst0:dst1, :, :] = arr[0:n, :, :]
        # optional: print could be verbose, leave to caller if needed

    # ensure xarray dims attr so opens cleanly
    if arr.attrs.get("_ARRAY_DIMENSIONS") != ["time", "lat", "lon"]:
        arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]


def repeat_first_n_years_to_end(
    store: str | Path,
    var_name: str | None = None,
    *,
    n_years: int = 20,
    days_per_year: int = 365,
    consolidated: bool = False,
) -> None:
    """
    After the first `n_years` are written, copy that block forward to fill the rest of time.
    Works in-place; copies one year at a time to keep memory bounded.
    Assumes noleap calendar (365-day years) by default.
    """
    store = Path(store)
    block_days = n_years * days_per_year

    with xr.open_zarr(store, consolidated=consolidated) as ds:
        T = int(ds.sizes["time"])

    root = zarr.open_group(str(store), mode="a")
    if var_name is None:
        var_name = _detect_data_var_name(store)
    arr = root[var_name]

    if arr.shape[0] < block_days:
        raise RuntimeError(
            f"First {n_years} years not fully written in {store} ({var_name}): "
            f"time={arr.shape[0]} < {block_days}"
        )

    total_years = (T + days_per_year - 1) // days_per_year  # ceil

    # Copy year-by-year using modulo within the first n_years
    for y_idx in range(n_years, total_years):
        dst0 = y_idx * days_per_year
        dst1 = min(dst0 + days_per_year, T)
        n = dst1 - dst0
        if n <= 0:
            continue

        src_idx = y_idx % n_years  # 0..n_years-1
        src0 = src_idx * days_per_year
        src1 = src0 + n

        arr[dst0:dst1, :, :] = arr[src0:src1, :, :]

    if arr.attrs.get("_ARRAY_DIMENSIONS") != ["time", "lat", "lon"]:
        arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]
        
