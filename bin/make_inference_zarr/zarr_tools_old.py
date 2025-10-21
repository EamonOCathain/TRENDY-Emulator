from __future__ import annotations
import zarr
from cftime import DatetimeNoLeap
import numpy as np
import xarray as xr
from numcodecs import Blosc
from typing import List
from pathlib import Path
import cftime
from typing import Iterable, Sequence
import shutil
import xarray as xr
import zarr

# ==================== Helpers ======================

def _ensure_written_mask(store: str | Path, var_name: str, T: int) -> zarr.core.Array:
    """
    Ensure a 1D boolean array '__written__{var}' of length T exists in the group.
    Returns the zarr array handle.
    """
    store = Path(store)
    root = zarr.open_group(str(store), mode="a")
    key = f"__written__{var_name}"
    if key in root:
        arr = root[key]
        # repair attrs if needed
        if "_ARRAY_DIMENSIONS" not in arr.attrs:
            arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        return arr

    arr = root.create_dataset(
        name=key,
        shape=(T,),
        chunks=(min(T, 4096),),
        dtype="|b1",        # boolean
        fill_value=False,
        overwrite=False,
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]
    return arr

def _get_time_length(store: str | Path, consolidated: bool = True) -> int:
    with xr.open_zarr(store, consolidated=consolidated) as ds:
        return int(ds.sizes["time"])

def _mark_written(store: str | Path, var_name: str, i0: int, i1: int, consolidated: bool = True):
    T = _get_time_length(store, consolidated=consolidated)
    mask = _ensure_written_mask(store, var_name, T)
    mask[i0:i1] = True  # zarr write

def _slice_is_written(store: str | Path, var_name: str, i0: int, i1: int, consolidated: bool = True) -> bool:
    T = _get_time_length(store, consolidated=consolidated)
    mask = _ensure_written_mask(store, var_name, T)
    view = mask[i0:i1]
    # Fast path: if any False -> not fully written
    return bool(view.all())

def set_complete(store: str | Path, var_name: str, consolidated: bool = True):
    """Mark a variable as fully written."""
    store = Path(store)
    root = zarr.open_group(str(store), mode="a")
    root.attrs[f"complete:{var_name}"] = True

def is_complete(store: str | Path, var_name: str) -> bool:
    store = Path(store)
    root = zarr.open_group(str(store), mode="r")
    return bool(root.attrs.get(f"complete:{var_name}", False))

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

# ==================== Make Inference Zarr ======================

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

def netcdf_to_zarr_var(
    nc_path: str | Path,
    zarr_store: str | Path,
    var_name: str,
    overwrite: bool = False,
    lat_chunk: int = 6,
    lon_chunk: int = 12,
    time_chunk: int | None = None,   # <-- will be honored below
    consolidate: bool = False,
) -> None:
    import xarray as xr
    import zarr

    nc_path = Path(nc_path)
    zarr_store = Path(zarr_store)
    zarr_store.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(zarr_store), mode="a")

    # If already marked complete and not overwriting -> skip fast
    if not overwrite and bool(root.attrs.get(f"complete:{var_name}", False)):
        print(f"[SKIP] {zarr_store.name}:{var_name} marked complete", flush=True)
        return

    # If overwriting, drop the variable and clear flag
    if overwrite and var_name in root:
        del root[var_name]
        if f"complete:{var_name}" in root.attrs:
            del root.attrs[f"complete:{var_name}"]
        print(f"[INFO] Removed existing '{var_name}' from {zarr_store.name}", flush=True)

    print(f"[START] write {zarr_store.name}:{var_name} from {nc_path.name}", flush=True)

    with xr.open_dataset(nc_path, decode_times=False) as ds_in:
        if var_name not in ds_in.data_vars:
            raise KeyError(f"'{var_name}' not found in {nc_path}. Available: {list(ds_in.data_vars)}")

        da = ds_in[var_name].astype("float32")

        # Build chunk mapping (ONLY set dims that exist; honor provided time_chunk)
        chunks: dict[str, int] = {}
        if "time" in da.dims and time_chunk is not None:
            chunks["time"] = int(time_chunk)
        if "lat" in da.dims and lat_chunk is not None:
            chunks["lat"] = int(lat_chunk)
        if "lon" in da.dims and lon_chunk is not None:
            chunks["lon"] = int(lon_chunk)
        if chunks:
            da = da.chunk(chunks)

        # Write
        ds_var = da.to_dataset(name=var_name)
        mode = "w" if overwrite else "a"
        ds_var.to_zarr(str(zarr_store), mode=mode, consolidated=consolidate)

    # Mark complete only after success
    root.attrs[f"complete:{var_name}"] = True

    if consolidate:
        try:
            zarr.consolidate_metadata(str(zarr_store))
        except Exception as e:
            print(f"[WARN] consolidate_metadata failed: {e}", flush=True)

    print(f"[DONE] {zarr_store.name}:{var_name}", flush=True)

def initialise_var_in_store(
    store: str | Path,
    var_name: str,
    dtype: str = "float32",
    lat_chunks: int = 6,
    lon_chunks: int = 12,
    consolidated: bool = True,
):
    store = Path(store)

    # Read coord sizes
    ds = xr.open_zarr(store, consolidated=consolidated)
    T = int(ds.sizes["time"])
    Y = int(ds.sizes["lat"])
    X = int(ds.sizes["lon"])
    ds.close()

    root = zarr.open_group(str(store), mode="a")
    if var_name in root:
        # Ensure the attr exists; repair if needed
        arr = root[var_name]
        if "_ARRAY_DIMENSIONS" not in arr.attrs:
            arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]
        return

    compressor = Blosc(cname="zstd", clevel=8, shuffle=Blosc.BITSHUFFLE)

    arr = root.create_dataset(
        name=var_name,
        shape=(T, Y, X),
        chunks=(min(T, 365), lat_chunks, lon_chunks),
        dtype=dtype,
        fill_value=np.nan,
        compressor=compressor,
        overwrite=False,
    )
    # ➜ critical for xarray:
    arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]

    # Optional but nice for speed:
    try:
        zarr.consolidate_metadata(str(store))
    except Exception:
        pass

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
    lat_chunks: int = 6,
    lon_chunks: int = 12,
    dtype: str = "float32",
    consolidated: bool = True,
    overwrite: bool = False,
    mark_complete_when_done: bool = True,
    time_chunks: int | None = None,   # <-- NEW: honor requested time chunking
):
    import xarray as xr
    import numpy as np
    import zarr

    store = Path(store)
    year_files = sorted(map(Path, year_files))

    # Ensure target var exists (shape/time/chunks) — your existing helper
    initialise_var_in_store(
        store=store, var_name=var_name,
        lat_chunks=lat_chunks, lon_chunks=lon_chunks,
        consolidated=consolidated
    )

    # Optional: skip the whole var if previously completed and overwrite=False
    if not overwrite and is_complete(store, var_name):
        print(f"[SKIP] {store.name}:{var_name} already complete")
        return

    # Normalize target time to integer days since 1901-01-01
    with xr.open_zarr(store, consolidated=consolidated) as tgt:
        time_target_days = _to_days_since_1901(tgt["time"].values)
    if not np.all(np.diff(time_target_days) > 0):
        raise ValueError("Target Zarr time is not strictly increasing.")
    value_to_idx = {int(v): i for i, v in enumerate(time_target_days)}

    total_written = 0

    for fp in year_files:
        with xr.open_dataset(fp, decode_times=False) as ds:
            if var_name not in ds:
                raise KeyError(f"'{var_name}' not in {fp}. Available: {list(ds.data_vars)}")
            if "time" not in ds.dims:
                raise ValueError(f"{fp} has no 'time' dimension.")

            da = ds[var_name].astype(dtype)

            # Build chunk mapping (use requested time_chunks if provided)
            chunks: dict[str, int] = {}
            if "time" in da.dims and time_chunks is not None:
                chunks["time"] = int(time_chunks)
            if "lat" in da.dims and lat_chunks is not None:
                chunks["lat"] = int(lat_chunks)
            if "lon" in da.dims and lon_chunks is not None:
                chunks["lon"] = int(lon_chunks)
            if chunks:
                da = da.chunk(chunks)

            tvals = _to_days_since_1901(ds["time"].values)
            if tvals.ndim != 1 or tvals.size == 0:
                raise ValueError(f"{fp}: unexpected time axis shape {tvals.shape}")
            if not np.all(np.diff(tvals) == 1):
                raise ValueError(f"{fp}: daily time not consecutive by 1 day.")

            try:
                i0 = value_to_idx[int(tvals[0])]
                i1 = value_to_idx[int(tvals[-1])] + 1
            except KeyError as e:
                raise ValueError(
                    f"{fp}: time value {int(e.args[0])} not present in target Zarr. "
                    "Units/calendar mismatch or wrong skeleton time axis."
                )

            if not np.array_equal(tvals, time_target_days[i0:i1]):
                raise ValueError(
                    f"{fp}: file time values don’t match target slice [{i0}:{i1}). "
                    "Rebuild skeleton with correct daily noleap axis if needed."
                )

            # Skip already-written slices when overwrite=False
            if not overwrite and _slice_is_written(store, var_name, i0, i1, consolidated=consolidated):
                print(f"[SKIP] {fp.name} -> [{i0}:{i1}) already written")
                continue

            ds_payload = xr.Dataset({var_name: (da.dims, da.data)})
            ds_payload.to_zarr(store, mode="a", region={"time": slice(i0, i1)})
            _mark_written(store, var_name, i0, i1, consolidated=consolidated)

            total_written += (i1 - i0)
            print(f"[OK] wrote {fp.name} -> [{i0}:{i1})")

    if mark_complete_when_done:
        # Consider a write "complete" only if all timesteps are written
        T = _get_time_length(store, consolidated=consolidated)
        mask = _ensure_written_mask(store, var_name, T)
        if bool(mask[:].all()):
            set_complete(store, var_name)
            print(f"[OK] {store.name}:{var_name} marked complete")

    # Optional consolidate
    try:
        zarr.consolidate_metadata(str(store))
    except Exception:
        pass

    print(f"[DONE] {store.name}:{var_name}, new timesteps written: {total_written}")


# ================== TRAINING ZARR FUNCTIONS =======================

# ----------------------------- Time helpers -----------------------------

def make_time_axis_days_since_1901(time_res: str, start: str, end: str) -> np.ndarray:
    """
    Return integer time coordinate = days since 1901-01-01 (noleap), inclusive.
    """
    import cftime
    ref = cftime.DatetimeNoLeap(1901, 1, 1)

    if time_res == "daily":
        dates = xr.cftime_range(start=start, end=end, freq="D",  calendar="noleap")
    elif time_res == "monthly":
        dates = xr.cftime_range(start=start, end=end, freq="MS", calendar="noleap")
    elif time_res == "annual":
        dates = xr.cftime_range(start=start, end=end, freq="YS", calendar="noleap")
    else:
        raise ValueError("time_res must be one of: daily | monthly | annual")

    return np.asarray([(d - ref).days for d in dates], dtype="int32")


def slice_for_period(time_days: np.ndarray, start: str, end: str) -> slice:
    """
    Given a 1D int32 time axis (days since 1901-01-01 noleap),
    return a Python slice [i0:i1) covering [start, end] inclusive.
    """
    target = make_time_axis_days_since_1901(
        time_res=("daily" if len(time_days) > 4000 else "annual"),  # not used; we just need endpoints below
        start=start, end=end
    )
    # align by searching the first and last values present in the target axis
    i0 = int(np.searchsorted(time_days, target[0], side="left"))
    i1 = int(np.searchsorted(time_days, target[-1], side="right"))
    return slice(i0, i1)


# ----------------------------- Mask helpers -----------------------------

def build_indices_from_mask(
    mask_nc: Path | str,
    code: int,
    *,
    shuffle: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Load a 2D mask (lat, lon) with integer codes; return 1D *flat* indices
    (row-major) where mask==code. Optionally shuffle.
    """
    rng = np.random.default_rng(seed)
    with xr.open_dataset(mask_nc, decode_times=False) as ds:
        # assume variable is named 'tvt_mask'; adjust if needed
        varname = next((v for v in ds.data_vars if ds[v].ndim == 2), None)
        if varname is None:
            raise ValueError("Mask file must contain a 2D variable (lat, lon).")
        mask = ds[varname].values  # (lat, lon)
    flat = np.flatnonzero(mask.ravel(order="C") == code).astype(np.int64)
    if shuffle:
        rng.shuffle(flat)
    return flat


def lat_lon_1d_from_grid(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given 1D lat, lon vectors for a rectilinear grid, return stacked 1D arrays
    aligned with np.ravel(order='C'): location index k corresponds to
    (i = k // lon.size, j = k % lon.size).
    """
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    return lat2d.ravel(order="C"), lon2d.ravel(order="C")


# --------------------------- Target skeleton ----------------------------

def make_tensor_skeleton(
    out_store: Path | str,
    *,
    time_days: np.ndarray,
    lat_all: np.ndarray,
    lon_all: np.ndarray,
    location_index: np.ndarray,
    scenario_labels: Sequence[str] = ("S0", "S1", "S2", "S3"),
    chunks: tuple[int, int, int] = (-1, 1, 70),
    overwrite: bool = False,
) -> Path:
    """
    Create a coords-only Zarr store with dims (time, scenario, location) and
    coords time/lat/lon aligned to location order given by 'location_index'.
    """
    out_store = Path(out_store)
    if out_store.exists():
        if not overwrite:
            print(f"[SKIP] skeleton exists: {out_store}")
            return out_store
        import shutil; shutil.rmtree(out_store)

    # Build 1D lat/lon for the *selected* locations, in selected order
    lat_flat, lon_flat = lat_lon_1d_from_grid(lat_all, lon_all)
    lat_sel = lat_flat[location_index].astype("float32")
    lon_sel = lon_flat[location_index].astype("float32")

    ds = xr.Dataset(
        coords=dict(
            time=("time", time_days.astype("int32")),
            scenario=("scenario", np.arange(len(scenario_labels), dtype="int32")),
            location=("location", np.arange(location_index.size, dtype="int32")),
            lat=("location", lat_sel),
            lon=("location", lon_sel),
        )
    )
    ds["time"].attrs.update({
        "units": "days since 1901-01-01 00:00:00",
        "calendar": "noleap",
        "standard_name": "time",
        "axis": "T",
    })
    ds["scenario"].attrs.update({
        "labels": list(scenario_labels),
    })
    compressor = Blosc(cname="zstd", clevel=8, shuffle=Blosc.BITSHUFFLE)
    enc = {
        "time": {"compressor": compressor, "chunks": (chunks[0],)},
        "scenario": {"compressor": compressor, "chunks": (chunks[1],)},
        "location": {"compressor": compressor, "chunks": (chunks[2],)},
        "lat": {"compressor": compressor, "chunks": (chunks[2],)},
        "lon": {"compressor": compressor, "chunks": (chunks[2],)},
    }

    out_store.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(out_store, mode="w", encoding=enc)
    zarr.consolidate_metadata(str(out_store))
    return out_store


def ensure_variable_in_store(
    store: Path | str,
    var_name: str,
    dtype: str = "float32",
    chunks: tuple[int, int, int] = (-1, 1, 70),
) -> None:
    """
    Ensure a (time, scenario, location) array exists in the destination store,
    matching the store's coord sizes. If missing, create it with NaNs and set
    xarray's required _ARRAY_DIMENSIONS attribute.
    """
    store = Path(store)

    # Read sizes from coords in the tensor store
    with xr.open_zarr(store, consolidated=True) as ds:
        T = int(ds.sizes["time"])
        S = int(ds.sizes["scenario"])
        L = int(ds.sizes["location"])

    root = zarr.open_group(str(store), mode="a")
    if var_name in root:
        arr = root[var_name]
        if "_ARRAY_DIMENSIONS" not in arr.attrs:
            arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "scenario", "location"]
        return

    from numcodecs import Blosc
    compressor = Blosc(cname="zstd", clevel=8, shuffle=Blosc.BITSHUFFLE)

    # Resolve -1 placeholders against actual sizes
    c_time = T if chunks[0] == -1 else chunks[0]
    c_scen = 1 if chunks[1] == -1 else chunks[1]
    c_loc  = min(70, L) if chunks[2] == -1 else chunks[2]

    arr = root.create_dataset(
        name=var_name,
        shape=(T, S, L),
        chunks=(c_time, c_scen, c_loc),
        dtype=dtype,
        fill_value=np.nan,
        compressor=compressor,
        overwrite=False,
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "scenario", "location"]


# ------------------------------ Copy logic ------------------------------
def _clear_progress_for_var(store: Path | str, var_name: str):
    root = zarr.open_group(str(store), mode="a")
    # delete complete flag if present
    if f"complete:{var_name}" in root.attrs:
        del root.attrs[f"complete:{var_name}"]
    # delete loc-written mask array if present
    key = f"__written_loc__{var_name}"
    if key in root:
        del root[key]

# --------------------- Overwrite Logic Helpers ---------------------

def _dst_sizes(store: Path | str):
    with xr.open_zarr(store, consolidated=True, decode_times=False) as ds:
        return int(ds.sizes["time"]), int(ds.sizes["scenario"]), int(ds.sizes["location"])

def _set_complete(store: Path | str, var_name: str):
    root = zarr.open_group(str(store), mode="a")
    root.attrs[f"complete:{var_name}"] = True

def _is_complete(store: Path | str, var_name: str) -> bool:
    root = zarr.open_group(str(store), mode="a")
    return bool(root.attrs.get(f"complete:{var_name}", False))

# ---------- Per-scenario progress helpers (treat scenarios independently) ----------

def _mask_key(var_name: str, scen_idx: int) -> str:
    # one mask per scenario, easy to delete once done
    return f"__written_loc__{var_name}:scen{scen_idx}"

def _ensure_loc_written_mask_scen(store: Path | str, var_name: str, scen_idx: int) -> zarr.core.Array:
    """Ensure a 1-D boolean mask over location exists for this (var, scenario)."""
    store = Path(store)
    root = zarr.open_group(str(store), mode="a")
    key = _mask_key(var_name, scen_idx)
    if key in root:
        arr = root[key]
        if "_ARRAY_DIMENSIONS" not in arr.attrs:
            arr.attrs["_ARRAY_DIMENSIONS"] = ["location"]
        return arr

    # infer L from destination store
    with xr.open_zarr(store, consolidated=True, decode_times=False) as ds:
        L = int(ds.sizes["location"])

    comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    arr = root.create_dataset(
        name=key,
        shape=(L,),
        chunks=(min(4096, L),),
        dtype="|b1",
        fill_value=False,
        compressor=comp,
        overwrite=False,
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["location"]
    return arr

def _mark_loc_written_scen(store: Path | str, var_name: str, scen_idx: int, start: int, stop: int):
    mask = _ensure_loc_written_mask_scen(store, var_name, scen_idx)
    mask[start:stop] = True

def _loc_slice_is_written_scen(store: Path | str, var_name: str, scen_idx: int, start: int, stop: int) -> bool:
    mask = _ensure_loc_written_mask_scen(store, var_name, scen_idx)
    return bool(mask[start:stop].all())

def _is_scenario_complete(store: Path | str, var_name: str, scen_idx: int) -> bool:
    root = zarr.open_group(str(store), mode="a")
    return bool(root.attrs.get(f"complete:{var_name}:scen{scen_idx}", False))

def _set_scenario_complete(store: Path | str, var_name: str, scen_idx: int):
    root = zarr.open_group(str(store), mode="a")
    root.attrs[f"complete:{var_name}:scen{scen_idx}"] = True

def _cleanup_mask_scen(store: Path | str, var_name: str, scen_idx: int):
    """Delete the per-scenario mask array for this var/scenario, if present."""
    root = zarr.open_group(str(store), mode="a")
    key = _mask_key(var_name, scen_idx)
    if key in root:
        del root[key]

def _is_complete_global(store: Path | str, var_name: str) -> bool:
    root = zarr.open_group(str(store), mode="a")
    return bool(root.attrs.get(f"complete:{var_name}", False))

def _set_complete_global_if_all_scenarios_done(store: Path | str, var_name: str):
    """If all scenarios have per-scenario complete flags, set the global complete flag."""
    with xr.open_zarr(store, consolidated=True, decode_times=False) as ds:
        S = int(ds.sizes["scenario"])
    root = zarr.open_group(str(store), mode="a")
    for s in range(S):
        if not root.attrs.get(f"complete:{var_name}:scen{s}", False):
            return  # some scenario still incomplete
    root.attrs[f"complete:{var_name}"] = True  # now globally complete

def _clear_progress_for_var_scen(store: Path | str, var_name: str, scen_idx: int):
    """Clear ONLY this scenario’s progress (mask + per-scenario flag)."""
    root = zarr.open_group(str(store), mode="a")
    # clear per-scenario flag
    key_attr = f"complete:{var_name}:scen{scen_idx}"
    if key_attr in root.attrs:
        del root.attrs[key_attr]
    # remove per-scenario mask
    _cleanup_mask_scen(store, var_name, scen_idx)
    
def all_vars_complete(store: str | Path, var_names) -> bool:
    """
    True if *global* complete flag is set for every variable in var_names.
    (We already set global complete when all scenarios are done.)
    """
    store = Path(store)
    root = zarr.open_group(str(store), mode="r")
    return all(bool(root.attrs.get(f"complete:{v}", False)) for v in var_names)

def copy_variables_from_source(
    src_store: Path | str,
    dst_store: Path | str,
    *,
    vars_keep,
    location_index: np.ndarray,
    time_slice_src: slice,
    scenario_index: int,
    location_block: int = 70,
    overwrite_data: bool = False,
    verbose: bool = True,
):
    with xr.open_zarr(dst_store, consolidated=True, decode_times=False) as ds_dst:
        T_dst = int(ds_dst.sizes["time"])

    with xr.open_zarr(src_store, consolidated=True, decode_times=False) as ds_src:
        ds_src = ds_src.isel(time=time_slice_src)

        # Ensure (time, lat, lon) ordering for robustness before stacking
        expected = ("time", "lat", "lon")
        for v in vars_keep:
            if v in ds_src.data_vars:
                ds_src[v] = ds_src[v].transpose(*[d for d in expected if d in ds_src[v].dims])

        time_len = int(ds_src.sizes["time"])
        if time_len != T_dst:
            raise ValueError(f"time length mismatch dst={T_dst} vs src-slice={time_len}")

        ds_stacked = ds_src.stack(location=("lat", "lon"))  # (time, location)
        loc_len = location_index.size

        from src.utils.zarr_tools import ensure_variable_in_store as _ensure_var

        for var in vars_keep:
            if var not in ds_stacked.data_vars:
                if verbose: print(f"[WARN] {Path(src_store).name} missing {var}", flush=True)
                continue

            # Overwrite behavior: clear ONLY this scenario’s progress if requested
            if overwrite_data:
                _clear_progress_for_var_scen(dst_store, var, scenario_index)

            # Fast skip if this scenario is already complete (and not overwriting)
            if (not overwrite_data) and _is_scenario_complete(dst_store, var, scenario_index):
                if verbose: print(f"[SKIP] {var} scenario {scenario_index} already complete", flush=True)
                continue

            # If global complete is set (legacy or previous run), skip too
            if (not overwrite_data) and _is_complete_global(dst_store, var):
                if verbose: print(f"[SKIP] {var} globally complete", flush=True)
                continue

            if verbose:
                nblocks = int(np.ceil(loc_len / location_block))
                print(f"[START] {var} scen={scenario_index} blocks={nblocks} block_size={location_block} -> {Path(dst_store).name}",
                      flush=True)

            _ensure_var(dst_store, var)

            # Copy in aligned location blocks
            blocks_total = int(np.ceil(loc_len / location_block))
            for bi, start in enumerate(range(0, loc_len, location_block), start=1):
                stop = min(start + location_block, loc_len)

                if (not overwrite_data) and _loc_slice_is_written_scen(dst_store, var, scenario_index, start, stop):
                    if verbose and (bi % 20 == 0 or bi == blocks_total):
                        print(f"[PROG] {var} scen={scenario_index}: skipped {bi}/{blocks_total}", flush=True)
                    continue

                src_locs = location_index[start:stop]
                da_blk = ds_stacked[var].isel(location=xr.DataArray(src_locs, dims="location"))
                da_blk = da_blk.astype("float32").chunk({"time": -1, "location": -1})

                ds_payload = xr.Dataset({var: (("time", "scenario", "location"),
                                               da_blk.data[:, None, :])})
                region = {"time": slice(0, time_len),
                          "scenario": slice(scenario_index, scenario_index + 1),
                          "location": slice(start, stop)}
                ds_payload.to_zarr(dst_store, mode="a", region=region)
                _mark_loc_written_scen(dst_store, var, scenario_index, start, stop)

                if verbose and (bi % 20 == 0 or bi == blocks_total):
                    print(f"[PROG] {var} scen={scenario_index}: wrote {bi}/{blocks_total}", flush=True)

            # Scenario-level completion + cleanup
            # mask is per-scenario; if all True -> set per-scenario flag and delete the mask
            m = _ensure_loc_written_mask_scen(dst_store, var, scenario_index)
            if bool(m[:].all()):
                _set_scenario_complete(dst_store, var, scenario_index)
                _cleanup_mask_scen(dst_store, var, scenario_index)

            # Optional: set global complete if all scenario flags done
            _set_complete_global_if_all_scenarios_done(dst_store, var)

            # Consolidate metadata to keep loads snappy
            try:
                zarr.consolidate_metadata(str(dst_store))
            except Exception:
                pass

            if verbose:
                done_scen = _is_scenario_complete(dst_store, var, scenario_index)
                done_global = _is_complete_global(dst_store, var)
                print(f"[DONE] {var} scen={scenario_index} -> scenario_complete={done_scen} global_complete={done_global}",
                      flush=True)


# Rechunking Functions
def rechunk_location_store(
    store: str | Path,
    target_loc_chunk: int = 70,
    *,
    tmp_suffix: str = ".tmp_rechunk",
    consolidated: bool = True,
    verbose: bool = True,
) -> None:
    """
    Rechunk a (time, scenario, location) Zarr so that *location* uses `target_loc_chunk`.
    Leaves other dims as: time=-1 (full), scenario=1.
    Only applies explicit chunk encodings to variables with dims exactly ('time','scenario','location').
    """
    store = Path(store)
    tmp = store.with_name(store.name + tmp_suffix)
    if tmp.exists():
        shutil.rmtree(tmp)
        
    with xr.open_zarr(store, consolidated=True) as ds:
        # pick any 3D var
        v = next((k for k, da in ds.data_vars.items() if tuple(da.dims)==("time","scenario","location")), None)
        if v and ds[v].chunks and ds[v].chunks[-1] == (target_loc_chunk,):
            if verbose: print("[RECHUNK] already at target location chunk; skipping")
            return

    # Open lazily; no dask chunking on read
    ds = xr.open_zarr(store, consolidated=consolidated, chunks={})

    # Dataset-level rechunk (guidance for writing)
    ds_rechunked = ds.chunk({"time": -1, "scenario": 1, "location": target_loc_chunk})

    # Encoding: coords and the 3D tensors
    enc = {}
    enc["time"] = {"chunks": (int(ds.sizes["time"]),)}   # full time
    enc["scenario"] = {"chunks": (1,)}
    enc["location"] = {"chunks": (target_loc_chunk,)}
    if "lat" in ds.coords: enc["lat"] = {"chunks": (target_loc_chunk,)}
    if "lon" in ds.coords: enc["lon"] = {"chunks": (target_loc_chunk,)}

    for v, da in ds.data_vars.items():
        if tuple(da.dims) == ("time", "scenario", "location"):
            enc[v] = {"chunks": (int(ds.sizes["time"]), 1, target_loc_chunk)}

    if verbose:
        print(f"[RECHUNK] → {store.name} (location={target_loc_chunk}) -> {tmp}")

    ds_rechunked.to_zarr(tmp, mode="w", encoding=enc)
    try:
        zarr.consolidate_metadata(str(tmp))
    except Exception:
        pass

    # Atomic-ish swap
    bak = store.with_name(store.name + ".bak_rechunk")
    try:
        if bak.exists(): shutil.rmtree(bak)
        store.rename(bak)
        tmp.rename(store)
        shutil.rmtree(bak)
        if verbose: print(f"[RECHUNK] swapped in {store.name} with location={target_loc_chunk}")
    except Exception as e:
        if verbose:
            print(f"[RECHUNK][WARN] swap failed: {e}")
            print(f"[RECHUNK] temp left at: {tmp}")

def rechunk_latlon_store(
    store: str | Path,
    target_lat_chunk: int = 5,
    target_lon_chunk: int = 5,
    *,
    target_time_chunk: int | None = None,   # None -> leave as-is; -1 -> full time
    tmp_suffix: str = ".tmp_rechunk",
    consolidated: bool = True,
    verbose: bool = True,
) -> None:
    """
    Rechunk a (time, lat, lon) Zarr so that lat/lon use the provided chunk sizes.
    Only applies explicit chunk encodings to variables with dims exactly ('time','lat','lon').
    """
    store = Path(store)
    tmp = store.with_name(store.name + tmp_suffix)
    if tmp.exists():
        shutil.rmtree(tmp)

    ds = xr.open_zarr(store, consolidated=consolidated, chunks={})

    # Build dataset-level chunk map
    chunk_map = {"lat": target_lat_chunk, "lon": target_lon_chunk}
    if target_time_chunk is not None:
        chunk_map["time"] = (int(ds.sizes["time"]) if target_time_chunk == -1 else target_time_chunk)

    ds_rechunked = ds.chunk(chunk_map)

    # Encoding: coords and (time,lat,lon) vars
    enc = {}
    # coords
    if "time" in ds.coords and target_time_chunk is not None:
        tchunk = int(ds.sizes["time"]) if target_time_chunk == -1 else int(target_time_chunk)
        enc["time"] = {"chunks": (tchunk,)}
    if "lat" in ds.coords: enc["lat"] = {"chunks": (target_lat_chunk,)}
    if "lon" in ds.coords: enc["lon"] = {"chunks": (target_lon_chunk,)}

    # data vars
    for v, da in ds.data_vars.items():
        if tuple(da.dims) == ("time", "lat", "lon"):
            tchunk = int(ds.sizes["time"]) if target_time_chunk in (None, -1) else int(target_time_chunk)
            if target_time_chunk is None:
                # If not specified, default to full time for a clean write
                tchunk = int(ds.sizes["time"])
            enc[v] = {"chunks": (tchunk, target_lat_chunk, target_lon_chunk)}

    if verbose:
        tt = ("leave" if target_time_chunk is None else ("full" if target_time_chunk == -1 else target_time_chunk))
        print(f"[RECHUNK] → {store.name} (time={tt}, lat={target_lat_chunk}, lon={target_lon_chunk}) -> {tmp}")

    ds_rechunked.to_zarr(tmp, mode="w", encoding=enc)
    try:
        zarr.consolidate_metadata(str(tmp))
    except Exception:
        pass

    # Swap
    bak = store.with_name(store.name + ".bak_rechunk")
    try:
        if bak.exists(): shutil.rmtree(bak)
        store.rename(bak)
        tmp.rename(store)
        shutil.rmtree(bak)
        if verbose:
            print(f"[RECHUNK] swapped in {store.name} with lat={target_lat_chunk}, lon={target_lon_chunk}")
    except Exception as e:
        if verbose:
            print(f"[RECHUNK][WARN] swap failed: {e}")
            print(f"[RECHUNK] temp left at: {tmp}")