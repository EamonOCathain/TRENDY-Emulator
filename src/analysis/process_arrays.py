from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, overload, Optional, Union, Literal

import numpy as np
import xarray as xr
import zarr
import cftime
import argparse
from numcodecs import Blosc
import datetime
# ---------------- Project imports ---------------- #
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.paths.paths import masks_dir

# open the dataset (all vars)
def open_zarr(path: Path) -> xr.Dataset:
    """Open a Zarr store lazily, inheriting on-disk chunks."""
    return xr.open_zarr(path, consolidated=True, decode_times=False, chunks="auto")

# open a variable from the dataset
def get_var(ds: xr.Dataset, var: str) -> xr.DataArray:
    """Get a variable from a dataset, erroring if not found."""
    if var not in ds.data_vars:
        raise SystemExit(f"variable '{var}' not found. available: {list(ds.data_vars)}")
    return ds[var]

def multiply_land_mask(da: xr.DataArray) -> xr.DataArray:
    # land mask path
    land_mask_path = masks_dir / "inference_land_mask.nc"
    land_mask_da = xr.open_dataarray(land_mask_path, decode_times=False).squeeze()

    # Standardise coordinates if needed
    land_mask_da = normalise_coords(land_mask_da)

    # Ensure land mask is binary (1 = land, 0 = non-land)
    land_mask_da = xr.where(land_mask_da > 0, 1, 0)

    # Apply mask: keep values where land=1, else NaN
    masked = da.where(land_mask_da == 1, np.nan)

    return masked
    
# construct a standard time axis for given length
def construct_time_axis(length: int) -> np.ndarray:
    ref = cftime.DatetimeNoLeap(1901, 1, 1)

    if length == 44895:
        # 123 years * 365 noleap days
        time_vals = np.arange(0, 123 * 365, dtype="int32")
    elif length == 1476:
        # first day of each month 1901-01..2023-12
        vals = []
        for y in range(1901, 2024):
            for m in range(1, 13):
                dt = cftime.DatetimeNoLeap(y, m, 1)
                vals.append((dt - ref).days)
        time_vals = np.asarray(vals, dtype="int32")
    elif length == 123:
        # Jan 1 each year 1901..2023
        vals = []
        for y in range(1901, 2024):
            dt = cftime.DatetimeNoLeap(y, 1, 1)
            vals.append((dt - ref).days)
        time_vals = np.asarray(vals, dtype="int32")
    else:
        raise ValueError(
            f"Unsupported time length {length}. "
            "Expected one of {44895 (daily), 1476 (monthly), 123 (annual)}."
        )
        
    return time_vals

def normalise_coords(da: xr.DataArray) -> xr.DataArray:
    """
    Replace coords with standard values of time, lat and lon
    """
    da = da.copy()
    
    # Define lat and lon arrays
    std_lat = np.arange(-89.75, 90.0, 0.5, dtype="float32")
    std_lon = np.arange(0.0, 360.0, 0.5, dtype="float32")


    # time
    if "time" in da.dims:
        ntime = int(da.sizes["time"])
        da = da.assign_coords(time=("time", construct_time_axis(ntime)))
        da["time"].attrs.update({
            "units": "days since 1901-01-01 00:00:00",
            "calendar": "noleap",
        })

    # lat (only load two scalars to check order)
    if "lat" in da.dims:
        lat0 = float(da["lat"].isel(lat=0))
        latN = float(da["lat"].isel(lat=-1))
        if lat0 > latN:
            da = da.sortby("lat")
        da = da.assign_coords(lat=("lat", std_lat))

    # lon
    if "lon" in da.dims:
        lon0 = float(da["lon"].isel(lon=0))
        lonN = float(da["lon"].isel(lon=-1))
        if lon0 > lonN:
            da = da.sortby("lon")
        da = da.assign_coords(lon=("lon", std_lon))

    return da

# Create date string from time integer
def int_to_date_string(day_val: int) -> str:
    """Convert a time integer into a date string"""
    units = "days since 1901-01-01 00:00:00"
    cal = "noleap"
    dt = cftime.num2date(int(day_val), units=units, calendar=cal)
    return dt.strftime("%Y-%m-%d")

def date_string_to_int(date_str: str) -> int:
    """
    Convert 'YYYY-MM-DD' string to integer days since 1901-01-01 (noleap calendar).
    """
    y, m, d = map(int, date_str.split("-"))
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    dt = cftime.DatetimeNoLeap(y, m, d)
    return (dt - ref).days

def parse_time_slice_arg(arg: str):
    """
    Parse --time_slice argument as a tuple of two YYYY-MM-DD strings.
    Example: --time_slice "1902-02-02,1905-02-02"
    """
    try:
        parts = [p.strip() for p in arg.split(",")]
        if len(parts) != 2:
            raise ValueError("Must provide exactly two dates separated by a comma")
        # validate format
        for p in parts:
            datetime.datetime.strptime(p, "%Y-%m-%d")
        return tuple(parts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid --time_slice: {e}")

# Put it all together
def open_and_standardise(path: Path, var: str) -> xr.DataArray:
    ds = open_zarr(path)
    da = get_var(ds, var)
    da = normalise_coords(da)
    da = multiply_land_mask(da)
    
    return da

def combine_to_dataset(preds: List[Tuple[str, xr.DataArray]], labels: List[Tuple[str, xr.DataArray]]) -> xr.Dataset:
    """Combine a list of (scenario, DataArray) pairs into a single Dataset, renaming variables."""
    all_das = []
    for scenario, da in preds:
        var = da.name
        new_var = f"Predicted_{var}_{scenario}"
        all_das.append(da.rename(new_var))
    for scenario, da in labels:
        var = da.name
        new_var = f"Label_{var}_{scenario}"
        all_das.append(da.rename(new_var))
    
    ds = xr.merge(all_das)
    return ds

# --- Overloads for type checkers ---
@overload
def subset_time(
    obj: xr.Dataset | xr.DataArray,
    test_subset: Literal[True],
    time_slice: None = ...,
) -> tuple[xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray]: ...

@overload
def subset_time(
    obj: xr.Dataset | xr.DataArray,
    test_subset: Literal[False] = ...,
    time_slice: Optional[Tuple[str, str]] = ...,
) -> xr.Dataset | xr.DataArray: ...


# --- Implementation ---
def subset_time(
    obj: xr.Dataset | xr.DataArray,
    test_subset: bool = False,
    time_slice: Optional[Tuple[str, str]] = None,
) -> xr.Dataset | xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Subset along 'time'.

    - If test_subset=True:
        * For Dataset: return a single object with early (1901–1918) and late (2018–2023)
          concatenated along 'time'.
        * For DataArray: return TWO arrays (early, late), each renamed with
          '_test_period_early' or '_test_period_late'.
    - If time_slice is provided: select [start, end] as 'YYYY-MM-DD'.
    - Else: return unchanged.
    """
    if "time" not in obj.dims:
        raise ValueError("Object has no 'time' dimension")

    if test_subset:
        early = obj.sel(time=slice("1901-01-01", "1918-12-31"))
        late  = obj.sel(time=slice("2018-01-01", "2023-12-31"))

        if isinstance(obj, xr.DataArray):
            # rename each subset
            name = obj.name or "var"
            early = early.rename(f"{name}_test_period_early")
            late  = late.rename(f"{name}_test_period_late")
            return early, late

        elif isinstance(obj, xr.Dataset):
            out = xr.concat([early, late], dim="time")
            # ensure 'time' is strictly increasing after concat
            out = out.sortby("time")
            return out

    if time_slice is not None:
        start, end = time_slice
        return obj.sel(time=slice(start, end))

    return obj

def find_var_groups(ds: xr.Dataset) -> List[Tuple[str, ...]]:
    """
    Find matching variable groups in a dataset.
    - (Label, Predicted)
    - (Label, Predicted, Bias) if a Bias variable is present.
    """
    grouped: Dict[str, Dict[str, str]] = {}

    for name in ds.data_vars:
        m = re.match(r"(Label|Predicted|Bias)_(.+)", name)
        if not m:
            continue
        prefix, suffix = m.groups()
        grouped.setdefault(suffix, {})[prefix] = name

    groups: List[Tuple[str, ...]] = []
    for suffix, g in grouped.items():
        if "Label" in g and "Predicted" in g:
            if "Bias" in g:
                groups.append((g["Label"], g["Predicted"], g["Bias"]))
            else:
                groups.append((g["Label"], g["Predicted"]))

    return groups

def save_to_zarr(
    obj: xr.Dataset | xr.DataArray,
    store_path: Path,
    *,
    overwrite: bool = False,
    clevel: int = 3,
    cast_float32: bool = True,
    var_name: Optional[str] = None,   # used only if obj is a DataArray
    consolidated: bool = True,
) -> None:
    """
    Save an xarray Dataset or DataArray to a Zarr store using the object's existing chunks.

    - If 'time' is present, encode with:
        units   = "days since 1901-01-01 00:00:00"
        calendar= "noleap"
      (values are not changed)
    - If obj is a DataArray, it is wrapped into a Dataset (name from .name or var_name).
    - All data variables are written as float32 (unless cast_float32=False) with Blosc/Zstd.
    """
    # Make out path
    store_path.parent.mkdir(exist_ok=True, parents=True)

    def _ensure_time_attrs(ds: xr.Dataset) -> xr.Dataset:
        if "time" in ds.coords:
            # keep values, just coerce dtype and set attrs
            ds = ds.assign_coords(time=("time", ds["time"].astype("int32").values))
            ds["time"].attrs.update({"units": "days since 1901-01-01 00:00:00", "calendar": "noleap"})
        return ds

    def _build_zarr_encodings(ds: xr.Dataset, clevel: int) -> Dict[str, Dict]:
        comp = Blosc(cname="zstd", clevel=int(clevel), shuffle=Blosc.SHUFFLE)
        enc: Dict[str, Dict] = {}
        for v in ds.data_vars:
            enc[v] = {
                "compressor": comp,
                "dtype": "float32" if cast_float32 else ds[v].dtype,
                "_FillValue": np.float32(np.nan) if cast_float32 else np.nan,
            }
        # coords: leave uncompressed; time attrs handled above
        return enc

    # Wrap DataArray → Dataset (so we can set per-var encoding cleanly)
    if isinstance(obj, xr.DataArray):
        name = obj.name or var_name or "variable"
        ds = obj.to_dataset(name=name)
    elif isinstance(obj, xr.Dataset):
        ds = obj
    else:
        raise TypeError(f"Expected Dataset or DataArray, got {type(obj)}")

    # Optionally cast vars to float32
    if cast_float32:
        ds = ds.map(lambda da: da.astype("float32") if isinstance(da, xr.DataArray) else da)

    # Enforce time attrs if time exists
    ds = _ensure_time_attrs(ds)

    # Per-var encodings (use existing dask chunks; we don't set chunksizes here)
    encoding = _build_zarr_encodings(ds, clevel)

    store_path = Path(store_path)
    mode = "w" if overwrite else "w-"

    ds.to_zarr(
        zarr.DirectoryStore(str(store_path)),
        mode=mode,
        encoding=encoding,
        consolidated=consolidated,
        compute=True,
    )

    # Ensure consolidated metadata exists (fast open)
    zarr.consolidate_metadata(zarr.DirectoryStore(str(store_path)))
    print(f"[ZARR] Wrote {store_path} | overwrite={overwrite} | clevel={clevel} | "
          f"vars={list(ds.data_vars)} | chunks=existing")

def compute_cache_and_open(*, func, ds, store_path, overwrite=False, **kwargs):
    """
    Compute obj = func(ds, **kwargs), cache to Zarr, then reopen and return it.
    Works for functions that return a DA or DS.
    """
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or (not store_path.exists()):
        obj = func(ds, **kwargs)             # <- pass extra args here
        save_to_zarr(obj=obj, store_path=store_path, overwrite=overwrite)
        del obj

    return open_zarr(store_path)
    
# --------------------- TILES --------------------
def tile_bounds_and_indices(
    tile_index: int,
    *,
    nlat: int = 360, nlon: int = 720,     # grid size
    tile_h: int = 30, tile_w: int = 30,   # tile size in *cells*
    lat0: float = -89.75, lon0: float = 0.0,  # first cell centers
    dlat: float = 0.5, dlon: float = 0.5      # grid spacing
):
    """
    Return (lat_min, lat_max, lon_min, lon_max, ilat_start, ilat_end, ilon_start, ilon_end)
    for a 30x30-cells tiling (12x24 = 288 tiles) on a 360x720 grid at 0.5° resolution.

    Index ranges are half-open [start, end), while lat/lon are the *center* bounds
    of the first and last cells covered by the tile.
    """
    nrows = nlat // tile_h   # 360/30 = 12
    ncols = nlon // tile_w   # 720/30 = 24
    total = nrows * ncols    # 288

    if not (0 <= tile_index < total):
        raise ValueError(f"tile_index must be in [0, {total-1}], got {tile_index}")

    row = tile_index // ncols
    col = tile_index %  ncols

    ilat_start = row * tile_h
    ilat_end   = min(ilat_start + tile_h, nlat)
    ilon_start = col * tile_w
    ilon_end   = min(ilon_start + tile_w, nlon)

    # Convert index range to lat/lon range (of cell centers)
    lat_min = lat0 + ilat_start * dlat
    lat_max = lat0 + (ilat_end - 1) * dlat
    lon_min = lon0 + ilon_start * dlon
    lon_max = lon0 + (ilon_end - 1) * dlon

    return (lat_min, lat_max, lon_min, lon_max,
            ilat_start, ilat_end, ilon_start, ilon_end)

# --------------------- METRICS --------------------

def _preserve_coords_after_reduce_ds(ds: xr.Dataset, drop_dims=("time",)) -> Dict[str, xr.DataArray]:
    """
    Preserve non-reduced coords after a Dataset reduction.
    """
    coords = {}
    for name, coord in ds.coords.items():
        if not any(d in coord.dims for d in drop_dims):
            coords[name] = coord
    return coords

# Averaging
def time_avg(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Average over 'time'. Returns same type as input (i.e da or ds)
    """
    if isinstance(obj, xr.DataArray):
        if "time" not in obj.dims:
            raise ValueError("time_avg(DataArray): 'time' dimension not found.")
        out = obj.mean(dim="time", skipna=True)
        print("Took time avg (DataArray).")
        return out

    if isinstance(obj, xr.Dataset):
        if "time" not in obj.dims:
            raise ValueError("time_avg(Dataset): 'time' dimension not found.")
        data_vars = {name: da.mean(dim="time", skipna=True) for name, da in obj.data_vars.items()}
        coords = _preserve_coords_after_reduce_ds(obj, drop_dims=("time",))
        out = xr.Dataset(data_vars, coords=coords)
        print("Took time avg (Dataset).")
        return out

    raise TypeError(f"Expected Dataset or DataArray, got {type(obj)}")

def space_avg(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Take a spatial average (over lat/lon) of either a Dataset or a DataArray.
    Returns the same type as received.
    """
    if isinstance(obj, xr.DataArray):
        out = obj.mean(dim=("lat", "lon"), skipna=True)
        print("Took space average (DataArray).")
        return out

    elif isinstance(obj, xr.Dataset):
        out = {name: da.mean(dim=("lat", "lon"), skipna=True)
               for name, da in obj.data_vars.items()}
        out_ds = xr.Dataset(out, coords={"time": obj["time"]})
        print("Took space average (Dataset).")
        return out_ds

    else:
        raise TypeError(f"Expected xarray.Dataset or xarray.DataArray, got {type(obj)}")

def scenario_avg(ds: xr.Dataset) -> xr.Dataset:
    """
    Take an average across scenarios and store as a new array.
    """
    out = ds.copy()

    # Group by variable type
    predicted_vars = [v for v in ds.data_vars if "Predicted" in v]
    label_vars     = [v for v in ds.data_vars if "Label" in v]

    # Compute averages
    if predicted_vars:
        pred_mean = ds[predicted_vars].to_array().mean("variable")
        for v in predicted_vars:
            base = v.replace("Predicted_", "").split("_")[0]
        out[f"Predicted_{base}_avg"] = pred_mean

    if label_vars:
        lab_mean = ds[label_vars].to_array().mean("variable")
        for v in label_vars:
            base = v.replace("Label_", "").split("_")[0]
        out[f"Label_{base}_avg"] = lab_mean
        
    print("Took scenario average and added to ds")
    
    return out