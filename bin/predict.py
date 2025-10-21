#!/usr/bin/env python3
"""
Tile-wise daily inference for a single scenario (no scenario arg needed).
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc
import torch
from contextlib import contextmanager
import time
import shutil
from datetime import datetime

# ---------------- Project imports ---------------- #
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.paths.paths import predictions_dir
from src.dataset.variables import var_names
from src.models.custom_transformer import YearProcessor  # adjust if needed
from src.training.stats import load_and_filter_standardisation
from src.paths.paths import predictions_dir, std_dict_path, masks_dir

# ---------------- Variable groups (match training) ---------------- #

# Load std dict and prune variables exactly like training
std_dict, pruned = load_and_filter_standardisation(
    standardisation_path=std_dict_path,
    all_vars=var_names["all"],
    daily_vars=var_names["daily_forcing"],
    monthly_vars=var_names["monthly_forcing"],
    annual_vars=var_names["annual_forcing"],
    monthly_states=var_names["monthly_states"],
    annual_states=var_names["annual_states"],
    exclude_vars=set(),   # or the same exclude set you used in training
)

# Use the pruned sets from here on
DAILY_FORCING   = list(pruned["daily_vars"])
MONTHLY_FORCING = list(pruned["monthly_vars"])
MONTHLY_STATES  = list(pruned["monthly_states"])
ANNUAL_FORCING  = list(pruned["annual_vars"])
ANNUAL_STATES   = list(pruned["annual_states"])

# States to persist/carry between years: monthly + annual (this order matters!)
STATE_VARS = MONTHLY_STATES + ANNUAL_STATES

# Outputs and their order must match training:
# monthly_fluxes + monthly_states + annual_states
OUTPUT_ORDER = list(var_names["monthly_fluxes"]) + MONTHLY_STATES + ANNUAL_STATES

# For the Zarr schema, use the exact same list
OUTPUT_VARS = OUTPUT_ORDER

nin = (len(DAILY_FORCING) + len(MONTHLY_FORCING) + len(MONTHLY_STATES)
       + len(ANNUAL_FORCING) + len(ANNUAL_STATES))
print(f"[INFO] Inference input_dim={nin} (should match training)")

# ---------------- Zarr Store Initialisation Helpers ---------------- #
# ==== CONFIG (match your script) ====
TILE_T, TILE_Y, TILE_X = 365, 30, 30

# ==== TIME AXES (noleap) ====
def _days_since_1901_noleap_daily(start: str, end: str) -> np.ndarray:
    import cftime, xarray as xr
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    dates = xr.date_range(start=start, end=end, freq="D", calendar="noleap", use_cftime=True)
    return np.asarray([(d - ref).days for d in dates], dtype="int32")

def _days_since_1901_noleap_monthly(start_year: int, end_year: int) -> np.ndarray:
    """Use the first of each month as the timestamp (days since 1901-01-01)."""
    import cftime
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    vals = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            dt = cftime.DatetimeNoLeap(y, m, 1)
            vals.append((dt - ref).days)
    return np.asarray(vals, dtype="int32")

def _days_since_1901_noleap_annual(start_year: int, end_year: int) -> np.ndarray:
    """Use Jan-01 of each year as the timestamp (days since 1901-01-01)."""
    import cftime
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    vals = []
    for y in range(start_year, end_year + 1):
        dt = cftime.DatetimeNoLeap(y, 1, 1)
        vals.append((dt - ref).days)
    return np.asarray(vals, dtype="int32")

def _period_meta(p0: str, p1: str):
    """Return (start_year, end_year, n_days_total, n_months_total, n_years_total)."""
    sy, ey = int(p0[:4]), int(p1[:4])
    n_years = ey - sy + 1
    n_months = n_years * 12
    n_days = n_years * 365  # noleap
    return sy, ey, n_days, n_months, n_years

# ==== GRID ====
def _global_halfdeg_grid():
    lat = np.arange(-89.75, 90.0, 0.5, dtype="float32")  # 360
    lon = np.arange(0.0, 360.0, 0.5, dtype="float32")    # 720
    return lat, lon

# ==== CORE INIT HELPERS ====
def _ensure_coords_(root, *, time_arr: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                    calendar_note: str):
    """
    Create/repair coord arrays with proper attrs and finite values.
    Ensures _ARRAY_DIMENSIONS exists on all coords, and values are finite.
    """
    T, Y, X = len(time_arr), len(lat), len(lon)

    # --- time ---
    if "time" not in root:
        d = root.create_dataset("time", shape=(T,), chunks=(T,), dtype="i4",
                                compressor=None, overwrite=False)
        d[:] = np.asarray(time_arr, dtype="int32")
    else:
        d = root["time"]
        # Repair shape/type if needed
        if d.shape != (T,) or d.dtype.kind not in ("i","u"):
            d.resize((T,))
            d[:] = np.asarray(time_arr, dtype="int32")
    d.attrs.setdefault("units", "days since 1901-01-01 00:00:00")
    d.attrs.setdefault("calendar", calendar_note)
    d.attrs.setdefault("_ARRAY_DIMENSIONS", ["time"])

    # --- lat ---
    if "lat" not in root:
        d = root.create_dataset("lat", shape=(Y,), chunks=(Y,), dtype="f4",
                                compressor=None, overwrite=False)
        d[:] = lat.astype("float32")
    else:
        d = root["lat"]
        # If any non-finite or wrong length, rebuild canonical half-degree lat
        need_fix = (d.shape != (Y,)) or (not np.isfinite(np.asarray(d[:])).all())
        if need_fix:
            d.resize((Y,))
            d[:] = lat.astype("float32")
    d.attrs.setdefault("_ARRAY_DIMENSIONS", ["lat"])

    # --- lon ---
    if "lon" not in root:
        d = root.create_dataset("lon", shape=(X,), chunks=(X,), dtype="f4",
                                compressor=None, overwrite=False)
        d[:] = lon.astype("float32")
    else:
        d = root["lon"]
        need_fix = (d.shape != (X,)) or (not np.isfinite(np.asarray(d[:])).all())
        if need_fix:
            d.resize((X,))
            d[:] = lon.astype("float32")
    d.attrs.setdefault("_ARRAY_DIMENSIONS", ["lon"])

def _ensure_vars_(root, *, var_names: list[str], shape, chunks, clevel=4):
    comp = Blosc(cname="zstd", clevel=int(clevel), shuffle=Blosc.SHUFFLE)
    for v in var_names:
        if v in root:
            # make sure the CF hint is present even for pre-existing arrays
            root[v].attrs.setdefault("_ARRAY_DIMENSIONS", ["time", "lat", "lon"])
            # (optional) sanity shape check
            if tuple(root[v].shape) != tuple(shape):
                raise RuntimeError(
                    f"Existing variable '{v}' has shape {tuple(root[v].shape)}; expected {tuple(shape)}."
                )
            continue
        d = root.create_dataset(
            v, shape=shape, chunks=chunks, dtype="f4",
            compressor=comp, fill_value=np.float32(np.nan), overwrite=False
        )
        d.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]

def ensure_all_stores(
    *,
    run_root: Path,
    period: tuple[str, str],
    daily_vars: list[str],
    monthly_vars: list[str],
    annual_vars: list[str],
    tile_h: int = TILE_Y,
    tile_w: int = TILE_X,
):
    """
    Create/open three zarr stores under run_root / 'zarr':
      - daily.zarr    (time = daily noleap)
      - monthly.zarr  (time = first day of each month noleap)
      - annual.zarr   (time = Jan 1 each year noleap)

    Returns (daily_store_path, monthly_store_path, annual_store_path, tiles_json_path, meta_dict)
    where meta_dict contains period and sizes useful for indexing.
    """
    import zarr

    zarr_dir = run_root / "zarr"
    zarr_dir.mkdir(parents=True, exist_ok=True)

    p0, p1 = period
    sy, ey, N_days, N_months, N_years = _period_meta(p0, p1)
    lat, lon = _global_halfdeg_grid()
    NY, NX = len(lat), len(lon)

    # --- daily store ---
    daily_store = zarr_dir / "daily.zarr"
    ds = zarr.open_group(store=zarr.DirectoryStore(str(daily_store)), mode="a")
    time_daily = _days_since_1901_noleap_daily(p0, p1)
    _ensure_coords_(ds, time_arr=time_daily, lat=lat, lon=lon, calendar_note="noleap")
    _ensure_vars_(ds, var_names=daily_vars,
                  shape=(N_days, NY, NX),
                  chunks=(TILE_T, tile_h, tile_w))
    zarr.consolidate_metadata(zarr.DirectoryStore(str(daily_store)))

    # --- monthly store ---
    monthly_store = zarr_dir / "monthly.zarr"
    ms = zarr.open_group(store=zarr.DirectoryStore(str(monthly_store)), mode="a")
    time_monthly = _days_since_1901_noleap_monthly(sy, ey)
    _ensure_coords_(ms, time_arr=time_monthly, lat=lat, lon=lon, calendar_note="noleap")
    # choose a chunk of (12, tile, tile) so each year’s write is one chunk row
    _ensure_vars_(ms, var_names=monthly_vars,
                  shape=(N_months, NY, NX),
                  chunks=(12, tile_h, tile_w))
    zarr.consolidate_metadata(zarr.DirectoryStore(str(monthly_store)))

    # --- annual store ---
    annual_store = zarr_dir / "annual.zarr"
    as_ = zarr.open_group(store=zarr.DirectoryStore(str(annual_store)), mode="a")
    time_annual = _days_since_1901_noleap_annual(sy, ey)
    _ensure_coords_(as_, time_arr=time_annual, lat=lat, lon=lon, calendar_note="noleap")
    _ensure_vars_(as_, var_names=annual_vars,
                  shape=(N_years, NY, NX),
                  chunks=(1, tile_h, tile_w))
    zarr.consolidate_metadata(zarr.DirectoryStore(str(annual_store)))

    # --- tiles json (same layout as before, but filename independent of store name) ---
    tiles_json = zarr_dir / f"tiles_{tile_h}x{tile_w}.json"
    if not tiles_json.exists():
        tiles = []
        for y0 in range(0, NY, tile_h):
            y1 = min(y0 + tile_h, NY)
            for x0 in range(0, NX, tile_w):
                x1 = min(x0 + tile_w, NX)
                tiles.append((y0, y1, x0, x1))
        with open(tiles_json, "w") as f:
            json.dump({"ny": NY, "nx": NX,
                       "tile_lat": tile_h, "tile_lon": tile_w,
                       "tiles": tiles}, f)

    meta = {
        "start_year": sy,
        "end_year": ey,
        "n_days": N_days,
        "n_months": N_months,
        "n_years": N_years,
        "tile_h": tile_h,
        "tile_w": tile_w,
    }
    return daily_store, monthly_store, annual_store, tiles_json, meta

# ---------------- Time helpers (noleap) ---------------- #
def years_in_range(start: str, end: str) -> List[int]:
    sY = int(start[:4]); eY = int(end[:4])
    return list(range(sY, eY + 1))

def year_bounds(y: int) -> Tuple[str, str]:
    return f"{y}-01-01", f"{y}-12-31"

# ---------------- Zarr helpers ---------------- #
def open_forcing_stores(forcing_dir: Path) -> Dict[str, xr.Dataset]:
    print(f"[INFO] Opening forcing from {forcing_dir}")
    ds_daily   = xr.open_zarr(forcing_dir / "daily.zarr",   consolidated=True)
    ds_monthly = xr.open_zarr(forcing_dir / "monthly.zarr", consolidated=True)
    ds_annual  = xr.open_zarr(forcing_dir / "annual.zarr",  consolidated=True)
    print(f"[INFO] Forcing datasets opened: daily={list(ds_daily.data_vars)}, "
          f"monthly={list(ds_monthly.data_vars)}, annual={list(ds_annual.data_vars)}")
    return {"daily": ds_daily, "monthly": ds_monthly, "annual": ds_annual}

def _infer_dims_from_state_dict(sd: dict) -> tuple[int | None, int | None, dict]:
    """
    Infer (input_dim, output_dim, cfg) from layer shapes in a raw state_dict.
    Returns (in_dim, out_dim, cfg) where any could be None if not inferrable.

    Expecting module names like:
      pre_conv.0.weight : Conv1d(in_dim, h, 1)       -> shape [h, in_dim, 1]
      pre_conv.2.weight : Conv1d(h, 3*d, 1)          -> shape [3*d, h, 1]
      post_conv.0.weight: Conv1d(d, g, 1)            -> shape [g, d, 1]
      post_conv.2.weight: Conv1d(g, out_dim, 1)      -> shape [out_dim, g, 1]
    Or with the inner model prefix: "inner.pre_conv.*", "inner.post_conv.*".
    """
    def find_shape(keys):
        for k in keys:
            if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].ndim >= 2:
                return tuple(sd[k].shape)
        return None

    # Try both top-level and "inner."-prefixed names
    pre0 = find_shape(["pre_conv.0.weight", "inner.pre_conv.0.weight"])
    pre2 = find_shape(["pre_conv.2.weight", "inner.pre_conv.2.weight"])
    post0 = find_shape(["post_conv.0.weight", "inner.post_conv.0.weight"])
    post2 = find_shape(["post_conv.2.weight", "inner.post_conv.2.weight"])

    in_dim = None
    out_dim = None
    cfg = {}

    # input_dim from pre_conv.0: [h, in_dim, 1]
    if pre0 is not None and len(pre0) >= 2:
        in_dim = int(pre0[1])
        h = int(pre0[0])
        cfg["h"] = h

    # d from pre_conv.2: [3*d, h, 1]
    if pre2 is not None and len(pre2) >= 1:
        c = int(pre2[0])
        if c % 3 == 0:
            cfg["d"] = c // 3

    # g from post_conv.0: [g, d, 1]
    if post0 is not None and len(post0) >= 1:
        cfg["g"] = int(post0[0])

    # output_dim from post_conv.2: [out_dim, g, 1]
    if post2 is not None and len(post2) >= 1:
        out_dim = int(post2[0])

    # Fill safe defaults for anything we couldn't infer
    cfg.setdefault("d", 128)
    cfg.setdefault("h", 1024)
    cfg.setdefault("g", 256)
    cfg.setdefault("num_layers", 4)
    cfg.setdefault("nhead", 8)
    cfg.setdefault("dropout", 0.1)
    cfg.setdefault("transformer_kwargs", {"max_len": 31})

    return in_dim, out_dim, cfg

# ---------------- Data loading for one year & tile ---------------- #
def _sel_year(ds: xr.Dataset, y0: str, y1: str) -> xr.Dataset:
    """Slice a dataset to [y0, y1] along time."""
    return ds.sel(time=slice(y0, y1))

def _expand_monthly_to_daily(arr_m: np.ndarray) -> np.ndarray:
    """
    Expand monthly (12, Y, X, C) to daily (365, Y, X, C) using a noleap calendar.
    """
    days_per_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=int)
    pieces = []
    for m in range(12):
        pieces.append(np.repeat(arr_m[m:m+1, ...], days_per_month[m], axis=0))
    return np.concatenate(pieces, axis=0)

def _expand_annual_to_daily(arr_a: np.ndarray) -> np.ndarray:
    """Expand annual (1, Y, X, C) to daily (365, Y, X, C) by repetition."""
    return np.repeat(arr_a, 365, axis=0)

def load_tile_year_inputs(
    forc: Dict[str, xr.Dataset],
    year_bounds_tuple: Tuple[str, str],
    ys: slice, xs: slice,
    carry_states: np.ndarray | None,
) -> np.ndarray:
    """
    Return inputs for one year & tile as (365, Y, X, Nin) in *training order*:
      daily_forcing + monthly_forcing + monthly_states + annual_forcing + annual_states

    If carry_states is provided (Y, X, Ns), it overwrites the monthly/annual state
    channels (broadcasted to daily), preserving the original input order.
    """
    y0, y1 = year_bounds_tuple

    parts: list[np.ndarray] = []

    # DAILY forcings
    if DAILY_FORCING:
        dsD = _sel_year(forc["daily"], y0, y1)
        arrs = []
        for v in DAILY_FORCING:
            if v in dsD:
                # to (time, Y, X, 1)
                arrs.append(
                    dsD[v].isel(lat=ys, lon=xs).transpose("time", "lat", "lon").values[..., None]
                )
        if arrs:
            D = np.concatenate(arrs, axis=-1).astype("float32")  # (365, Y, X, Nd)
            parts.append(D)

    # MONTHLY → daily
    if MONTHLY_FORCING or MONTHLY_STATES:
        dsM = _sel_year(forc["monthly"], y0, y1)

        if MONTHLY_FORCING:
            arrs = []
            for v in MONTHLY_FORCING:
                if v in dsM:
                    arrs.append(
                        dsM[v].isel(lat=ys, lon=xs).transpose("time", "lat", "lon").values[..., None]
                    )
            if arrs:
                M_forc = np.concatenate(arrs, axis=-1).astype("float32")  # (12, Y, X, Nm_f)
                parts.append(_expand_monthly_to_daily(M_forc))

        if MONTHLY_STATES:
            arrs = []
            for v in MONTHLY_STATES:
                if v in dsM:
                    arrs.append(
                        dsM[v].isel(lat=ys, lon=xs).transpose("time", "lat", "lon").values[..., None]
                    )
            if arrs:
                M_state = np.concatenate(arrs, axis=-1).astype("float32")  # (12, Y, X, Ns_m)
                parts.append(_expand_monthly_to_daily(M_state))

    # ANNUAL → daily
    if ANNUAL_FORCING or ANNUAL_STATES:
        dsA = _sel_year(forc["annual"], y0, y1)

        if ANNUAL_FORCING:
            arrs = []
            for v in ANNUAL_FORCING:
                if v in dsA:
                    arrs.append(
                        dsA[v].isel(lat=ys, lon=xs).transpose("time", "lat", "lon").values[..., None]
                    )
            if arrs:
                A_forc = np.concatenate(arrs, axis=-1).astype("float32")  # (1, Y, X, Na_f)
                parts.append(_expand_annual_to_daily(A_forc))

        if ANNUAL_STATES:
            arrs = []
            for v in ANNUAL_STATES:
                if v in dsA:
                    arrs.append(
                        dsA[v].isel(lat=ys, lon=xs).transpose("time", "lat", "lon").values[..., None]
                    )
            if arrs:
                A_state = np.concatenate(arrs, axis=-1).astype("float32")  # (1, Y, X, Ns_a)
                parts.append(_expand_annual_to_daily(A_state))

    # Concatenate along channel
    if not parts:
        raise RuntimeError("No input variables were found for this year/tile.")
    X = np.concatenate(parts, axis=-1)  # (365, Y, X, Nin)

    # Overwrite STATE channels with carry (if provided)
    if carry_states is not None and (len(MONTHLY_STATES) + len(ANNUAL_STATES)) > 0:
        c = 0
        c += len(DAILY_FORCING)
        ms_start = c;                 c += len(MONTHLY_FORCING)
        mstate_start = c;             c += len(MONTHLY_STATES)
        a_forcing_start = c;          c += len(ANNUAL_FORCING)
        astate_start = c              # + len(ANNUAL_STATES) is the end

        # monthly states part
        if len(MONTHLY_STATES) > 0:
            X[:, :, :, mstate_start:mstate_start+len(MONTHLY_STATES)] = carry_states[:, :, :len(MONTHLY_STATES)]
        # annual states part
        if len(ANNUAL_STATES) > 0:
            X[:, :, :, astate_start:astate_start+len(ANNUAL_STATES)] = carry_states[:, :, len(MONTHLY_STATES):]

    return X.astype("float32")

# ---------------- Model I/O shape helpers ---------------- #
def flatten_tile_time_to_batch(X: np.ndarray) -> np.ndarray:
    """
    Convert (365, Y, X, C) → (B, 365, C), where B = Y*X.
    """
    T, Y, X_, C = X.shape
    return X.reshape(T, Y * X_, C).transpose(1, 0, 2).copy()

def unflatten_batch_to_tile(pred: np.ndarray, Y: int, X: int) -> np.ndarray:
    """
    Convert (B, 365, C) → (365, Y, X, C).
    """
    B, T, C = pred.shape
    return pred.transpose(1, 0, 2).reshape(T, Y, X, C)

# Logic to skip any batches with nans in them 
def _select_valid_batch(Xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Xn: (365, Y, X, C) standardized inputs (NaNs allowed)
    Returns: (Xb_valid, valid_flat)
      - Xb_valid: (B_valid, 365, C)
      - valid_flat: (Y*X,) boolean mask in B order (row-major)
    A pixel is valid if it has finite inputs for all 365 days and all channels.
    """
    T, Y, X, C = Xn.shape
    # valid if ALL finite along time and channel for each Y,X
    valid = np.isfinite(Xn).all(axis=(0, 3))     # (Y, X)
    valid_flat = valid.reshape(Y * X)
    if not np.any(valid_flat):
        return np.empty((0, T, C), dtype=Xn.dtype), valid_flat
    Xb = flatten_tile_time_to_batch(Xn)          # (B, 365, C)
    return Xb[valid_flat], valid_flat

def _scatter_back(pred_valid_btc: np.ndarray, valid_flat: np.ndarray, Y: int, X: int, C: int) -> np.ndarray:
    """
    pred_valid_btc: (B_valid, 365, C)
    valid_flat: (Y*X,) boolean mask
    Returns pred_full_btc: (B, 365, C) filled with NaN, with valid rows inserted.
    """
    B_valid, T, Cout = pred_valid_btc.shape
    pred_full = np.full((Y * X, T, C), np.nan, dtype=np.float32)
    if B_valid:
        pred_full[valid_flat] = pred_valid_btc
    return pred_full

# ---------------- Writing & carry update ---------------- #
def write_year_daily(
    g: zarr.hierarchy.Group,
    pred_year: np.ndarray,
    year_idx0: int,
    ys: slice, xs: slice,
    *,
    log: bool = False,
):
    year_len = pred_year.shape[0]
    C = pred_year.shape[-1]
    if C != len(OUTPUT_ORDER):
        raise RuntimeError(
            "Prediction channel count does not match OUTPUT_ORDER.\n"
            f"  - pred channels: {C}\n"
            f"  - len(OUTPUT_ORDER): {len(OUTPUT_ORDER)}\n"
            f"  - OUTPUT_ORDER: {', '.join(OUTPUT_ORDER)}"
        )

    if log:
        print(f"[WRITE] Writing {year_len} days to Zarr at offset {year_idx0} "
              f"for lat={ys.start}:{ys.stop}, lon={xs.start}:{xs.stop}")

    for k, v in enumerate(OUTPUT_ORDER):
        if v not in g:
            raise KeyError(
                f"Output variable '{v}' not found in Zarr store. "
                "Ensure ensure_all_stores() was called with the same variable names."
            )
        g[v].oindex[year_idx0:year_idx0+year_len, ys, xs] = pred_year[..., k].astype("float32")

def make_carry_from_states(pred_year: np.ndarray) -> np.ndarray:
    idx = [OUTPUT_ORDER.index(v) for v in STATE_VARS]
    if not idx:
        return None
    states_daily = pred_year[..., idx]
    with np.errstate(invalid="ignore"):
        mean_states = np.nanmean(states_daily, axis=0)
    return mean_states.astype("float32")

def _extract_state_dict(ckpt: dict | object) -> dict:
    """
    Accepts either a raw state_dict or a training checkpoint and returns a clean state_dict
    with common prefixes (module./model.) removed.
    """
    # 1) If it's already a state_dict (all values are tensors), just use it
    if isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        sd = ckpt
    elif isinstance(ckpt, dict):
        # 2) Common containers used in training
        for key in ("model_state", "state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
        else:
            raise RuntimeError(
                "Checkpoint does not contain a recognized model state dict "
                "(looked for: model_state/state_dict/model_state_dict/model)"
            )
    else:
        raise RuntimeError("Unrecognized checkpoint format")

    # 3) Strip common prefixes
    def strip_prefix(d: dict, prefix: str) -> dict:
        if not any(k.startswith(prefix) for k in d.keys()):
            return d
        return {k[len(prefix):]: v for k, v in d.items()}

    sd = strip_prefix(sd, "module.")
    sd = strip_prefix(sd, "model.")
    return sd

def _load_landmask_2d(ys: slice, xs: slice) -> np.ndarray:
    """
    Load land fraction from Zarr only (no NetCDF fallback).
    Expects masks_dir / 'inference_land_mask.zarr' with var 'land_fraction'
    on dims ('lat','lon') (and optional 'time').
    Returns (Y, X) float32 in [0,1].
    """
    zarr_path = masks_dir / "inference_land_mask.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(
            f"Land mask Zarr not found at {zarr_path}. "
            "Please build it (see mask builder script)."
        )

    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except Exception as e:
        raise RuntimeError(f"Failed to open land mask Zarr at {zarr_path}: {e}")

    if "land_fraction" not in ds:
        raise KeyError(f"'land_fraction' variable not found in {zarr_path}")

    da = ds["land_fraction"]

    # drop/ignore time if present
    if "time" in da.dims:
        da = da.isel(time=0)
    # enforce lat/lon dims and slice
    if not {"lat", "lon"}.issubset(set(da.dims)):
        raise ValueError(
            f"Expected 'lat' and 'lon' dims in land mask; got {tuple(da.dims)}"
        )

    da = da.isel(lat=ys, lon=xs).transpose("lat", "lon")
    m = np.asarray(da.values, dtype=np.float32)
    m = np.nan_to_num(m, nan=0.0)
    m = np.clip(m, 0.0, 1.0)
    return m

def _extract_dims_and_cfg(ckpt: dict) -> tuple[int, int, dict]:
    """
    Return (input_dim, output_dim, model_cfg).
    Prefer metadata; if missing, infer from the state_dict weights.
    """
    def _get(d, *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    in_dim  = ckpt.get("input_dim")
    out_dim = ckpt.get("output_dim")

    # Look into nested config blobs
    if in_dim is None or out_dim is None:
        mk = _get(ckpt, "config", "extra_cfg", "model_kwargs", default={}) or {}
        in_dim  = in_dim  if in_dim  is not None else mk.get("input_dim")
        out_dim = out_dim if out_dim is not None else mk.get("output_dim")

    # Fallback: args
    if in_dim is None or out_dim is None:
        in_dim  = in_dim  if in_dim  is not None else _get(ckpt, "config", "args", "input_dim")
        out_dim = out_dim if out_dim is not None else _get(ckpt, "config", "args", "output_dim")

    # If still missing, try to infer from a raw state_dict
    if in_dim is None or out_dim is None:
        # Get a clean state_dict from the checkpoint (or treat the ckpt as sd)
        try:
            sd = _extract_state_dict(ckpt)
        except Exception:
            sd = ckpt if isinstance(ckpt, dict) else {}
        if not sd or not all(isinstance(v, torch.Tensor) for v in sd.values()):
            raise RuntimeError(
                "Checkpoint doesn't contain input_dim/output_dim metadata, and no usable state_dict to infer from. "
                "Make sure you pass a training checkpoint (checkpoints/best.pt or epoch*.pt), not a bare weights file."
            )
        in_i, out_i, inferred_cfg = _infer_dims_from_state_dict(sd)
        if in_dim is None:
            in_dim = in_i
        if out_dim is None:
            out_dim = out_i
        base_cfg = inferred_cfg
    else:
        # Build cfg from metadata when available
        base_cfg = {
            "d":         _get(ckpt, "config", "extra_cfg", "model_kwargs", "d",         default=128),
            "h":         _get(ckpt, "config", "extra_cfg", "model_kwargs", "h",         default=1024),
            "g":         _get(ckpt, "config", "extra_cfg", "model_kwargs", "g",         default=256),
            "num_layers":_get(ckpt, "config", "extra_cfg", "model_kwargs", "num_layers",default=4),
            "nhead":     _get(ckpt, "config", "extra_cfg", "model_kwargs", "nhead",     default=8),
            "dropout":   _get(ckpt, "config", "extra_cfg", "model_kwargs", "dropout",   default=0.1),
            "transformer_kwargs": {"max_len": 31},
        }

    if in_dim is None or out_dim is None:
        # Final hard error (nothing left to infer)
        meta_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else ["<non-dict>"]
        raise RuntimeError(
            "Could not determine input/output dimensions from checkpoint.\n"
            f"  - checkpoint top-level keys: {meta_keys[:20]}{'...' if len(meta_keys)>20 else ''}\n"
            "Expected metadata fields or recognizable Conv1d shapes."
        )

    return int(in_dim), int(out_dim), base_cfg

def _check_dims_or_die(
    *,
    ck_input_dim: int,
    ck_output_dim: int,
    nin_expected: int,
    out_names_expected: list[str],
) -> None:
    """
    HARD ERROR if input or output dims differ.
    """
    exp_out_dim = len(out_names_expected)

    if ck_input_dim != nin_expected:
        raise RuntimeError(
            "Input dimension mismatch between checkpoint and inference script.\n"
            f"  - ck_input_dim: {ck_input_dim}\n"
            f"  - expected (from forcings): {nin_expected}\n\n"
            "This usually means exclude_vars or variable pruning was inconsistent "
            "between training and inference."
        )

    if ck_output_dim != exp_out_dim:
        names_str = ", ".join(out_names_expected)
        raise RuntimeError(
            "Output dimension mismatch between checkpoint and inference script.\n"
            f"  - ck_output_dim: {ck_output_dim}\n"
            f"  - expected (from OUTPUT_ORDER): {exp_out_dim}\n"
            f"  - OUTPUT_ORDER names: [{names_str}]\n\n"
            "Refusing to run to avoid writing wrong variables to Zarr."
        )
      
# Function to create standardisation vectors   
def _vector_from_std_dict(names: list[str], std_dict: dict, key_mean="mean", key_std="std"):
    """
    Build per-channel mean/std vectors matching `names` order from std_dict.
    Falls back to mean=0, std=1 if missing.
    """
    mu = []
    sd = []
    for v in names:
        stats = std_dict.get(v, {})
        m = stats.get(key_mean, 0.0)
        s = stats.get(key_std,  1.0)
        # guard against zeros/NaNs
        if not np.isfinite(s) or s == 0:
            s = 1.0
        if not np.isfinite(m):
            m = 0.0
        mu.append(m)
        sd.append(s)
    return np.array(mu, dtype=np.float32), np.array(sd, dtype=np.float32)

# ---------------- Lock helper ---------------- #
@contextmanager
def simple_lock(lock_path: Path, timeout=120):
    lock_f = lock_path.open("w")
    start = time.time()
    waited = False
    while True:
        try:
            os.lockf(lock_f.fileno(), os.F_TLOCK, 0)
            if waited:
                print(f"[LOCK] Acquired: {lock_path}")
            break
        except OSError:
            if not waited:
                print(f"[LOCK] Waiting for {lock_path} ...")
                waited = True
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout acquiring lock {lock_path}")
            time.sleep(0.5)
    try:
        yield
    finally:
        os.lockf(lock_f.fileno(), os.F_ULOCK, 0)
        lock_f.close()

# Helpers for taking the averages
# ---- helpers to map names -> channel indices in OUTPUT_ORDER ----
def _name_idx_map(names: list[str]) -> dict[str, int]:
    return {n: OUTPUT_ORDER.index(n) for n in names}

def _subset_channels(arr_365_y_x_c: np.ndarray, idxs: list[int]) -> np.ndarray:
    # returns (365, Y, X, Csub)
    return arr_365_y_x_c[..., idxs]

# ---- noleap month slicing ----
_NOLEAP_MLEN = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=np.int32)

def _month_slices():
    # returns list of (start, end) day indices for months in a noleap year
    starts = np.cumsum(np.r_[0, _NOLEAP_MLEN[:-1]])
    ends   = np.cumsum(_NOLEAP_MLEN)
    return list(zip(starts.tolist(), ends.tolist()))

# ---- monthly mean for a single year's daily block ----
def monthly_mean_from_daily_year(daily_365_y_x_csub: np.ndarray) -> np.ndarray:
    out = []
    with np.errstate(invalid="ignore"):
        for s, e in _month_slices():
            out.append(np.nanmean(daily_365_y_x_csub[s:e, ...], axis=0, dtype=np.float32))
    return np.stack(out, axis=0).astype("float32")

# ---- annual mean for a single year's daily block ----
def annual_mean_from_daily_year(daily_365_y_x_csub: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(daily_365_y_x_csub, axis=0, dtype=np.float32)[None, ...].astype("float32")

def write_months_one_year(
    g_monthly,                      # zarr group for monthly.zarr
    monthly_block_12_y_x_csub,      # (12, Y, X, Csub)
    y_global: int,                  # 0-based index of the year in the full period
    ys: slice, xs: slice,
    monthly_vars: list[str],
    var_to_chan_idx: dict[str, int],
):
    # monthly_block corresponds to monthly_vars order; write one var at a time
    start = y_global * 12
    end   = start + 12
    for k, v in enumerate(monthly_vars):
        arr = g_monthly[v]
        arr.oindex[start:end, ys, xs] = monthly_block_12_y_x_csub[..., k].astype("float32")

def write_annual_one_year(
    g_annual,
    annual_block_1_y_x_csub,        # (1, Y, X, Csub)
    y_global: int,
    ys: slice, xs: slice,
    annual_vars: list[str],
):
    for k, v in enumerate(annual_vars):
        arr = g_annual[v]
        arr.oindex[y_global:y_global+1, ys, xs] = annual_block_1_y_x_csub[..., k].astype("float32")

# Process a single tile (all years)
def _process_one_tile(
    *,
    tile_index: int,
    tj: dict,
    forc: Dict[str, xr.Dataset],
    g_daily,                       # zarr group for daily.zarr
    g_monthly,                     # zarr group for monthly.zarr
    g_annual,                      # zarr group for annual.zarr
    monthly_vars: list[str],
    annual_vars: list[str],
    years: list[int],
    period_str: str,
    model: torch.nn.Module,
    device: torch.device,
    MU_IN_B: np.ndarray,
    SD_IN_B: np.ndarray,
    MU_OUT_B: np.ndarray,
    SD_OUT_B: np.ndarray,
    tiles_done_before: int | None = None, 
    ntiles_total: int | None = None
):
    """
    One tile over all years:
      - load inputs (+carry), standardise
      - predict -> de-standardise (physical)
      - update carry from UNMASKED, PHYSICAL state channels
      - mask everything by land fraction
      - write masked daily to g_daily
      - compute monthly/annual means from masked daily and write to g_monthly / g_annual
    """
    tiles: List[Tuple[int, int, int, int]] = [tuple(t) for t in tj["tiles"]]
    y0, y1, x0, x1 = tiles[tile_index]
    ys, xs = slice(y0, y1), slice(x0, x1)
    Y, X = (y1 - y0), (x1 - x0)

    print(f"[INFO] Tile {tile_index}: lat={ys.start}:{ys.stop}, lon={xs.start}:{xs.stop}")

    # Land-fraction mask (Y, X) -> broadcast to (1, Y, X, 1)
    mask2d = _load_landmask_2d(ys, xs).astype("float32")
    mask4d = mask2d[None, :, :, None]

    # Precompute which output channels feed monthly/annual aggregates
    monthly_idxs = [OUTPUT_ORDER.index(v) for v in monthly_vars] if monthly_vars else []
    annual_idxs  = [OUTPUT_ORDER.index(v) for v in annual_vars]  if annual_vars  else []

    # Carry in PHYSICAL units
    carry = np.zeros((Y, X, len(STATE_VARS)), dtype="float32") if STATE_VARS else None
    day_ptr = 0  # offset into daily time axis (days since start of period)

    for yi, y in enumerate(years):
        do_log = ((yi + 1) % 10 == 0)
        if do_log:
            print(f"[YEAR] Processing year {y} ({yi+1}/{len(years)})")
        y0s, y1s = year_bounds(y)

        # Load inputs for this year (PHYSICAL units) and inject carry
        X_year = load_tile_year_inputs(forc, (y0s, y1s), ys, xs, carry)   # (365, Y, X, Cin)

        # Standardise inputs (match training)
        Xn = (X_year - MU_IN_B) / SD_IN_B       # (365, Y, X, Cin) may contain NaNs

        # Select only valid pixels for this year/tile
        Xb_valid, valid_flat = _select_valid_batch(Xn)   # (B_valid, 365, Cin), (Y*X,)
        n_valid = Xb_valid.shape[0]

        # If nothing valid: fill outputs with NaN and continue
        if n_valid == 0:
            pred_daily_std = np.full((365, Y, X, len(OUTPUT_ORDER)), np.nan, dtype=np.float32)
        else:
            # Predict only valid pixels
            with torch.no_grad():
                xb_t = torch.from_numpy(Xb_valid).to(device, non_blocking=True)
                pred_valid_btc = model(xb_t).detach().cpu().numpy().astype("float32")  # (B_valid, 365, Cout)

            # Scatter back into full (B,365,C), then to (365,Y,X,C)
            pred_full_btc = _scatter_back(pred_valid_btc, valid_flat, Y, X, len(OUTPUT_ORDER))
            pred_daily_std = unflatten_batch_to_tile(pred_full_btc, Y, X)   # (365, Y, X, Cout)

        # De-standardise to PHYSICAL units
        pred_daily = pred_daily_std * SD_OUT_B + MU_OUT_B   # NaNs stay NaN

        # Update carry from UNMASKED, PHYSICAL states (NaNs propagate for invalid pixels)
        if STATE_VARS:
            carry = make_carry_from_states(pred_daily)      # (Y, X, Ns) with NaNs where invalid

        # Mask EVERYTHING before writing
        pred_daily_masked = pred_daily * mask4d             # (365, Y, X, Cout); NaN stays NaN

        # --- Write daily block ---
        write_year_daily(g_daily, pred_daily_masked, day_ptr, ys, xs, log=do_log)  
        day_ptr += 365

        # --- Monthly means (NaN-aware) ---
        if monthly_idxs:
            daily_monthly_sub = pred_daily_masked[..., monthly_idxs]
            monthly_block = monthly_mean_from_daily_year(daily_monthly_sub)   # NaN if all days NaN
            write_months_one_year(
                g_monthly, monthly_block, y_global=yi,
                ys=ys, xs=xs, monthly_vars=monthly_vars, var_to_chan_idx=None
            )

        # --- Annual means (NaN-aware) ---
        if annual_idxs:
            daily_annual_sub = pred_daily_masked[..., annual_idxs]
            annual_block = annual_mean_from_daily_year(daily_annual_sub)
            write_annual_one_year(
                g_annual, annual_block, y_global=yi, ys=ys, xs=xs, annual_vars=annual_vars
            )

    # replace the existing final print with:
    suffix = ""
    if tiles_done_before is not None and ntiles_total is not None:
        suffix = f" | completed {tiles_done_before+1}/{ntiles_total} tiles"
    print(f"[DONE] Tile {tile_index}: wrote {len(years)} years (daily/monthly/annual){suffix}")
        
def _load_ocean_only_indices(json_path: Path) -> set[int]:
    """
    Load a JSON file describing fully-ocean tiles and return a set of indices.
    Supports a few shapes:
      - {"tile_indices": [..]}
      - {"ocean_only_indices": [..]}
      - {"tiles": [...], "tile_indices": [...]}  (we use tile_indices)
      - A bare JSON list [..]
    """
    if not json_path.exists():
        print(f"[INFO] No ocean-tiles file found at {json_path}; running all tiles.")
        return set()

    with open(json_path, "r") as f:
        data = json.load(f)

    # bare list?
    if isinstance(data, list):
        return set(int(i) for i in data)

    # common keys
    for key in ("tile_indices", "ocean_only_indices"):
        if key in data and isinstance(data[key], (list, tuple)):
            return set(int(i) for i in data[key])

    print(f"[WARN] Unrecognized schema in {json_path}; ignoring skip list.")
    return set()

def _done_dir(run_root: Path) -> Path:
    return run_root / "zarr" / ".done"

def _tile_done_path(run_root: Path, tile_index: int) -> Path:
    return _done_dir(run_root) / f"tile_{tile_index}.ok"

def is_tile_done(run_root: Path, tile_index: int) -> bool:
    return _tile_done_path(run_root, tile_index).exists()

def mark_tile_done(run_root: Path, tile_index: int, *, years: list[int]) -> None:
    p = _tile_done_path(run_root, tile_index)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tile_index": int(tile_index),
        "years": [int(y) for y in years],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(p, "w") as f:
        json.dump(payload, f)

def clear_tile_done(run_root: Path, tile_index: int) -> None:
    p = _tile_done_path(run_root, tile_index)
    if p.exists():
        p.unlink()
        
# ---------------- Main ---------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_name",     required=True)
    ap.add_argument("--forcing_dir",  required=True, type=Path)
    ap.add_argument("--weights",      required=True, type=Path)
    ap.add_argument("--period",       default="1901-01-01:2023-12-31")
    ap.add_argument("--device",       default='cuda')

    # NEW: either process one tile or do round-robin sharding
    ap.add_argument("--tile_index",   type=int, default=None,
                    help="If set, process only this tile and exit.")
    ap.add_argument("--shards",       type=int, default=None,
                    help="Total workers for round-robin distribution (e.g., 8).")
    ap.add_argument("--shard_id",     type=int, default=None,
                    help="This worker id [0..shards-1]. Defaults to $SLURM_ARRAY_TASK_ID when --shards is set.")
    ap.add_argument("--overwrite_skeleton", action="store_true",
                help="Delete and recreate the Zarr stores and tiles JSON before running.")
    ap.add_argument("--overwrite_data", action="store_true",
                    help="Reprocess tiles even if a per-tile done flag exists.")
    args = ap.parse_args()

    # --- paths & period ---
    run_root = predictions_dir / args.job_name          # NOTE: no '/zarr' here
    p0, p1 = args.period.split(":")
    years = years_in_range(p0, p1)
    
    # Optionally blow away existing stores & tiles metadata before re-init
    if args.overwrite_skeleton:
        zroot = run_root / "zarr"
        print(f"[WARN] --overwrite_skeleton requested: removing {zroot} (if exists)")
        if zroot.exists():
            shutil.rmtree(zroot)

    # Which variables go to monthly / annual stores
    MONTHLY_OUT_VARS = list(var_names["monthly_outputs"])  # must be subset of OUTPUT_ORDER
    ANNUAL_OUT_VARS  = list(var_names["annual_outputs"])   # must be subset of OUTPUT_ORDER
    
    assert set(MONTHLY_OUT_VARS).issubset(OUTPUT_ORDER), "monthly_outputs must be a subset of OUTPUT_ORDER"
    assert set(ANNUAL_OUT_VARS).issubset(OUTPUT_ORDER), "annual_outputs must be a subset of OUTPUT_ORDER"

    # --- create/open stores (daily/monthly/annual) & tiles file ---
    lock_file = run_root / "zarr" / ".init.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with simple_lock(lock_file):
        daily_path, monthly_path, annual_path, tiles_json, meta = ensure_all_stores(
            run_root=run_root,
            period=(p0, p1),
            daily_vars=OUTPUT_ORDER,
            monthly_vars=MONTHLY_OUT_VARS,
            annual_vars=ANNUAL_OUT_VARS,
            tile_h=TILE_Y, tile_w=TILE_X,
        )

    # open the three groups
    g_daily   = zarr.open_group(store=zarr.DirectoryStore(str(daily_path)),   mode="a")
    g_monthly = zarr.open_group(store=zarr.DirectoryStore(str(monthly_path)), mode="a")
    g_annual  = zarr.open_group(store=zarr.DirectoryStore(str(annual_path)),  mode="a")

    # --- tiles ---
    with open(tiles_json) as f:
        tj = json.load(f)
    ntiles = len(tj["tiles"])

    # --- ocean-only skip list (optional) ---
    ocean_json = masks_dir / "tiles_ocean_only.json"
    ocean_only = _load_ocean_only_indices(ocean_json)

    print(f"[INFO] Predicting for {len(years)} years ({p0}–{p1})")
    forc = open_forcing_stores(args.forcing_dir)

    # --- device & model ---
    device = torch.device(args.device if args.device in ["cpu", "cuda"] else "cpu")
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading checkpoint (with metadata) from {args.weights}")
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    ck_input_dim, ck_output_dim, ck_model_cfg = _extract_dims_and_cfg(ckpt)
    print(f"[INFO] Checkpoint dims: input_dim={ck_input_dim}, output_dim={ck_output_dim}")

    _check_dims_or_die(
        ck_input_dim=ck_input_dim,
        ck_output_dim=ck_output_dim,
        nin_expected=nin,
        out_names_expected=OUTPUT_ORDER,
    )

    model = YearProcessor(
        input_dim=ck_input_dim,
        output_dim=ck_output_dim,
        **ck_model_cfg
    ).float().to(device)

    state_dict = _extract_state_dict(ckpt if isinstance(ckpt, dict) else {})
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("[WARN] load_state_dict(strict=False) differences:")
        if missing:    print("  Missing:", missing)
        if unexpected: print("  Unexpected:", unexpected)
    model.eval()

    # ---- standardisation vectors (build once) ----
    INPUT_NAMES  = DAILY_FORCING + MONTHLY_FORCING + MONTHLY_STATES + ANNUAL_FORCING + ANNUAL_STATES
    MU_IN,  SD_IN  = _vector_from_std_dict(INPUT_NAMES,  std_dict)
    MU_OUT, SD_OUT = _vector_from_std_dict(OUTPUT_ORDER, std_dict)

    def _bcast(mu, sd):
        return mu.reshape(1,1,1,-1), sd.reshape(1,1,1,-1)

    MU_IN_B, SD_IN_B   = _bcast(MU_IN,  SD_IN)
    MU_OUT_B, SD_OUT_B = _bcast(MU_OUT, SD_OUT)

    # --------- execution modes (balanced land/ocean distribution) ---------
    if args.tile_index is not None:
        # Single-tile debug/rehydrate mode: we still report whether it's land/ocean.
        idxs = [args.tile_index]
        # Defer "todo/skipped" decision to the ocean list below
        assigned_land  = [i for i in idxs if i not in ocean_only]
        assigned_ocean = [i for i in idxs if i in ocean_only]
        shard_id = None
        total_shards = None
    else:
        if not args.shards:
            raise SystemExit("Either provide --tile_index or set --shards for round-robin distribution.")
        shard_id = args.shard_id
        if shard_id is None:
            shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        if not (0 <= shard_id < args.shards):
            raise SystemExit(f"Bad shard_id {shard_id} for shards={args.shards}")

        # Split tiles into land vs ocean
        all_tiles     = list(range(ntiles))
        land_tiles    = [i for i in all_tiles if i not in ocean_only]
        ocean_tiles   = [i for i in all_tiles if i in ocean_only]

        # Distribute *land* tiles evenly across shards (these are the ones we actually process)
        assigned_land = [t for j, t in enumerate(land_tiles)  if (j % args.shards) == shard_id]
        # Distribute ocean-only tiles evenly as well (we still skip them, but keep logging even)
        assigned_ocean = [t for j, t in enumerate(ocean_tiles) if (j % args.shards) == shard_id]

        total_shards = args.shards

    # Build final work lists
    todo    = assigned_land                       # we only process land tiles
    skipped = assigned_ocean                      # ocean-only tiles (balanced across shards)
    
    # Respect per-tile done flags unless user forces overwrite_data
    if not args.overwrite_data:
        todo_before = len(todo)
        todo = [t for t in todo if not is_tile_done(run_root, t)]
        skipped_done = todo_before - len(todo)
        if skipped_done > 0:
            print(f"[INFO] Skipping {skipped_done} tiles already marked done (use --overwrite_data to reprocess).")
    else:
        # If we're going to reprocess, clear stale done flags for tiles we're assigned
        for t in todo:
            clear_tile_done(run_root, t)

    if shard_id is not None:
        print(f"[INFO] Shard {shard_id}/{total_shards}: "
            f"{len(assigned_land)} land assigned, {len(assigned_ocean)} ocean assigned "
            f"(of total {len([i for i in range(ntiles) if i not in ocean_only])} land / "
            f"{len([i for i in range(ntiles) if i in ocean_only])} ocean).")
    else:
        # Single-tile mode
        if skipped:
            print(f"[INFO] Tile {idxs[0]} is ocean-only; it will be skipped.")
        else:
            print(f"[INFO] Tile {idxs[0]} is land; it will be processed.")

    if skipped:
        print(f"[INFO] Skipping {len(skipped)} ocean-only tiles per {ocean_json}")

    failures = []
    ntiles_total = len(todo)  
    tiles_done = 0

    for ti in todo:
        try:
            _process_one_tile(
                tile_index=ti,
                tj=tj,
                forc=forc,
                g_daily=g_daily,
                g_monthly=g_monthly,
                g_annual=g_annual,
                monthly_vars=MONTHLY_OUT_VARS,
                annual_vars=ANNUAL_OUT_VARS,
                years=years,
                period_str=args.period,
                model=model,
                device=device,
                MU_IN_B=MU_IN_B,
                SD_IN_B=SD_IN_B,
                MU_OUT_B=MU_OUT_B,
                SD_OUT_B=SD_OUT_B,
                tiles_done_before=tiles_done,
                ntiles_total=ntiles_total,
            )
            tiles_done += 1
            mark_tile_done(run_root, ti, years=years)
        except Exception as e:
            print(f"[ERR] tile {ti} failed: {e}")
            failures.append(ti)

    if failures:
        print(f"[DONE] Completed with {len(failures)} failures: {failures}")
        raise SystemExit(1)
    print("[DONE] All assigned tiles completed.")
    
if __name__ == "__main__":
    try:
        print("[BOOT] starting predict.py")
        main()
        print("[BOOT] finished predict.py")
    except Exception as e:
        import traceback
        print("[FATAL] Unhandled exception in predict.py:", repr(e))
        traceback.print_exc()
        sys.exit(1)