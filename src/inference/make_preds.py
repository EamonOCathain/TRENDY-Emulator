#!/usr/bin/env python3
"""
src.inference.make_preds
Helpers for tile-wise inference, aggregation, and NetCDF export.

This module is intentionally free of global training-variable state.
Pass an `InferenceSpec` instance to helpers that depend on variable order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import json
import numpy as np
import torch
import xarray as xr
import zarr
import pandas as pd

# =============================================================================
# Spec describing variable layout
# =============================================================================

@dataclass(frozen=True)
class InferenceSpec:
    """Names and ordering used by the model & outputs."""
    DAILY_FORCING: List[str]
    MONTHLY_FORCING: List[str]
    MONTHLY_STATES: List[str]
    ANNUAL_FORCING: List[str]
    ANNUAL_STATES: List[str]
    OUTPUT_ORDER: List[str]

    @property
    def STATE_VARS(self) -> List[str]:
        """Persisted/carry-over states: monthly + annual (this order matters)."""
        return list(self.MONTHLY_STATES) + list(self.ANNUAL_STATES)

    @property
    def nin(self) -> int:
        """Total number of input channels for the model."""
        return (
            len(self.DAILY_FORCING)
            + len(self.MONTHLY_FORCING)
            + len(self.MONTHLY_STATES)
            + len(self.ANNUAL_FORCING)
            + len(self.ANNUAL_STATES)
        )


# =============================================================================
# Simple time helpers (noleap)
# =============================================================================

def years_in_range(start: str, end: str) -> List[int]:
    """Return a list of integer years inclusive from 'YYYY-..' start to end."""
    sY = int(start[:4])
    eY = int(end[:4])
    return list(range(sY, eY + 1))

def flatten_tile_time_to_batch(arr: np.ndarray) -> np.ndarray:
    """
    (T, Y, X, C) -> (B, T, C) with B = Y*X, row-major [y,x] order.
    """
    T, H, W, C = arr.shape
    return arr.reshape(T, H * W, C).transpose(1, 0, 2).copy()

def unflatten_batch_to_tile(btc: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    (B, T, C) -> (T, Y, X, C) where B = Y*X, row-major.
    """
    B, T, C = btc.shape
    assert B == H * W, f"B={B} must equal Y*X={H*W}"
    return btc.transpose(1, 0, 2).reshape(T, H, W, C).copy()

def _expand_monthly_to_daily(m12_y_x_c: np.ndarray) -> np.ndarray:
    """
    (12, Y, X, C) -> (365, Y, X, C) by repeating each month over its noleap day-count.
    """
    mslices = _month_slices()
    Y, X, C = m12_y_x_c.shape[1], m12_y_x_c.shape[2], m12_y_x_c.shape[3]
    out = np.empty((365, Y, X, C), dtype=np.float32)
    for m, (a, b) in enumerate(mslices):
        out[a:b, ...] = m12_y_x_c[m][None, ...]
    return out

def annual_mean_from_daily_year(daily_365_y_x_csub: np.ndarray) -> np.ndarray:
    """(365, Y, X, C) -> (1, Y, X, C) annual mean."""
    with np.errstate(invalid="ignore"):
        m = np.nanmean(daily_365_y_x_csub, axis=0, dtype=np.float32)
    return m[None, ...].astype("float32")


# =============================================================================
# Forcing I/O
# =============================================================================

def open_forcing_stores(forcing_dir: Path) -> Dict[str, xr.Dataset]:
    """
    Open daily/monthly/annual forcing Zarr datasets (consolidated).
    Returns a dict: {"daily": ds, "monthly": ds, "annual": ds}.
    """
    print(f"[INFO] Opening forcing from {forcing_dir}")
    ds_daily = xr.open_zarr(forcing_dir / "daily.zarr", consolidated=True)
    ds_monthly = xr.open_zarr(forcing_dir / "monthly.zarr", consolidated=True)
    ds_annual = xr.open_zarr(forcing_dir / "annual.zarr", consolidated=True)
    print(
        "[INFO] Forcing datasets opened: "
        f"daily={list(ds_daily.data_vars)}, "
        f"monthly={list(ds_monthly.data_vars)}, "
        f"annual={list(ds_annual.data_vars)}"
    )
    return {"daily": ds_daily, "monthly": ds_monthly, "annual": ds_annual}

def _present_names(ds: xr.Dataset, names: List[str]) -> List[str]:
    """Keep only variable names that actually exist in the Dataset."""
    dvars = set(ds.data_vars)
    return [v for v in names if v in dvars]


# =============================================================================
# Checkpoint shape inference / extraction
# =============================================================================

def _infer_dims_from_state_dict(sd: dict) -> tuple[Optional[int], Optional[int], dict]:
    """
    Infer (input_dim, output_dim, cfg) from layer shapes in a raw state_dict.

    Looks for Conv1d-like weights with shapes:
      pre_conv.0.weight : [h, in_dim, 1]
      pre_conv.2.weight : [3*d, h, 1]
      post_conv.0.weight: [g, d, 1]
      post_conv.2.weight: [out_dim, g, 1]
    Supports optional "inner." prefix.
    """
    def find_shape(keys):
        for k in keys:
            t = sd.get(k, None)
            if isinstance(t, torch.Tensor) and t.ndim >= 2:
                return tuple(t.shape)
        return None

    pre0 = find_shape(["pre_conv.0.weight", "inner.pre_conv.0.weight"])
    pre2 = find_shape(["pre_conv.2.weight", "inner.pre_conv.2.weight"])
    post0 = find_shape(["post_conv.0.weight", "inner.post_conv.0.weight"])
    post2 = find_shape(["post_conv.2.weight", "inner.post_conv.2.weight"])

    in_dim = None
    out_dim = None
    cfg: dict = {}

    if pre0 is not None and len(pre0) >= 2:
        in_dim = int(pre0[1])
        cfg["h"] = int(pre0[0])

    if pre2 is not None and len(pre2) >= 1:
        c = int(pre2[0])
        if c % 3 == 0:
            cfg["d"] = c // 3

    if post0 is not None and len(post0) >= 1:
        cfg["g"] = int(post0[0])

    if post2 is not None and len(post2) >= 1:
        out_dim = int(post2[0])

    # Sensible defaults
    cfg.setdefault("d", 128)
    cfg.setdefault("h", 1024)
    cfg.setdefault("g", 256)
    cfg.setdefault("num_layers", 4)
    cfg.setdefault("nhead", 8)
    cfg.setdefault("dropout", 0.1)
    cfg.setdefault("transformer_kwargs", {"max_len": 31})

    return in_dim, out_dim, cfg


def _extract_state_dict(ckpt: dict | object) -> dict:
    """
    Accepts either a raw state_dict or a training checkpoint and returns a clean
    state_dict with common prefixes (module./model.) removed.
    """
    # If it's already a state_dict (all values tensors), just use it
    if isinstance(ckpt, dict) and ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd = ckpt
    elif isinstance(ckpt, dict):
        # Common containers used in training
        for key in ("model_state", "state_dict", "model_state_dict", "model"):
            blob = ckpt.get(key)
            if isinstance(blob, dict):
                sd = blob
                break
        else:
            raise RuntimeError(
                "Checkpoint does not contain a recognized model state dict "
                "(tried: model_state/state_dict/model_state_dict/model)"
            )
    else:
        raise RuntimeError("Unrecognized checkpoint format for state dict extraction.")

    def strip_prefix(d: dict, prefix: str) -> dict:
        if not any(k.startswith(prefix) for k in d.keys()):
            return d
        return {k[len(prefix):]: v for k, v in d.items()}

    sd = strip_prefix(sd, "module.")
    sd = strip_prefix(sd, "model.")
    return sd


def _extract_dims_and_cfg(ckpt: dict) -> tuple[int, int, dict]:
    """
    Return (input_dim, output_dim, model_cfg). Prefer explicit metadata; if
    missing, infer from state_dict weights.
    """
    def _get(d, *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    in_dim = ckpt.get("input_dim")
    out_dim = ckpt.get("output_dim")

    # Look into nested config blobs
    if in_dim is None or out_dim is None:
        mk = _get(ckpt, "config", "extra_cfg", "model_kwargs", default={}) or {}
        in_dim = in_dim if in_dim is not None else mk.get("input_dim")
        out_dim = out_dim if out_dim is not None else mk.get("output_dim")

    # Fallback: args
    if in_dim is None or out_dim is None:
        in_dim = in_dim if in_dim is not None else _get(ckpt, "config", "args", "input_dim")
        out_dim = out_dim if out_dim is not None else _get(ckpt, "config", "args", "output_dim")

    # Final fallback: infer from weights
    if in_dim is None or out_dim is None:
        try:
            sd = _extract_state_dict(ckpt)
        except Exception:
            sd = ckpt if isinstance(ckpt, dict) else {}
        if not sd or not all(isinstance(v, torch.Tensor) for v in sd.values()):
            raise RuntimeError(
                "Checkpoint lacks input/output dim metadata and no usable state_dict to infer from."
            )
        in_i, out_i, inferred_cfg = _infer_dims_from_state_dict(sd)
        if in_dim is None:
            in_dim = in_i
        if out_dim is None:
            out_dim = out_i
        base_cfg = inferred_cfg
    else:
        base_cfg = {
            "d": _get(ckpt, "config", "extra_cfg", "model_kwargs", "d", default=128),
            "h": _get(ckpt, "config", "extra_cfg", "model_kwargs", "h", default=1024),
            "g": _get(ckpt, "config", "extra_cfg", "model_kwargs", "g", default=256),
            "num_layers": _get(ckpt, "config", "extra_cfg", "model_kwargs", "num_layers", default=4),
            "nhead": _get(ckpt, "config", "extra_cfg", "model_kwargs", "nhead", default=8),
            "dropout": _get(ckpt, "config", "extra_cfg", "model_kwargs", "dropout", default=0.1),
            "transformer_kwargs": {"max_len": 31},
        }

    if in_dim is None or out_dim is None:
        meta_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else ["<non-dict>"]
        raise RuntimeError(
            "Could not determine input/output dimensions from checkpoint. "
            f"Top-level keys: {meta_keys[:20]}{'...' if len(meta_keys) > 20 else ''}"
        )

    return int(in_dim), int(out_dim), base_cfg


# =============================================================================
# Forcing offsets (counterfactuals)
# =============================================================================

def _apply_forcing_offsets_to_sliced(
    dsd_yx: xr.Dataset,
    dsm_yx: xr.Dataset,
    dsa_yx: xr.Dataset,
    forcing_offsets: Optional[Dict[str, Dict[str, dict]]],
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Apply per-layer offsets. Each var rule can be:
      {"mode":"percent","value":-20.0}  -> scale by (1 - 0.20)
      {"mode":"add","value":0.3}        -> add +0.3 (physical units)
    After applying, clamp non-negative variables at 0.
    """
    if not forcing_offsets:
        return dsd_yx, dsm_yx, dsa_yx

    # Variables that should never be < 0 (extend if you like)
    NONNEG = {
        "pre", "mrro", "evapotrans", "mrso",
        "lai", "cVeg", "cSoil", "cLitter", "cTotal",
        "population", "potential_radiation", "tswrf", "dlwrf", "fd", "cld"
    }

    def _apply(ds: xr.Dataset, rules: Dict[str, dict]) -> xr.Dataset:
        if not rules:
            return ds
        updates: Dict[str, xr.DataArray] = {}
        for v, cfg in rules.items():
            if v not in ds:
                continue
            arr = ds[v]
            mode = (cfg.get("mode") or "add").lower()
            val = float(cfg.get("value", 0.0))
            if mode == "percent":
                arr = arr * (1.0 + val / 100.0)
            elif mode == "add":
                arr = arr + val
            else:
                print(f"[WARN] Unknown offset mode '{mode}' for var '{v}' (skipped)")
                continue
            if v in NONNEG:
                arr = xr.where(arr < 0, 0, arr)
            updates[v] = arr
        return ds.assign(**updates) if updates else ds

    dsd_yx = _apply(dsd_yx, forcing_offsets.get("daily", {}))
    dsm_yx = _apply(dsm_yx, forcing_offsets.get("monthly", {}))
    dsa_yx = _apply(dsa_yx, forcing_offsets.get("annual", {}))
    return dsd_yx, dsm_yx, dsa_yx


def _check_dims_or_die(
    *,
    ck_input_dim: int,
    ck_output_dim: int,
    nin_expected: int,
    out_names_expected: List[str],
) -> None:
    """Hard error if input or output dims differ."""
    exp_out_dim = len(out_names_expected)

    if ck_input_dim != nin_expected:
        raise RuntimeError(
            "Input dimension mismatch between checkpoint and inference script.\n"
            f"  - ck_input_dim: {ck_input_dim}\n"
            f"  - expected (from forcings): {nin_expected}\n\n"
            "This usually means variable pruning/exclusion differed between training and inference."
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


# =============================================================================
# State seeding utilities
# =============================================================================

def _build_annual_broadcast_seed_from_prev_year(
    spec: InferenceSpec,
    daily_states_prev_year: np.ndarray,  # (365, Y, X, Ns) in STATE_VARS order
) -> np.ndarray:
    """Annual mean (per state channel) across year t-1 → broadcast to all 365 days of year t."""
    assert daily_states_prev_year.ndim == 4
    _, Y, X, _ = daily_states_prev_year.shape
    with np.errstate(invalid="ignore"):
        ann_mean = np.nanmean(daily_states_prev_year, axis=0, dtype=np.float32)  # (Y, X, Ns)
    return np.broadcast_to(ann_mean[None, ...], (365, Y, X, ann_mean.shape[-1])).copy().astype(np.float32)


def _build_january_seed_from_prev_year(
    spec: InferenceSpec,
    daily_states_prev_year: np.ndarray,  # (365, Y, X, Ns)
) -> np.ndarray:
    """
    Build daily template for the new year:
      - MONTHLY states: fill only January with December(t-1) mean
      - ANNUAL states: broadcast annual mean to all days
    Returns (365, Y, X, Ns) in STATE_VARS order.
    """
    assert daily_states_prev_year.ndim == 4
    _, Y, X, Ns = daily_states_prev_year.shape
    mslices = _month_slices()
    jan_s, jan_e = mslices[0]
    dec_s, dec_e = mslices[11]

    out = np.zeros((365, Y, X, Ns), dtype=np.float32)
    with np.errstate(invalid="ignore"):
        dec_mean = np.nanmean(daily_states_prev_year[dec_s:dec_e, ...], axis=0, dtype=np.float32)  # (Y, X, Ns)
        ann_mean = np.nanmean(daily_states_prev_year, axis=0, dtype=np.float32)                    # (Y, X, Ns)

    for si, name in enumerate(getattr(spec, "STATE_VARS", [])):
        if name in getattr(spec, "MONTHLY_STATES", []):
            out[jan_s:jan_e, :, :, si] = dec_mean[:, :, si][None, ...]
        elif name in getattr(spec, "ANNUAL_STATES", []):
            out[:, :, :, si] = ann_mean[:, :, si]
        else:
            out[:, :, :, si] = 0.0

    return out


# =============================================================================
# Input assembly helpers
# =============================================================================

def _expand_monthly_vars_to_daily(
    ds_monthly: xr.Dataset,
    year: int,
    ys: slice,
    xs: slice,
    var_names: List[str],
) -> Optional[np.ndarray]:
    """
    Expand selected monthly vars (12, Y, X, C) to daily (365, Y, X, C) by
    repeating each month over its day count. Returns None if nothing found.
    """
    if not var_names:
        return None
    mslices = _month_slices()
    cols = []
    for v in var_names:
        if v not in ds_monthly:
            continue
        arr = (
            ds_monthly[v]
            .sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
            .isel(lat=ys, lon=xs)
            .values
            .astype("float32")
        )  # (12, Y, X)
        cols.append(arr[..., None])  # (12, Y, X, 1)
    if not cols:
        return None
    monthly_12_y_x_c = np.concatenate(cols, axis=-1)  # (12, Y, X, C)
    Y, X, C = monthly_12_y_x_c.shape[1], monthly_12_y_x_c.shape[2], monthly_12_y_x_c.shape[3]
    daily = np.empty((365, Y, X, C), dtype=np.float32)
    for m, (a, b) in enumerate(mslices):
        daily[a:b, ...] = monthly_12_y_x_c[m, ...][None, ...]
    return daily


def _expand_annual_vars_to_daily(
    ds_annual: xr.Dataset,
    year: int,
    ys: slice,
    xs: slice,
    var_names: List[str],
) -> Optional[np.ndarray]:
    """
    Broadcast selected annual vars (1, Y, X, C) to daily (365, Y, X, C).
    Returns None if nothing found.
    """
    if not var_names:
        return None
    cols = []
    for v in var_names:
        if v not in ds_annual:
            continue
        arr = (
            ds_annual[v]
            .sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
            .isel(lat=ys, lon=xs)
            .values
            .astype("float32")
        )  # (1, Y, X)
        cols.append(arr[0, ...][..., None])  # (Y, X, 1)
    if not cols:
        return None
    a_1_y_x_c = np.concatenate(cols, axis=-1)  # (Y, X, C)
    return np.broadcast_to(a_1_y_x_c[None, ...], (365,) + a_1_y_x_c.shape).copy()


def _preslice_forcing_for_year_and_tile(
    forc: Dict[str, xr.Dataset],
    year: int,
    ys: slice,
    xs: slice,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Return daily/monthly/annual datasets sliced to this year and tile for faster
    per-day `.isel(time=...)` operations downstream.
    """
    y0, y1 = f"{year}-01-01", f"{year}-12-31"
    dsd_yx = forc["daily"].sel(time=slice(y0, y1)).isel(lat=ys, lon=xs)
    dsm_yx = forc["monthly"].sel(time=slice(y0, y1)).isel(lat=ys, lon=xs)
    dsa_yx = forc["annual"].sel(time=slice(y0, y1)).isel(lat=ys, lon=xs)
    return dsd_yx, dsm_yx, dsa_yx


# =============================================================================
# Valid-mask selection & scatter
# =============================================================================

def _select_valid_batch(Xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Select valid pixels (finite for all 365 days and channels).
    Returns:
      - Xb_valid: (B_valid, 365, C)
      - valid_flat: (Y*X,) boolean mask in B order (row-major)
    """
    T, Y, X, C = Xn.shape
    valid = np.isfinite(Xn).all(axis=(0, 3))  # (Y, X)
    valid_flat = valid.reshape(Y * X)
    if not np.any(valid_flat):
        return np.empty((0, T, C), dtype=Xn.dtype), valid_flat
    Xb = flatten_tile_time_to_batch(Xn)
    return Xb[valid_flat], valid_flat


def _scatter_back(pred_valid_btc: np.ndarray, valid_flat: np.ndarray, Y: int, X: int, C: int) -> np.ndarray:
    """Scatter valid predictions back into full (B,365,C); fill others with NaN."""
    pred_full = np.full((Y * X, pred_valid_btc.shape[1], C), np.nan, dtype=np.float32)
    if pred_valid_btc.shape[0]:
        pred_full[valid_flat] = pred_valid_btc
    return pred_full


# =============================================================================
# Writing helpers
# =============================================================================

def write_year_daily(
    *,
    spec: InferenceSpec,
    g: zarr.hierarchy.Group,
    pred_year: np.ndarray,
    year_idx0: int,
    ys: slice,
    xs: slice,
    log: bool = False,
) -> None:
    """
    Write a full year's daily outputs into the daily Zarr group variable-by-variable.
    """
    year_len = pred_year.shape[0]
    C = pred_year.shape[-1]
    if C != len(spec.OUTPUT_ORDER):
        raise RuntimeError(
            "Prediction channel count does not match OUTPUT_ORDER.\n"
            f"  - pred channels: {C}\n"
            f"  - len(OUTPUT_ORDER): {len(spec.OUTPUT_ORDER)}\n"
            f"  - OUTPUT_ORDER: {', '.join(spec.OUTPUT_ORDER)}"
        )

    for k, v in enumerate(spec.OUTPUT_ORDER):
        if v not in g:
            raise KeyError(
                f"Output variable '{v}' not found in Zarr store. "
                "Ensure stores were initialised with the same variable names."
            )
        g[v].oindex[year_idx0:year_idx0 + year_len, ys, xs] = pred_year[..., k].astype("float32")


# =============================================================================
# Aggregation (monthly/annual)
# =============================================================================

_NOLEAP_MLEN = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int32)


def _month_slices() -> List[tuple[int, int]]:
    """Return list of (start_index, end_index) for each month in no-leap year."""
    starts = np.cumsum(np.r_[0, _NOLEAP_MLEN[:-1]])
    ends = np.cumsum(_NOLEAP_MLEN)
    return list(zip(starts.tolist(), ends.tolist()))


def monthly_mean_from_daily_year(daily_365_y_x_csub: np.ndarray) -> np.ndarray:
    """Compute monthly means from daily series (noleap calendar)."""
    out = []
    with np.errstate(invalid="ignore"):
        for s, e in _month_slices():
            out.append(np.nanmean(daily_365_y_x_csub[s:e, ...], axis=0, dtype=np.float32))
    return np.stack(out, axis=0).astype("float32")


def write_months_one_year(
    *,
    g_monthly: zarr.hierarchy.Group,
    monthly_block_12_y_x_csub: np.ndarray,
    y_global: int,                  # 0-based index of the year in the full period
    ys: slice,
    xs: slice,
    monthly_vars: List[str],
) -> None:
    """Write a single year's monthly means into the monthly Zarr."""
    start = y_global * 12
    end = start + 12
    for k, v in enumerate(monthly_vars):
        g_monthly[v].oindex[start:end, ys, xs] = monthly_block_12_y_x_csub[..., k].astype("float32")


def write_annual_one_year(
    *,
    g_annual: zarr.hierarchy.Group,
    annual_block_1_y_x_csub: np.ndarray,   # (1, Y, X, Csub)
    y_global: int,
    ys: slice,
    xs: slice,
    annual_vars: List[str],
) -> None:
    """Write a single year's annual means into the annual Zarr."""
    for k, v in enumerate(annual_vars):
        g_annual[v].oindex[y_global:y_global + 1, ys, xs] = annual_block_1_y_x_csub[..., k].astype("float32")


# =============================================================================
# Ocean-only skip list
# =============================================================================

def _load_ocean_only_indices(json_path: Path) -> set[int]:
    """
    Load a JSON file describing fully-ocean tiles and return a set of indices.
    Supports:
      - {"tile_indices": [..]}
      - {"ocean_only_indices": [..]}
      - {"tiles": [...], "tile_indices": [...]} (uses tile_indices)
      - bare list [..]
    """
    if not json_path.exists():
        print(f"[INFO] No ocean-tiles file found at {json_path}; running all tiles.")
        return set()

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return set(int(i) for i in data)

    for key in ("tile_indices", "ocean_only_indices"):
        if key in data and isinstance(data[key], (list, tuple)):
            return set(int(i) for i in data[key])

    print(f"[WARN] Unrecognized schema in {json_path}; ignoring skip list.")
    return set()


# =============================================================================
# Per-tile done flags
# =============================================================================

def _done_dir(run_root: Path) -> Path:
    return run_root / ".done"


def _tile_done_path(run_root: Path, tile_index: int) -> Path:
    return _done_dir(run_root) / f"tile_{tile_index}.ok"


def is_tile_done(run_root: Path, tile_index: int) -> bool:
    return _tile_done_path(run_root, tile_index).exists()


def mark_tile_done(run_root: Path, tile_index: int, *, years: List[int]) -> None:
    """Mark a tile as completed by writing an .ok JSON payload."""
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
    """Remove the done marker for a tile (if it exists)."""
    p = _tile_done_path(run_root, tile_index)
    if p.exists():
        p.unlink()

def _slice_year_range(ds: xr.Dataset, y0: int, y1: int) -> xr.Dataset:
    """
    Slice a Dataset with MultiIndex time(year, month) by year range.
    Falls back to datetime slice if no 'year' coord.
    """
    if "year" in ds.coords:
        return ds.sel(year=slice(int(y0), int(y1)))
    # fallback for datetime monthly
    return ds.sel(time=slice(f"{int(y0)}-01-01", f"{int(y1)}-12-31"))

# =============================================================================
# Tile window + misc input helpers
# =============================================================================

def _get_tile_window(tiles_json: dict, tile_index: int) -> tuple[slice, slice, int, int]:
    """
    Resolve (ys, xs) slices for a tile and return (ys, xs, Y, X).

    Supported schemas for tiles_json["tiles"][i]:
      A) [y0, y1, x0, x1]  (list/tuple of 4 ints)
      B) dict with explicit spans:
         {"ys":..,"ye":..,"xs":..,"xe":..} or {"y0":..,"y1":..,"x0":..,"x1":..}
         or {"row_start":..,"row_end":..,"col_start":..,"col_end":..}
      C) dict with (row,col) and tile sizes (either at root or per-tile):
         {"row": r, "col": c}  and  tiles_json has {"tile_h": H, "tile_w": W}
         (or per-tile {"tile_h": H, "tile_w": W}). Optional overall size:
         tiles_json may have {"nlat":NLAT,"nlon":NLON} or {"height":..,"width":..} or {"shape":[NLAT,NLON]}.
    """
    # Robustly fetch the tile entry
    tiles = tiles_json["tiles"] if isinstance(tiles_json, dict) and "tiles" in tiles_json else tiles_json
    t = tiles[int(tile_index)]

    # --- Schema A: list/tuple of [y0,y1,x0,x1]
    if isinstance(t, (list, tuple)) and len(t) >= 4:
        try:
            y0, y1, x0, x1 = map(int, t[:4])
            ys, xs = slice(y0, y1), slice(x0, x1)
            Y, X = y1 - y0, x1 - x0
            if Y <= 0 or X <= 0:
                raise RuntimeError(f"Empty tile window from list schema: {t}")
            return ys, xs, Y, X
        except Exception as e:
            raise RuntimeError(f"Invalid list/tuple tile entry at index {tile_index}: {t}") from e

    # Helper for dict schemas
    def _pick(d, *keys):
        for k in keys:
            if isinstance(d, dict) and (k in d) and (d[k] is not None):
                return int(d[k])
        return None

    # --- Schema B: dict with explicit spans
    if isinstance(t, dict):
        ys0 = _pick(t, "ys", "y0", "row_start")
        ye0 = _pick(t, "ye", "y1", "row_end")
        xs0 = _pick(t, "xs", "x0", "col_start")
        xe0 = _pick(t, "xe", "x1", "col_end")
        if None not in (ys0, ye0, xs0, xe0):
            Y = int(ye0 - ys0); X = int(xe0 - xs0)
            if Y <= 0 or X <= 0:
                raise RuntimeError(f"Empty tile window from explicit-span schema: {t}")
            return slice(ys0, ye0), slice(xs0, xe0), Y, X

        # --- Schema C: grid (row, col) + sizes
        row = _pick(t, "row", "iy", "tile_row", "r")
        col = _pick(t, "col", "ix", "tile_col", "c")
        if row is None or col is None:
            # Better error that doesn't assume dict.keys() exists
            raise RuntimeError(
                f"tiles_json entry at index {tile_index} lacks spans or (row,col). "
                f"Entry type={type(t).__name__}, value={t}"
            )

        tile_h = _pick(tiles_json, "tile_h", "tileH", "tile_height") or _pick(t, "tile_h", "tileH", "tile_height")
        tile_w = _pick(tiles_json, "tile_w", "tileW", "tile_width")  or _pick(t, "tile_w", "tileW", "tile_width")
        if tile_h is None or tile_w is None:
            raise RuntimeError("Missing tile_h/tile_w in tiles_json or per-tile entry.")

        ys0 = int(row) * int(tile_h)
        xs0 = int(col) * int(tile_w)
        ye0 = ys0 + int(tile_h)
        xe0 = xs0 + int(tile_w)

        # Optional clamping to full grid size
        nlat = _pick(tiles_json, "nlat", "height")
        nlon = _pick(tiles_json, "nlon", "width")
        if (nlat is None) or (nlon is None):
            shape = tiles_json.get("shape") if isinstance(tiles_json, dict) else None
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                nlat, nlon = int(shape[0]), int(shape[1])

        if nlat is not None:
            ye0 = min(ye0, int(nlat))
        if nlon is not None:
            xe0 = min(xe0, int(nlon))

        Y = int(ye0 - ys0); X = int(xe0 - xs0)
        if Y <= 0 or X <= 0:
            raise RuntimeError(f"Computed empty tile window: (ys,ye,xs,xe)=({ys0},{ye0},{xs0},{xe0})")

        return slice(ys0, ye0), slice(xs0, xe0), Y, X

    # If we got here, the entry is of an unknown type
    raise RuntimeError(
        f"Unsupported tile entry type at index {tile_index}: type={type(t).__name__}, value={t}"
    )


def _compute_nfert_fill_indices(input_names: list[str], nfert: Optional[list[str]]) -> list[int]:
    """
    Return indices in `input_names` that should be zero-filled (e.g. fertilizer channels).
    If `nfert` is None/empty, returns [].
    """
    if not nfert:
        return []
    s = set(nfert)
    return [i for i, name in enumerate(input_names) if name in s]


# =============================================================================
# Misc helpers
# =============================================================================

def _build_tl_seed_daily_from_year(
    *,
    spec: InferenceSpec,
    forc: Dict[str, xr.Dataset],
    seed_year: int,
    ys: slice,
    xs: slice,
    state_targets: List[str],  # e.g. ["lai_avh15c1"] or ["lai_modis"]
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Return (daily, kept_targets):
      - daily: (365, Y, X, K) for the requested TL monthly state targets expanded to daily
      - kept_targets: the subset of `state_targets` that exist in the monthly forcing
    Only monthly states are supported. Returns None if nothing found.
    """
    if not state_targets:
        return None

    mon_targets = [t for t in state_targets if t in getattr(spec, "MONTHLY_STATES", [])]
    if not mon_targets:
        return None

    dsm = forc.get("monthly")
    if dsm is None:
        raise RuntimeError("Monthly forcing dataset missing but TL seeding requested.")

    mon_present = _present_names(dsm, mon_targets)
    if not mon_present:
        return None

    daily = _expand_monthly_vars_to_daily(dsm, seed_year, ys, xs, mon_present)
    return (daily, mon_present) if daily is not None else None


# =============================================================================
# State template composers  (Y/X fix applied here)
# =============================================================================

def _compose_pinned_states_year0(
    spec: InferenceSpec,
    forc: Dict[str, xr.Dataset],
    year: int,
    ys: slice,
    xs: slice,
) -> Optional[np.ndarray]:
    """
    Build (365, Y, X, Ns) in STATE_VARS order for the *current* year by pinning
    states to the forcing:
      - MONTHLY_STATES: take this year's monthly series (12,Y,X) and expand to daily
      - ANNUAL_STATES : take this year's annual value (1,Y,X) and broadcast daily
    Missing variables are zero-filled.
    """
    Ns = len(spec.STATE_VARS)
    if Ns == 0:
        return None

    # FIX: Y from ys, X from xs (previously swapped)
    Y = ys.stop - ys.start
    X = xs.stop - xs.start

    out = np.zeros((365, Y, X, Ns), dtype=np.float32)

    # Monthly piece
    if spec.MONTHLY_STATES:
        keep_m = [v for v in spec.MONTHLY_STATES if v in forc["monthly"].data_vars]
        if keep_m:
            m_daily = _expand_monthly_vars_to_daily(
                forc["monthly"], year, ys, xs, keep_m
            )  # (365, Y, X, Cm)
            if m_daily is not None:
                for j, name in enumerate(keep_m):
                    si = spec.STATE_VARS.index(name)
                    out[..., si] = m_daily[..., j]

    # Annual piece
    if spec.ANNUAL_STATES:
        keep_a = [v for v in spec.ANNUAL_STATES if v in forc["annual"].data_vars]
        if keep_a:
            a_daily = _expand_annual_vars_to_daily(
                forc["annual"], year, ys, xs, keep_a
            )  # (365, Y, X, Ca)
            if a_daily is not None:
                for j, name in enumerate(keep_a):
                    si = spec.STATE_VARS.index(name)
                    out[..., si] = a_daily[..., j]

    return out


def _compose_states_from_forcing_prev_year(
    spec: InferenceSpec,
    forc: Dict[str, xr.Dataset],
    year: int,
    ys: slice,
    xs: slice,
) -> Optional[np.ndarray]:
    """
    As above, but uses year-1 forcing to seed the new year.
      - MONTHLY_STATES: expand monthly series from (t-1) to daily
      - ANNUAL_STATES: broadcast the (t-1) annual value.
    """
    if not getattr(spec, "STATE_VARS", []):
        return None

    y_prev = int(year) - 1
    if y_prev < 0:
        return None

    Ns = len(spec.STATE_VARS)
    # FIX: Y from ys, X from xs (previously swapped)
    Y = ys.stop - ys.start
    X = xs.stop - xs.start

    out = np.zeros((365, Y, X, Ns), dtype=np.float32)

    # Monthly prev-year
    if spec.MONTHLY_STATES:
        keep_m = [v for v in spec.MONTHLY_STATES if v in forc["monthly"].data_vars]
        if keep_m:
            m_daily = _expand_monthly_vars_to_daily(
                forc["monthly"], y_prev, ys, xs, keep_m
            )
            if m_daily is not None:
                for j, name in enumerate(keep_m):
                    si = spec.STATE_VARS.index(name)
                    out[..., si] = m_daily[..., j]

    # Annual prev-year
    if spec.ANNUAL_STATES:
        keep_a = [v for v in spec.ANNUAL_STATES if v in forc["annual"].data_vars]
        if keep_a:
            a_daily = _expand_annual_vars_to_daily(
                forc["annual"], y_prev, ys, xs, keep_a
            )
            if a_daily is not None:
                for j, name in enumerate(keep_a):
                    si = spec.STATE_VARS.index(name)
                    out[..., si] = a_daily[..., j]

    return out


# =============================================================================
# Nudging (PHYSICAL space) — restored original semantics, STATES ONLY
# =============================================================================

def _apply_nudge_states_in_phys(
    day_outputs_phys: np.ndarray,   # (Y, X, Cout)
    spec: InferenceSpec,
    std_dict: dict,
    nudge_lambda: float,
    nudge_mode: str,                # "none" | "original" | "z_shrink" | "z_mirror" | "z_adaptive"
) -> np.ndarray:
    """
    Apply nudging to STATE channels only, in physical units. Returns a modified copy.

    Modes (exactly as in your original):
      - "original": out[..., k] = mu + (nudge_lambda * sigma - (S - mu) / sigma) * sigma
      - "z_shrink": z -> (1 - λ) * z
      - "z_mirror": z -> λ * clip(z,[-3,3]) + (1-λ) * (sign(z)*3)   (reflect toward ±3 then shrink)
      - "z_adaptive": shrink inside ±3; mirror outside
    """
    if (
        nudge_mode == "none"
        or nudge_lambda is None
        or float(nudge_lambda) == 0.0
        or not getattr(spec, "STATE_VARS", [])
    ):
        return day_outputs_phys

    out = day_outputs_phys.copy()

    for v in spec.STATE_VARS:
        # output channel index for this state
        k = spec.OUTPUT_ORDER.index(v)
        stats = std_dict.get(v, {}) or {}
        mu = float(stats.get("mean", 0.0))
        sigma = float(stats.get("std", 1.0)) or 1.0
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

        S = out[..., k]

        if nudge_mode == "original":
            # Restore the original formula exactly
            out[..., k] = mu + (float(nudge_lambda) * sigma - (S - mu) / sigma) * sigma

        elif nudge_mode == "z_shrink":
            z = (S - mu) / sigma
            out[..., k] = mu + (1.0 - float(nudge_lambda)) * z * sigma

        elif nudge_mode == "z_mirror":
            z = (S - mu) / sigma
            z_clip = np.clip(z, -3.0, 3.0)
            z_ref = np.where(np.abs(z) > 3.0, 3.0 * np.sign(z), z)
            z_new = float(nudge_lambda) * z_clip + (1.0 - float(nudge_lambda)) * z_ref
            out[..., k] = mu + z_new * sigma

        elif nudge_mode == "z_adaptive":
            z = (S - mu) / sigma
            abs_z = np.abs(z)
            lam = float(max(0.0, min(1.0, nudge_lambda)))
            inside = abs_z <= 3.0
            z_new = np.where(inside, (1.0 - lam) * z, np.sign(z) * (3.0 - (3.0 - abs_z) * (1.0 - lam)))
            out[..., k] = mu + z_new * sigma

        else:
            raise ValueError("nudge_mode must be one of: 'none', 'original', 'z_shrink', 'z_mirror', 'z_adaptive'")

    return out


# ---------------------------------------------------------------------------
# Tiny local replacement for _vector_from_std_dict (exported)
# ---------------------------------------------------------------------------
def mu_sd_from_std(names: List[str], std_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build mean/std vectors in `names` order from std_dict; fall back to 0/1."""
    mu = []
    sd = []
    for v in names:
        s = std_dict.get(v, {}) or {}
        m = s.get("mean", 0.0)
        st = s.get("std", 1.0)
        if not np.isfinite(m):
            m = 0.0
        if not np.isfinite(st) or float(st) == 0.0:
            st = 1.0
        mu.append(float(m))
        sd.append(float(st))
    return np.asarray(mu, dtype=np.float32), np.asarray(sd, dtype=np.float32)


# =============================================================================
# Main per-tile processing
# =============================================================================

def process_one_tile(
    *,
    spec: InferenceSpec,
    tile_index: int,
    tiles_json: dict,
    forc: Dict[str, xr.Dataset],
    g_daily: zarr.hierarchy.Group,
    g_monthly: zarr.hierarchy.Group,
    g_annual: zarr.hierarchy.Group,
    monthly_vars: List[str],
    annual_vars: List[str],
    years: List[int],
    model: torch.nn.Module,
    device: torch.device,
    MU_IN_B: np.ndarray,
    SD_IN_B: np.ndarray,
    MU_OUT_B: np.ndarray,
    SD_OUT_B: np.ndarray,
    tiles_done_before: Optional[int] = None,
    ntiles_total: Optional[int] = None,
    nfert: Optional[List[str]] = None,
    store_start_year: Optional[int] = None,
    std_dict: Optional[dict] = None,
    nudge_lambda: Optional[float] = None,
    nudge_mode: str = "none",
    carry_forward_states: bool = True,
    sequential_months: bool = False,
    tl_seed_cfg: Optional[dict] = None,
    forcing_offsets: Optional[Dict[str, Dict[str, dict]]] = None, 
) -> None:
    """
    Tile-wise rollout (no land masking), vectorized per year:
      1) Build full-year inputs (365, Y, X, Cin) in PHYSICAL space
      2) Standardize → batch/valid mask → model → destandardize
      3) State overwrite (year0) / nudging (>=1) in PHYSICAL space
      4) Write daily outputs; aggregate monthly/annual as requested
    """
    if store_start_year is None:
        raise RuntimeError("process_one_tile requires store_start_year with split store/write periods.")

    ys, xs, Y, X = _get_tile_window(tiles_json, tile_index)
    print(f"[INFO] Tile {tile_index}: lat={ys.start}:{ys.stop}, lon={xs.start}:{xs.stop}")

    monthly_idxs = [spec.OUTPUT_ORDER.index(v) for v in monthly_vars] if monthly_vars else []
    annual_idxs = [spec.OUTPUT_ORDER.index(v) for v in annual_vars] if annual_vars else []

    first_write_year = int(years[0])
    if first_write_year < store_start_year:
        raise RuntimeError("Write window begins before store start year.")
    day_ptr = (first_write_year - store_start_year) * 365  # daily write offset

    # Channel bookkeeping
    input_names = (
        list(spec.DAILY_FORCING)
        + list(spec.MONTHLY_FORCING)
        + list(spec.MONTHLY_STATES)
        + list(spec.ANNUAL_FORCING)
        + list(spec.ANNUAL_STATES)
    )
    Cin = len(input_names)
    Cout = len(spec.OUTPUT_ORDER)
    n_mstates = len(spec.MONTHLY_STATES)
    n_astates = len(spec.ANNUAL_STATES)

    # Precompute which input channels to zero-fill for nfert once
    fill_idxs: List[int] = _compute_nfert_fill_indices(input_names, nfert)
 
    prev_year_daily_states_ns: Optional[np.ndarray] = None  # (365, Y, X, Ns)

    for yi, year in enumerate(years):
        do_log = ((yi + 1) % 10 == 0)
        if do_log:
            print(f"[YEAR] Processing {year} ({yi + 1}/{len(years)})")

        # Pre-slice forcing to this year & tile (fast indexed reads)
        dsd_yx, dsm_yx, dsa_yx = _preslice_forcing_for_year_and_tile(forc, year, ys, xs)
        
        # Apply any physical offsets to the forcing (e.g., co2 += 100)
        dsd_yx, dsm_yx, dsa_yx = _apply_forcing_offsets_to_sliced(
            dsd_yx, dsm_yx, dsa_yx, forcing_offsets
        )

        # ---- Build state input template (daily, STATE_VARS order)
        if yi == 0:
            # Year-0: start from pinned states of the simulation start year
            state_template_daily = _compose_pinned_states_year0(spec, forc, year, ys, xs)

            # If TL seeding requested: overlay ONLY the requested monthly state targets
            if tl_seed_cfg:
                seed_year = int(tl_seed_cfg.get("seed_year"))
                state_targets = list(tl_seed_cfg.get("state_targets", []))  # e.g., ["lai_avh15c1"]

                tl_ret = _build_tl_seed_daily_from_year(
                    spec=spec,
                    forc=forc,
                    seed_year=seed_year,
                    ys=ys,
                    xs=xs,
                    state_targets=state_targets,
                )
                if tl_ret is not None:
                    tl_daily, kept_targets = tl_ret  # tl_daily: (365,Y,X,K), kept_targets: List[str]
                    for j, name in enumerate(kept_targets):
                        if name in getattr(spec, "STATE_VARS", []):
                            si = spec.STATE_VARS.index(name)  # channel in state template
                            state_template_daily[..., si] = tl_daily[..., j]
                else:
                    print("[WARN] TL seeding requested but no TL seed data found for the targets/year; proceeding without overlay.")
        else:
            # Subsequent years: carry from previous predictions or fallback to forcing(t-1)
            if carry_forward_states:
                if prev_year_daily_states_ns is None and getattr(spec, "STATE_VARS", []):
                    raise RuntimeError("Missing previous-year states for carry_forward_states=True.")
                src = prev_year_daily_states_ns  # PHYSICAL space
                state_template_daily = (
                    _build_january_seed_from_prev_year(spec, src)
                    if sequential_months else
                    _build_annual_broadcast_seed_from_prev_year(spec, src)
                )
            else:
                state_template_daily = _compose_states_from_forcing_prev_year(spec, forc, year, ys, xs)
                if state_template_daily is None:
                    raise RuntimeError(
                        f"Year {year-1} forcing states not available, cannot build inputs for year {year}."
                    )

        # ---- Assemble full-year inputs in PHYSICAL units
        X_year_phys = np.empty((365, Y, X, Cin), dtype=np.float32)
        chan = 0

        # DAILY_FORCING
        if spec.DAILY_FORCING:
            cols = [dsd_yx[v].transpose("time", "lat", "lon").values[..., None].astype("float32")
                    for v in spec.DAILY_FORCING]
            D = np.concatenate(cols, axis=-1) if cols else np.empty((365, Y, X, 0), dtype=np.float32)
            X_year_phys[..., chan:chan + D.shape[-1]] = D
            chan += D.shape[-1]

        # MONTHLY_FORCING (12 → 365)
        if spec.MONTHLY_FORCING:
            cols = []
            for v in spec.MONTHLY_FORCING:
                m12 = dsm_yx[v].transpose("time", "lat", "lon").values[..., None].astype("float32")  # (12, Y, X, 1)
                cols.append(_expand_monthly_to_daily(m12))  # (365, Y, X, 1)
            M_forc = np.concatenate(cols, axis=-1) if cols else np.empty((365, Y, X, 0), dtype=np.float32)
            X_year_phys[..., chan:chan + M_forc.shape[-1]] = M_forc
            chan += M_forc.shape[-1]

        # MONTHLY_STATES (from template)
        if n_mstates > 0:
            X_year_phys[..., chan:chan + n_mstates] = 0.0 if state_template_daily is None else state_template_daily[..., :n_mstates]
            chan += n_mstates

        # ANNUAL_FORCING (broadcast)
        if spec.ANNUAL_FORCING:
            cols = []
            for v in spec.ANNUAL_FORCING:
                a = dsa_yx[v].isel(time=0).values.astype("float32")[..., None]  # (Y, X, 1)
                cols.append(np.broadcast_to(a[None, ...], (365, Y, X, 1)).copy())
            A_forc = np.concatenate(cols, axis=-1) if cols else np.empty((365, Y, X, 0), dtype=np.float32)
            X_year_phys[..., chan:chan + A_forc.shape[-1]] = A_forc
            chan += A_forc.shape[-1]

        # ANNUAL_STATES (from template)
        if n_astates > 0:
            X_year_phys[..., chan:chan + n_astates] = 0.0 if state_template_daily is None else state_template_daily[..., n_mstates:]
            chan += n_astates

        # Optional zero-fill for specified inputs (e.g., nfert)
        if fill_idxs:
            X_year_phys[..., fill_idxs] = np.nan_to_num(X_year_phys[..., fill_idxs], nan=0.0)

        # ---- Standardize → select valid → model → destandardize
        Xn_year = (X_year_phys - MU_IN_B) / SD_IN_B                      # (365, Y, X, Cin)
        Xb_valid, valid_flat = _select_valid_batch(Xn_year)              # (B_valid, 365, Cin), (B,)

        if Xb_valid.shape[0] == 0:
            y_phys_year = np.full((365, Y, X, Cout), np.nan, dtype=np.float32)
        else:
            with torch.no_grad():
                xb = torch.from_numpy(Xb_valid).to(device, non_blocking=True)   # (B_valid, 365, Cin)
                y_std_valid = model(xb).detach().cpu().numpy().astype("float32")
            y_std_full_btc = _scatter_back(y_std_valid, valid_flat, Y, X, Cout)  # (B, 365, C)
            y_std_year = unflatten_batch_to_tile(y_std_full_btc, Y, X)           # (365, Y, X, C)
            y_phys_year = y_std_year * SD_OUT_B + MU_OUT_B                       # back to physical

        # ---- State overwrite / nudging (PHYSICAL space)
        if getattr(spec, "STATE_VARS", []):
            if yi == 0 and state_template_daily is not None:
                # Overwrite state outputs with pinned values for all days
                for si, v in enumerate(spec.STATE_VARS):
                    k = spec.OUTPUT_ORDER.index(v)
                    y_phys_year[..., k] = state_template_daily[..., si]
            else:
                if std_dict is not None and nudge_mode != "none" and (nudge_lambda or 0.0) != 0.0:
                    for t in range(365):
                        y_phys_year[t, ...] = _apply_nudge_states_in_phys(
                            day_outputs_phys=y_phys_year[t, ...],
                            spec=spec,
                            std_dict=std_dict,
                            nudge_lambda=float(nudge_lambda),
                            nudge_mode=nudge_mode,
                        )

        # ---- Write daily
        write_year_daily(
            spec=spec,
            g=g_daily,
            pred_year=y_phys_year,
            year_idx0=day_ptr,
            ys=ys,
            xs=xs,
        )
        day_ptr += 365

        # ---- Aggregate & write monthly / annual
        y_global = int(year) - int(store_start_year)

        if monthly_idxs:
            monthly_src = np.empty((365, Y, X, len(monthly_idxs)), dtype=np.float32)
            for i, name in enumerate(monthly_vars):
                monthly_src[..., i] = y_phys_year[..., spec.OUTPUT_ORDER.index(name)]
            monthly_block = monthly_mean_from_daily_year(monthly_src)  # (12, Y, X, C_m)
            write_months_one_year(
                g_monthly=g_monthly,
                monthly_block_12_y_x_csub=monthly_block,
                y_global=y_global,
                ys=ys,
                xs=xs,
                monthly_vars=monthly_vars,
            )

        if annual_idxs:
            annual_src = np.empty((365, Y, X, len(annual_idxs)), dtype=np.float32)
            for i, name in enumerate(annual_vars):
                annual_src[..., i] = y_phys_year[..., spec.OUTPUT_ORDER.index(name)]
            annual_block = annual_mean_from_daily_year(annual_src)  # (1, Y, X, C_a)
            write_annual_one_year(
                g_annual=g_annual,
                annual_block_1_y_x_csub=annual_block,
                y_global=y_global,
                ys=ys,
                xs=xs,
                annual_vars=annual_vars,
            )

        # ---- Prepare state template source for next year
        if getattr(spec, "STATE_VARS", []):
            Ns = len(spec.STATE_VARS)
            prev_year_daily_states_ns = np.empty((365, Y, X, Ns), dtype=np.float32)
            for si, v in enumerate(spec.STATE_VARS):
                k = spec.OUTPUT_ORDER.index(v)
                prev_year_daily_states_ns[..., si] = y_phys_year[..., k]

    suffix = ""
    if tiles_done_before is not None and ntiles_total is not None:
        suffix = f" | completed {tiles_done_before + 1}/{ntiles_total} tiles"
    print(f"[DONE] Tile {tile_index}: wrote {len(years)} year(s) (window) into global store{suffix}")


# =============================================================================
# NetCDF Export (monthly)
# =============================================================================

# ---- Hard-coded attributes for variables ----
ATTRS = {
    # Carbon fluxes (kg m-2 s-1)
    "nbp": {
        "units": "kg m-2 s-1",
        "long_name": "Net Biome Productivity",
        "standard_name": "net_biome_productivity_of_biomass_expressed_as_carbon_mass_flux",
    },
    "gpp": {
        "units": "kg m-2 s-1",
        "long_name": "Gross Primary Production",
        "standard_name": "gross_primary_productivity_of_biomass_expressed_as_carbon_mass_flux",
    },
    "npp": {
        "units": "kg m-2 s-1",
        "long_name": "Net Primary Production",
        "standard_name": "net_primary_productivity_of_biomass_expressed_as_carbon_mass_flux",
    },
    "ra": {
        "units": "kg m-2 s-1",
        "long_name": "Autotrophic Respiration",
        "standard_name": "autotrophic_respiration_carbon_mass_flux",
    },
    "rh": {
        "units": "kg m-2 s-1",
        "long_name": "Heterotrophic Respiration",
        "standard_name": "heterotrophic_respiration_carbon_mass_flux",
    },
    "fLuc": {
        "units": "kg m-2 s-1",
        "long_name": "Land-Use Change Emissions",
        "standard_name": "land_use_change_carbon_mass_flux",
    },
    "fFire": {
        "units": "kg m-2 s-1",
        "long_name": "Fire Emissions",
        "standard_name": "fire_carbon_mass_flux",
    },

    # Water fluxes (kg m-2 s-1)
    "mrro": {
        "units": "kg m-2 s-1",
        "long_name": "Total Runoff",
        "standard_name": "total_runoff_flux",
    },
    "evapotrans": {
        "units": "kg m-2 s-1",
        "long_name": "Evapotranspiration",
        "standard_name": "evapotranspiration_flux",
    },

    # Carbon stocks / states (kg m-2)
    "cLitter": {
        "units": "kg m-2",
        "long_name": "Carbon in Litter Pool",
        "standard_name": "carbon_mass_content_of_litter",
    },
    "cSoil": {
        "units": "kg m-2",
        "long_name": "Carbon in Soil Pool",
        "standard_name": "carbon_mass_content_of_soil",
    },
    "cVeg": {
        "units": "kg m-2",
        "long_name": "Carbon in Vegetation",
        "standard_name": "carbon_mass_content_of_vegetation",
    },
    "cTotal": {
        "units": "kg m-2",
        "long_name": "Carbon in Ecosystem",
        "standard_name": "carbon_mass_content_of_ecosystem",
    },

    # Water state (kg m-2)
    "mrso": {
        "units": "kg m-2",
        "long_name": "Total Soil Moisture Content",
        "standard_name": "soil_moisture_content",
    },

    # Dimensionless index
    "lai": {
        "units": "m2 m-2",
        "long_name": "Leaf Area Index",
        "standard_name": "leaf_area_index",
    }, 
}


def _ensure_dir(p: Path) -> None:
    """Create directory (and parents) if needed."""
    p.mkdir(parents=True, exist_ok=True)


def _export_one_var_nc(ds: xr.Dataset, var_in: str, var_out: str, out_path: Path) -> None:
    """
    Export a single variable from an xarray Dataset to NetCDF, renaming
    var_in → var_out and applying attributes if defined.
    """
    sub = ds[[var_in]].rename({var_in: var_out})
    if var_out in ATTRS:
        sub[var_out].attrs.update(ATTRS[var_out])

    out_path.unlink(missing_ok=True)
    sub.to_netcdf(out_path)


# ---- Monthly export policy helpers ----

def _annual_mean_then_repeat_monthly(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert a monthly Dataset to a monthly-like series that is flat within each year:
      - If MultiIndex (year, month): groupby('year').mean → expand months → restack.
      - Otherwise (datetime64 or CFTimeIndex monthly): group by time.year, take the annual
        mean, then select per-original time.year and realign to the original monthly index.
    """
    if "time" not in ds.dims:
        raise ValueError("Dataset must have a 'time' dimension")

    var_names = list(ds.data_vars)

    # Case 1: MultiIndex (year, month)
    idx = ds.indexes.get("time", None)
    has_year = ("year" in ds.coords)
    has_month = ("month" in ds.coords)
    is_multi = (idx is not None) and isinstance(idx, pd.MultiIndex) and has_year and has_month
    if is_multi:
        annual = ds.groupby("year").mean(dim="time", skipna=True)
        months = xr.DataArray(np.arange(1, 13, dtype=int), dims=("month",), name="month")
        repeated = annual.expand_dims(month=months)  # dims: year, month, ...
        out = repeated.stack(time=("year", "month")).transpose("time", ...)
        out = out.set_index(time=["year", "month"])
        return out[var_names]

    # Case 2: datetime64 or CFTimeIndex monthly
    # Use groupby('time.year') which works for both numpy datetime64 and cftime indices.
    try:
        years = ds["time"].dt.year  # DataArray of per-timestamp years
        annual = ds.groupby("time.year").mean(skipna=True)  # dims: year, ...
        # Re-expand to the original monthly stamps by selecting each row's year
        repeated = annual.sel(year=years)
        # Align dims/coords back to the original "time"
        repeated = repeated.rename({"year": "time"}).assign_coords(time=ds["time"])
        return repeated[var_names]
    except Exception as e:
        # As a last resort, try resample which also supports CFTimeIndex in modern xarray
        try:
            annual = ds.resample(time="AS").mean(skipna=True)
            return annual.reindex(time=ds.time, method="ffill")[var_names]
        except Exception:
            pass

    raise ValueError(
        "Unsupported time coordinate for export; expected MultiIndex(year,month), "
        "datetime64 monthly, or cftime monthly."
    )


def _should_treat_as_annual(var_name: str, annual_vars: Optional[List[str]]) -> bool:
    """
    Decide if a variable should be exported as 'annual mean then repeated'.
    If `annual_vars` is None, use a conservative default set of common annual states.
    """
    default_annual = {"cVeg", "cSoil", "cLitter", "cTotal"}
    if annual_vars is None:
        return var_name in default_annual
    return var_name in set(annual_vars)


def export_monthly_netcdf_for_scenario(
    *,
    scenario: str,
    nc_root: Path,
    monthly_zarr: Path,
    overwrite: bool = False,
    annual_vars: Optional[List[str]] = None,
) -> None:
    """
    Export monthly NetCDFs from monthly_zarr for a scenario.

    Policy:
      - For variables listed in `annual_vars`, compute the annual mean per year and
        repeat 12× (flat within year).
      - For all other variables, export the stored monthly means directly.

    If `annual_vars` is None, a conservative default set is used: {"cVeg","cSoil","cLitter","cTotal"}.

    Output layout:
      <nc_root>/full/<var>.nc
      <nc_root>/test/early/<var>.nc   (1901-01-01..1918-12-31 if available)
      <nc_root>/test/late/<var>.nc    (2018-01-01..2023-12-31 if available)

    Aliases:
      lai_avh15c1 -> lai
      lai_modis   -> lai
      cTotal_monthly -> cTotal
    """
    print(f"[EXPORT] Opening monthly.zarr for scenario {scenario}: {monthly_zarr}")
    ds = xr.open_zarr(monthly_zarr, consolidated=True)

    # Output dirs
    full_dir  = nc_root / "full"
    early_dir = nc_root / "test" / "early"
    late_dir  = nc_root / "test" / "late"
    for d in (full_dir, early_dir, late_dir):
        _ensure_dir(d)

    # Time slices
    early_slice = slice("1901-01-01", "1918-12-31")
    late_slice  = slice("2018-01-01", "2023-12-31")

    # Map Zarr var -> exported name. Force all LAI variants to 'lai'.
    alias = {
        "lai": "lai",
        "lai_avh15c1": "lai",
        "lai_modis": "lai",
        "cTotal_monthly": "cTotal",
    }

    written_as: set[str] = set()

    for var in sorted(ds.data_vars):
        var_out = alias.get(var, var)

        # Avoid duplicate targets (e.g., multiple LAI sources)
        if var_out in written_as and var_out == "lai":
            print(f"[EXPORT] Skipping '{var}' → 'lai' (already written from another source)")
            continue

        p_full  = full_dir  / f"{var_out}.nc"
        p_early = early_dir / f"{var_out}.nc"
        p_late  = late_dir  / f"{var_out}.nc"

        if (not overwrite) and p_full.exists() and p_early.exists() and p_late.exists():
            print(f"[EXPORT] Skipping {var} (already exists as '{var_out}'; --overwrite_data not set)")
            written_as.add(var_out)
            continue

        def _maybe_transform(ds_sel: xr.Dataset) -> xr.Dataset:
            return _annual_mean_then_repeat_monthly(ds_sel) if _should_treat_as_annual(var_out, annual_vars) else ds_sel

        # -------- Full series --------
        try:
            sub_full = _maybe_transform(ds[[var]].rename({var: var_out}))
            _export_one_var_nc(sub_full, var_out, var_out, p_full)
        except Exception as e:
            print(f"[EXPORT][WARN] Full export failed for {var}: {e}")

        # -------- Early subset --------
        try:
            ds_early = ds.sel(time=early_slice)
            if ds_early.time.size:
                sub_early = _maybe_transform(ds_early[[var]].rename({var: var_out}))
                _export_one_var_nc(sub_early, var_out, var_out, p_early)
        except Exception as e:
            print(f"[EXPORT][WARN] Early subset failed for {var}: {e}")

        # -------- Late subset --------
        try:
            ds_late = ds.sel(time=late_slice)
            if ds_late.time.size:
                sub_late = _maybe_transform(ds_late[[var]].rename({var: var_out}))
                _export_one_var_nc(sub_late, var_out, var_out, p_late)
        except Exception as e:
            print(f"[EXPORT][WARN] Late subset failed for {var}: {e}")

        written_as.add(var_out)

    print(f"[EXPORT] Wrote NetCDFs to:\n  - {full_dir}\n  - {early_dir}\n  - {late_dir}")


def export_netcdf_sharded(
    *,
    monthly_zarr: Path,
    nc_root: Path,
    shards: int,
    shard_id: int,
    overwrite: bool = False,
    var_order: Optional[list[str]] = None,
    annual_vars: Optional[List[str]] = None,
):
    """
    Export per-variable NetCDFs from monthly.zarr, splitting work across shards.

    Policy:
      - For each source variable:
          * if in `annual_vars`: compute annual means and repeat 12× (flat within year)
          * else: export monthly means as stored.
      - Apply aliases:
          lai_avh15c1 -> lai
          lai_modis   -> lai
          cTotal_monthly -> cTotal
      - Deduplicate by final *target* name (first occurrence wins).
      - Deterministic sharding by index in the final target list:
          (index % shards) == shard_id
      - Write to:
          <nc_root>/full/<var>.nc
          <nc_root>/test/early/<var>.nc   (1901-01-01..1918-12-31, if present)
          <nc_root>/test/late/<var>.nc    (2018-01-01..2023-12-31, if present)
    """
    import xarray as xr
    from pathlib import Path

    alias = {
        "lai": "lai",
        "lai_avh15c1": "lai",
        "lai_modis": "lai",
        "cTotal_monthly": "cTotal",
    }

    nc_root = Path(nc_root)
    (nc_root / "full").mkdir(parents=True, exist_ok=True)
    (nc_root / "test" / "early").mkdir(parents=True, exist_ok=True)
    (nc_root / "test" / "late").mkdir(parents=True, exist_ok=True)

    print(f"[EXPORT][{shard_id}/{shards}] Opening {monthly_zarr}")
    ds = xr.open_zarr(monthly_zarr, consolidated=True)

    present = set(ds.data_vars)

    # Preserve caller-provided order if given, else sorted store vars
    if var_order is not None:
        sources_ordered = [v for v in var_order if v in present]
    else:
        sources_ordered = sorted(present)

    # Map source->target with de-duplication on target name (first wins)
    pairs: list[tuple[str, str]] = []
    seen_targets: set[str] = set()
    for src in sources_ordered:
        tgt = alias.get(src, src)
        if tgt in seen_targets:
            continue
        seen_targets.add(tgt)
        pairs.append((src, tgt))

    # Deterministic sharding by target index
    my_pairs = [(src, tgt) for i, (src, tgt) in enumerate(pairs) if (i % shards) == shard_id]
    print(f"[EXPORT][{shard_id}/{shards}] Assigned {len(my_pairs)} targets: {[t for _, t in my_pairs]}")

    def _maybe_write(path: Path, subset: xr.Dataset) -> bool:
        if path.exists() and not overwrite:
            return False
        subset.to_netcdf(path)
        return True

    early_slice = slice("1901-01-01", "1918-12-31")
    late_slice  = slice("2018-01-01", "2023-12-31")

    n_written = 0
    for v_in, v_out in my_pairs:
        print(f"[EXPORT][{shard_id}/{shards}] {v_in} -> {v_out}")

        # Select and rename
        sel = ds[[v_in]].rename({v_in: v_out})

        # Optional: attach attrs if provided
        if "ATTRS" in globals() and isinstance(ATTRS, dict) and v_out in ATTRS:
            try:
                sel[v_out].attrs.update(ATTRS[v_out])
            except Exception:
                pass

        # Transform only for annual vars
        if _should_treat_as_annual(v_out, annual_vars):
            sel = _annual_mean_then_repeat_monthly(sel)

        # write full series (skip if exists unless overwrite=True)
        wrote_any = False
        wrote_any |= _maybe_write(nc_root / "full" / f"{v_out}.nc", sel)

        # early / late ranges by YEAR (handles MultiIndex cleanly)
        try:
            se = _slice_year_range(sel, 1901, 1918)
            if getattr(se, "time", None) is not None and se.time.size:
                wrote_any |= _maybe_write(nc_root / "test" / "early" / f"{v_out}.nc", se)
        except Exception as e:
            print(f"[EXPORT][WARN] early subset failed for {v_in}: {e}")

        try:
            sl = _slice_year_range(sel, 2018, 2023)
            if getattr(sl, "time", None) is not None and sl.time.size:
                wrote_any |= _maybe_write(nc_root / "test" / "late" / f"{v_out}.nc", sl)
        except Exception as e:
            print(f"[EXPORT][WARN] late subset failed for {v_in}: {e}")

        if wrote_any:
            n_written += 1

    print(f"[EXPORT][{shard_id}/{shards}] Done. Variables written or updated: {n_written}")


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # data classes
    "InferenceSpec",
    # time helpers
    "years_in_range",
    # forcing I/O
    "open_forcing_stores",
    # checkpoint helpers
    "_extract_state_dict", "_infer_dims_from_state_dict", "_extract_dims_and_cfg", "_check_dims_or_die",
    # input assembly & reshaping
    "flatten_tile_time_to_batch", "unflatten_batch_to_tile",
    "_select_valid_batch", "_scatter_back",
    # writing & carry
    "write_year_daily",
    # stats
    "mu_sd_from_std",
    # aggregation
    "monthly_mean_from_daily_year",
    "write_months_one_year",
    # ocean-only and done flags
    "_load_ocean_only_indices",
    "is_tile_done", "mark_tile_done", "clear_tile_done",
    # main per-tile driver
    "process_one_tile",
    # exporter
    "export_monthly_netcdf_for_scenario",
    "export_netcdf_sharded",
]