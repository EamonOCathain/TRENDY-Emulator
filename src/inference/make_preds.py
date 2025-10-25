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

def _expand_monthly_to_daily(arr_m: np.ndarray) -> np.ndarray:
    """Expand monthly (12, Y, X, C) → daily (365, Y, X, C) using a no-leap calendar."""
    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=int)
    pieces = [np.repeat(arr_m[m:m + 1, ...], days_per_month[m], axis=0) for m in range(12)]
    return np.concatenate(pieces, axis=0)


def _expand_annual_to_daily(arr_a: np.ndarray) -> np.ndarray:
    """Expand annual (1, Y, X, C) → daily (365, Y, X, C) by repetition."""
    return np.repeat(arr_a, 365, axis=0)


# =============================================================================
# Batch reshape utilities
# =============================================================================

def flatten_tile_time_to_batch(X: np.ndarray) -> np.ndarray:
    """(365, Y, X, C) → (B, 365, C), where B = Y * X."""
    T, Y, X_, C = X.shape
    return X.reshape(T, Y * X_, C).transpose(1, 0, 2).copy()


def unflatten_batch_to_tile(pred: np.ndarray, Y: int, X: int) -> np.ndarray:
    """(B, 365, C) → (365, Y, X, C)."""
    B, T, C = pred.shape
    return pred.transpose(1, 0, 2).reshape(T, Y, X, C)


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
# Stats helpers
# =============================================================================

def _vector_from_std_dict(names: List[str], std_dict: dict, key_mean="mean", key_std="std") -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-channel mean/std vectors matching `names` order from std_dict.
    Falls back to mean=0, std=1 if missing.
    """
    mu: List[float] = []
    sd: List[float] = []
    for v in names:
        stats = std_dict.get(v, {}) or {}
        m = stats.get(key_mean, 0.0)
        s = stats.get(key_std, 1.0)
        if not np.isfinite(s) or s == 0:
            s = 1.0
        if not np.isfinite(m):
            m = 0.0
        mu.append(float(m))
        sd.append(float(s))
    return np.array(mu, dtype=np.float32), np.array(sd, dtype=np.float32)


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


def annual_mean_from_daily_year(daily_365_y_x_csub: np.ndarray) -> np.ndarray:
    """Compute annual mean of daily series (returns shape (1, Y, X, C))."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(daily_365_y_x_csub, axis=0, dtype=np.float32)[None, ...].astype("float32")


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


# =============================================================================
# Tile window + forcing helpers
# =============================================================================

def _get_tile_window(tiles_json: dict, tile_index: int) -> Tuple[slice, slice, int, int]:
    """Return (ys, xs, Y, X) for a tile index from the tiles JSON layout."""
    y0, y1, x0, x1 = map(int, tiles_json["tiles"][tile_index])
    ys, xs = slice(y0, y1), slice(x0, x1)
    Y, X = y1 - y0, x1 - x0
    return ys, xs, Y, X


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


def _compose_pinned_states_year0(
    spec: InferenceSpec,
    forc: Dict[str, xr.Dataset],
    year: int,
    ys: slice,
    xs: slice,
) -> Optional[np.ndarray]:
    """
    Build (365, Y, X, Ns) pinned states for Year 0 in STATE_VARS order:
      - monthly states: expand to daily (12→365)
      - annual states: broadcast to daily (1→365)
    Returns None if there are no state vars.
    """
    state_names = list(getattr(spec, "STATE_VARS", []))
    if not state_names:
        return None

    mon_state_names = [v for v in state_names if v in getattr(spec, "MONTHLY_STATES", [])]
    ann_state_names = [v for v in state_names if v in getattr(spec, "ANNUAL_STATES", [])]

    parts = []
    dsm = forc.get("monthly")
    dsa = forc.get("annual")
    
    if mon_state_names and dsm is not None:
        mon_present = _present_names(dsm, mon_state_names)
        if mon_present:
            m_daily = _expand_monthly_vars_to_daily(dsm, year, ys, xs, mon_present)  # (365,Y,X,Cm)
            if m_daily is not None:
                parts.append((mon_present, m_daily))
                
    if ann_state_names and dsa is not None:
        ann_present = _present_names(dsa, ann_state_names)
        if ann_present:
            a_daily = _expand_annual_vars_to_daily(dsa, year, ys, xs, ann_present)   # (365,Y,X,Ca)
            if a_daily is not None:
                parts.append((ann_present, a_daily))

    if not parts:
        return None

    Y, X = parts[0][1].shape[1], parts[0][1].shape[2]
    Ns = len(state_names)
    out = np.empty((365, Y, X, Ns), dtype=np.float32)

    for si, name in enumerate(state_names):
        src_block = None
        src_idx = None
        for names, arr in parts:
            if name in names:
                src_block = arr
                src_idx = names.index(name)
                break
        out[..., si] = 0.0 if src_block is None else src_block[..., src_idx]

    return out


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
# Nudging (applied in physical space)
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
        k = spec.OUTPUT_ORDER.index(v)
        stats = std_dict.get(v, {}) or {}
        mu = float(stats.get("mean", 0.0))
        sigma = float(stats.get("std", 1.0)) or 1.0
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

        S = out[..., k]

        if nudge_mode == "original":
            out[..., k] = nudge_lambda * sigma - (S - mu) / sigma

        elif nudge_mode == "z_shrink":
            z = (S - mu) / sigma
            out[..., k] = mu + (1.0 - float(nudge_lambda)) * z * sigma

        elif nudge_mode == "z_mirror":
            z = (S - mu) / sigma
            out[..., k] = mu + (float(nudge_lambda) - z) * sigma

        elif nudge_mode == "z_adaptive":
            z = (S - mu) / sigma
            abs_z = np.abs(z)
            lam = float(max(0.0, min(1.0, nudge_lambda)))
            alpha = (lam * (abs_z / (1.0 + abs_z))).astype(np.float32)  # alpha(|z|)
            out[..., k] = mu + (1.0 - alpha) * z * sigma

        else:
            raise ValueError("nudge_mode must be one of: 'none', 'original', 'z_shrink', 'z_mirror', 'z_adaptive'")

    return out


# =============================================================================
# Misc helpers
# =============================================================================

def _compute_nfert_fill_indices(input_names: List[str], nfert: Optional[List[str]]) -> List[int]:
    """
    Given the model's input_names (in channel order) and a list of variable names `nfert`,
    return the column indices to zero-fill. Returns [] if nothing matches.
    """
    if not nfert:
        return []
    name_to_idx = {name: i for i, name in enumerate(input_names)}
    return [name_to_idx[n] for n in nfert if n in name_to_idx]


def _compose_states_from_forcing_prev_year(
    spec: InferenceSpec,
    forc: Dict[str, xr.Dataset],
    year: int,
    ys: slice,
    xs: slice,
) -> Optional[np.ndarray]:
    """
    Build (365, Y, X, Ns) state inputs for year=t using forcing from year=t-1:
      - Monthly state vars: read 12 months from t-1, expand to daily.
      - Annual  state vars: read 1 value from t-1, repeat to 365 days.
    Returns None if there are no state vars.
    """
    state_names = list(getattr(spec, "STATE_VARS", []))
    if not state_names:
        return None

    y_prev = year - 1
    mon_state_names = [v for v in state_names if v in getattr(spec, "MONTHLY_STATES", [])]
    ann_state_names = [v for v in state_names if v in getattr(spec, "ANNUAL_STATES", [])]

    parts = {}

    if mon_state_names:
        dsm = forc.get("monthly")
        if dsm is None:
            raise RuntimeError("Monthly forcing dataset missing but monthly states requested.")
        mon_present = _present_names(dsm, mon_state_names)
        if mon_present:
            m_daily = _expand_monthly_vars_to_daily(dsm, y_prev, ys, xs, mon_present)  # (365,Y,X,Cm)
            if m_daily is not None:
                parts["monthly"] = (mon_present, m_daily)

    if ann_state_names:
        dsa = forc.get("annual")
        if dsa is None:
            raise RuntimeError("Annual forcing dataset missing but annual states requested.")
        ann_present = _present_names(dsa, ann_state_names)
        if ann_present:
            a_daily = _expand_annual_vars_to_daily(dsa, y_prev, ys, xs, ann_present)   # (365,Y,X,Ca)
            if a_daily is not None:
                parts["annual"] = (ann_present, a_daily)

    if not parts:
        return None

    Y, X = next(iter(parts.values()))[1].shape[1:3]
    Ns = len(state_names)
    out = np.empty((365, Y, X, Ns), dtype=np.float32)

    for si, name in enumerate(state_names):
        if "monthly" in parts and name in parts["monthly"][0]:
            arr = parts["monthly"][1]
            idx = parts["monthly"][0].index(name)
            out[..., si] = arr[..., idx]
        elif "annual" in parts and name in parts["annual"][0]:
            arr = parts["annual"][1]
            idx = parts["annual"][0].index(name)
            out[..., si] = arr[..., idx]
        else:
            out[..., si] = 0.0

    return out


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


def _iter_vars_to_export(ds: xr.Dataset, only_vars: Optional[List[str]]) -> List[str]:
    """
    Decide which variables to export:
      - if only_vars is None: all ds.data_vars
      - else: intersection (and keep order from only_vars)
    """
    if only_vars is None:
        return list(ds.data_vars)
    present = set(ds.data_vars)
    return [v for v in only_vars if v in present]


def export_monthly_netcdf_for_scenario(
    *,
    scenario: str,
    nc_root: Path,
    monthly_zarr: Path,
    overwrite: bool = False,
) -> None:
    print(f"[EXPORT] Opening monthly.zarr for scenario {scenario}: {monthly_zarr}")
    ds = xr.open_zarr(monthly_zarr, consolidated=True)

    # Output dirs
    full_dir = nc_root / "full"
    early_dir = nc_root / "test" / "early"
    late_dir = nc_root / "test" / "late"
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
        # keep your special cTotal rename too:
        "cTotal_monthly": "cTotal",
    }

    # Keep track so we don’t export 'lai' twice if multiple sources exist
    written_as: set[str] = set()

    # Iterate variables in the monthly Zarr in a stable order
    for var in sorted(ds.data_vars):
        var_out = alias.get(var, var)

        # Skip duplicate target names (e.g., if lai_avh15c1 and lai both present)
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

        # Full series
        _export_one_var_nc(ds, var, var_out, p_full)

        # Early subset
        try:
            ds_early = ds.sel(time=early_slice)
            if ds_early.time.size:
                _export_one_var_nc(ds_early, var, var_out, p_early)
        except Exception as e:
            print(f"[EXPORT][WARN] Early subset failed for {var}: {e}")

        # Late subset
        try:
            ds_late = ds.sel(time=late_slice)
            if ds_late.time.size:
                _export_one_var_nc(ds_late, var, var_out, p_late)
        except Exception as e:
            print(f"[EXPORT][WARN] Late subset failed for {var}: {e}")

        written_as.add(var_out)

    print(f"[EXPORT] Wrote NetCDFs to:\n  - {full_dir}\n  - {early_dir}\n  - {late_dir}")


def export_netcdf_sharded(monthly_zarr: Path, nc_root: Path, shards: int, shard_id: int, overwrite: bool):
    """
    Export per-variable NetCDFs from the monthly.zarr, splitting the work across shards.
    Each shard writes a disjoint subset of *target* variable names to avoid collisions.

    Rules:
      - Alias lai_* -> 'lai'; cTotal_monthly -> 'cTotal'
      - Dedup targets so only the first source that maps to a target is exported
      - Shard by modulo on the final target name to guarantee disjoint work
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

    # Build (source_var -> target_var) pairs with de-dup on target_var
    vars_all = sorted(ds.data_vars)
    pairs = []
    seen_targets = set()
    for v in vars_all:
        tgt = alias.get(v, v)
        if tgt in seen_targets:
            continue
        seen_targets.add(tgt)
        pairs.append((v, tgt))

    # Assign work by modulo on target name (stable, avoids collisions)
    assigned = [(src, tgt) for (src, tgt) in pairs if (hash(tgt) % shards) == shard_id]
    print(f"[EXPORT][{shard_id}/{shards}] Assigned {len(assigned)} targets: {[t for _, t in assigned]}")

    # Export helpers
    def _maybe_write(path: Path, sub):
        if path.exists() and not overwrite:
            return False
        sub.to_netcdf(path)
        return True

    n_written = 0
    for v_in, v_out in assigned:
        print(f"[EXPORT][{shard_id}/{shards}] {v_in} -> {v_out}")
        sel = ds[[v_in]].rename({v_in: v_out})

        wrote = False
        wrote |= _maybe_write(nc_root / "full" / f"{v_out}.nc", sel)

        # early split
        try:
            se = sel.sel(time=slice("1901-01-01", "1918-12-31"))
            if se.time.size:
                wrote |= _maybe_write(nc_root / "test" / "early" / f"{v_out}.nc", se)
        except Exception as e:
            print(f"[EXPORT][WARN] early subset failed for {v_in}: {e}")

        # late split
        try:
            sl = sel.sel(time=slice("2018-01-01", "2023-12-31"))
            if sl.time.size:
                wrote |= _maybe_write(nc_root / "test" / "late" / f"{v_out}.nc", sl)
        except Exception as e:
            print(f"[EXPORT][WARN] late subset failed for {v_in}: {e}")

        if wrote:
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
    "_vector_from_std_dict",
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
]