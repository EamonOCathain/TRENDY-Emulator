# src/training/carry.py

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401  (imported elsewhere)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG = logging.getLogger("carry")

# Fallback month lengths for a no-leap calendar. If the model exposes its own
# calendar, we'll use that instead.
MONTH_LENGTHS_FALLBACK = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

POPULATION_NORMALISE = True 

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _ceil_years(x: float) -> int:
    """Ceil a float horizon to an integer number of years. Safe fallback to 0."""
    try:
        return int(math.ceil(float(x)))
    except Exception:
        return 0


def _year_bounds(y0: int, y1: int) -> tuple[int, int]:
    """Return [day_start, day_end) indices for years [y0..y1) in a 365*n layout."""
    return (y0 * 365, y1 * 365)


def _slice_last_year_bounds(t: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ([month_start, month_end), [year_start, year_end)) for target year t."""
    return (t * 12, (t + 1) * 12), (t, t + 1)


def _prev_monthly_from_preds(
    preds_m: torch.Tensor,
    out_m_idx: list[int],
    granularity: str,
) -> Optional[torch.Tensor]:
    """
    From pooled monthly predictions (absolutes) return the vector to carry.

    - preds_m shape: [B, 12, nm]
    - returns: [B, len(out_m_idx)] or None

    Strategy:
      * monthly  → take last month: preds_m[:, -1, out_m_idx]
      * annual   → take annual mean: preds_m.mean(dim=1)[:, out_m_idx]
    """
    if (preds_m is None) or (not out_m_idx):
        return None
    if granularity == "annual":
        annual_mean = preds_m.mean(dim=1)  # [B, nm]
        return annual_mean[:, out_m_idx]
    return preds_m[:, -1, out_m_idx]


def _inject_monthly_carry(
    x_year: torch.Tensor,
    carry_vec: torch.Tensor,
    in_m_idx: list[int],
    first_month_len: int,
    granularity: str,
    logger: Optional[logging.Logger] = None,
):
    """
    Inject the carry vector into a year's inputs.

    Args:
      x_year:    [B, 365, nin]
      carry_vec: [B, len(in_m_idx)]
      monthly  → write only into the first month's days
      annual   → broadcast into all 365 days
    """
    logger = logger or _LOG

    if (carry_vec is None) or (not in_m_idx):
        return
    if not torch.isfinite(carry_vec).all():
        logger.warning(
            "[carry] non-finite CARRY(monthly) vector before injection (granularity=%s, shape=%s)",
            granularity, tuple(carry_vec.shape)
        )
        # Skip injection to avoid contaminating inputs
        return

    B = x_year.size(0)
    if granularity == "annual":
        _safe_index_copy(
            x_year, 2, in_m_idx,
            carry_vec.unsqueeze(1).expand(B, 365, len(in_m_idx)),
            where_len=365, logger=logger, name="monthly carry→inputs(annual)",
        )
    else:
        _safe_index_copy(
            x_year[:, :first_month_len, :], 2, in_m_idx,
            carry_vec.unsqueeze(1).expand(B, first_month_len, len(in_m_idx)),
            where_len=first_month_len, logger=logger, name="monthly carry→inputs(monthly)",
        )


def _validate_granularity(granularity: str, H: float):
    """Guard: in annual mode, horizon must be 0 or ≥ 1 year (no fractional carry)."""
    if granularity == "annual":
        if H not in (0.0,) and H < 1.0:
            raise AssertionError("In annual mode, carry_horizon must be 0 or ≥ 1 year.")


# ---------------------------------------------------------------------------
# Month helpers & pooling
# ---------------------------------------------------------------------------

def _month_id_from_bounds(bounds: list[int], device, dtype=torch.long):
    """bounds [0,31,59,...,365] → month_id: [0,0,..,0, 1,1,.., 11]"""
    month_id = torch.empty(bounds[-1], dtype=dtype, device=device)
    for m in range(12):
        s, e = bounds[m], bounds[m + 1]
        month_id[s:e] = m
    return month_id


def pool_from_daily_abs(
    y_daily_abs: torch.Tensor,
    nm: int,
    na: int,
    bounds: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pool daily absolute predictions to (monthly means, annual means).

    Args:
      y_daily_abs: [B, 365, nm+na]
      nm, na:      monthly/annual head sizes
      bounds:      month day bounds (len=13)

    Returns:
      preds_m: [B, 12, nm]  (monthly means)
      preds_a: [B,  1, na]  (annual mean)
    """
    B, D, _ = y_daily_abs.shape  # D=365
    y_m_daily = y_daily_abs[..., :nm]        # [B,365,nm]
    y_a_daily = y_daily_abs[..., nm:nm+na]   # [B,365,na]

    month_id = _month_id_from_bounds(bounds, y_daily_abs.device)      # [365]
    month_id_exp = month_id.view(1, D, 1).expand(B, D, nm)            # [B,365,nm]
    denom = torch.as_tensor(
        [bounds[i + 1] - bounds[i] for i in range(12)],
        device=y_daily_abs.device,
        dtype=y_daily_abs.dtype,
    ).view(1, 12, 1)  # [1,12,1]

    preds_m = torch.zeros(B, 12, nm, device=y_daily_abs.device, dtype=y_daily_abs.dtype)
    preds_m.scatter_add_(1, month_id_exp, y_m_daily)  # sum days→months
    preds_m = preds_m / denom                         # mean per month

    preds_a = y_a_daily.mean(dim=1, keepdim=True)     # [B,1,na]

    return preds_m, preds_a


@contextmanager
def _model_mode(model: torch.nn.Module, granularity: str, enabled: bool = True):
    """
    Temporarily switch model mode based on carry granularity and restore it.

    - "monthly" → sequential_months
    - "annual"  → batch_months
    """
    if not enabled or not hasattr(model, "set_mode"):
        yield
        return

    prev = getattr(model, "mode", None)
    target_mode = "sequential_months" if granularity == "monthly" else "batch_months"

    try:
        model.set_mode(target_mode)
        yield
    finally:
        if prev is not None:
            model.set_mode(prev)


# ---------------------------------------------------------------------------
# Index guards & safe indexed writes
# ---------------------------------------------------------------------------

def _sanitize_idx(idx_list, size, name, logger=None):
    """Clamp/validate index list against a dimension size; warn if dropping."""
    if not idx_list:
        return []
    good = [int(i) for i in idx_list
            if isinstance(i, (int, np.integer)) and 0 <= int(i) < size]
    if logger and len(good) != len(idx_list):
        bad = [i for i in idx_list if i not in good]
        logger.warning(f"[carry] Dropping invalid {name} indices {bad} for size={size}")
    return good


def _safe_index_copy(x, dim, idx_list, values, where_len=None, logger=None, name=""):
    """
    Safely copy `values` into `x` along dimension `dim` at `idx_list`.

    If idx/value width mismatches or idx is empty, silently skip (with logs).
    """
    size = x.size(dim)
    idx_list = _sanitize_idx(idx_list, size, name, logger)
    if not idx_list:
        return
    idx = torch.as_tensor(idx_list, device=x.device, dtype=torch.long)
    need = idx.numel()
    got = values.size(-1)
    if need != got:
        if logger:
            logger.error(f"[carry] Value width ({got}) != len(idx) ({need}) for {name}; skipping injection.")
        return
    x.index_copy_(dim, idx, values)

# ---------------------------------------------------------------------------
# Month metadata (slices and model-aware calendar)
# ---------------------------------------------------------------------------

def month_slices_from_lengths(month_lengths: list[int]) -> list[tuple[int, int]]:
    """Convert 12 month lengths (sum=365) into [(start, end), ...] slices."""
    assert len(month_lengths) == 12, "month_lengths must have 12 entries"
    assert sum(month_lengths) == 365, "noleap calendar expected"
    s = 0
    out = []
    for L in month_lengths:
        out.append((s, s + L))
        s += L
    return out


def get_month_metadata(model: Optional[torch.nn.Module] = None) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return (month_lengths, month_slices), using model.month_lengths if present."""
    if (model is not None) and hasattr(model, "month_lengths"):
        ml = list(getattr(model, "month_lengths"))
    else:
        ml = list(MONTH_LENGTHS_FALLBACK)
    return ml, month_slices_from_lengths(ml)


# ---------------------------------------------------------------------------
# Rollout configuration builder
# ---------------------------------------------------------------------------

def build_rollout_cfg(
    input_order: list[str],
    output_order: list[str],
    var_names: dict[str, list[str]],
    carry_horizon: float = 0.0,
) -> dict:
    """
    Build a canonical rollout configuration with indices and name lists.

    Returns keys:
      - in_monthly_state_idx, in_annual_state_idx                (INPUT space)
      - out_monthly_state_idx, out_annual_state_idx              (HEAD-LOCAL)
      - out_monthly_all_idx, out_monthly_names (head order)
      - out_annual_names     (head order)
      - month_lengths, month_slices
      - carry_horizon, tbptt_years (0), output_order (global reference)
    """
    in_idx = {n: i for i, n in enumerate(input_order)}

    monthly_states = list(var_names.get("monthly_states", []))
    annual_states = list(var_names.get("annual_states", []))
    monthly_fluxes = list(var_names.get("monthly_fluxes", []))

    # Sanity: all required variables must exist in their spaces
    def _missing(names: list[str], space: set[str]) -> list[str]:
        return [n for n in names if n not in space]

    missing = (
        _missing(monthly_states, set(input_order)) +
        _missing(annual_states, set(input_order)) +
        _missing(monthly_states, set(output_order)) +
        _missing(annual_states, set(output_order)) +
        _missing(monthly_fluxes, set(output_order))
    )
    if missing:
        raise ValueError(f"[build_rollout_cfg] some variables missing from heads: {missing}")

    # INPUT indices (against input feature dimension)
    in_monthly_state_idx = [in_idx[n] for n in monthly_states]
    in_annual_state_idx = [in_idx[n] for n in annual_states]

    # OUTPUT name lists (per head, head-local order)
    monthly_all_set = set(monthly_fluxes) | set(monthly_states)
    out_monthly_names = [n for n in output_order if n in monthly_all_set]      # len = nm
    out_annual_names = [n for n in output_order if n in set(annual_states)]    # len = na

    # name → local index maps
    out_monthly_local = {n: i for i, n in enumerate(out_monthly_names)}
    out_annual_local = {n: i for i, n in enumerate(out_annual_names)}

    # OUTPUT indices relative to heads
    out_monthly_state_idx = [out_monthly_local[n] for n in monthly_states]     # 0..nm-1
    out_annual_state_idx = [out_annual_local[n] for n in annual_states]        # 0..na-1

    # Full monthly head indices (useful for plotting, etc.)
    out_monthly_all_idx = [out_monthly_local[n] for n in out_monthly_names]    # 0..nm-1

    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_slices = month_slices_from_lengths(month_lengths)

    return {
        "in_monthly_state_idx": in_monthly_state_idx,
        "in_annual_state_idx": in_annual_state_idx,
        "out_monthly_state_idx": out_monthly_state_idx,  # HEAD-LOCAL
        "out_annual_state_idx": out_annual_state_idx,    # HEAD-LOCAL
        "out_monthly_all_idx": out_monthly_all_idx,      # HEAD-LOCAL
        "out_monthly_names": out_monthly_names,          # HEAD ORDER
        "out_annual_names": out_annual_names,            # HEAD ORDER
        "month_lengths": month_lengths,
        "month_slices": month_slices,
        "carry_horizon": float(carry_horizon),
        "tbptt_years": 0,
        "output_order": list(output_order),              # global order (reference)
    }


# ---------------------------------------------------------------------------
# Canonical rollout core (used by eval/test; can also be used with training=True)
# ---------------------------------------------------------------------------

def _rollout_core(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor,        # [nin, 365*Y, L]
    labels_m: torch.Tensor,      # [nm (or nm_all), 12*Y, L]  (normalized)
    labels_a: torch.Tensor,      # [na, 1*Y, L]               (normalized)
    device: torch.device,
    rollout_cfg: dict,
    training: bool,              # True enables autograd/backward in this core
    loss_func: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    carry_on: bool,              # True => inject cross-year carries
    return_pairs: bool = False,  # True => return dict[var] = (y, yhat)
    logger: Optional[logging.Logger] = None,
):
    """
    Roll over locations + years with optional state carry, computing loss and/or
    collecting (y, ŷ) pairs. Works in training or no-grad mode.
    """
    assert (not training) or (loss_func is not None), "loss_func required when training=True"

    # Calendar & month bounds
    month_lengths = rollout_cfg.get("month_lengths", MONTH_LENGTHS_FALLBACK)
    bounds = [0]
    for Lm in month_lengths:
        bounds.append(bounds[-1] + Lm)
    first_month_len = int(month_lengths[0])

    # Indices (raw from cfg)
    in_m_idx = rollout_cfg.get("in_monthly_state_idx", []) or []
    in_a_idx = rollout_cfg.get("in_annual_state_idx", []) or []
    out_m_idx = rollout_cfg.get("out_monthly_state_idx", []) or []
    out_a_idx = rollout_cfg.get("out_annual_state_idx", []) or []

    monthly_names = list(rollout_cfg.get("out_monthly_names", []))
    annual_names = list(rollout_cfg.get("out_annual_names", []))

    # Granularity & horizon guard
    granularity = str(rollout_cfg.get("carry_granularity", "monthly"))
    H = float(rollout_cfg.get("carry_horizon", 0.0) or 0.0)
    _validate_granularity(granularity, H)

    # Shapes / guards
    nin, Ttot, L = int(inputs.shape[0]), int(inputs.shape[1]), int(inputs.shape[2])
    Y = Ttot // 365
    if Y <= 1:
        if return_pairs:
            return {}
        return (0.0, 0) if loss_func is not None else None

    # Move once
    inputs = inputs.to(device, non_blocking=True)
    labels_m = labels_m.to(device, non_blocking=True)
    labels_a = labels_a.to(device, non_blocking=True)

    # Loss accumulators
    sum_loss = 0.0
    n_windows = 0

    # Pair buffers
    if return_pairs:
        buf_y = {n: [] for n in (monthly_names + annual_names)}
        buf_p = {n: [] for n in (monthly_names + annual_names)}

    # Toggle model mode depending on granularity (only when carrying across years)
    with _model_mode(model, granularity, enabled=carry_on), (torch.enable_grad() if training else torch.no_grad()):
        for loc in range(L):
            xb_full = inputs[:, :, loc].T.unsqueeze(0)     # [1, 365*Y, nin]
            ym_full = labels_m[:, :, loc].T.unsqueeze(0)   # [1, 12*Y,  nm]
            ya_full = labels_a[:, :, loc].T.unsqueeze(0)   # [1,   Y,   na]

            B = 1
            nm = int(ym_full.shape[-1])
            na = int(ya_full.shape[-1])

            # sanitize output indices once per location (only when we actually carry)
            if carry_on:
                out_m_idx_sane = _sanitize_idx(out_m_idx, nm, "out_monthly_state_idx", logger)
                out_a_idx_sane = _sanitize_idx(out_a_idx, na, "out_annual_state_idx", logger)
            else:
                out_m_idx_sane = []
                out_a_idx_sane = []

            prev_monthly_state = None
            prev_annual_state = None

            # Warm-up year 0 (only if carrying)
            if carry_on:
                x_prev = xb_full[:, 0:365, :]
                if not torch.isfinite(x_prev).all():
                    if logger:
                        logger.warning("[eval-core] non-finite warm-up INPUT (loc=%d, shape=%s)",
                                       loc, tuple(x_prev.shape))
                    del x_prev
                    continue

                preds_abs_daily_prev = model(x_prev)  # [1,365,nm+na]
                if not torch.isfinite(preds_abs_daily_prev).all():
                    if logger:
                        logger.warning("[eval-core] non-finite warm-up PRED (loc=%d, shape=%s)",
                                       loc, tuple(preds_abs_daily_prev.shape))
                    del preds_abs_daily_prev
                    continue

                preds_m_prev, preds_a_prev = pool_from_daily_abs(preds_abs_daily_prev, nm, na, bounds)

                out_m_idx_warm = _sanitize_idx(out_m_idx, nm, "out_monthly_state_idx", logger)
                out_a_idx_warm = _sanitize_idx(out_a_idx, na, "out_annual_state_idx", logger)

                prev_monthly_state = _prev_monthly_from_preds(preds_m_prev, out_m_idx_warm, granularity)
                if prev_monthly_state is not None:
                    if not torch.isfinite(prev_monthly_state).all():
                        if logger:
                            logger.warning("[eval-core] non-finite warm-up CARRY(monthly) (loc=%d, shape=%s)",
                                           loc, tuple(prev_monthly_state.shape))
                        prev_monthly_state = None
                    else:
                        prev_monthly_state = prev_monthly_state.detach()

                prev_annual_state = (preds_a_prev[:, 0, out_a_idx_warm] if out_a_idx_warm else None)
                if prev_annual_state is not None:
                    if not torch.isfinite(prev_annual_state).all():
                        if logger:
                            logger.warning("[eval-core] non-finite warm-up CARRY(annual) (loc=%d, shape=%s)",
                                           loc, tuple(prev_annual_state.shape))
                        prev_annual_state = None
                    else:
                        prev_annual_state = prev_annual_state.detach()

                del preds_abs_daily_prev, preds_m_prev, preds_a_prev

            # Loop over years 1..Y-1
            for y in range(1, Y):
                s_days = y * 365
                e_days = s_days + 365
                x_year = xb_full[:, s_days:e_days, :]  # [1,365,nin]

                # Inject carries
                if carry_on:
                    if prev_monthly_state is not None and in_m_idx:
                        _inject_monthly_carry(
                            x_year,
                            (prev_monthly_state if training else prev_monthly_state.detach()),
                            in_m_idx=in_m_idx,
                            first_month_len=first_month_len,
                            granularity=granularity,
                            logger=logger,
                        )
                    if prev_annual_state is not None and in_a_idx:
                        if not torch.isfinite(prev_annual_state).all():
                            if logger:
                                logger.warning("[carry] non-finite CARRY(annual) vector before injection (shape=%s)",
                                               tuple(prev_annual_state.shape))
                        else:
                            _safe_index_copy(
                                x_year, 2, in_a_idx,
                                (prev_annual_state if training else prev_annual_state.detach())
                                .unsqueeze(1).expand(B, 365, len(in_a_idx)),
                                where_len=365, logger=logger, name="annual carry→inputs",
                            )

                # INPUT guard
                if not torch.isfinite(x_year).all():
                    if logger:
                        logger.warning("[eval-core] non-finite INPUT (loc=%d, y=%d, shape=%s)",
                                       loc, y, tuple(x_year.shape))
                    del x_year
                    continue

                # Forward (daily absolutes)
                preds_abs_daily = model(x_year)  # [1,365,nm+na]
                if not torch.isfinite(preds_abs_daily).all():
                    if logger:
                        logger.warning("[eval-core] non-finite PRED (loc=%d, y=%d, x_shape=%s, pred_shape=%s)",
                                       loc, y, tuple(x_year.shape), tuple(preds_abs_daily.shape))
                    del preds_abs_daily, x_year
                    continue

                # Labels (target year y)
                yb_m = ym_full[:, y * 12:(y + 1) * 12, :]
                yb_a = ya_full[:, y:y + 1, :]

                # LABELS guard
                if (not torch.isfinite(yb_m).all()) or (not torch.isfinite(yb_a).all()):
                    if logger:
                        logger.warning("[eval-core] non-finite LABELS (loc=%d, y=%d)", loc, y)
                    del preds_abs_daily, yb_m, yb_a
                    continue

                # Loss
                if loss_func is not None:
                    loss = loss_func(preds_abs_daily, yb_m, yb_a)
                    if training:
                        loss.backward()
                    sum_loss += float(loss.detach().cpu())
                    n_windows += 1

                # (y, ŷ) pairs
                if return_pairs:
                    preds_m, preds_a = pool_from_daily_abs(preds_abs_daily, nm, na, bounds)
                    for j, name in enumerate(monthly_names):
                        buf_y[name].append(yb_m[..., j].reshape(-1).detach().cpu().numpy())
                        buf_p[name].append(preds_m[..., j].reshape(-1).detach().cpu().numpy())
                    for j, name in enumerate(annual_names):
                        buf_y[name].append(yb_a[..., j].reshape(-1).detach().cpu().numpy())
                        buf_p[name].append(preds_a[..., j].reshape(-1).detach().cpu().numpy())

                # Prepare next carry
                if carry_on:
                    preds_m, preds_a = pool_from_daily_abs(preds_abs_daily, nm, na, bounds)
                    next_m = _prev_monthly_from_preds(preds_m, out_m_idx_sane, granularity)
                    prev_monthly_state = (next_m if training else (next_m.detach() if next_m is not None else None))
                    if prev_monthly_state is not None and (not torch.isfinite(prev_monthly_state).all()):
                        if logger:
                            logger.warning("[eval-core] non-finite CARRY(monthly) after pooling (loc=%d, y=%d)", loc, y)
                        prev_monthly_state = None

                    if out_a_idx_sane:
                        a_vec = preds_a[:, 0, out_a_idx_sane]
                        prev_annual_state = (a_vec if training else a_vec.detach())
                        if prev_annual_state is not None and (not torch.isfinite(prev_annual_state).all()):
                            if logger:
                                logger.warning("[eval-core] non-finite CARRY(annual) after pooling (loc=%d, y=%d)", loc, y)
                            prev_annual_state = None
                    else:
                        prev_annual_state = None

                del preds_abs_daily, yb_m, yb_a

    # Returns
    if return_pairs:
        out = {}
        for name in (monthly_names + annual_names):
            ys = buf_y.get(name, [])
            ps = buf_p.get(name, [])
            y = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=float)
            p = np.concatenate(ps, axis=0) if ps else np.empty((0,), dtype=float)
            out[name] = (y, p)
        return out

    if loss_func is not None:
        return (sum_loss, n_windows)

    return None


# ---------------------------------------------------------------------------
# Windowed training with carry (tail-only loss)
# ---------------------------------------------------------------------------

def rollout_train_outer_batch(
    *,
    model: torch.nn.Module,
    loss_func,
    opt: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    inputs: torch.Tensor,        # [nin, 365*Y, L]   (on device)
    labels_m: torch.Tensor,      # [nm,  12*Y, L]    (on device)
    labels_a: torch.Tensor,      # [na,   1*Y, L]    (on device)
    mb_size: int,                # interpreted as *windows per microbatch*
    eff_accum: int,
    eff_clip: Optional[float],
    history,                     # History
    global_opt_step: int,
    device: torch.device,
    rollout_cfg: dict,
) -> tuple[float, int]:
    """
    Windowed carry training with tail-only loss:

      - D = ceil(carry_horizon), W = D+1 years per window
      - windows are (loc, target_year=t), t in [D ... Y-1]
      - loss is computed ONLY for the last year of each window (unique ownership)

    Supports carry_granularity in {"monthly","annual"}.
    """
    model.train()

    # Config & guards
    granularity = str(rollout_cfg.get("carry_granularity", "monthly"))
    H = float(rollout_cfg.get("carry_horizon", 0.0) or 0.0)
    _validate_granularity(granularity, H)

    nin, Ttot, L = int(inputs.shape[0]), int(inputs.shape[1]), int(inputs.shape[2])
    Y = int(labels_a.shape[1])
    if Y <= 0 or L <= 0:
        return float("inf"), global_opt_step

    # Horizon → window extents
    D = _ceil_years(H)  # dependency depth
    W = D + 1           # years per window
    if Y <= D:
        _LOG.warning(f"[carry-windowed] Skipping batch: Y={Y} <= D={D}")
        return float("inf"), global_opt_step

    # Output dims
    nm = int(labels_m.shape[0])
    na = int(labels_a.shape[0])

    # Indices (sanitize once with correct sizes)
    in_m_idx = _sanitize_idx(rollout_cfg.get("in_monthly_state_idx", []) or [], nin, "in_monthly_state_idx", _LOG)
    in_a_idx = _sanitize_idx(rollout_cfg.get("in_annual_state_idx", []) or [], nin, "in_annual_state_idx", _LOG)
    out_m_idx = _sanitize_idx(rollout_cfg.get("out_monthly_state_idx", []) or [], nm, "out_monthly_state_idx", _LOG)
    out_a_idx = _sanitize_idx(rollout_cfg.get("out_annual_state_idx", []) or [], na, "out_annual_state_idx", _LOG)

    # Month metadata
    month_lengths = rollout_cfg.get("month_lengths", MONTH_LENGTHS_FALLBACK)
    bounds = [0]
    for m in month_lengths:
        bounds.append(bounds[-1] + m)
    first_month_len = int(month_lengths[0])

    # Build the global list of windows (loc, t) with t as target year
    windows: list[tuple[int, int]] = []
    for loc in range(L):
        for t in range(D, Y):
            windows.append((loc, t))

    total_windows = len(windows)
    if total_windows == 0:
        return float("inf"), global_opt_step

    # Microbatch over windows — scale down with window length W = D+1
    base_mb = max(1, int(mb_size))
    micro = min(base_mb, total_windows)
    rollout_cfg["autotuned_mb_size_windows"] = int(micro)

    # DDP consistency: broadcast chosen micro from rank 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t_mb = torch.tensor([micro], dtype=torch.int64, device=device)
        torch.distributed.broadcast(t_mb, src=0)
        micro = int(t_mb.item())
        rollout_cfg["autotuned_mb_size_windows"] = int(micro)

    _LOG.info(
        "[carry-windowed][train][%s] D=%d, W=%d, windows_per_loc=%d, base_mb=%d, scale=%.4f, mb_size_windows=%d",
        granularity, D, W, (Y - D), base_mb, scale, micro,
    )

    running_loss_sum = 0.0
    windows_done = 0
    microbatches_done = 0
    
    

    # Helper: pool monthly/annual from DAILY ABSOLUTES
    def _pool_from_daily_abs(y_daily_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_m_daily = y_daily_abs[..., :nm]
        y_a_daily = y_daily_abs[..., nm:nm+na]
        pieces = [y_m_daily[:, bounds[i]:bounds[i + 1], :].mean(dim=1, keepdim=True) for i in range(12)]
        preds_m = torch.cat(pieces, dim=1)            # [B,12,nm]
        preds_a = y_a_daily.mean(dim=1, keepdim=True) # [B,1, na]
        return preds_m, preds_a

    # === Training loop over windowed microbatches ===
    with _model_mode(model, granularity, enabled=True):
        w_idx = 0
        while w_idx < total_windows:
            s = w_idx
            e = min(w_idx + micro, total_windows)
            w_idx = e
            B = e - s  # windows in this microbatch

            # Precompute locs/targets once per microbatch
            locs = [windows[k][0] for k in range(s, e)]
            targets = [windows[k][1] for k in range(s, e)]

            prev_m = None  # [B, len(out_m_idx)]
            prev_a = None  # [B, len(out_a_idx)]

            # Iterate years within window (granularity controls carry behavior)
            for y_off in range(W):
                y0s = [t - D + y_off for t in targets]  # absolute year index for this offset

                # Build a [B,365,nin] slice for this year across windows
                xb_list = []
                for loc, y_abs in zip(locs, y0s):
                    ds, de = _year_bounds(y_abs, y_abs + 1)
                    xb_list.append(inputs[:, ds:de, loc].T)  # [365,nin]
                xb = torch.stack(xb_list, dim=0).to(device, non_blocking=True)  # [B,365,nin]

                # Inject carries (from previous year within the window)
                if y_off > 0:
                    if (prev_m is not None) and in_m_idx:
                        _inject_monthly_carry(
                            xb, prev_m, in_m_idx=in_m_idx,
                            first_month_len=first_month_len, granularity=granularity,
                            logger=_LOG,
                        )
                    if (prev_a is not None) and in_a_idx:
                        if not torch.isfinite(prev_a).all():
                            _LOG.warning("[carry-windowed/train] non-finite CARRY(annual) pre-inject (B=%d, y_off=%d, shape=%s)",
                                         B, y_off, tuple(prev_a.shape))
                        else:
                            _safe_index_copy(
                                xb, 2, in_a_idx,
                                prev_a.unsqueeze(1).expand(B, 365, len(in_a_idx)),
                                where_len=365, name="annual carry→inputs", logger=_LOG,
                            )

                # INPUT guard (before forward)
                if not torch.isfinite(xb).all():
                    _LOG.warning("[carry-windowed/train] non-finite INPUT (B=%d, y_off=%d, shape=%s)",
                                 xb.shape[0], y_off, tuple(xb.shape))
                    del xb
                    continue

                # Forward one whole year (predictions are daily absolutes)
                preds_abs_daily = model(xb)  # [B,365,nm+na]

                # PRED guard
                if not torch.isfinite(preds_abs_daily).all():
                    _LOG.warning("[carry-windowed/train] non-finite PRED (B=%d, y_off=%d, x_shape=%s, pred_shape=%s)",
                                 preds_abs_daily.shape[0], y_off, tuple(xb.shape), tuple(preds_abs_daily.shape))
                    del xb, preds_abs_daily
                    continue

                # Pool to month/annual and prep next prev_* from pooled absolutes
                preds_m, preds_a = _pool_from_daily_abs(preds_abs_daily)  # [B,12,nm], [B,1,na]
                next_m = _prev_monthly_from_preds(preds_m, out_m_idx, granularity)
                prev_m = (next_m.detach() if next_m is not None else None)
                prev_a = (preds_a[:, 0, out_a_idx].detach() if out_a_idx else None)

                # Tail-year loss (y_off == W-1)
                if y_off == W - 1:
                    ybm_list, yba_list = [], []
                    for loc, t in zip(locs, targets):
                        (ms, me), (ys, ye) = _slice_last_year_bounds(t)
                        ybm_list.append(labels_m[:, ms:me, loc].T)  # [12,nm]
                        yba_list.append(labels_a[:, ys:ye, loc].T)  # [1,na]
                    ybm = torch.stack(ybm_list, dim=0).to(device, non_blocking=True)  # [B,12,nm]
                    yba = torch.stack(yba_list, dim=0).to(device, non_blocking=True)  # [B,1, na]

                    # LABEL guards
                    if (not torch.isfinite(ybm).all()) or (not torch.isfinite(yba).all()):
                        _LOG.warning("[carry-windowed/train] non-finite LABELS (B=%d)", B)
                        del xb, preds_abs_daily, preds_m, preds_a, ybm, yba
                        continue

                    loss = loss_func(preds_abs_daily, ybm, yba)
                    (loss / eff_accum).backward()
                    microbatches_done += 1
                    batch_weight = B if POPULATION_NORMALISE else 1
                    windows_done     += batch_weight
                    running_loss_sum += float(loss.detach().cpu()) * batch_weight

                    # Optimizer step (gradient accumulation aware)
                    if microbatches_done % eff_accum == 0:
                        if eff_clip is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), eff_clip)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        if scheduler is not None:
                            scheduler.step()
                        history.lr_values.append(float(opt.param_groups[0]["lr"]))
                        history.lr_steps.append(global_opt_step)
                        global_opt_step += 1

                # free
                del xb, preds_abs_daily, preds_m, preds_a

        # Flush remainder if gradients are pending
        if microbatches_done % eff_accum != 0:
            if eff_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), eff_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            history.lr_values.append(float(opt.param_groups[0]["lr"]))
            history.lr_steps.append(global_opt_step)
            global_opt_step += 1

    avg_loss = running_loss_sum / max(1, windows_done)
    return avg_loss, global_opt_step


# ---------------------------------------------------------------------------
# Windowed evaluation with carry (tail-only loss)
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_eval_outer_batch(
    *,
    model: torch.nn.Module,
    loss_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,        # [nin, 365*Y, L]
    labels_m: torch.Tensor,      # [nm,  12*Y, L]
    labels_a: torch.Tensor,      # [na,   1*Y, L]
    device: torch.device,
    rollout_cfg: dict,
    mb_size: Optional[int] = None,
) -> tuple[float, int]:
    """
    Windowed carry evaluation with tail-only loss; now micro-batches over windows.
    Supports carry_granularity in {"monthly","annual"}.
    """
    model.eval()

    # Config & guards
    granularity = str(rollout_cfg.get("carry_granularity", "monthly"))
    H = float(rollout_cfg.get("carry_horizon", 0.0) or 0.0)
    _validate_granularity(granularity, H)

    nin, Ttot, L = int(inputs.shape[0]), int(inputs.shape[1]), int(inputs.shape[2])
    Y = int(labels_a.shape[1])

    D = _ceil_years(H)
    W = D + 1
    if Y <= D or L <= 0:
        return 0.0, 0

    # Dims and indices (sanitize once)
    nm = int(labels_m.shape[0])
    na = int(labels_a.shape[0])
    in_m_idx  = _sanitize_idx(rollout_cfg.get("in_monthly_state_idx", []) or [], nin, "in_monthly_state_idx", _LOG)
    in_a_idx  = _sanitize_idx(rollout_cfg.get("in_annual_state_idx", [])  or [], nin, "in_annual_state_idx",  _LOG)
    out_m_idx = _sanitize_idx(rollout_cfg.get("out_monthly_state_idx", []) or [], nm,  "out_monthly_state_idx", _LOG)
    out_a_idx = _sanitize_idx(rollout_cfg.get("out_annual_state_idx", [])  or [], na,  "out_annual_state_idx",  _LOG)

    # Month metadata
    month_lengths = rollout_cfg.get("month_lengths", MONTH_LENGTHS_FALLBACK)
    bounds = [0]
    for m in month_lengths:
        bounds.append(bounds[-1] + m)
    first_month_len = int(month_lengths[0])

    # Helper for pooling monthly/annual from DAILY ABSOLUTES
    def _pool_from_daily_abs(y_daily_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_m_daily = y_daily_abs[..., :nm]
        y_a_daily = y_daily_abs[..., nm:nm+na]
        pieces = [y_m_daily[:, bounds[i]:bounds[i + 1], :].mean(dim=1, keepdim=True) for i in range(12)]
        preds_m = torch.cat(pieces, dim=1)            # [B,12,nm]
        preds_a = y_a_daily.mean(dim=1, keepdim=True) # [B,1, na]
        return preds_m, preds_a

    # Build global list of evaluation windows (loc, target_year=t)
    windows: list[tuple[int, int]] = []
    for loc in range(L):
        for t in range(D, Y):
            windows.append((loc, t))

    total_windows = len(windows)
    if total_windows == 0:
        return 0.0, 0

    # Choose microbatch size in *windows* (scale by window length to keep memory reasonable)
    base_mb = int(mb_size) if (mb_size is not None and mb_size > 0) else 2048
    micro   = min(max(1, base_mb), total_windows)

    # DDP consistency (if active): broadcast chosen micro
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t_mb = torch.tensor([micro], dtype=torch.int64, device=device)
        torch.distributed.broadcast(t_mb, src=0)
        micro = int(t_mb.item())

    _LOG.info(
        "[carry-windowed][eval][%s] D=%d, W=%d, windows_per_loc=%d, mb_size_windows=%d",
        granularity, D, W, (Y - D), micro
    )

    total_loss = 0.0
    windows_done = 0

    # === Evaluation over windowed microbatches ===
    with _model_mode(model, granularity, enabled=True):
        w_idx = 0
        while w_idx < total_windows:
            s = w_idx
            e = min(w_idx + micro, total_windows)
            w_idx = e
            B = e - s  # windows in this microbatch

            # Precompute locs/targets once per microbatch
            locs    = [windows[k][0] for k in range(s, e)]
            targets = [windows[k][1] for k in range(s, e)]

            prev_m = None  # [B, len(out_m_idx)]   carry vector (monthly)
            prev_a = None  # [B, len(out_a_idx)]   carry vector (annual)

            # Iterate years within the window [t-D .. t]
            for y_off in range(W):
                y0s = [t - D + y_off for t in targets]  # absolute year index for this offset

                # Build a [B,365,nin] tensor for this year across windows
                xb_list = []
                for loc, y_abs in zip(locs, y0s):
                    ds, de = _year_bounds(y_abs, y_abs + 1)
                    xb_list.append(inputs[:, ds:de, loc].T)  # [365,nin]
                xb = torch.stack(xb_list, dim=0).to(device, non_blocking=True)  # [B,365,nin]

                # Inject carries from previous year within the window
                if y_off > 0:
                    if (prev_m is not None) and in_m_idx:
                        _inject_monthly_carry(
                            xb, prev_m, in_m_idx=in_m_idx,
                            first_month_len=first_month_len, granularity=granularity,
                            logger=_LOG,
                        )
                    if (prev_a is not None) and in_a_idx:
                        if not torch.isfinite(prev_a).all():
                            _LOG.warning("[carry-windowed/eval] non-finite CARRY(annual) pre-inject (B=%d, y_off=%d, shape=%s)",
                                         B, y_off, tuple(prev_a.shape))
                        else:
                            _safe_index_copy(
                                xb, 2, in_a_idx,
                                prev_a.unsqueeze(1).expand(B, 365, len(in_a_idx)),
                                where_len=365, name="annual carry→inputs", logger=_LOG,
                            )

                # INPUT guard
                if not torch.isfinite(xb).all():
                    _LOG.warning("[carry-windowed/eval] non-finite INPUT (B=%d, y_off=%d, shape=%s)",
                                 xb.shape[0], y_off, tuple(xb.shape))
                    del xb
                    continue

                # Forward one whole year (daily absolutes)
                preds_abs_daily = model(xb)  # [B,365,nm+na]

                # PRED guard
                if not torch.isfinite(preds_abs_daily).all():
                    _LOG.warning("[carry-windowed/eval] non-finite PRED (B=%d, y_off=%d, x_shape=%s, pred_shape=%s)",
                                 preds_abs_daily.shape[0], y_off, tuple(xb.shape), tuple(preds_abs_daily.shape))
                    del xb, preds_abs_daily
                    continue

                # Pool to month/annual and prep next prev_* from pooled absolutes
                preds_m, preds_a = _pool_from_daily_abs(preds_abs_daily)  # [B,12,nm], [B,1,na]
                next_m = _prev_monthly_from_preds(preds_m, out_m_idx, granularity)
                prev_m = (next_m if next_m is not None else None)  # already no-grad
                prev_a = (preds_a[:, 0, out_a_idx] if out_a_idx else None)

                # Tail-year loss when y_off == W-1 (i.e., target year)
                if y_off == W - 1:
                    # Build labels for each window in this microbatch
                    ybm_list, yba_list = [], []
                    for loc, t in zip(locs, targets):
                        (ms, me), (ys, ye) = _slice_last_year_bounds(t)
                        ybm_list.append(labels_m[:, ms:me, loc].T)  # [12,nm]
                        yba_list.append(labels_a[:, ys:ye, loc].T)  # [1, na]
                    ybm = torch.stack(ybm_list, dim=0).to(device, non_blocking=True)  # [B,12,nm]
                    yba = torch.stack(yba_list, dim=0).to(device, non_blocking=True)  # [B,1, na]

                    if (not torch.isfinite(ybm).all()) or (not torch.isfinite(yba).all()):
                        _LOG.warning("[carry-windowed/eval] non-finite LABELS (B=%d)", B)
                        del xb, preds_abs_daily, preds_m, preds_a, ybm, yba
                        continue

                    loss = loss_func(preds_abs_daily, ybm, yba)
                    batch_weight = B if POPULATION_NORMALISE else 1
                    total_loss   += float(loss.detach().cpu()) * batch_weight
                    windows_done += batch_weight

                # free
                del xb, preds_abs_daily, preds_m, preds_a

    return total_loss, windows_done

# ---------------------------------------------------------------------------
# gather (y, ŷ) pairs for metrics/plots with the same carry behavior
# ---------------------------------------------------------------------------
@torch.no_grad()
def gather_pred_label_pairs(
    *,
    model: torch.nn.Module,
    test_dl,
    device: torch.device,
    rollout_cfg: dict,
    eval_mode: str = "teacher_forced",            # NEW: {"teacher_forced","full_sequence","windowed_tail_only"}
    mb_size: Optional[int] = None,                # windows per microbatch (used only for windowed_tail_only)
    max_points_per_var: int | None = 2_000_000,   # optional downsampling cap per variable
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Collect {var_name: (y_norm, yhat_norm)} pairs using one of three explicit modes:

      - teacher_forced:
          No carry at all. Evaluate each year independently (including year 0).
      - full_sequence:
          Warm-up on year 0, then carry predictions forward and evaluate every target year 1..Y-1.
      - windowed_tail_only:
          Existing windowed semantics using carry_horizon (ceil→D), collect pairs only for tail year.

    Returns:
      dict[var_name] = (y_concat, yhat_concat) in NORMALIZED space.
    """
    assert eval_mode in {"teacher_forced", "full_sequence", "windowed_tail_only"}, \
        f"Unsupported eval_mode={eval_mode}"

    model.eval()

    # Names (head order)
    monthly_names = list(rollout_cfg.get("out_monthly_names", []))
    annual_names  = list(rollout_cfg.get("out_annual_names", []))

    # Buffers for concatenation
    buf = {n: ([], []) for n in (monthly_names + annual_names)}  # var -> (list[y], list[yhat])

    for batch_inputs, batch_monthly, batch_annual in test_dl:
        # [nin, 365*Y, L], [nm, 12*Y, L], [na, Y, L] in NORMALIZED units
        inputs   = batch_inputs.squeeze(0).float().to(device, non_blocking=True)
        labels_m = batch_monthly.squeeze(0).float().to(device, non_blocking=True)
        labels_a = batch_annual.squeeze(0).float().to(device, non_blocking=True)

        nin, Ttot, L = int(inputs.shape[0]), int(inputs.shape[1]), int(inputs.shape[2])
        Y = int(labels_a.shape[1])
        if L <= 0 or Y <= 0:
            continue

        nm = int(labels_m.shape[0])
        na = int(labels_a.shape[0])

        # Indices (sanitize per outer batch)
        in_m_idx  = _sanitize_idx(rollout_cfg.get("in_monthly_state_idx", []) or [], nin, "in_monthly_state_idx", _LOG)
        in_a_idx  = _sanitize_idx(rollout_cfg.get("in_annual_state_idx", [])  or [], nin, "in_annual_state_idx",  _LOG)
        out_m_idx = _sanitize_idx(rollout_cfg.get("out_monthly_state_idx", []) or [], nm,  "out_monthly_state_idx", _LOG)
        out_a_idx = _sanitize_idx(rollout_cfg.get("out_annual_state_idx", [])  or [], na,  "out_annual_state_idx",  _LOG)

        # Month metadata
        month_lengths = rollout_cfg.get("month_lengths", MONTH_LENGTHS_FALLBACK)
        bounds = [0]
        for m in month_lengths:
            bounds.append(bounds[-1] + m)
        first_month_len = int(month_lengths[0])

        # Helper: pool M/A from daily ABSOLUTES
        def _pool_from_daily_abs(y_daily_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            y_m_daily = y_daily_abs[..., :nm]
            y_a_daily = y_daily_abs[..., nm:nm+na]
            pieces = [y_m_daily[:, bounds[i]:bounds[i + 1], :].mean(dim=1, keepdim=True) for i in range(12)]
            preds_m = torch.cat(pieces, dim=1)            # [B,12,nm]
            preds_a = y_a_daily.mean(dim=1, keepdim=True) # [B,1, na]
            return preds_m, preds_a

        # ---- MODE 1: teacher_forced (no carry, evaluate ALL years including year 0) ----
        if eval_mode == "teacher_forced":
            # We can microbatch per (loc,year) by stacking across locs for each year.
            for y in range(Y):
                # Build [L,365,nin] (one year slice across all locs)
                xb_list = []
                for loc in range(L):
                    ds, de = _year_bounds(y, y + 1)
                    xb_list.append(inputs[:, ds:de, loc].T)  # [365,nin]
                xb = torch.stack(xb_list, dim=0)  # [L,365,nin]

                if not torch.isfinite(xb).all():
                    _LOG.warning("[gather/TF] non-finite INPUT at year=%d; skipping this year", y)
                    del xb
                    continue

                preds_abs_daily = model(xb)  # [L,365,nm+na]
                if not torch.isfinite(preds_abs_daily).all():
                    _LOG.warning("[gather/TF] non-finite PRED at year=%d; skipping this year", y)
                    del xb, preds_abs_daily
                    continue

                preds_m, preds_a = _pool_from_daily_abs(preds_abs_daily)  # [L,12,nm], [L,1,na]

                # Year-specific labels
                ybm_list, yba_list = [], []
                (ms, me), (ys, ye) = _slice_last_year_bounds(y)
                for loc in range(L):
                    ybm_list.append(labels_m[:, ms:me, loc].T)  # [12,nm]
                    yba_list.append(labels_a[:, ys:ye, loc].T)  # [1, na]
                ybm = torch.stack(ybm_list, dim=0)  # [L,12,nm]
                yba = torch.stack(yba_list, dim=0)  # [L, 1, na]

                if not (torch.isfinite(ybm).all() and torch.isfinite(yba).all()):
                    _LOG.warning("[gather/TF] non-finite LABELS at year=%d; skipping this year", y)
                    del xb, preds_abs_daily, preds_m, preds_a, ybm, yba
                    continue

                # Append per-variable (flatten [L,12] or [L,1])
                for j, name in enumerate(monthly_names):
                    y_true = ybm[..., j].reshape(-1).detach().cpu().numpy()
                    y_pred = preds_m[..., j].reshape(-1).detach().cpu().numpy()
                    buf[name][0].append(y_true); buf[name][1].append(y_pred)
                for j, name in enumerate(annual_names):
                    y_true = yba[..., j].reshape(-1).detach().cpu().numpy()
                    y_pred = preds_a[..., j].reshape(-1).detach().cpu().numpy()
                    buf[name][0].append(y_true); buf[name][1].append(y_pred)

                del xb, preds_abs_daily, preds_m, preds_a, ybm, yba

            continue  # next outer batch

        # ---- MODE 2: full_sequence (warm-up year 0, then carry and evaluate all years 1..Y-1) ----
        if eval_mode == "full_sequence":
            # Process one location at a time (keeps the carry chain correct per loc)
            with _model_mode(model, str(rollout_cfg.get("carry_granularity", "monthly")), enabled=True):
                for loc in range(L):
                    # Warm-up on year 0
                    ds0, de0 = _year_bounds(0, 1)
                    xb0 = inputs[:, ds0:de0, loc].T.unsqueeze(0)  # [1,365,nin]
                    if not torch.isfinite(xb0).all():
                        _LOG.warning("[gather/FS] non-finite warm-up INPUT (loc=%d)", loc)
                        del xb0
                        continue
                    preds0 = model(xb0)                              # [1,365,nm+na]
                    if not torch.isfinite(preds0).all():
                        _LOG.warning("[gather/FS] non-finite warm-up PRED (loc=%d)", loc)
                        del xb0, preds0
                        continue
                    pm0, pa0 = _pool_from_daily_abs(preds0)          # [1,12,nm], [1,1,na]

                    # Prepare carry vectors
                    next_m = _prev_monthly_from_preds(pm0, out_m_idx, str(rollout_cfg.get("carry_granularity","monthly")))
                    prev_m = (next_m if next_m is not None else None)
                    prev_a = (pa0[:, 0, out_a_idx] if out_a_idx else None)
                    del xb0, preds0, pm0, pa0

                    # Evaluate years 1..Y-1 with injected carry from previous year
                    for y in range(1, Y):
                        ds, de = _year_bounds(y, y + 1)
                        xb = inputs[:, ds:de, loc].T.unsqueeze(0)  # [1,365,nin]

                        # Inject carries
                        if (prev_m is not None) and in_m_idx:
                            _inject_monthly_carry(
                                xb, prev_m, in_m_idx=in_m_idx,
                                first_month_len=first_month_len,
                                granularity=str(rollout_cfg.get("carry_granularity", "monthly")),
                                logger=_LOG,
                            )
                        if (prev_a is not None) and in_a_idx:
                            _safe_index_copy(
                                xb, 2, in_a_idx,
                                prev_a.unsqueeze(1).expand(1, 365, len(in_a_idx)),
                                where_len=365, logger=_LOG, name="annual carry→inputs",
                            )

                        # Guards
                        if not torch.isfinite(xb).all():
                            _LOG.warning("[gather/FS] non-finite INPUT (loc=%d, year=%d)", loc, y)
                            del xb
                            # carry remains from previous step; continue sequence
                            continue

                        preds = model(xb)  # [1,365,nm+na]
                        if not torch.isfinite(preds).all():
                            _LOG.warning("[gather/FS] non-finite PRED (loc=%d, year=%d)", loc, y)
                            del xb, preds
                            continue

                        pm, pa = _pool_from_daily_abs(preds)  # [1,12,nm], [1,1,na]

                        # Labels for this (loc, year)
                        (ms, me), (ys, ye) = _slice_last_year_bounds(y)
                        ybm = labels_m[:, ms:me, loc].T.unsqueeze(0)  # [1,12,nm]
                        yba = labels_a[:, ys:ye, loc].T.unsqueeze(0)  # [1, 1,na]

                        if torch.isfinite(ybm).all() and torch.isfinite(yba).all():
                            for j, name in enumerate(monthly_names):
                                y_true = ybm[..., j].reshape(-1).detach().cpu().numpy()
                                y_pred = pm[..., j].reshape(-1).detach().cpu().numpy()
                                buf[name][0].append(y_true); buf[name][1].append(y_pred)
                            for j, name in enumerate(annual_names):
                                y_true = yba[..., j].reshape(-1).detach().cpu().numpy()
                                y_pred = pa[..., j].reshape(-1).detach().cpu().numpy()
                                buf[name][0].append(y_true); buf[name][1].append(y_pred)
                        else:
                            _LOG.warning("[gather/FS] non-finite LABELS (loc=%d, year=%d) — skipped", loc, y)

                        # Update carry for next year
                        next_m = _prev_monthly_from_preds(pm, out_m_idx, str(rollout_cfg.get("carry_granularity","monthly")))
                        prev_m = (next_m if next_m is not None else None)
                        prev_a = (pa[:, 0, out_a_idx] if out_a_idx else None)

                        del xb, preds, pm, pa, ybm, yba

            continue  # next outer batch

        # ---- MODE 3: windowed_tail_only (existing semantics; pairs only on tail year) ----
        # Use your current micro-batched windowed loop (D = ceil(H); W = D+1)
        H = float(rollout_cfg.get("carry_horizon", 0.0) or 0.0)
        D = _ceil_years(H)
        W = D + 1

        # Build windows = all (loc, t) with t in [D..Y-1]
        windows: list[tuple[int, int]] = []
        for loc in range(L):
            for t in range(D, Y):
                windows.append((loc, t))
        if not windows:
            continue

        base_mb = int(mb_size) if (mb_size is not None and mb_size > 0) else 2048
        micro   = min(max(1, base_mb), len(windows))

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t_mb = torch.tensor([micro], dtype=torch.int64, device=device)
            torch.distributed.broadcast(t_mb, src=0)
            micro = int(t_mb.item())

        with _model_mode(model, str(rollout_cfg.get("carry_granularity", "monthly")), enabled=True):
            w_idx = 0
            while w_idx < len(windows):
                s = w_idx
                e = min(w_idx + micro, len(windows))
                w_idx = e
                B = e - s

                locs    = [windows[k][0] for k in range(s, e)]
                targets = [windows[k][1] for k in range(s, e)]

                prev_m = None
                prev_a = None

                for y_off in range(W):
                    y0s = [t - D + y_off for t in targets]

                    xb_list = []
                    for loc, y_abs in zip(locs, y0s):
                        ds, de = _year_bounds(y_abs, y_abs + 1)
                        xb_list.append(inputs[:, ds:de, loc].T)
                    xb = torch.stack(xb_list, dim=0)  # [B,365,nin]

                    # Inject carries
                    if y_off > 0:
                        if (prev_m is not None) and in_m_idx:
                            _inject_monthly_carry(
                                xb, prev_m, in_m_idx=in_m_idx,
                                first_month_len=first_month_len,
                                granularity=str(rollout_cfg.get("carry_granularity", "monthly")),
                                logger=_LOG,
                            )
                        if (prev_a is not None) and in_a_idx:
                            _safe_index_copy(
                                xb, 2, in_a_idx,
                                prev_a.unsqueeze(1).expand(B, 365, len(in_a_idx)),
                                where_len=365, logger=_LOG, name="annual carry→inputs",
                            )

                    # Guards
                    if not torch.isfinite(xb).all():
                        _LOG.warning("[gather/WIN] non-finite INPUT (B=%d, y_off=%d)", B, y_off)
                        del xb
                        continue

                    preds_abs_daily = model(xb)
                    if not torch.isfinite(preds_abs_daily).all():
                        _LOG.warning("[gather/WIN] non-finite PRED (B=%d, y_off=%d)", B, y_off)
                        del xb, preds_abs_daily
                        continue

                    pm, pa = _pool_from_daily_abs(preds_abs_daily)

                    # Prepare next carry
                    next_m = _prev_monthly_from_preds(pm, out_m_idx, str(rollout_cfg.get("carry_granularity","monthly")))
                    prev_m = (next_m if next_m is not None else None)
                    prev_a = (pa[:, 0, out_a_idx] if out_a_idx else None)

                    # Tail-year only → collect pairs
                    if y_off == W - 1:
                        ybm_list, yba_list = [], []
                        for loc, t in zip(locs, targets):
                            (ms, me), (ys, ye) = _slice_last_year_bounds(t)
                            ybm_list.append(labels_m[:, ms:me, loc].T)  # [12,nm]
                            yba_list.append(labels_a[:, ys:ye, loc].T)  # [1, na]
                        ybm = torch.stack(ybm_list, dim=0)
                        yba = torch.stack(yba_list, dim=0)

                        if torch.isfinite(ybm).all() and torch.isfinite(yba).all():
                            for j, name in enumerate(monthly_names):
                                y_true = ybm[..., j].reshape(-1).detach().cpu().numpy()
                                y_pred = pm[..., j].reshape(-1).detach().cpu().numpy()
                                buf[name][0].append(y_true); buf[name][1].append(y_pred)
                            for j, name in enumerate(annual_names):
                                y_true = yba[..., j].reshape(-1).detach().cpu().numpy()
                                y_pred = pa[..., j].reshape(-1).detach().cpu().numpy()
                                buf[name][0].append(y_true); buf[name][1].append(y_pred)
                        else:
                            _LOG.warning("[gather/WIN] non-finite LABELS (B=%d)", B)

                    del xb, preds_abs_daily, pm, pa

    # Concatenate & optional downsample per variable
    rng = np.random.default_rng(123)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k, (ys, ps) in buf.items():
        y = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=float)
        p = np.concatenate(ps, axis=0) if ps else np.empty((0,), dtype=float)
        n = y.size
        if (max_points_per_var is not None) and (n > max_points_per_var):
            idx = rng.choice(n, size=max_points_per_var, replace=False)
            y = y[idx]; p = p[idx]
        out[k] = (y, p)
    return out