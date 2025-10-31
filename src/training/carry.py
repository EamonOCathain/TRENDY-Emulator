# src/training/carry.py

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple, Dict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG = logging.getLogger("carry")

# Noleap month lengths
MONTH_LENGTHS_FALLBACK = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# If True, weight window losses by microbatch size (population-normalised)
POPULATION_NORMALISE = True

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _year_bounds(y0: int, y1: int) -> tuple[int, int]:
    """Return [day_start, day_end) indices for years [y0..y1) in a 365*n layout."""
    return (y0 * 365, y1 * 365)


def _slice_last_year_bounds(t: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ([month_start, month_end), [year_start, year_end)) for target year t."""
    return (t * 12, (t + 1) * 12), (t, t + 1)


def _month_id_from_bounds(bounds: List[int], device, dtype=torch.long):
    """bounds [0,31,59,...,365] → month_id: [0,0,.., 1,1,.., 11]"""
    month_id = torch.empty(bounds[-1], dtype=dtype, device=device)
    for m in range(12):
        s, e = bounds[m], bounds[m + 1]
        month_id[s:e] = m
    return month_id


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


def month_slices_from_lengths(month_lengths: List[int]) -> List[tuple[int, int]]:
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
# Pooling & carry-vector builders
# ---------------------------------------------------------------------------

def pool_from_daily_abs(
    y_daily_abs: torch.Tensor,
    nm: int,
    na: int,
    bounds: List[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pool daily absolute predictions to (monthly means, annual means).
      y_daily_abs: [B, 365, nm+na]
      Returns: preds_m [B,12,nm], preds_a [B,1,na]
    """
    B, D, _ = y_daily_abs.shape
    y_m_daily = y_daily_abs[..., :nm]        # [B,365,nm]
    y_a_daily = y_daily_abs[..., nm:nm+na]   # [B,365,na]

    month_id = _month_id_from_bounds(bounds, y_daily_abs.device)      # [365]
    month_id_exp = month_id.view(1, D, 1).expand(B, D, nm)            # [B,365,nm]
    denom = torch.as_tensor(
        [bounds[i + 1] - bounds[i] for i in range(12)],
        device=y_daily_abs.device, dtype=y_daily_abs.dtype
    ).view(1, 12, 1)

    preds_m = torch.zeros(B, 12, nm, device=y_daily_abs.device, dtype=y_daily_abs.dtype)
    preds_m.scatter_add_(1, month_id_exp, y_m_daily)  # sum days→months
    preds_m = preds_m / denom

    preds_a = y_a_daily.mean(dim=1, keepdim=True)     # [B,1,na]
    return preds_m, preds_a


def _prev_monthly_from_preds(
    preds_m: torch.Tensor,
    out_m_idx: List[int],
) -> Optional[torch.Tensor]:
    """From monthly (absolutes), take December (m=11) for the carry vector. Returns [B, len(out_m_idx)]."""
    if (preds_m is None) or (not out_m_idx):
        return None
    return preds_m[:, -1, out_m_idx]


# ---------------------------------------------------------------------------
# Mode forcing (always sequential for this file)
# ---------------------------------------------------------------------------

@contextmanager
def _model_mode(model: torch.nn.Module, enabled: bool):
    """
    Force sequential months while inside the context when enabled=True.
    Restores previous mode on exit. No-op if model has no set_mode().
    """
    if not hasattr(model, "set_mode"):
        yield
        return
    prev = getattr(model, "mode", None)
    try:
        if enabled:
            model.set_mode("sequential_months")
        yield
    finally:
        if prev is not None:
            model.set_mode(prev)


# ---------------------------------------------------------------------------
# Carry injection helpers
# ---------------------------------------------------------------------------

def _inject_monthly_carry(
    x_year: torch.Tensor,
    carry_vec: torch.Tensor,
    in_m_idx: List[int],
    first_month_len: int,
    logger: Optional[logging.Logger] = None,
):
    """
    Inject carried monthly state into the next year's inputs.
    - Write into January only (0:first_month_len).
    - Force NaN for these same features for Feb–Dec so any accidental use errors.

      x_year:    [B, 365, nin]
      carry_vec: [B, len(in_m_idx)]
    """
    logger = logger or _LOG
    if (carry_vec is None) or (not in_m_idx):
        return
    if not torch.isfinite(carry_vec).all():
        logger.warning("[carry] non-finite monthly carry vector; skipping injection.")
        return

    B = x_year.size(0)

    # 1) Inject January values
    _safe_index_copy(
        x_year[:, :first_month_len, :],  # January days only
        2, in_m_idx,
        carry_vec.unsqueeze(1).expand(B, first_month_len, len(in_m_idx)),
        where_len=first_month_len, logger=logger, name="monthly carry→inputs(january-only)",
    )

    # 2) Poison the rest of the year for these features with NaN
    try:
        idx = torch.as_tensor(in_m_idx, device=x_year.device, dtype=torch.long)
        x_year[:, first_month_len:, :].index_fill_(2, idx, float("nan"))
    except Exception as e:
        logger.error("[carry] failed to set NaNs for post-January monthly-carry inputs: %s", e)


def _isfinite_except_monthly_carry_mask(
    x_year: torch.Tensor,
    in_m_idx: List[int],
    first_month_len: int,
) -> bool:
    """
    Return True iff x_year is finite, except allow NaNs for the *monthly-carry*
    input features (in_m_idx) on days after January (first_month_len..364).
    Everything else must be finite.
    """
    if not in_m_idx:
        return torch.isfinite(x_year).all().item()

    finite = torch.isfinite(x_year)
    B, D, N = x_year.shape
    allowed = torch.zeros((B, D, N), dtype=torch.bool, device=x_year.device)
    if first_month_len < D:
        idx = torch.as_tensor(in_m_idx, device=x_year.device, dtype=torch.long)
        allowed[:, first_month_len:, :].index_fill_(2, idx, True)
    ok = finite | allowed
    return ok.all().item()


# ---------------------------------------------------------------------------
# Rollout configuration builder
# ---------------------------------------------------------------------------

def build_rollout_cfg(
    input_order: List[str],
    output_order: List[str],
    var_names: Dict[str, List[str]],
    carry_horizon: int = 0,   # ALWAYS an int now
) -> dict:
    """
    Build a canonical rollout configuration with indices and name lists.

    Returns keys:
      - in_monthly_state_idx, in_annual_state_idx                (INPUT space)
      - out_monthly_state_idx, out_annual_state_idx              (HEAD-LOCAL)
      - out_monthly_all_idx, out_monthly_names (head order)
      - out_annual_names     (head order)
      - month_lengths, month_slices
      - carry_horizon (int), tbptt_years (0), output_order (global reference)
    """
    in_idx = {n: i for i, n in enumerate(input_order)}

    monthly_states = list(var_names.get("monthly_states", []))
    annual_states  = list(var_names.get("annual_states", []))
    monthly_fluxes = list(var_names.get("monthly_fluxes", []))

    def _missing(names: List[str], space: set[str]) -> List[str]:
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
    in_annual_state_idx  = [in_idx[n] for n in annual_states]

    # OUTPUT name lists (per head, head-local order)
    monthly_all_set  = set(monthly_fluxes) | set(monthly_states)
    out_monthly_names = [n for n in output_order if n in monthly_all_set]   # len = nm
    out_annual_names  = [n for n in output_order if n in set(annual_states)]# len = na

    # name → local index
    out_monthly_local = {n: i for i, n in enumerate(out_monthly_names)}
    out_annual_local  = {n: i for i, n in enumerate(out_annual_names)}

    # OUTPUT indices relative to heads
    out_monthly_state_idx = [out_monthly_local[n] for n in monthly_states]  # 0..nm-1
    out_annual_state_idx  = [out_annual_local[n] for n in annual_states]    # 0..na-1

    # Full monthly head indices
    out_monthly_all_idx = [out_monthly_local[n] for n in out_monthly_names] # 0..nm-1

    month_lengths = list(MONTH_LENGTHS_FALLBACK)
    month_slices  = month_slices_from_lengths(month_lengths)

    return {
        "in_monthly_state_idx": in_monthly_state_idx,
        "in_annual_state_idx":  in_annual_state_idx,
        "out_monthly_state_idx": out_monthly_state_idx,   # HEAD-LOCAL
        "out_annual_state_idx":  out_annual_state_idx,    # HEAD-LOCAL
        "out_monthly_all_idx":   out_monthly_all_idx,     # HEAD-LOCAL
        "out_monthly_names":     out_monthly_names,       # HEAD ORDER
        "out_annual_names":      out_annual_names,        # HEAD ORDER
        "month_lengths":         month_lengths,
        "month_slices":          month_slices,
        "carry_horizon":         int(carry_horizon or 0), # ensure int
        "tbptt_years":           0,
        "output_order":          list(output_order),      # global reference (for sanity checks)
    }

# ---------------------------------------------------------------------------
# Windowed training with carry (tail-only loss)
# ---------------------------------------------------------------------------

def rollout_outer_batch(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor,        # [nin, 365*Y, L]   (on device or will be moved)
    labels_m: torch.Tensor,      # [nm,  12*Y, L]    (on device or will be moved)
    labels_a: torch.Tensor,      # [na,   1*Y, L]    (on device or will be moved)
    device: torch.device,
    rollout_cfg: dict,
    # training controls (set training=False to run "eval" mode)
    training: bool = False,
    # NOTE: loss may accept extra_daily kw (optional)
    loss_func: Optional[Callable[..., torch.Tensor]] = None,
    opt: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    mb_size: Optional[int] = None,      # windows per microbatch
    eff_accum: int = 1,
    eff_clip: Optional[float] = None,
    history=None,                       # History or None
    global_opt_step: int = 0,
    logger: Optional[logging.Logger] = None,
) -> tuple[float, int, int, Dict[str, float]]:
    """
    Unified windowed carry loop (training or eval):
      - carry_horizon = int(H) ; W = H+1 years per window
      - windows: (loc, target_year=t), t in [H .. Y-1]
      - loss computed ONLY on tail year (unique ownership)
    Returns: (total_loss, windows_done, global_opt_step, mb_sums)
    """
    log = logger or _LOG
    H = int(rollout_cfg.get("carry_horizon", 0) or 0)
    nin, Ttot, L = int(inputs.shape[0]), int(inputs.shape[1]), int(inputs.shape[2])
    Y = int(labels_a.shape[1])
    if L <= 0 or Y <= H:
        return 0.0, 0, global_opt_step, {}

    W = H + 1

    # sizes / indices
    nm = int(labels_m.shape[0]); na = int(labels_a.shape[0])
    in_m_idx  = _sanitize_idx(rollout_cfg.get("in_monthly_state_idx", []) or [], nin, "in_monthly_state_idx", log)
    in_a_idx  = _sanitize_idx(rollout_cfg.get("in_annual_state_idx", [])  or [], nin, "in_annual_state_idx",  log)
    out_m_idx = _sanitize_idx(rollout_cfg.get("out_monthly_state_idx", []) or [], nm,  "out_monthly_state_idx", log)
    out_a_idx = _sanitize_idx(rollout_cfg.get("out_annual_state_idx", [])  or [], na,  "out_annual_state_idx",  log)

    # month bounds
    month_lengths = rollout_cfg.get("month_lengths", MONTH_LENGTHS_FALLBACK)
    bounds = [0]; 
    for m in month_lengths:
        bounds.append(bounds[-1] + m)
    first_month_len = int(month_lengths[0])

    # helpers
    def _pool_from_daily_abs(y_daily_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_m_daily = y_daily_abs[..., :nm]
        y_a_daily = y_daily_abs[..., nm:nm+na]
        pieces = [y_m_daily[:, bounds[i]:bounds[i+1], :].mean(dim=1, keepdim=True) for i in range(12)]
        preds_m = torch.cat(pieces, dim=1)            # [B,12,nm]
        preds_a = y_a_daily.mean(dim=1, keepdim=True) # [B,1, na]
        return preds_m, preds_a

    # windows list
    windows: list[tuple[int, int]] = [(loc, t) for loc in range(L) for t in range(H, Y)]
    if not windows:
        return 0.0, 0, global_opt_step, {}

    # microbatch size (windows)
    if training:
        micro = max(1, int(mb_size or 1))
    else:
        base = int(mb_size) if (mb_size is not None and mb_size > 0) else 2048
        micro = max(1, base)
    micro = min(micro, len(windows))
    rollout_cfg["autotuned_mb_size_windows"] = int(micro)

    # DDP broadcast micro
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t_mb = torch.tensor([micro], dtype=torch.int64, device=device)
        torch.distributed.broadcast(t_mb, src=0)
        micro = int(t_mb.item())
        rollout_cfg["autotuned_mb_size_windows"] = int(micro)

    # move to device once
    inputs   = inputs.to(device, non_blocking=True)
    labels_m = labels_m.to(device, non_blocking=True)
    labels_a = labels_a.to(device, non_blocking=True)

    total_loss = 0.0
    windows_done = 0
    microbatches_done = 0
    mb_sums: Dict[str, float] = {}

    # modes
    if training:
        model.train()
    else:
        model.eval()

    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with _model_mode(model, enabled=True), grad_ctx:
        w_idx = 0
        while w_idx < len(windows):
            s = w_idx; e = min(w_idx + micro, len(windows)); w_idx = e
            B = e - s
            locs    = [windows[k][0] for k in range(s, e)]
            targets = [windows[k][1] for k in range(s, e)]

            prev_m = None
            prev_a = None

            # iterate the W years inside each window
            for y_off in range(W):
                y0s = [t - H + y_off for t in targets]

                # build xb [B,365,nin]
                xb_list = []
                for loc, y_abs in zip(locs, y0s):
                    ds, de = _year_bounds(y_abs, y_abs + 1)
                    xb_list.append(inputs[:, ds:de, loc].T)
                xb = torch.stack(xb_list, dim=0)  # [B,365,nin]

                # inject carries
                if y_off > 0:
                    if (prev_m is not None) and in_m_idx:
                        _inject_monthly_carry(xb, prev_m, in_m_idx=in_m_idx,
                                              first_month_len=first_month_len, logger=log)
                    if (prev_a is not None) and in_a_idx:
                        if not torch.isfinite(prev_a).all():
                            log.warning("[rollout] non-finite annual carry pre-inject (B=%d,y_off=%d,shape=%s)",
                                        B, y_off, tuple(prev_a.shape))
                        else:
                            _safe_index_copy(
                                xb, 2, in_a_idx,
                                prev_a.unsqueeze(1).expand(B, 365, len(in_a_idx)),
                                where_len=365, name="annual carry→inputs", logger=log,
                            )

                # INPUT guard (allow Feb–Dec NaNs for monthly carry vars)
                if not _isfinite_except_monthly_carry_mask(xb, in_m_idx, first_month_len):
                    log.warning("[rollout] non-finite INPUT (disallowed NaNs) (B=%d, y_off=%d)", B, y_off)
                    del xb
                    continue

                # forward 1 year → daily absolutes
                preds_abs_daily = model(xb)  # [B,365,nm+na]
                if not torch.isfinite(preds_abs_daily).all():
                    log.warning("[rollout] non-finite PRED (B=%d, y_off=%d)", B, y_off)
                    del xb, preds_abs_daily
                    continue

                # pool to monthly/annual and prep next carry
                preds_m, preds_a = _pool_from_daily_abs(preds_abs_daily)
                next_m = _prev_monthly_from_preds(preds_m, out_m_idx)
                if training:
                    prev_m = (next_m.detach() if next_m is not None else None)
                    prev_a = (preds_a[:, 0, out_a_idx].detach() if out_a_idx else None)
                else:
                    prev_m = (next_m if next_m is not None else None)
                    prev_a = (preds_a[:, 0, out_a_idx] if out_a_idx else None)

                # tail-year loss only
                if y_off == W - 1:
                    if loss_func is None:
                        raise RuntimeError("loss_func is required for loss computation")
                    ybm_list = [labels_m[:, (t*12):((t+1)*12), loc].T for loc, t in zip(locs, targets)]
                    yba_list = [labels_a[:, t:(t+1),        loc].T for loc, t in zip(locs, targets)]
                    ybm = torch.stack(ybm_list, dim=0)  # [B,12,nm]
                    yba = torch.stack(yba_list, dim=0)  # [B,1, na]

                    if (not torch.isfinite(ybm).all()) or (not torch.isfinite(yba).all()):
                        log.warning("[rollout] non-finite LABELS (B=%d)", B)
                        del xb, preds_abs_daily, preds_m, preds_a, ybm, yba
                        continue

                    # ---- Build optional extra_daily (e.g. physical Pre) ----
                    extra_daily = None
                    try:
                        in_order = (rollout_cfg or {}).get("input_order", [])
                        std_in   = ((rollout_cfg or {}).get("std_stats_in", {}) or {})
                        want_feats = (rollout_cfg or {}).get("extra_daily_features", ["pre"])
                        ed: Dict[str, torch.Tensor] = {}
                        for feat in want_feats:
                            if feat in in_order:
                                idx = in_order.index(feat)
                                mu  = float(((std_in.get(feat) or {}).get("mean", 0.0)))
                                sd  = float(((std_in.get(feat) or {}).get("std",  1.0)))
                                feat_phys = xb[:, :, idx] * sd + mu  # [B,365]
                                ed[feat] = feat_phys
                        if ed:
                            extra_daily = ed
                    except Exception:
                        extra_daily = None

                    # Loss (allow signature with extra_daily kw)
                    try:
                        loss = loss_func(preds_abs_daily, ybm, yba, extra_daily=extra_daily)
                    except TypeError:
                        loss = loss_func(preds_abs_daily, ybm, yba)

                    if training:
                        (loss / max(1, eff_accum)).backward()
                        microbatches_done += 1
                        if microbatches_done % max(1, eff_accum) == 0:
                            if eff_clip is not None:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), eff_clip)
                            if opt is not None:
                                opt.step()
                                opt.zero_grad(set_to_none=True)
                            if scheduler is not None:
                                scheduler.step()
                            if history is not None and opt is not None:
                                history.lr_values.append(float(opt.param_groups[0]["lr"]))
                                history.lr_steps.append(global_opt_step)
                            global_opt_step += 1

                    batch_weight = B if POPULATION_NORMALISE else 1
                    total_loss   += float(loss.detach().cpu()) * batch_weight
                    windows_done += batch_weight

                    # Optional mass-balance accumulation from loss_func
                    bd = getattr(loss_func, "last_breakdown", None)
                    if isinstance(bd, dict) and ("weighted" in bd):
                        for k, v in bd["weighted"].items():
                            if v is None:
                                continue
                            mb_sums[k] = mb_sums.get(k, 0.0) + float(v) * float(batch_weight)

                del xb, preds_abs_daily, preds_m, preds_a

        # flush remainder if gradients pending
        if training and (microbatches_done % max(1, eff_accum) != 0):
            if eff_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), eff_clip)
            if opt is not None:
                opt.step()
                opt.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if history is not None and opt is not None:
                history.lr_values.append(float(opt.param_groups[0]["lr"]))
                history.lr_steps.append(global_opt_step)
            global_opt_step += 1

    # Return totals (population-weighted loss sum and window count), next step, and MB sums
    return total_loss, windows_done, global_opt_step, mb_sums


# ---------------------------------------------------------------------------
# gather (y, ŷ) pairs for metrics/plots with the same carry behaviour
# ---------------------------------------------------------------------------

@torch.no_grad()
def gather_pred_label_pairs(
    *,
    model: torch.nn.Module,
    test_dl,
    device: torch.device,
    rollout_cfg: dict,
    eval_mode: str = "teacher_forced",            # {"teacher_forced","full_sequence","windowed_tail_only"}
    mb_size: Optional[int] = None,                # windows per microbatch (for windowed_tail_only)
    max_points_per_var: int | None = 2_000_000,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Collect {var_name: (y_norm, yhat_norm)} pairs using one of three explicit modes:

      - teacher_forced:
          No carry at all. Evaluate each year independently (including year 0).
      - full_sequence:
          Warm-up on year 0, then carry predictions forward and evaluate every target year 1..Y-1.
      - windowed_tail_only:
          Windowed semantics using carry_horizon (int → D), collect pairs only for tail year.
    """
    assert eval_mode in {"teacher_forced", "full_sequence", "windowed_tail_only"}, \
        f"Unsupported eval_mode={eval_mode}"

    model.eval()

    # Names (head order)
    monthly_names = list(rollout_cfg.get("out_monthly_names", []))
    annual_names  = list(rollout_cfg.get("out_annual_names", []))

    # Buffers
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
            preds_m = torch.cat(pieces, dim=1)            # [L,12,nm]
            preds_a = y_daily_abs[..., nm:nm+na].mean(dim=1, keepdim=True)  # [L,1,na]
            return preds_m, preds_a

        # ---- MODE 1: teacher_forced ----
        if eval_mode == "teacher_forced":
            with _model_mode(model, True):
                for y in range(Y):
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

        # ---- MODE 2: full_sequence ----
        if eval_mode == "full_sequence":
            with _model_mode(model, enabled=True):
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

                    next_m = _prev_monthly_from_preds(pm0, out_m_idx)
                    prev_m = (next_m if next_m is not None else None)
                    prev_a = (pa0[:, 0, out_a_idx] if out_a_idx else None)
                    del xb0, preds0, pm0, pa0

                    # Evaluate years 1..Y-1
                    for y in range(1, Y):
                        ds, de = _year_bounds(y, y + 1)
                        xb = inputs[:, ds:de, loc].T.unsqueeze(0)  # [1,365,nin]

                        # Inject carries
                        if (prev_m is not None) and in_m_idx:
                            _inject_monthly_carry(
                                xb, prev_m, in_m_idx=in_m_idx,
                                first_month_len=first_month_len, logger=_LOG,
                            )
                        if (prev_a is not None) and in_a_idx:
                            _safe_index_copy(
                                xb, 2, in_a_idx,
                                prev_a.unsqueeze(1).expand(1, 365, len(in_a_idx)),
                                where_len=365, logger=_LOG, name="annual carry→inputs",
                            )

                        # Guards (allow Feb–Dec NaNs for monthly-carry features)
                        if not _isfinite_except_monthly_carry_mask(xb, in_m_idx, first_month_len):
                            _LOG.warning("[gather/FS] non-finite INPUT (disallowed NaNs) (loc=%d, year=%d)", loc, y)
                            del xb
                            continue

                        preds = model(xb)  # [1,365,nm+na]
                        if not torch.isfinite(preds).all():
                            _LOG.warning("[gather/FS] non-finite PRED (loc=%d, year=%d)", loc, y)
                            del xb, preds
                            continue

                        pm, pa = _pool_from_daily_abs(preds)  # [1,12,nm], [1,1,na]

                        # Labels
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
                        next_m = _prev_monthly_from_preds(pm, out_m_idx)
                        prev_m = (next_m if next_m is not None else None)
                        prev_a = (pa[:, 0, out_a_idx] if out_a_idx else None)

                        del xb, preds, pm, pa, ybm, yba

            continue  # next outer batch

        # ---- MODE 3: windowed_tail_only ----
        D = int(rollout_cfg.get("carry_horizon", 0) or 0)
        W = D + 1

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

        with _model_mode(model, enabled=True):
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
                                first_month_len=first_month_len, logger=_LOG,
                            )
                        if (prev_a is not None) and in_a_idx:
                            _safe_index_copy(
                                xb, 2, in_a_idx,
                                prev_a.unsqueeze(1).expand(B, 365, len(in_a_idx)),
                                where_len=365, logger=_LOG, name="annual carry→inputs",
                            )

                    # Guards (allow Feb–Dec NaNs for monthly-carry features)
                    if not _isfinite_except_monthly_carry_mask(xb, in_m_idx, first_month_len):
                        _LOG.warning("[gather/WIN] non-finite INPUT (disallowed NaNs) (B=%d, y_off=%d)", B, y_off)
                        del xb
                        continue

                    preds_abs_daily = model(xb)
                    if not torch.isfinite(preds_abs_daily).all():
                        _LOG.warning("[gather/WIN] non-finite PRED (B=%d, y_off=%d)", B, y_off)
                        del xb, preds_abs_daily
                        continue

                    pm, pa = _pool_from_daily_abs(preds_abs_daily)

                    # Prepare next carry
                    next_m = _prev_monthly_from_preds(pm, out_m_idx)
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