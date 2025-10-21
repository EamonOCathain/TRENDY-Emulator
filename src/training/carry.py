# src/training/carry.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
import torch
from contextlib import contextmanager
import numpy as np
from collections import deque
from torch.nn.parallel import DistributedDataParallel as DDP
import math
from pathlib import Path
import sys
import logging

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))
from src.training.delta import DeltaContext 

# ------------------------- logging -------------------------
_LOG = logging.getLogger("carry")

# ------------------------- context manager for sequential mode -------------------------
@contextmanager
def _sequential_mode(model):
    target = model.module if isinstance(model, DDP) else model
    prev = getattr(target, "mode", None)
    try:
        if hasattr(target, "set_mode"):
            target.set_mode("sequential_months")
        yield
    finally:
        if (prev is not None) and hasattr(target, "set_mode"):
            target.set_mode(prev)

# Fallback month metadata (noleap) — we’ll reuse model’s if available
MONTH_LENGTHS_FALLBACK = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# ------------------------- core reductions -------------------------
def _reduce_monthly(preds_daily: torch.Tensor, month_bounds: list[int]) -> torch.Tensor:
    """preds_daily: [B,365,C] -> [B,12,C] by calendar-month mean."""
    pieces = [preds_daily[:, month_bounds[i]:month_bounds[i+1], :].mean(dim=1, keepdim=True)
              for i in range(12)]
    return torch.cat(pieces, dim=1)

# ------------------------- other helpers -------------------------
def _ceil_years(x: float) -> int:
    try:
        return int(math.ceil(float(x)))
    except Exception:
        return 0

def _year_bounds(y0: int, y1: int) -> tuple[int, int]:
    """Return [day_start, day_end) indices for years [y0..y1) in a 365*n layout."""
    return (y0 * 365, y1 * 365)

def _slice_last_year_bounds(t: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return [month_start, month_end), [year_start, year_end) for target year t."""
    return (t * 12, (t + 1) * 12), (t, t + 1)

def _prev_monthly_from_preds(preds_m: torch.Tensor, out_m_idx: list[int], granularity: str) -> Optional[torch.Tensor]:
    """
    preds_m: [B, 12, nm] pooled monthly predictions (absolutes).
    return vector for carry: [B, len(out_m_idx)] or None.
    monthly  -> take last month: preds_m[:, -1, out_m_idx]
    annual   -> take annual mean: preds_m.mean(dim=1)[:, out_m_idx]
    """
    if (preds_m is None) or (not out_m_idx):
        return None
    if granularity == "annual":
        annual_mean = preds_m.mean(dim=1)  # [B, nm]
        return annual_mean[:, out_m_idx]
    # default monthly:
    return preds_m[:, -1, out_m_idx]

def _inject_monthly_carry(x_year: torch.Tensor, carry_vec: torch.Tensor,
                          in_m_idx: list[int], first_month_len: int, granularity: str):
    """
    x_year: [B, 365, nin], carry_vec: [B, len(in_m_idx)]
    monthly  -> write only into the first month days
    annual   -> broadcast into all 365 days
    """
    if (carry_vec is None) or (not in_m_idx):
        return
    B = x_year.size(0)
    if granularity == "annual":
        _safe_index_copy(
            x_year, 2, in_m_idx,
            carry_vec.unsqueeze(1).expand(B, 365, len(in_m_idx)),
            where_len=365, name="monthly carry→inputs(annual)"
        )
    else:
        _safe_index_copy(
            x_year[:, :first_month_len, :], 2, in_m_idx,
            carry_vec.unsqueeze(1).expand(B, first_month_len, len(in_m_idx)),
            where_len=first_month_len, name="monthly carry→inputs(monthly)"
        )

# ------------------------- guards & safe index copy -------------------------
def _sanitize_idx(idx_list, size, name, logger=None):
    if not idx_list:
        return []
    good = [int(i) for i in idx_list
            if isinstance(i, (int, np.integer)) and 0 <= int(i) < size]
    if logger and len(good) != len(idx_list):
        bad = [i for i in idx_list if i not in good]
        logger.warning(f"[carry] Dropping invalid {name} indices {bad} for size={size}")
    return good

def _safe_index_copy(x, dim, idx_list, values, where_len=None, logger=None, name=""):
    size = x.size(dim)
    idx_list = _sanitize_idx(idx_list, size, name, logger)
    if not idx_list:
        return
    idx = torch.as_tensor(idx_list, device=x.device, dtype=torch.long)
    need = idx.numel()
    got  = values.size(-1)
    if need != got:
        if logger:
            logger.error(f"[carry] Value width ({got}) != len(idx) ({need}) for {name}; skipping injection.")
        return
    x.index_copy_(dim, idx, values)
    
def parse_carry_years_flag(flag: str | list[str]) -> Tuple[str, list[float]]:
    """
    Parses --carry_years flag into ("mode", [carry_values]).
    Accepts:
      - "progressive" → ("progressive", [])
      - "2" or "3/12" → ("static", [value])
      - list like ["1","2","3","6","9"] → ("multi", [1.0,2.0,3.0,6.0,9.0])
      - comma/space separated string "1 2 3" → same as above
    Returns:
      (mode, values)
    """
    # If already a list (from argparse with nargs="+")
    if isinstance(flag, (list, tuple)):
        if len(flag) == 1:
            s = flag[0].strip().lower()
            if s == "progressive":
                return "progressive", []
            if "/" in s:
                num, den = s.split("/", 1)
                return "static", [float(num)/float(den)]
            return "static", [float(s)]
        # len > 1 → multi
        vals = []
        for f in flag:
            s = f.strip().lower()
            if "/" in s:
                num, den = s.split("/", 1)
                vals.append(float(num) / float(den))
            else:
                vals.append(float(s))
        return "multi", vals

    # If plain string
    s = str(flag or "0").strip().lower()
    if s == "progressive":
        return "progressive", []
    if " " in s or "," in s:
        parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
        vals = []
        for p in parts:
            if "/" in p:
                num, den = p.split("/", 1)
                vals.append(float(num) / float(den))
            else:
                vals.append(float(p))
        return "multi", vals
    if "/" in s:
        num, den = s.split("/", 1)
        return "static", [float(num) / float(den)]
    return "static", [float(s)]

# ------------------------- ONE canonical rollout used by eval/test  -------------------------
# --- tiny tensor summary for logging ---
def _tsum(x: torch.Tensor, name: str) -> str:
    try:
        x = x.detach()
        finite = torch.isfinite(x)
        return (f"{name}[shape={tuple(x.shape)}, "
                f"finite={finite.all().item()}, "
                f"min={torch.nanmin(x).item():.4g}, "
                f"max={torch.nanmax(x).item():.4g}, "
                f"mean={torch.nanmean(x).item():.4g}]")
    except Exception:
        return f"{name}[unavailable]"

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
    assert (not training) or (loss_func is not None), "loss_func required when training=True"

    # Calendar & month bounds
    month_lengths = rollout_cfg.get("month_lengths", MONTH_LENGTHS_FALLBACK)
    bounds = [0]
    for Lm in month_lengths:
        bounds.append(bounds[-1] + Lm)
    first_month_len = int(month_lengths[0])

    # Indices (raw from cfg)
    in_m_idx  = rollout_cfg.get("in_monthly_state_idx", [])  or []
    in_a_idx  = rollout_cfg.get("in_annual_state_idx", [])   or []
    out_m_idx = rollout_cfg.get("out_monthly_state_idx", []) or []
    out_a_idx = rollout_cfg.get("out_annual_state_idx", [])  or []

    monthly_names = list(rollout_cfg.get("out_monthly_names", []))
    annual_names  = list(rollout_cfg.get("out_annual_names", []))

    # Granularity & horizon guard
    granularity = str(rollout_cfg.get("carry_granularity", "monthly"))
    H = float(rollout_cfg.get("carry_horizon", 0.0) or 0.0)
    if granularity == "annual":
        assert (H == 0.0) or (H >= 1.0), \
            "In annual carry mode, carry_horizon must be 0 or >= 1 year (no fractional carry)."

    # Shapes / guards
    nin, Ttot, L = int(inputs.shape[0]), int(inputs.shape[1]), int(inputs.shape[2])
    Y = Ttot // 365
    if Y <= 1:
        if return_pairs:
            return {}
        return (0.0, 0) if loss_func is not None else None

    # Move once
    inputs   = inputs.to(device, non_blocking=True)
    labels_m = labels_m.to(device, non_blocking=True)
    labels_a = labels_a.to(device, non_blocking=True)

    # Accumulators
    sum_loss = 0.0
    n_windows = 0

    # For pairs
    if return_pairs:
        buf_y = {n: [] for n in (monthly_names + annual_names)}
        buf_p = {n: [] for n in (monthly_names + annual_names)}

    # Autograd context
    ctx = torch.enable_grad() if training else torch.no_grad()

    # Toggle model mode depending on granularity (only when carrying across years)
    model_mode_prev = getattr(model, "mode", None)
    if carry_on and hasattr(model, "set_mode"):
        if granularity == "monthly":
            model.set_mode("sequential_months")  # within-year sequential (original behavior)
        else:
            model.set_mode("batch_months")       # full-year forward in one pass

    # Delta flag
    dc = rollout_cfg.get("delta_ctx", None)
    delta_enabled = (dc is not None) and getattr(dc, "enabled", False)

    # helper: compute monthly/annual pooled from DAILY absolutes
    def _pool_from_daily_abs(y_daily_abs: torch.Tensor, nm: int, na: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        y_daily_abs: [B,365,nm+na] absolute (already reconstructed if delta_enabled)
        returns:
          preds_m: [B,12,nm]  monthly means
          preds_a: [B,1, na]  annual mean
        """
        y_m_daily = y_daily_abs[..., :nm]
        y_a_daily = y_daily_abs[..., nm:nm+na]
        # monthly pool
        pieces = [y_m_daily[:, bounds[i]:bounds[i+1], :].mean(dim=1, keepdim=True) for i in range(12)]
        preds_m = torch.cat(pieces, dim=1)                  # [B,12,nm]
        # annual pool
        preds_a = y_a_daily.mean(dim=1, keepdim=True)       # [B,1,na]
        return preds_m, preds_a

    try:
        with ctx:
            for loc in range(L):
                xb_full = inputs[:, :, loc].T.unsqueeze(0)               # [1, 365*Y, nin]
                ym_full = labels_m[:, :, loc].T.unsqueeze(0)             # [1, 12*Y,  nm]
                ya_full = labels_a[:, :, loc].T.unsqueeze(0)             # [1,   Y,   na]

                B  = 1
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
                prev_annual_state  = None

                # --- WARM-UP: run year 0 once to seed prev_* for y=1 (carry only) ---
                if carry_on:
                    x_prev = xb_full[:, 0:365, :].clone()  # [1,365,nin]
                    if not torch.isfinite(x_prev).all():
                        if logger:
                            logger.warning("[eval-core] non-finite warm-up INPUT (loc=%d): %s",
                                        loc, _tsum(x_prev, "x_prev"))
                        del x_prev
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        continue

                    preds_prev = model(x_prev)  # [1,365,out]

                    if delta_enabled:
                        pm_delta_prev = preds_prev[..., :nm]                        # [1,365,nm]
                        pa_delta_prev = preds_prev[..., nm:nm+na]                   # [1,365,na]

                        # Reconstruct monthwise with intra-year anchors (Jan starts from 0 **within y=0**)
                        y_m_abs_prev = torch.empty_like(pm_delta_prev)
                        cur = torch.zeros((B, nm), device=preds_prev.device, dtype=preds_prev.dtype)
                        for m in range(12):
                            s_m, e_m = bounds[m], bounds[m+1]
                            seg_abs = cur.unsqueeze(1) + pm_delta_prev[:, s_m:e_m, :]
                            y_m_abs_prev[:, s_m:e_m, :] = seg_abs
                            cur = seg_abs.mean(dim=1)

                        # Annual: anchor=0 within y=0 window (this just seeds next year's carry)
                        y_a_abs_prev = pa_delta_prev
                        preds_abs_daily_prev = torch.cat([y_m_abs_prev, y_a_abs_prev], dim=-1)
                    else:
                        preds_abs_daily_prev = preds_prev

                    preds_m_prev, preds_a_prev = _pool_from_daily_abs(preds_abs_daily_prev, nm, na)

                    out_m_idx_warm = _sanitize_idx(out_m_idx, nm, "out_monthly_state_idx", logger)
                    out_a_idx_warm = _sanitize_idx(out_a_idx, na, "out_annual_state_idx", logger)

                    # monthly carry vector selection depends on granularity
                    prev_monthly_state = _prev_monthly_from_preds(preds_m_prev, out_m_idx_warm, granularity)
                    if prev_monthly_state is not None:
                        prev_monthly_state = prev_monthly_state.detach()

                    # annual carry still uses the annual pooled state
                    prev_annual_state  = (preds_a_prev[:, 0,  out_a_idx_warm].detach()
                                          if out_a_idx_warm else None)

                    del preds_prev, preds_abs_daily_prev, preds_m_prev, preds_a_prev

                for y in range(1, Y):
                    s_days = y * 365
                    e_days = s_days + 365
                    x_year = xb_full[:, s_days:e_days, :].clone()         # [1,365,nin]

                    # Inject carries ONLY if carry_on (as inputs)
                    if carry_on:
                        if prev_monthly_state is not None and in_m_idx:
                            _inject_monthly_carry(
                                x_year,
                                (prev_monthly_state if training else prev_monthly_state.detach()),
                                in_m_idx=in_m_idx,
                                first_month_len=first_month_len,
                                granularity=granularity,
                            )
                        if prev_annual_state is not None and in_a_idx:
                            _safe_index_copy(
                                x_year, 2, in_a_idx,
                                (prev_annual_state if training else prev_annual_state.detach())
                                    .unsqueeze(1).expand(B, 365, len(in_a_idx)),
                                where_len=365, logger=logger, name="annual carry→inputs"
                            )

                    # Guard inputs for finiteness per window (most common NaN source)
                    if not torch.isfinite(x_year).all():
                        if logger:
                            logger.warning(
                                "[eval-core] non-finite INPUT detected (loc=%d, y=%d): %s",
                                loc, y, _tsum(x_year, "x_year")
                            )
                        # Skip this window
                        del x_year
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        continue

                    preds_daily = model(x_year)  # [1,365,out]

                    # Immediate post-forward sanity check
                    if not torch.isfinite(preds_daily).all():
                        if logger:
                            logger.warning(
                                "[eval-core] model produced non-finite (loc=%d, y=%d): %s | %s",
                                loc, y, _tsum(x_year, "x_year"), _tsum(preds_daily, "preds_daily")
                            )
                        del preds_daily, x_year
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        continue

                    # ---------------- delta reconstruction to ABSOLUTES ----------------
                    if delta_enabled:
                        dc: DeltaContext = rollout_cfg["delta_ctx"]
                        # NOTE: use the *sanitized* head-local indices
                        preds_abs_daily = dc.reconstruct_groups_daily_segmentwise(
                            preds=preds_daily,                  # [B,365,nm+na] deltas (norm.)
                            nm=nm,
                            na=na,
                            month_slices=rollout_cfg["month_slices"],
                            mode="carry",                       # we're in the carry path here
                            out_m_idx=out_m_idx_sane,
                            out_a_idx=out_a_idx_sane,
                            prev_monthly_state=prev_monthly_state,  # [B,len(out_m_idx)]
                            prev_annual_state=prev_annual_state,    # [B,len(out_a_idx)]
                            yb_m_prev_last=None,
                            yb_a_prev=None,
                        )
                    else:
                        # non-delta path: predictions are already daily absolutes
                        preds_abs_daily = preds_daily

                    # Labels window for supervision
                    yb_m = ym_full[:, y*12:(y+1)*12, :]                    # [1,12,nm]
                    yb_a = ya_full[:, y:y+1, :]                            # [1,1, na]

                    # ---- LOSS (loss pools internally from daily absolutes) ----
                    if loss_func is not None:
                        loss = loss_func(preds_abs_daily, yb_m, yb_a)
                        
                        # ---- Sanity guards ----
                        if (not torch.isfinite(preds_abs_daily).all()
                            or not torch.isfinite(yb_m).all()
                            or not torch.isfinite(yb_a).all()):
                            if logger:
                                logger.warning(
                                    "[eval-core] non-finite detected "
                                    f"(loc={loc}, y={y}): "
                                    f"preds_abs_daily finite={torch.isfinite(preds_abs_daily).all().item()}, "
                                    f"yb_m finite={torch.isfinite(yb_m).all().item()}, "
                                    f"yb_a finite={torch.isfinite(yb_a).all().item()}"
                                )
                            # Skip this window
                            del preds_daily, preds_abs_daily, yb_m, yb_a
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                            continue
                        
                        if training:
                            loss.backward()
                        sum_loss += float(loss.detach().cpu())
                        n_windows += 1

                    # ---- Pairs (metrics/plots) from pooled monthly/annual ----
                    if return_pairs:
                        preds_m, preds_a = _pool_from_daily_abs(preds_abs_daily, nm, na)
                        for j, name in enumerate(monthly_names):
                            buf_y[name].append(yb_m[..., j].reshape(-1).detach().cpu().numpy())
                            buf_p[name].append(preds_m[..., j].reshape(-1).detach().cpu().numpy())
                        for j, name in enumerate(annual_names):
                            buf_y[name].append(yb_a[..., j].reshape(-1).detach().cpu().numpy())
                            buf_p[name].append(preds_a[..., j].reshape(-1).detach().cpu().numpy())

                    # ---- Prepare next-year carries from outputs ----
                    if carry_on:
                        # derive carries from pooled absolutes
                        preds_m, preds_a = _pool_from_daily_abs(preds_abs_daily, nm, na)

                        next_m = _prev_monthly_from_preds(preds_m, out_m_idx_sane, granularity)
                        prev_monthly_state = (next_m if training else (next_m.detach() if next_m is not None else None))

                        if out_a_idx_sane:
                            a_vec = preds_a[:, 0, out_a_idx_sane]
                            prev_annual_state = (a_vec if training else a_vec.detach())
                        else:
                            prev_annual_state = None

                    # free
                    del preds_daily, preds_abs_daily, yb_m, yb_a
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        if carry_on and (model_mode_prev is not None) and hasattr(model, "set_mode"):
            model.set_mode(model_mode_prev)

    # ---------- RETURNS ----------
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
# ------------------------- public: pred/label pairs (metrics/plots) -------------------------
@torch.no_grad()
def gather_pred_label_pairs(
    *,
    model: torch.nn.Module,
    test_dl,
    device: torch.device,
    rollout_cfg: dict,
    carry_horizon: float = 0.0,
    max_points_per_var: int | None = 2_000_000
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Returns {var: (y_norm, yhat_norm)} using the SAME carry behavior as eval/test.
    carry_horizon == 0.0 -> teacher-forced (no cross-year injection)
    carry_horizon  > 0.0 -> carry (inject monthly+annual states across years)
    """
    model.eval()
    carry_on = float(carry_horizon) > 0.0

    monthly_names = list(rollout_cfg.get("out_monthly_names", []))
    annual_names  = list(rollout_cfg.get("out_annual_names", []))
    buf = {n: ([], []) for n in (monthly_names + annual_names)}  # (y, yhat)

    for batch_inputs, batch_monthly, batch_annual in test_dl:
        pairs = _rollout_core(
            model=model,
            inputs=batch_inputs.squeeze(0).float(),
            labels_m=batch_monthly.squeeze(0).float(),
            labels_a=batch_annual.squeeze(0).float(),
            device=device,
            rollout_cfg=rollout_cfg,
            training=False,
            loss_func=None,
            carry_on=carry_on,
            return_pairs=True,
            logger=_LOG,
        )
        for k, (y, p) in pairs.items():
            buf[k][0].append(y)
            buf[k][1].append(p)

    # concat + optional subsample
    rng = np.random.default_rng(123)
    out = {}
    for k, (ys, ps) in buf.items():
        y = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=float)
        p = np.concatenate(ps, axis=0) if ps else np.empty((0,), dtype=float)
        n = y.size
        if (max_points_per_var is not None) and (n > max_points_per_var):
            idx = rng.choice(n, size=max_points_per_var, replace=False)
            y = y[idx]; p = p[idx]
        out[k] = (y, p)
    return out

# ------------------------- rollout kernels (train/eval API stays) -------------------------
def rollout_train_outer_batch(
    *,
    model: torch.nn.Module,
    loss_func,
    opt: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    inputs: torch.Tensor,        # [nin, 365*Y, L]
    labels_m: torch.Tensor,      # [nm,  12*Y, L]
    labels_a: torch.Tensor,      # [na,   1*Y, L]
    mb_size: int,
    eff_accum: int,
    eff_clip: Optional[float],
    history,                     # History object
    global_opt_step: int,
    device: torch.device,
    rollout_cfg: dict,
) -> tuple[float, int]:
    """
    Streaming training with carry (same behavior as core).
    Keeps your micro-batching/accum/step logic intact.
    """
    
    # if carry_horizon is set, use the windowed kernel
    H = float(rollout_cfg.get("carry_horizon", 0.0) or 0.0)
    if H > 0.0:
        avg, step, _ = rollout_train_outer_batch_windowed(
            model=model, loss_func=loss_func, opt=opt, scheduler=scheduler,
            inputs=inputs, labels_m=labels_m, labels_a=labels_a,
            mb_size=mb_size, eff_accum=eff_accum, eff_clip=eff_clip,
            history=history, global_opt_step=global_opt_step,
            device=device, rollout_cfg=rollout_cfg,
        )
        return avg, step
    
    model.train()

    # Shapes
    Y = int(labels_a.shape[1])
    L = int(inputs.shape[2])

    # Adaptive micro-batch seed
    auto_key = "autotuned_mb_size"
    if rollout_cfg.get(auto_key) is None:
        rollout_cfg[auto_key] = int(mb_size)
    micro = max(1, min(int(rollout_cfg[auto_key]), L))

    # Calendar for reductions
    month_lengths = rollout_cfg.get("month_lengths", MONTH_LENGTHS_FALLBACK)
    bounds = [0]
    for Lm in month_lengths:
        bounds.append(bounds[-1] + Lm)
    first_month_len = int(month_lengths[0])

    # Indices
    in_m_idx  = rollout_cfg.get("in_monthly_state_idx",  []) or []
    in_a_idx  = rollout_cfg.get("in_annual_state_idx",   []) or []
    out_m_idx = rollout_cfg.get("out_monthly_state_idx", []) or []
    out_a_idx = rollout_cfg.get("out_annual_state_idx",  []) or []

    running_loss_sum  = 0.0
    total_windows     = 0
    microbatches_done = 0

    # delta flag
    dc = rollout_cfg.get("delta_ctx", None)
    delta_enabled = (dc is not None) and getattr(dc, "enabled", False)

    # helper: pool monthly/annual from DAILY ABSOLUTES
    def _pool_from_daily_abs(y_daily_abs: torch.Tensor, nm: int, na: int) -> tuple[torch.Tensor, torch.Tensor]:
        y_m_daily = y_daily_abs[..., :nm]
        y_a_daily = y_daily_abs[..., nm:nm+na]
        pieces = [y_m_daily[:, bounds[i]:bounds[i+1], :].mean(dim=1, keepdim=True) for i in range(12)]
        preds_m = torch.cat(pieces, dim=1)                     # [B,12,nm]
        preds_a = y_a_daily.mean(dim=1, keepdim=True)          # [B,1, na]
        return preds_m, preds_a

    loc_idx = 0
    while loc_idx < L:
        s = loc_idx
        e = min(loc_idx + micro, L)
        loc_idx = e

        xb_seq = inputs[:, :, s:e].permute(2, 1, 0).to(device, non_blocking=True)   # [B, 365*Y, nin]
        ym_seq = labels_m[:, :, s:e].permute(2, 1, 0).to(device, non_blocking=True) # [B, 12*Y,  nm]
        ya_seq = labels_a[:, :, s:e].permute(2, 1, 0).to(device, non_blocking=True) # [B,   Y,   na]

        B = xb_seq.shape[0]
        nm = ym_seq.shape[-1]
        na = ya_seq.shape[-1]

        # Within-year carry on
        with _sequential_mode(model):
            prev_monthly_state = None
            prev_annual_state  = None
            
            # --- WARM-UP: run year 0 once to seed prev_* for y=1 ---
            x_prev = xb_seq[:, 0:365, :].clone()                 # [B,365,nin]
            preds_prev = model(x_prev)                            # [B,365,out]

            if delta_enabled:
                dc: DeltaContext = rollout_cfg["delta_ctx"]
                preds_abs_daily_prev = dc.reconstruct_groups_daily_segmentwise(
                    preds=preds_prev,                             # <-- use preds_prev
                    nm=nm, na=na,
                    month_slices=rollout_cfg["month_slices"],
                    mode="carry",                                 # y=0 has no prev states -> zeros
                    out_m_idx=out_m_idx, out_a_idx=out_a_idx,
                    prev_monthly_state=None, prev_annual_state=None,
                    yb_m_prev_last=None, yb_a_prev=None,
                )
            else:
                preds_abs_daily_prev = preds_prev                 # <-- name is *_prev

            preds_m_prev, preds_a_prev = _pool_from_daily_abs(preds_abs_daily_prev, nm, na)

            out_m_idx_sane = _sanitize_idx(out_m_idx, nm, "out_monthly_state_idx", _LOG)
            out_a_idx_sane = _sanitize_idx(out_a_idx, na, "out_annual_state_idx", _LOG)

            prev_monthly_state = (preds_m_prev[:, -1, out_m_idx_sane].detach()
                                if out_m_idx_sane else None)
            prev_annual_state  = (preds_a_prev[:, 0,  out_a_idx_sane].detach()
                                if out_a_idx_sane else None)

            del preds_prev, preds_abs_daily_prev, preds_m_prev, preds_a_prev
            
            for y in range(1, Y):
                x_year = xb_seq[:, y*365:(y+1)*365, :].clone() # [B,365,nin]

                # Inject carries (DETACHED across years for TBPTT=0)
                if (prev_monthly_state is not None) and in_m_idx:
                    _safe_index_copy(
                        x_year[:, :first_month_len, :], 2, in_m_idx,
                        prev_monthly_state.detach().unsqueeze(1).expand(B, first_month_len, len(in_m_idx)),
                        where_len=first_month_len, logger=_LOG, name="monthly carry→inputs"
                    )
                if (prev_annual_state is not None) and in_a_idx:
                    _safe_index_copy(
                        x_year, 2, in_a_idx,
                        prev_annual_state.detach().unsqueeze(1).expand(B, 365, len(in_a_idx)),
                        where_len=365, logger=_LOG, name="annual carry→inputs"
                    )

                # Forward: DAILY predictions (deltas if delta_enabled, else absolutes)
                preds_daily = model(x_year)                      # [B,365,out]

                if delta_enabled:
                    dc: DeltaContext = rollout_cfg["delta_ctx"]
                    preds_abs_daily = dc.reconstruct_groups_daily_segmentwise(
                        preds=preds_daily,
                        nm=nm, na=na,
                        month_slices=rollout_cfg["month_slices"],
                        mode="carry",
                        out_m_idx=out_m_idx, out_a_idx=out_a_idx,         # head-local indices
                        prev_monthly_state=prev_monthly_state,            # [B, len(out_m_idx)] or None
                        prev_annual_state=prev_annual_state,              # [B, len(out_a_idx)] or None
                        yb_m_prev_last=None, yb_a_prev=None,
                    )
                else:
                    preds_abs_daily = preds_daily

                # Labels slice for this supervised window
                yb_m = ym_seq[:, y*12:(y+1)*12, :]     # [B,12,nm]
                yb_a = ya_seq[:, y:y+1, :]             # [B,1, na]

                # --- LOSS on daily absolutes (loss pools internally) ---
                loss = loss_func(preds_abs_daily, yb_m, yb_a)
                (loss / eff_accum).backward()
                microbatches_done += 1

                running_loss_sum += float(loss.detach().cpu())
                total_windows    += 1

                # Prepare next carries (from pooled absolutes)
                preds_m, preds_a = _pool_from_daily_abs(preds_abs_daily, nm, na)

                prev_monthly_state = (preds_m[:, -1, out_m_idx] if out_m_idx else None)
                if prev_monthly_state is not None:
                    prev_monthly_state = prev_monthly_state.detach()

                prev_annual_state  = (preds_a[:, 0, out_a_idx] if out_a_idx else None)
                if prev_annual_state is not None:
                    prev_annual_state = prev_annual_state.detach()

                # Step if we hit accumulation
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

                # free per-year temporaries
                del preds_daily, preds_abs_daily, preds_m, preds_a, yb_m, yb_a, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # If the last group in this micro-batch didn't land on an eff_accum boundary
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

        history.add_batch(running_loss_sum / max(1, total_windows),
                          history.batch_step[-1] + 1 if history.batch_step else 0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_loss = running_loss_sum / max(1, total_windows)
    return avg_loss, global_opt_step

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
) -> tuple[float, int]:
    """Evaluation with carry using the same core implementation."""
    
    H = float(rollout_cfg.get("carry_horizon", 0.0) or 0.0)
    if H > 0.0:
        return rollout_eval_outer_batch_windowed(
            model=model, loss_func=loss_func,
            inputs=inputs, labels_m=labels_m, labels_a=labels_a,
            device=device, rollout_cfg=rollout_cfg,
        )
    
    model.eval()
    return _rollout_core(
        model=model,
        inputs=inputs,
        labels_m=labels_m,
        labels_a=labels_a,
        device=device,
        rollout_cfg=rollout_cfg,
        training=False,
        loss_func=loss_func,
        carry_on=True,            # this eval kernel is the "with carry" path
        return_pairs=False,
        logger=_LOG,
    )

# ------------------------- parsing & schedule -------------------------

def next_progressive_carry(current: float) -> float:
    """Heuristic schedule used by progressive training."""
    cap = 123.0
    if current < 1.0:   nxt = min(current + (3.0/12.0), 1.0)
    elif current < 10:  nxt = current + 1.0
    elif current < 20:  nxt = current + 2.0
    elif current < 40:  nxt = current + 5.0
    elif current < cap: nxt = current + 10.0
    else:               nxt = current
    return min(nxt, cap)

# ------------------------- month metadata -------------------------
def month_slices_from_lengths(month_lengths: list[int]) -> list[tuple[int, int]]:
    assert len(month_lengths) == 12, "month_lengths must have 12 entries"
    assert sum(month_lengths) == 365, "noleap calendar expected"
    s = 0
    out = []
    for L in month_lengths:
        out.append((s, s + L))
        s += L
    return out

def get_month_metadata(model: Optional[torch.nn.Module] = None) -> Tuple[List[int], List[Tuple[int,int]]]:
    if (model is not None) and hasattr(model, "month_lengths"):
        ml = list(getattr(model, "month_lengths"))
    else:
        ml = list(MONTH_LENGTHS_FALLBACK)
    return ml, month_slices_from_lengths(ml)

# ------------------------- rollout config -------------------------
def build_rollout_cfg(
    input_order: list[str],
    output_order: list[str],
    var_names: dict[str, list[str]],
    carry_horizon: float = 0.0,
) -> dict:
    """
    Returns keys:
      in_monthly_state_idx, in_annual_state_idx
      out_monthly_state_idx, out_annual_state_idx
      out_monthly_all_idx, out_monthly_names (output-head order)
      out_annual_names     (output-head order)
      month_lengths, month_slices
      carry_horizon, output_order
    """
    in_idx  = {n: i for i, n in enumerate(input_order)}
    out_idx = {n: i for i, n in enumerate(output_order)}

    monthly_states = list(var_names.get("monthly_states", []))
    annual_states  = list(var_names.get("annual_states", []))
    monthly_fluxes = list(var_names.get("monthly_fluxes", []))

    # sanity
    def _missing(names: list[str], space: set[str]) -> list[str]:
        return [n for n in names if n not in space]
    missing = (
        _missing(monthly_states, set(input_order)) +
        _missing(annual_states,  set(input_order)) +
        _missing(monthly_states, set(output_order)) +
        _missing(annual_states,  set(output_order)) +
        _missing(monthly_fluxes, set(output_order))
    )
    if missing:
        raise ValueError(f"[build_rollout_cfg] some variables missing from heads: {missing}")

    # --- INPUT indices (these are against the input feature dim) ---
    in_monthly_state_idx  = [in_idx[n]  for n in monthly_states]
    in_annual_state_idx   = [in_idx[n]  for n in annual_states]

    # --- OUTPUT name lists per head (head-local order) ---
    monthly_all_set   = set(monthly_fluxes) | set(monthly_states)
    out_monthly_names = [n for n in output_order if n in monthly_all_set]   # length = nm
    out_annual_names  = [n for n in output_order if n in set(annual_states)]# length = na

    # Build name → local-index maps (0..nm-1, 0..na-1)
    out_monthly_local = {n: i for i, n in enumerate(out_monthly_names)}
    out_annual_local  = {n: i for i, n in enumerate(out_annual_names)}

    # --- OUTPUT indices *relative to their heads* (THIS IS THE IMPORTANT CHANGE) ---
    out_monthly_state_idx = [out_monthly_local[n] for n in monthly_states]  # 0..nm-1
    out_annual_state_idx  = [out_annual_local[n]  for n in annual_states]   # 0..na-1

    # Also useful to keep the full monthly head indices for plotting etc.
    out_monthly_all_idx = [out_monthly_local[n] for n in out_monthly_names] # 0..nm-1 in order

    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_slices  = month_slices_from_lengths(month_lengths)

    return {
        "in_monthly_state_idx":   in_monthly_state_idx,
        "in_annual_state_idx":    in_annual_state_idx,

        "out_monthly_state_idx":  out_monthly_state_idx,  # HEAD-LOCAL
        "out_annual_state_idx":   out_annual_state_idx,   # HEAD-LOCAL

        "out_monthly_all_idx":    out_monthly_all_idx,    # HEAD-LOCAL
        "out_monthly_names":      out_monthly_names,      # HEAD ORDER
        "out_annual_names":       out_annual_names,       # HEAD ORDER

        "month_lengths":          month_lengths,
        "month_slices":           month_slices,
        "carry_horizon":          float(carry_horizon),
        "tbptt_years":            0,
        "output_order":           list(output_order),     # global order (for reference only)
    }

# ------------------------- progressive driver -------------------------
def progressive_train(
    *,
    mode: str,
    carry_values: list[float],
    fit_fn: Callable[[float], Tuple[object, float, bool]],
    reload_best_fn: Callable[[], None],
    next_carry_fn: Callable[[float], float] = next_progressive_carry,
    log: Optional[object] = None,
    max_cap: float = 123.0,
    carry_granularity: str = "monthly",
    min_annual_carry: float = 1.0,
) -> Tuple[object, float]:
    """
    Extended progressive training loop:
      - mode ∈ {"static", "multi", "progressive"}
      - carry_values = list of carry horizons to iterate over if mode == "multi"
      - in "annual" mode, all carries are clamped to ≥ 1.0
    """
    carry_granularity = str(carry_granularity).lower()
    if carry_granularity not in ("monthly", "annual"):
        raise ValueError(f"carry_granularity must be 'monthly' or 'annual', got {carry_granularity!r}")

    # Enforce floor for annual mode
    def _clamp(c: float) -> float:
        if carry_granularity == "annual":
            if c <= 0.0:
                return 0.0         # keep true zero-carry
            return max(1.0, c)      # clamp only positive fractional carries
        return c

    # Determine sequence of carry values
    if mode == "static":
        sequence = [_clamp(carry_values[0] if carry_values else 0.0)]
    elif mode == "multi":
        sequence = [_clamp(c) for c in sorted(carry_values)]
    elif mode == "progressive":
        # start with first carry or default 0
        start = _clamp(carry_values[0] if carry_values else 0.0)
        sequence = [start]  # dynamically extended below
    else:
        raise ValueError(f"Unknown mode: {mode}")

    stage_idx = 0
    last_history, last_best_val = None, float("inf")

    if log:
        log.info(f"[Progressive] mode={mode}, carries={sequence}, granularity={carry_granularity}")
        log.info("Definitely using the progressive training loop..")

    while True:
        current_carry = sequence[stage_idx]
        if log:
            log.info(f"[Progressive] Stage {stage_idx}: carry={current_carry}")

        history, best_val, stopped_early = fit_fn(current_carry)
        last_history, last_best_val = history, best_val

        if stopped_early:
            reload_best_fn()

            if mode == "progressive":
                nxt = _clamp(next_carry_fn(current_carry))
                if nxt <= current_carry or nxt >= max_cap:
                    break
                sequence.append(nxt)
                stage_idx += 1
                continue
            elif mode == "multi":
                stage_idx += 1
                if stage_idx >= len(sequence):
                    break
                continue
            else:  # static
                break
        else:
            # Early stop not triggered — still end for static/multi modes
            if mode == "progressive":
                nxt = _clamp(next_carry_fn(current_carry))
                if nxt > current_carry and nxt < max_cap:
                    sequence.append(nxt)
                    stage_idx += 1
                    continue
            break

    return last_history, last_best_val

# ------------------------- Windowed Parallel Carry Logic -------------------------

def rollout_train_outer_batch_windowed(
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
) -> tuple[float, int, int]:
    """
    Windowed carry training with tail-only loss:
      - D = ceil(carry_horizon), W = D+1 years per window
      - windows are (loc, target_year=t), t in [D ... Y-1]
      - loss is computed ONLY for the last year of each window (unique ownership)
    Supports carry_granularity in {"monthly","annual"}.
    In annual mode, monthly carry uses the *annual mean* of monthly states and
    is broadcast to all days of the next year. Model runs each year as a single
    forward (batch_months mode).
    """
    model.train()

    # Config & guards
    granularity = str(rollout_cfg.get("carry_granularity", "monthly"))
    H = float(rollout_cfg.get("carry_horizon", 0.0) or 0.0)
    if granularity == "annual":
        assert (H == 0.0) or (H >= 1.0), \
            "In annual carry mode, carry_horizon must be 0 or >= 1 year (no fractional carry)."

    nin, Ttot, L = int(inputs.shape[0]), int(inputs.shape[1]), int(inputs.shape[2])
    Y = int(labels_a.shape[1])
    if Y <= 0 or L <= 0:
        return float("inf"), global_opt_step, 0

    # Horizon -> window
    D = _ceil_years(H)                 # dependency depth
    W = D + 1                          # years per window
    if Y <= D:
        _LOG.warning(f"[carry-windowed] Skipping batch: Y={Y} <= D={D}")
        return float("inf"), global_opt_step, 0

    # Output dims
    nm = int(labels_m.shape[0])
    na = int(labels_a.shape[0])

    # Indices (sanitize once with correct sizes)
    in_m_idx  = _sanitize_idx(rollout_cfg.get("in_monthly_state_idx",  []) or [], nin, "in_monthly_state_idx",  _LOG)
    in_a_idx  = _sanitize_idx(rollout_cfg.get("in_annual_state_idx",   []) or [], nin, "in_annual_state_idx",   _LOG)
    out_m_idx = _sanitize_idx(rollout_cfg.get("out_monthly_state_idx", []) or [], nm,  "out_monthly_state_idx", _LOG)
    out_a_idx = _sanitize_idx(rollout_cfg.get("out_annual_state_idx",  []) or [], na,  "out_annual_state_idx",  _LOG)

    # Month metadata + delta
    month_lengths = rollout_cfg.get("month_lengths", MONTH_LENGTHS_FALLBACK)
    bounds = [0]
    for m in month_lengths:
        bounds.append(bounds[-1] + m)
    first_month_len = int(month_lengths[0])

    dc = rollout_cfg.get("delta_ctx", None)
    delta_enabled = (dc is not None) and getattr(dc, "enabled", False)

    # Build the global list of windows (loc, t) with t as target year
    windows: list[tuple[int, int]] = []
    for loc in range(L):
        for t in range(D, Y):
            windows.append((loc, t))

    total_windows = len(windows)
    if total_windows == 0:
        return float("inf"), global_opt_step, 0

    # Microbatch over windows — scale down with window length W = D+1
    base_mb = max(1, int(mb_size))
    scale = 1.0 / float(max(1, W))  # ~ 1/(ceil(H)+1)
    micro = max(1, int(base_mb * scale))
    micro = min(micro, total_windows)

    # Remember what we used (handy for logs)
    rollout_cfg["autotuned_mb_size_windows"] = int(micro)

    # DDP consistency: broadcast chosen micro from rank 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t_mb = torch.tensor([micro], dtype=torch.int64, device=device)
        torch.distributed.broadcast(t_mb, src=0)
        micro = int(t_mb.item())
        rollout_cfg["autotuned_mb_size_windows"] = int(micro)

    _LOG.info(
        "[carry-windowed][train][%s] D=%d, W=%d, windows_per_loc=%d, base_mb=%d, scale=%.4f, mb_size_windows=%d",
        granularity, D, W, (Y - D), base_mb, scale, micro
    )

    running_loss_sum = 0.0
    windows_done = 0
    microbatches_done = 0

    # helper: pool monthly/annual from DAILY ABSOLUTES
    def _pool_from_daily_abs(y_daily_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_m_daily = y_daily_abs[..., :nm]
        y_a_daily = y_daily_abs[..., nm:nm+na]
        pieces = [y_m_daily[:, bounds[i]:bounds[i+1], :].mean(dim=1, keepdim=True) for i in range(12)]
        preds_m = torch.cat(pieces, dim=1)                  # [B,12,nm]
        preds_a = y_a_daily.mean(dim=1, keepdim=True)       # [B,1, na]
        return preds_m, preds_a

    # Preserve/override model mode for speed/semantics
    model_mode_prev = getattr(model, "mode", None)
    if hasattr(model, "set_mode"):
        model.set_mode("sequential_months" if granularity == "monthly" else "batch_months")

    try:
        # Process in chunks of windows
        w_idx = 0
        while w_idx < total_windows:
            s = w_idx
            e = min(w_idx + micro, total_windows)
            w_idx = e
            B = e - s  # number of windows in this microbatch

            # Precompute locs/targets once per microbatch
            locs = [windows[k][0] for k in range(s, e)]
            targets = [windows[k][1] for k in range(s, e)]

            prev_m = None  # [B, len(out_m_idx)]
            prev_a = None  # [B, len(out_a_idx)]

            # Iterate years within window (granularity controls carry behavior)
            for y_off in range(W):
                y0s = [t - D + y_off for t in targets]  # absolute year index for this offset

                # Build a [B,365,nin] slice for this year across windows
                # Build year slice [B,365,nin]
                xb_list = []
                for loc, y_abs in zip(locs, y0s):
                    ds, de = _year_bounds(y_abs, y_abs + 1)
                    xb_list.append(inputs[:, ds:de, loc].T)  # [365,nin]
                xb = torch.stack(xb_list, dim=0).to(device, non_blocking=True)  # [B,365,nin]

                # Inject carries
                if y_off > 0:
                    if (prev_m is not None) and in_m_idx:
                        _inject_monthly_carry(
                            xb, prev_m, in_m_idx=in_m_idx,
                            first_month_len=first_month_len, granularity=granularity
                        )
                    if (prev_a is not None) and in_a_idx:
                        _safe_index_copy(
                            xb, 2, in_a_idx,
                            prev_a.unsqueeze(1).expand(B, 365, len(in_a_idx)),
                            where_len=365, name="annual carry→inputs"
                        )

                # INPUT guard (before forward)
                if not torch.isfinite(xb).all():
                    _LOG.warning("[carry-windowed/train] non-finite INPUT (B=%d, y_off=%d): %s",
                                xb.shape[0], y_off, _tsum(xb, "xb"))
                    del xb
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue

                # Forward one whole year
                preds_daily = model(xb)  # [B,365,nm+na] (deltas if enabled)

                # PRED guard
                if not torch.isfinite(preds_daily).all():
                    _LOG.warning("[carry-windowed/train] non-finite PRED (B=%d, y_off=%d): %s | %s",
                                preds_daily.shape[0], y_off, _tsum(xb, "xb"), _tsum(preds_daily, "preds_daily"))
                    del xb, preds_daily
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue
            

                # Reconstruct to daily ABSOLUTES if delta-enabled
                if delta_enabled:
                    preds_abs_daily = dc.reconstruct_groups_daily_segmentwise(
                        preds=preds_daily, nm=nm, na=na,
                        month_slices=rollout_cfg["month_slices"],
                        mode="carry",
                        out_m_idx=out_m_idx, out_a_idx=out_a_idx,
                        prev_monthly_state=(prev_m if (y_off > 0) else None),
                        prev_annual_state=(prev_a if (y_off > 0) else None),
                        yb_m_prev_last=None, yb_a_prev=None,
                    )
                else:
                    preds_abs_daily = preds_daily

                # Pool to month/annual and prep next prev_* from pooled absolutes
                preds_m, preds_a = _pool_from_daily_abs(preds_abs_daily)  # [B,12,nm], [B,1,na]
                # monthly carry vector choice depends on granularity
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

                    loss = loss_func(preds_abs_daily, ybm, yba)
                    (loss / eff_accum).backward()
                    microbatches_done += 1
                    windows_done += B
                    running_loss_sum += float(loss.detach().cpu())

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
                del xb, preds_daily, preds_abs_daily, preds_m, preds_a
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Flush remainder
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

    finally:
        if hasattr(model, "set_mode") and (model_mode_prev is not None):
            model.set_mode(model_mode_prev)

    avg_loss = running_loss_sum / max(1, windows_done)
    return avg_loss, global_opt_step, windows_done

@torch.no_grad()
def rollout_eval_outer_batch_windowed(
    *,
    model: torch.nn.Module,
    loss_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,        # [nin, 365*Y, L]
    labels_m: torch.Tensor,      # [nm,  12*Y, L]
    labels_a: torch.Tensor,      # [na,   1*Y, L]
    device: torch.device,
    rollout_cfg: dict,
) -> tuple[float, int]:
    """
    Windowed carry evaluation with tail-only loss; same semantics as training.
    Supports carry_granularity in {"monthly","annual"} with the same broadcast rules.
    """
    model.eval()

    # Config & guards
    granularity = str(rollout_cfg.get("carry_granularity", "monthly"))
    H = float(rollout_cfg.get("carry_horizon", 0.0) or 0.0)
    if granularity == "annual":
        assert (H == 0.0) or (H >= 1.0), \
            "In annual carry mode, carry_horizon must be 0 or >= 1 year (no fractional carry)."

    nin, Ttot, L = int(inputs.shape[0]), int(inputs.shape[1]), int(inputs.shape[2])
    Y = int(labels_a.shape[1])

    D = _ceil_years(H)
    W = D + 1
    if Y <= D or L <= 0:
        return 0.0, 0

    # Dims and indices (sanitize once)
    nm = int(labels_m.shape[0]); na = int(labels_a.shape[0])
    in_m_idx  = _sanitize_idx(rollout_cfg.get("in_monthly_state_idx",  []) or [], nin, "in_monthly_state_idx",  _LOG)
    in_a_idx  = _sanitize_idx(rollout_cfg.get("in_annual_state_idx",   []) or [], nin, "in_annual_state_idx",   _LOG)
    out_m_idx = _sanitize_idx(rollout_cfg.get("out_monthly_state_idx", []) or [], nm,  "out_monthly_state_idx", _LOG)
    out_a_idx = _sanitize_idx(rollout_cfg.get("out_annual_state_idx",  []) or [], na,  "out_annual_state_idx",  _LOG)

    # Month metadata + delta
    month_lengths = rollout_cfg.get("month_lengths", MONTH_LENGTHS_FALLBACK)
    bounds = [0]
    for m in month_lengths:
        bounds.append(bounds[-1] + m)
    first_month_len = int(month_lengths[0])

    dc = rollout_cfg.get("delta_ctx", None)
    delta_enabled = (dc is not None) and getattr(dc, "enabled", False)

    _LOG.info("[carry-windowed][eval][%s] D=%d, W=%d, windows_per_loc=%d", granularity, D, W, (Y - D))

    def _pool_from_daily_abs(y_daily_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_m_daily = y_daily_abs[..., :nm]
        y_a_daily = y_daily_abs[..., nm:nm+na]
        pieces = [y_m_daily[:, bounds[i]:bounds[i+1], :].mean(dim=1, keepdim=True) for i in range(12)]
        preds_m = torch.cat(pieces, dim=1)
        preds_a = y_a_daily.mean(dim=1, keepdim=True)
        return preds_m, preds_a

    # Preserve/override model mode for speed/semantics
    model_mode_prev = getattr(model, "mode", None)
    if hasattr(model, "set_mode"):
        model.set_mode("sequential_months" if granularity == "monthly" else "batch_months")

    total_loss = 0.0
    total_windows = 0

    try:
        for loc in range(L):
            for t in range(D, Y):
                prev_m = None
                prev_a = None

                # iterate over years inside the window [t-D .. t]
                for y_abs in range(t - D, t + 1):
                    ds, de = _year_bounds(y_abs, y_abs + 1)
                    xb = inputs[:, ds:de, loc].T.unsqueeze(0).to(device, non_blocking=True)  # [1,365,nin]

                    if prev_m is not None and in_m_idx:
                        _inject_monthly_carry(
                            xb, prev_m, in_m_idx=in_m_idx,
                            first_month_len=first_month_len, granularity=granularity
                        )
                    if prev_a is not None and in_a_idx:
                        _safe_index_copy(
                            xb, 2, in_a_idx,
                            prev_a.unsqueeze(1).expand(1, 365, len(in_a_idx)),
                            where_len=365, name="annual carry→inputs"
                        )

                    # INPUT guard (before forward)
                    if not torch.isfinite(xb).all():
                        _LOG.warning("[carry-windowed/eval] non-finite INPUT (loc=%d, y_abs=%d): %s",
                                    loc, y_abs, _tsum(xb, "xb"))
                        del xb
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        continue

                    # Forward
                    preds_daily = model(xb)

                    # PRED guard
                    if not torch.isfinite(preds_daily).all():
                        _LOG.warning("[carry-windowed/eval] non-finite PRED (loc=%d, y_abs=%d): %s | %s",
                                    loc, y_abs, _tsum(xb, "xb"), _tsum(preds_daily, "preds_daily"))
                        del xb, preds_daily
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        continue

                    if delta_enabled:
                        preds_abs_daily = dc.reconstruct_groups_daily_segmentwise(
                            preds=preds_daily, nm=nm, na=na,
                            month_slices=rollout_cfg["month_slices"],
                            mode="carry",
                            out_m_idx=out_m_idx, out_a_idx=out_a_idx,
                            prev_monthly_state=prev_m, prev_annual_state=prev_a,
                            yb_m_prev_last=None, yb_a_prev=None,
                        )
                    else:
                        preds_abs_daily = preds_daily

                    preds_m, preds_a = _pool_from_daily_abs(preds_abs_daily)
                    # choose next monthly carry according to granularity
                    next_m = _prev_monthly_from_preds(preds_m, out_m_idx, granularity)
                    prev_m = (next_m.detach() if next_m is not None else None)
                    prev_a = (preds_a[:, 0, out_a_idx].detach() if out_a_idx else None)

                    # Tail-year loss (y_abs == t)
                    if y_abs == t:
                        ms, me = t * 12, (t + 1) * 12
                        ybm = labels_m[:, ms:me, loc].T.unsqueeze(0).to(device, non_blocking=True)  # [1,12,nm]
                        yba = labels_a[:, t:(t + 1), loc].T.unsqueeze(0).to(device, non_blocking=True)  # [1,1,na]
                        
                        if (not torch.isfinite(preds_abs_daily).all()
                            or not torch.isfinite(ybm).all()
                            or not torch.isfinite(yba).all()):
                            _LOG.warning(
                                "[carry-windowed/eval] non-finite detected (loc=%d, target_year=%d, y_abs=%d): "
                                "preds_abs_daily=%s, ybm=%s, yba=%s",
                                loc, t, y_abs,
                                torch.isfinite(preds_abs_daily).all().item(),
                                torch.isfinite(ybm).all().item(),
                                torch.isfinite(yba).all().item(),
                            )
                            # Skip tail-loss for this window
                            del xb, preds_daily, preds_abs_daily, preds_m, preds_a
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                            continue
                        
                        loss = loss_func(preds_abs_daily, ybm, yba)
                        total_loss += float(loss.detach().cpu())
                        total_windows += 1

                    del xb, preds_daily, preds_abs_daily, preds_m, preds_a
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        if hasattr(model, "set_mode") and (model_mode_prev is not None):
            model.set_mode(model_mode_prev)

    return total_loss, total_windows