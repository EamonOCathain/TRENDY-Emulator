# src/training/trainer.py
from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import logging
import random
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project imports / path setup
# ---------------------------------------------------------------------------
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.training.distributed import ddp_mean_scalar
from src.training.history import History

# Logger
_LOG = logging.getLogger("carry")

MONTH_LENGTHS_FALLBACK = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
POPULATION_NORMALISE = True
FULL_SEQUENCE_TEST_MB_SIZE = 35
# ---------------------------------------------------------------------------
# Helpers for rollout 
# ---------------------------------------------------------------------------

def _year_bounds(y0: int, y1: int) -> tuple[int, int]:
    """Return [day_start, day_end) indices for years [y0..y1) in a 365*n layout."""
    return (y0 * 365, y1 * 365)

def _slice_last_year_bounds(t: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ([month_start, month_end), [year_start, year_end)) for target year t."""
    return (t * 12, (t + 1) * 12), (t, t + 1)

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

def _is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

def _log_main(log, level: str, msg: str, *args):
    if _is_main_process():
        getattr(log, level)(msg, *args)

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

def _prev_monthly_from_preds(
    preds_m: torch.Tensor,
    out_m_idx: List[int],
) -> Optional[torch.Tensor]:
    """From monthly (absolutes), take December (m=11) for the carry vector. Returns [B, len(out_m_idx)]."""
    if (preds_m is None) or (not out_m_idx):
        return None
    return preds_m[:, -1, out_m_idx]

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

def _mask_ignored_nans(
    tensor: torch.Tensor,
    name_list: list[str],
    ignore_vars: set[str] | None = None
) -> torch.Tensor:
    """Return tensor with NaNs in ignored variables replaced by finite dummy (0)."""
    if tensor is None or not name_list:
        return tensor
    n_chan = tensor.shape[1] if tensor.ndim >= 2 else tensor.shape[0]
    if len(name_list) != n_chan or not ignore_vars:
        return tensor
    mask = torch.tensor([v in ignore_vars for v in name_list], dtype=torch.bool)
    if not mask.any():
        return tensor
    t_copy = tensor.clone()
    t_copy[:, mask, ...] = torch.nan_to_num(t_copy[:, mask, ...], nan=0.0)
    return t_copy

# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop when the monitored metric fails to improve by at least `min_delta`
    for `patience` epochs (after an optional warmup). Assumes 'min' mode.
    """
    def __init__(self, patience: int, min_delta: float = 0.0, warmup_epochs: int = 0):
        assert patience > 0
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup_epochs = int(warmup_epochs)
        self.best = float("inf")
        self.bad_epochs = 0
        self.should_stop = False
        self.best_epoch = -1

    def step(self, value: float, epoch_idx: int) -> None:
        if epoch_idx < self.warmup_epochs:
            return
        if value < self.best - self.min_delta:
            self.best = value
            self.best_epoch = epoch_idx
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True

    def step_on_check(self, value: float, epoch_idx: int) -> None:
        if epoch_idx < self.warmup_epochs:
            return
        if value < self.best - self.min_delta:
            self.best = value
            self.best_epoch = epoch_idx
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True

# ---------------------------------------------------------------------------
# Helpers for Validation 
# ---------------------------------------------------------------------------

def plan_validation(
    train_stats: Dict[str, int],
    valid_dl,
    validation_frequency: float,
    validation_size: float,
) -> Dict[str, int]:
    """
    Plan how often and how many batches to use for validation, and (optionally)
    precompute a fixed subset of validation batch indices to reuse every time.
    """
    # frequency -> every N train batches
    train_batches = int(train_stats["batches"])
    validate_every_batches = max(1, int(round(validation_frequency * train_batches)))

    # how many val batches exist / to use each probe
    val_total_batches = len(valid_dl)
    val_batches_to_use = int(round(validation_size * val_total_batches))
    val_batches_to_use = min(val_total_batches, max(1, val_batches_to_use))

    # fixed subset of val batch indices (same across the run)
    rng = random.Random(42)  # constant seed (deterministic subset)
    indices = list(range(val_total_batches))
    rng.shuffle(indices)
    fixed_ids = sorted(indices[:val_batches_to_use])

    return {
        "validate_every_batches": validate_every_batches,
        "val_batches_to_use": val_batches_to_use,
        "train_batches": train_batches,
        "val_total_batches": val_total_batches,
        "fixed_val_batch_ids": fixed_ids,
    }

def _sample_validation_batch_indices(total_batches: int, k: int) -> List[int]:
    """Randomly select k validation batch indices from total_batches (sorted)."""
    k = min(max(0, int(k)), int(total_batches))
    if k == 0:
        return []
    if k >= total_batches:
        return list(range(total_batches))
    return random.sample(range(total_batches), k)

# ---------------------------------------------------------------------------
# Helpers for Mass Balance Logging 
# ---------------------------------------------------------------------------
def _accum_bd_sums(acc: Dict[str, float], bd: Optional[dict], mult: float) -> None:
    """
    Accumulate per-call weighted contributions from loss_fn.last_breakdown["weighted"].
    `mult` should be the number of windows represented by this loss call.
    """
    if not bd or "weighted" not in bd:
        return
    for k, v in bd["weighted"].items():
        if v is None:
            continue
        acc[k] = acc.get(k, 0.0) + float(v) * float(mult)

def _normalize_bd_sums(acc: Dict[str, float], denom: float) -> Dict[str, float]:
    """Convert accumulated sums to averages per window."""
    if denom <= 0:
        return {k: float("nan") for k in acc.keys()}
    return {k: (v / denom) for k, v in acc.items()}

def log_mb_series(log: Optional[logging.Logger], tag: str, mb_dict: Dict[str, float]) -> None:
    """
    Pretty print mass-balance key/values in a stable, compact format.
    """
    if not log or not mb_dict:
        return
    line = " | ".join(f"{k}={mb_dict[k]:.6f}" for k in sorted(mb_dict))
    log.info("%s: %s", tag, line)

# ---------------------------------------------------------------------------
# DDP Helpers
# ---------------------------------------------------------------------------

def unwrap(model):
    """Return the underlying model when wrapped in DDP; otherwise return as-is."""
    return model.module if isinstance(model, DDP) else model

def is_main_rank() -> bool:
    return (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)


def broadcast_indices_for_ddp(batch_ids: List[int], device: torch.device) -> List[int]:
    """
    Ensure all ranks use the same list of validation/test batch indices.
    On rank 0: send; others: receive.
    Returns a plain Python list on every rank.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return list(batch_ids)

    rank0 = (dist.get_rank() == 0)
    if rank0:
        idx = torch.tensor(batch_ids, dtype=torch.int64, device=device)
        n   = torch.tensor([idx.numel()], dtype=torch.int64, device=device)
    else:
        n   = torch.zeros(1, dtype=torch.int64, device=device)

    dist.broadcast(n, src=0)
    if not rank0:
        idx = torch.empty(int(n.item()), dtype=torch.int64, device=device)

    dist.broadcast(idx, src=0)
    return idx.tolist()

@contextmanager
def model_mode(model, mode: Optional[str]):
    log = logging.getLogger("monthly_mode")
    target = unwrap(model)
    if mode is None or not hasattr(target, "set_mode"):
        yield
        return
    prev = getattr(target, "mode", None)
    log.info("[mode] enter: requested=%s | before=%s | ddp=%s",
             mode, prev, isinstance(model, DDP))
    try:
        target.set_mode(mode)
        log.info("[mode] applied: actual_now=%s", getattr(target, "mode", None))
        yield
    finally:
        if prev is not None:
            target.set_mode(prev)
        log.info("[mode] exit: restored_to=%s | actual_now=%s",
                 prev, getattr(target, "mode", None))


# =============================================================================
# Rollout Over a Single Batch (for train/val/test)
# =============================================================================

def rollout(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor,        # [nin, 365*Y, L]
    labels_m: torch.Tensor,      # [nm,  12*Y, L]
    labels_a: torch.Tensor,      # [na,   1*Y, L]
    device: torch.device,
    rollout_cfg: dict,
    # mode & optimisation toggles
    training: bool = False,
    loss_func: Optional[Callable[..., torch.Tensor]] = None,
    opt: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    mb_size: Optional[int] = None,      # windows per microbatch
    eff_accum: int = 1,
    eff_clip: Optional[float] = None,
    history=None,
    global_opt_step: int = 0,
    logger: Optional[logging.Logger] = None,
) -> tuple[float, int, int, Dict[str, float]]:
    """
    Unified windowed loop for training/validation/testing.

    Policy knobs in rollout_cfg (all optional; defaults shown):
      - carry_horizon: int H (default 0). Window spans W=H+1 years.
      - loss_policy: "tail_only" | "all_years"   (default "tail_only")
      - min_target_year: int (default 1)         skip scoring year 0 everywhere
      - weighting_policy: "per_target" | "per_window"
            For "all_years": how to count denominator units.
            Default "per_target" (recommended for comparability).
    Returns:
      (sum_loss, units_done, next_global_opt_step, mb_sums)
    """
    log = logger or _LOG

    # ---- read policy knobs ----
    H = int(rollout_cfg.get("carry_horizon", 0) or 0)
    loss_policy = (rollout_cfg.get("loss_policy") or "tail_only").lower()
    min_target_year = int(rollout_cfg.get("min_target_year", 1) or 1)
    weighting_policy = (rollout_cfg.get("weighting_policy") or
                        ("per_target" if loss_policy == "all_years" else "per_window")).lower()

    assert loss_policy in {"tail_only", "all_years"}
    assert weighting_policy in {"per_target", "per_window"}
    
    # Decide monthly model mode
    user_mode = rollout_cfg.get("monthly_mode", "batch_months")
    assert user_mode in ("batch_months", "sequential_months"), \
        f"Invalid monthly_mode='{user_mode}' (expected 'batch_months' or 'sequential_months')"

    # Always use sequential when carry_horizon > 0
    target_mode = "sequential_months" if (H > 0) else user_mode

    nin, Ttot, L = int(inputs.shape[0]), int(inputs.shape[1]), int(inputs.shape[2])
    Y = int(labels_a.shape[1])
    if L <= 0 or Y <= 0:
        return 0.0, 0, global_opt_step, {}

    nm = int(labels_m.shape[0]); na = int(labels_a.shape[0])

    # indices (sanitised against current outer batch)
    in_m_idx  = _sanitize_idx(rollout_cfg.get("in_monthly_state_idx", []) or [], nin, "in_monthly_state_idx", log)
    in_a_idx  = _sanitize_idx(rollout_cfg.get("in_annual_state_idx", [])  or [], nin, "in_annual_state_idx",  log)
    out_m_idx = _sanitize_idx(rollout_cfg.get("out_monthly_state_idx", []) or [], nm,  "out_monthly_state_idx", log)
    out_a_idx = _sanitize_idx(rollout_cfg.get("out_annual_state_idx", [])  or [], na,  "out_annual_state_idx",  log)

    # month metadata
    month_lengths = rollout_cfg.get("month_lengths", MONTH_LENGTHS_FALLBACK)
    bounds = [0]
    for m in month_lengths:
        bounds.append(bounds[-1] + m)
    first_month_len = int(month_lengths[0])

    # helper: pool from daily absolutes → monthly means & annual means
    def _pool_from_daily_abs(y_daily_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_m_daily = y_daily_abs[..., :nm]
        y_a_daily = y_daily_abs[..., nm:nm+na]
        pieces = [y_m_daily[:, bounds[i]:bounds[i+1], :].mean(dim=1, keepdim=True) for i in range(12)]
        preds_m = torch.cat(pieces, dim=1)            # [B,12,nm]
        preds_a = y_a_daily.mean(dim=1, keepdim=True) # [B,1, na]
        return preds_m, preds_a

    # choose valid target years: must satisfy t >= H and t >= min_target_year
    t_min = max(H, min_target_year)
    if t_min >= Y:
        return 0.0, 0, global_opt_step, {}

    # windows = (loc, target_year) for t in [t_min .. Y-1]
    windows: list[tuple[int, int]] = [(loc, t) for loc in range(L) for t in range(t_min, Y)]
    if not windows:
        return 0.0, 0, global_opt_step, {}

    # microbatch size in windows
    if training:
        micro = max(1, int(mb_size or 1))
    else:
        base = int(mb_size) if (mb_size is not None and mb_size > 0) else 2048
        micro = max(1, base)
    micro = min(micro, len(windows))
    rollout_cfg["autotuned_mb_size_windows"] = int(micro)

    # DDP: broadcast micro so all ranks use same (only while training)
    if training and torch.distributed.is_available() and torch.distributed.is_initialized():
        t_mb = torch.tensor([micro], dtype=torch.int64, device=device)
        torch.distributed.broadcast(t_mb, src=0)
        micro = int(t_mb.item())
        rollout_cfg["autotuned_mb_size_windows"] = int(micro)

    # move tensors once
    inputs   = inputs.to(device, non_blocking=True)
    labels_m = labels_m.to(device, non_blocking=True)
    labels_a = labels_a.to(device, non_blocking=True)

    total_loss = 0.0
    units_done = 0           # denominator units (per-window or per-target)
    microbatches_done = 0
    mb_sums: Dict[str, float] = {}

    # modes
    if training: model.train()
    else:        model.eval()

    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with model_mode(model, target_mode), grad_ctx:
        w_idx = 0
        while w_idx < len(windows):
            s = w_idx; e = min(w_idx + micro, len(windows)); w_idx = e
            B = e - s
            locs    = [windows[k][0] for k in range(s, e)]
            targets = [windows[k][1] for k in range(s, e)]

            prev_m = None
            prev_a = None

            # iterate year-by-year inside each window
            W = H + 1
            for y_off in range(W):
                y_abs_list = [t - H + y_off for t in targets]

                # build xb [B,365,nin]
                xb_list = []
                for loc, y_abs in zip(locs, y_abs_list):
                    ds, de = _year_bounds(y_abs, y_abs + 1)
                    xb_list.append(inputs[:, ds:de, loc].T)
                xb = torch.stack(xb_list, dim=0)  # [B,365,nin]

                # inject carries for y_off > 0
                if y_off > 0:
                    if (prev_m is not None) and in_m_idx:
                        _inject_monthly_carry(xb, prev_m, in_m_idx=in_m_idx,
                                              first_month_len=first_month_len, logger=log)
                    if (prev_a is not None) and in_a_idx:
                        if torch.isfinite(prev_a).all():
                            _safe_index_copy(
                                xb, 2, in_a_idx,
                                prev_a.unsqueeze(1).expand(B, 365, len(in_a_idx)),
                                where_len=365, name="annual carry→inputs", logger=log,
                            )
                        else:
                            log.warning("[carry] non-finite annual carry vector; skipping injection.")

                # guard inputs (allow NaNs only for monthly-carry vars after Jan)
                if not _isfinite_except_monthly_carry_mask(xb, in_m_idx, first_month_len):
                    # Extra one-shot debug on first failure
                    if y_off == 1 and B > 0:
                        bad = ~torch.isfinite(xb)
                        # collapse batch
                        bad_any = bad.any(dim=0)  # [365, nin]
                        # show first few offending (day, channel)
                        where = bad_any.nonzero(as_tuple=False)
                        log.warning("[debug] non-finite INPUT at %d positions; first 10 (day,chan): %s",
                                    where.size(0), where[:10].tolist())
                    log.warning("[rollout] non-finite INPUT (disallowed NaNs) (B=%d, y_off=%d)", B, y_off)
                    del xb
                    continue

                # forward → daily absolutes
                preds_abs_daily = model(xb)  # [B,365,nm+na]
                if not torch.isfinite(preds_abs_daily).all():
                    log.warning("[rollout] non-finite PRED (B=%d, y_off=%d)", B, y_off)
                    del xb, preds_abs_daily
                    continue

                preds_m, preds_a = _pool_from_daily_abs(preds_abs_daily)

                # next carry (detach during training)
                next_m = _prev_monthly_from_preds(preds_m, out_m_idx)
                if training:
                    prev_m = (next_m.detach() if next_m is not None else None)
                    prev_a = (preds_a[:, 0, out_a_idx].detach() if out_a_idx else None)
                else:
                    prev_m = (next_m if next_m is not None else None)
                    prev_a = (preds_a[:, 0, out_a_idx] if out_a_idx else None)

                # decide whether to score this y_off (tail_only vs all_years)
                t_abs_vec = torch.as_tensor([t - H + y_off for t in targets], device=device)
                score_this_step = False
                if loss_policy == "tail_only":
                    score_this_step = (y_off == W - 1)  # only tail year
                else:
                    # all target years where t_abs >= min_target_year
                    score_this_step = bool(torch.any(t_abs_vec >= min_target_year).item())

                if score_this_step:
                    # labels for current target year(s)
                    # tail_only: one target per window (t)
                    # all_years: if min_target_year cuts, only those >= threshold count
                    ybm_list = []
                    yba_list = []
                    valid_mask = []
                    for loc, t in zip(locs, targets):
                        t_abs = t - H + y_off
                        if loss_policy == "all_years" and t_abs < min_target_year:
                            valid_mask.append(False)
                            ybm_list.append(None); yba_list.append(None)
                            continue
                        (ms, me), (ys, ye) = _slice_last_year_bounds(t_abs)
                        ybm_list.append(labels_m[:, ms:me, loc].T)  # [12,nm]
                        yba_list.append(labels_a[:, ys:ye, loc].T)  # [1, na]
                        valid_mask.append(True)

                    if not any(valid_mask):
                        del xb, preds_abs_daily, preds_m, preds_a
                        continue

                    # pack only valid rows
                    xb_sel = xb  # <- ensure defined for both branches
                    if all(valid_mask):
                        ybm = torch.stack(ybm_list, dim=0)              # [B,12,nm]
                        yba = torch.stack(yba_list, dim=0)              # [B, 1,na]
                        B_eff = B
                        # xb_sel stays as xb
                    else:
                        sel = [i for i, ok in enumerate(valid_mask) if ok]
                        ybm = torch.stack([ybm_list[i] for i in sel], dim=0)
                        yba = torch.stack([yba_list[i] for i in sel], dim=0)
                        preds_abs_daily = preds_abs_daily[sel, ...]
                        xb_sel = xb[sel, :, :]                          
                        B_eff = len(sel)

                    if (not torch.isfinite(ybm).all()) or (not torch.isfinite(yba).all()):
                        log.warning("[rollout] non-finite LABELS (B_eff=%d)", B_eff)
                        del xb, preds_abs_daily, preds_m, preds_a, ybm, yba
                        continue

                    # extra_daily (optional)
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
                                feat_phys = preds_abs_daily.new_empty((B_eff, 365))
                                feat_phys[:] = xb_sel[:, :, idx] * sd + mu
                                ed[feat] = feat_phys
                        if ed:
                            extra_daily = ed
                    except Exception:
                        extra_daily = None

                    # compute loss on daily absolutes (same signature)
                    try:
                        loss = loss_func(preds_abs_daily, ybm, yba, extra_daily=extra_daily)
                    except TypeError:
                        loss = loss_func(preds_abs_daily, ybm, yba)

                    # weighting: what is one “unit”?
                    if loss_policy == "tail_only" or weighting_policy == "per_window":
                        unit_contrib = (B_eff if POPULATION_NORMALISE else 1)
                    else:  # all_years & per_target
                        unit_contrib = (B_eff if POPULATION_NORMALISE else 1)

                    total_loss += float(loss.detach().cpu()) * unit_contrib
                    units_done += unit_contrib

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

                    # mass-balance breakdown
                    bd = getattr(loss_func, "last_breakdown", None)
                    if isinstance(bd, dict) and ("weighted" in bd):
                        for k, v in bd["weighted"].items():
                            if v is None:
                                continue
                            mb_sums[k] = mb_sums.get(k, 0.0) + float(v) * float(unit_contrib)

                del xb, preds_abs_daily, preds_m, preds_a

        # flush pending grads
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

    return total_loss, units_done, global_opt_step, mb_sums

def train(
    epochs: int,
    model: torch.nn.Module,
    loss_func,
    opt: Optimizer,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    log: Optional[logging.Logger] = None,
    save_cb: Optional[Callable[[int, bool, float, "History"], None]] = None,
    accum_steps: Optional[int] = None,
    grad_clip: Optional[float] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    val_plan: Optional[dict] = None,
    ddp: bool = False,
    early_stop_patience: Optional[int] = None,
    early_stop_min_delta: float = 0.0,
    early_stop_warmup_epochs: int = 0,
    start_dt: Optional[datetime] = None,
    run_dir: Optional[Path] = None,
    args: Optional[object] = None,   # <- REQUIRED now (must carry mb sizes)
    start_epoch: int = 0,
    best_val_init: float = float("inf"),
    history_seed: Optional[dict] = None,
    samples_seen_seed: int = 0,
    rollout_cfg: Optional[dict] = None,
    validate_only: bool = False,
) -> Tuple["History", float, bool]:
    """
    Unified training loop using a single rollout() engine for train/val.

    Policy (for both train and val):
      • carry_horizon = rollout_cfg["carry_horizon"] (0 ⇒ TF, >0 ⇒ carry)
      • loss_policy   = "tail_only"
      • min_target_year = 1  (skip scoring year 0)
      • weighting     = per-window (implicit for tail_only)

    Returns: (history, best_val, stopped_early_flag)
    """

    history = History(model)
    stopped_early_flag = False
    
    # Make sure args passed in
    assert args is not None, "trainer.train/validate require `args` (with train_mb_size/eval_mb_size)."

    train_mb = int(args.train_mb_size)

    # --- Rehydrate history (resume) ---
    if history_seed:
        history.train_loss       = list(history_seed.get("train_loss", []))
        history.val_loss         = list(history_seed.get("val_loss", []))
        history.batch_loss       = list(history_seed.get("batch_loss", []))
        history.batch_step       = list(history_seed.get("batch_step", []))
        history.epoch_edges      = list(history_seed.get("epoch_edges", [0]))
        history.val_loss_batches = list(history_seed.get("val_loss_batches", []))
        history.val_loss_steps   = list(history_seed.get("val_loss_steps", []))
        history.lr_values        = list(history_seed.get("lr_values", []))
        history.lr_steps         = list(history_seed.get("lr_steps", []))

    history.samples_seen = int(samples_seen_seed)
    best_val = float(best_val_init)
    
    # --- Effective knobs ---
    eff_accum = 1 if (accum_steps is None or accum_steps <= 1) else int(accum_steps)
    eff_clip  = grad_clip if (grad_clip is not None and grad_clip > 0) else None

    # Helper flags / device
    is_main_fit = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
    model_device = next(model.parameters()).device

    # Log schema signature if provided
    if log and is_main_fit:
        sig = (rollout_cfg or {}).get("schema_sig")
        if sig:
            log.info("[schema] signature=%s", sig)

    # ---------- Validation-only short-circuit ----------
    if validate_only:
        batch_indices = None
        batch_indices = list(val_plan.get("fixed_val_batch_ids", [])) if val_plan else None

        avg_val_loss, val_cnt, val_mb_avgs = validate(
            model, loss_func, valid_dl,
            device=model_device,
            batch_indices=batch_indices,
            rollout_cfg=rollout_cfg,
            args=args
        )
        
        history.update(train_loss=float("nan"), val_loss=avg_val_loss)
        if log and is_main_fit:
            log.info("Validation-only: val_loss=%.6f (windows=%d)", avg_val_loss, val_cnt)
            if getattr(args, "use_mass_balances", False) and val_mb_avgs:
                log.info("Val-only MB (weighted avg per-window): %s",
                         " | ".join(f"{k}={v:.6f}" for k, v in sorted(val_mb_avgs.items())))
        return history, avg_val_loss, False

    # --- Early stopping state (rehydrate if provided) ---
    early = None
    if early_stop_patience is not None and early_stop_patience > 0:
        early = EarlyStopping(
            patience=early_stop_patience,
            min_delta=early_stop_min_delta,
            warmup_epochs=early_stop_warmup_epochs,
        )
    if early is not None and history_seed is not None:
        es = history_seed.get("_early_state") or history_seed.get("early_state")
        if es and es.get("enabled", False):
            early.best       = float(es.get("best", early.best))
            early.bad_epochs = int(es.get("bad_epochs", 0))
            early.best_epoch = int(es.get("best_epoch", -1))

    # --- Nothing to do if already finished ---
    if start_epoch >= epochs:
        if log and is_main_fit:
            log.info("Resume requested at epoch %d (>= total epochs %d). Nothing to do.", start_epoch, epochs)
        return history, best_val, False

    global_batch_idx = len(history.batch_step) if history.batch_step else 0
    global_opt_step  = len(history.lr_steps)   if history.lr_steps   else 0

    # -----------------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):

        # DDP per-epoch reshuffle
        if ddp:
            if hasattr(train_dl.sampler, "set_epoch"):
                train_dl.sampler.set_epoch(epoch)
            if hasattr(valid_dl.sampler, "set_epoch"):
                valid_dl.sampler.set_epoch(epoch)

        if log and is_main_fit:
            log.info(
                "Starting Training - [Epoch %d/%d] accum_steps=%s, train_mb_size=%s",
                epoch + 1, epochs, eff_accum, train_mb
            )

        every_n = max(1, len(train_dl) // 20)

        model.train()
        train_losses: List[float] = []
        opt.zero_grad(set_to_none=True)

        # Mass-balance epoch accumulators (train)
        epoch_train_mb_sums: Dict[str, float] = {}
        epoch_train_units: int = 0

        # ============================== outer-batch loop ==============================
        for batch_idx, (batch_inputs, batch_monthly, batch_annual) in enumerate(train_dl):
            # Move once per outer batch
            inputs   = batch_inputs.squeeze(0).float().to(model_device)
            labels_m = batch_monthly.squeeze(0).float().to(model_device)
            labels_a = batch_annual.squeeze(0).float().to(model_device)

            # Build *train* rollout policy (TF or carry depending on cfg)
            train_cfg = dict(rollout_cfg or {})
            train_cfg.update({
                "loss_policy": "tail_only",
                "min_target_year": 1,
                "monthly_mode": getattr(args, "model_monthly_mode", "batch_months"),
            })

            # Call the rollout for train
            sum_loss, units_done, global_opt_step, mb_step = rollout(
                model=model,
                loss_func=loss_func,
                opt=opt,
                scheduler=scheduler,
                inputs=inputs,
                labels_m=labels_m,
                labels_a=labels_a,
                mb_size=train_mb,   
                eff_accum=eff_accum,
                eff_clip=eff_clip,
                history=history,
                global_opt_step=global_opt_step,
                device=model_device,
                rollout_cfg=train_cfg,
                training=True,
            )

            avg_batch_loss = sum_loss / max(1, units_done)
            train_losses.append(float(avg_batch_loss))
            history.add_batch(float(avg_batch_loss), global_batch_idx)
            global_batch_idx += 1

            # samples_seen consistent with scored units
            history.samples_seen += int(units_done)

            # Mass-balance tallies
            if mb_step:
                for k, v in mb_step.items():
                    epoch_train_mb_sums[k] = epoch_train_mb_sums.get(k, 0.0) + float(v)
            epoch_train_units += int(units_done)

            # --------------------------- in-epoch validation ---------------------------
            if val_plan is not None:
                veb = val_plan["validate_every_batches"]
                if (veb < len(train_dl)) and ((batch_idx + 1) % veb == 0):
                    total_val_batches = val_plan["val_total_batches"]

                    if "fixed_val_batch_ids" in val_plan:
                        batch_ids = list(val_plan["fixed_val_batch_ids"])
                    else:
                        use_k = max(1, int(round(val_plan["val_batches_to_use"])))
                        batch_ids = _sample_validation_batch_indices(total_val_batches, use_k)

                    # DDP: broadcast chosen val batches to all ranks
                    if ddp and dist.is_available() and dist.is_initialized():
                        is_rank0 = (dist.get_rank() == 0)
                        dev = model_device
                        if is_rank0:
                            ids  = torch.tensor(batch_ids, dtype=torch.int64, device=dev)
                            size = torch.tensor([ids.numel()], dtype=torch.int64, device=dev)
                        else:
                            size = torch.zeros(1, dtype=torch.int64, device=dev)
                        dist.broadcast(size, src=0)
                        if not is_rank0:
                            ids = torch.empty(int(size.item()), dtype=torch.int64, device=dev)
                        dist.broadcast(ids, src=0)
                        batch_ids = ids.tolist()

                    interim_avg, interim_cnt, _ = validate(
                        model, loss_func, valid_dl, device=model_device, batch_indices=batch_ids,
                        rollout_cfg=rollout_cfg,
                        args = args
                    )

                    # Mid-epoch early stopping check
                    if early is not None:
                        early.step_on_check(interim_avg, epoch)
                        stop_now = torch.tensor(
                            [1 if (is_main_fit and early.should_stop) else 0],
                            device=model_device,
                            dtype=torch.int32,
                        )
                        if ddp and dist.is_available() and dist.is_initialized():
                            dist.broadcast(stop_now, src=0)

                        if int(stop_now.item()) == 1:
                            if log and is_main_fit:
                                log.info(
                                    "[Early stop mid-epoch] epoch %d (best @ epoch %d, best_val=%.6f)",
                                    epoch + 1, early.best_epoch + 1, early.best
                                )
                            stopped_early_flag = True
                            break  # break outer-batch loop for this epoch

                    if log and is_main_fit:
                        log.info("[Validation @ batch %d] avg_loss=%.6f on %d selected val batches (windows=%d)",
                                 batch_idx + 1, interim_avg, len(batch_ids), interim_cnt)

                    history.add_val_batch(interim_avg, global_batch_idx)

                    # Snapshot early state for resume
                    if early is not None:
                        history._early_state = {
                            "enabled": True,
                            "patience": early.patience,
                            "min_delta": early.min_delta,
                            "warmup_epochs": early.warmup_epochs,
                            "best": early.best,
                            "bad_epochs": early.bad_epochs,
                            "best_epoch": early.best_epoch,
                        }
                    else:
                        history._early_state = {"enabled": False}

                    # Save best-on-interim if desired
                    if interim_avg < best_val and save_cb:
                        best_val = interim_avg
                        save_cb(epoch, best=True, val=best_val, history=history)

            # ------------------------------ progress log ------------------------------
            if log and is_main_fit and (
                ((batch_idx + 1) % every_n == 0) or ((batch_idx + 1) == len(train_dl))
            ):
                total_outer_batches = max(1, epochs * len(train_dl))
                overall_done = epoch * len(train_dl) + (batch_idx + 1)
                percent_done = 100.0 * overall_done / total_outer_batches
                log.info(
                    "Progress %5.1f%% — epoch %d/%d, batch %d/%d, avg_batch_loss=%.6f",
                    percent_done, epoch + 1, epochs, batch_idx + 1, len(train_dl), train_losses[-1]
                )

        # ============================ end outer-batch loop ============================
        history.close_epoch()

        # --- HARD CLEANUP before epoch-end validation ---
        for _name in (
            "inputs","labels_m","labels_a","mb_step",
        ):
            if _name in locals():
                try: del locals()[_name]
                except Exception: pass
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Break out if we stopped mid-epoch
        if stopped_early_flag:
            break

        # Epoch-level metrics (+ DDP reduce)
        avg_train_loss = float(np.mean(train_losses)) if len(train_losses) else float("nan")
        avg_train_loss = ddp_mean_scalar(avg_train_loss, model_device)

        # Broadcast validation plan so DDP all use the same set
        batch_indices = val_plan.get("fixed_val_batch_ids") if val_plan else None
        if ddp and dist.is_available() and dist.is_initialized() and batch_indices is not None:
            batch_indices = broadcast_indices_for_ddp(batch_indices, device=model_device)

        avg_val_loss, val_cnt, val_mb_avgs = validate(
            model, loss_func, valid_dl,
            device=model_device,
            batch_indices=batch_indices,  
            rollout_cfg=rollout_cfg,
            args=args,
        )

        if val_cnt == 0:
            if log and is_main_fit:
                log.warning("Validation produced 0 windows — check batch_indices/filters.")
            avg_val_loss = float("inf")

        # Train MB averages for this epoch (weighted per-unit)
        train_mb_avgs = _normalize_bd_sums(epoch_train_mb_sums, max(1, epoch_train_units))

        if log and is_main_fit:
            log.info("Epoch average train loss=%.6f, val loss=%.6f (val_cnt=%d)",
                     avg_train_loss, avg_val_loss, val_cnt)
            if getattr(args, "use_mass_balances", False):
                if train_mb_avgs:
                    log.info("Epoch MB train (weighted avg per-unit): %s",
                             " | ".join(f"{k}={v:.6f}" for k, v in sorted(train_mb_avgs.items())))
                if val_mb_avgs:
                    log.info("Epoch MB val   (weighted avg per-window): %s",
                             " | ".join(f"{k}={v:.6f}" for k, v in sorted(val_mb_avgs.items())))

        history.update(train_loss=avg_train_loss, val_loss=avg_val_loss)

        # Persist MB series for plotting (if supported)
        if getattr(args, "use_mass_balances", False) and hasattr(history, "add_mass_balance_epoch"):
            history.add_mass_balance_epoch(train_mb_avgs, val_mb_avgs)

        # Early stopping update on epoch end
        if early is not None:
            early.step_on_check(avg_val_loss, epoch)

        # Per-epoch plots (best-effort)
        if log and is_main_fit and (run_dir is not None and args is not None and start_dt is not None):
            try:
                elapsed_seconds = (datetime.now() - start_dt).total_seconds()
                history.save_epoch_plots_overwrite(run_dir, args, start_dt, elapsed_seconds)
            except Exception as e:
                log.warning("Per-epoch plotting failed: %s", e)

        # Snapshot early state before saving
        if early is not None:
            history._early_state = {
                "enabled": True,
                "patience": early.patience,
                "min_delta": early.min_delta,
                "warmup_epochs": early.warmup_epochs,
                "best": early.best,
                "bad_epochs": early.bad_epochs,
                "best_epoch": early.best_epoch,
            }
        else:
            history._early_state = {"enabled": False}

        # Checkpointing (best + rolling)
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            if save_cb:
                save_cb(epoch, best=True, val=best_val, history=history)
        if save_cb:
            save_cb(epoch, best=False, val=avg_val_loss, history=history)

        # Final early-stopping decision for this epoch
        if early is not None:
            stop_flag = torch.tensor(
                [1 if (is_main_fit and early.should_stop) else 0],
                device=model_device,
                dtype=torch.int32,
            )
            if ddp and dist.is_available() and dist.is_initialized():
                dist.broadcast(stop_flag, src=0)

            if int(stop_flag.item()) == 1:
                if log and is_main_fit:
                    log.info(
                        "Early stopping triggered at epoch %d (best @ epoch %d, best_val=%.6f)",
                        epoch + 1, early.best_epoch + 1, early.best
                    )
                stopped_early_flag = True
                break

    # DDP: make samples_seen global
    if ddp and dist.is_available() and dist.is_initialized():
        t = torch.tensor([history.samples_seen], device=next(model.parameters()).device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        history.samples_seen = int(t.item())

    return history, best_val, stopped_early_flag

def validate(
    model: torch.nn.Module,
    loss_func,
    valid_dl: DataLoader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    batch_indices: Optional[Iterable[int]] = None,
    rollout_cfg: Optional[dict] = None,
    args: Optional[object] = None, 
) -> Tuple[float, int, Dict[str, float]]:
    
    # Make sure args passed in
    assert args is not None, "trainer.train/validate require `args` (with train_mb_size/eval_mb_size)."
    
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device

    # Use same policy as training: tail_only over t>=1; H from cfg (0 for TF val)
    base_cfg = dict(rollout_cfg or {})
    base_cfg.update({
        "loss_policy": "tail_only",
        "min_target_year": 1,
        "monthly_mode": getattr(args, "model_monthly_mode", "batch_months"),
    })
    
    mb_size_eval = int(args.eval_mb_size)

    use_index_subset = batch_indices is not None
    index_set = set(batch_indices) if use_index_subset else None
    total_loss = 0.0
    total_cnt  = 0
    mb_sums: Dict[str, float] = {}

    with torch.inference_mode():
        for b_idx, (batch_inputs, batch_monthly, batch_annual) in enumerate(valid_dl):
            if use_index_subset and (b_idx not in index_set):
                continue
            if (not use_index_subset) and (max_batches is not None) and (b_idx >= max_batches):
                break

            inputs   = batch_inputs.squeeze(0).float().to(device, non_blocking=True)
            labels_m = batch_monthly.squeeze(0).float().to(device, non_blocking=True)
            labels_a = batch_annual.squeeze(0).float().to(device, non_blocking=True)

            s, n, _step, mb_step = rollout(
                model=model,
                loss_func=loss_func,
                inputs=inputs,
                labels_m=labels_m,
                labels_a=labels_a,
                device=device,
                rollout_cfg=base_cfg,
                training=False,
                mb_size=mb_size_eval,
            )

            total_loss += float(s)
            total_cnt  += int(n)
            if mb_step:
                for k, v in mb_step.items():
                    mb_sums[k] = mb_sums.get(k, 0.0) + float(v)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # DDP reduction (sum/count and mb_sums) — keep your existing code here
    if dist.is_available() and dist.is_initialized():
        dev = device
        t_sum = torch.tensor([total_loss], device=dev, dtype=torch.float64)
        t_cnt = torch.tensor([total_cnt],  device=dev, dtype=torch.float64)
        dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_cnt, op=dist.ReduceOp.SUM)

        if mb_sums:
            keys = sorted(mb_sums.keys())
            buf  = torch.tensor([mb_sums[k] for k in keys], device=dev, dtype=torch.float64)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            mb_sums = {k: float(buf[i].item()) for i, k in enumerate(keys)}

        total_loss = float(t_sum.item())
        total_cnt  = int(t_cnt.item())

    avg = (total_loss / max(1, total_cnt)) if total_cnt > 0 else float("inf")
    mb_avgs = {k: (v / max(1, total_cnt)) for k, v in mb_sums.items()}
    return avg, total_cnt, mb_avgs

def test_once(
    *,
    model: torch.nn.Module,
    loss_func,
    test_dl: DataLoader,
    device: torch.device,
    base_cfg: Optional[dict],
    loss_policy: str,                    # "tail_only" | "all_years"
    carry_mode: str,                     # "fixed" | "full_sequence"
    args: Optional[object] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """
    Generic test pass. Returns dict with sum_loss, count, num_batches, and avg.

    carry_mode:
      - "fixed"         → use base_cfg["carry_horizon"] as H (teacher-forced or tail-only)
      - "full_sequence" → derive H = Y-1 per outer batch (long-history sequential test)

    Short-series skipping is applied **only** for full_sequence mode.
    """
    log = logger or _LOG

    # ---- main-rank guard for logging ----
    is_main = True
    if dist.is_available() and dist.is_initialized():
        try:
            is_main = (dist.get_rank() == 0)
        except Exception:
            is_main = True

    cfg = dict(base_cfg or {})
    cfg["loss_policy"] = loss_policy
    cfg["min_target_year"] = int(cfg.get("min_target_year", 1) or 1)

    # Monthly mode rule (keep caller override if present)
    user_mode = getattr(args, "model_monthly_mode", "batch_months")
    if "monthly_mode" not in cfg:
        if (carry_mode == "full_sequence") or (int(cfg.get("carry_horizon", 0) or 0) > 0):
            cfg["monthly_mode"] = "sequential_months"
        else:
            cfg["monthly_mode"] = user_mode

    # Eval microbatch size
    if carry_mode == "full_sequence":
        mb_size_eval = int(globals().get("FULL_SEQUENCE_TEST_MB_SIZE", 2048))
    else:
        mb_size_eval = int(getattr(args, "eval_mb_size", 2048))

    # ---- short-series policy (only for full_sequence) ----
    # Use args.min_full_sequence_years if present; else default 120 (after dataset drop).
    min_years_full = int(getattr(args, "min_full_sequence_years", 120))
    enforce_short_series = (carry_mode == "full_sequence" and min_years_full > 0)

    sum_loss = 0.0
    count = 0
    num_batches = 0

    skipped_batches = 0
    skipped_reason_counts = {"x": 0, "m": 0, "a": 0, "short": 0}

    def _report_nonfinite(tensor, label: str):
        if tensor is None:
            return []
        mask = ~torch.isfinite(tensor)
        if not mask.any():
            return []
        bad_vars = mask.any(dim=tuple(range(mask.ndim - 1))).nonzero(as_tuple=False).flatten()
        bad_vars = bad_vars.tolist()[:10]
        if is_main:
            log.info(
                "[test] %s non-finite in %d elements (showing first 10 var idx): %s",
                label, mask.sum().item(), bad_vars,
            )
        return bad_vars

    with torch.no_grad():
        for batch_idx, (x_cpu, m_cpu, a_cpu) in enumerate(test_dl):
            # --- Short-series guard (ONLY for full_sequence) ---
            if enforce_short_series:
                Y_cpu = None
                try:
                    if a_cpu is not None:
                        Y_cpu = int(a_cpu.shape[2])      # labels_a is [1, na, Y_postdrop, L]
                    elif x_cpu is not None:
                        T = int(x_cpu.shape[2])          # inputs is [1, nin, 365*Y_postdrop, L]
                        Y_cpu = T // 365
                except Exception:
                    Y_cpu = None

                if (Y_cpu is not None) and (Y_cpu < min_years_full):
                    skipped_batches += 1
                    skipped_reason_counts["short"] += 1
                    if is_main:
                        log.info(
                            "[test] Skipping batch %d due to short series (Y=%d < %d)",
                            batch_idx, Y_cpu, min_years_full
                        )
                    continue
            # ---------------------------------------------------

            # --- Batch-level finiteness guard (CPU, before .to(device)) ---
            IGNORE_VARS = {"lai_avh15c1", "lai_modis"}

            names_x = list(cfg.get("in_names", []))
            names_m = list(cfg.get("out_monthly_names", []))
            names_a = list(cfg.get("out_annual_names", []))

            x_cpu_masked = _mask_ignored_nans(x_cpu, names_x, IGNORE_VARS)
            m_cpu_masked = _mask_ignored_nans(m_cpu, names_m, IGNORE_VARS)
            a_cpu_masked = _mask_ignored_nans(a_cpu, names_a, IGNORE_VARS)

            fx = bool(torch.isfinite(x_cpu_masked).all().item()) if x_cpu_masked is not None else True
            fm = bool(torch.isfinite(m_cpu_masked).all().item()) if m_cpu_masked is not None else True
            fa = bool(torch.isfinite(a_cpu_masked).all().item()) if a_cpu_masked is not None else True

            if not (fx and fm and fa):
                skipped_batches += 1
                if not fx:
                    skipped_reason_counts["x"] += 1
                    _report_nonfinite(x_cpu, "inputs (x)")
                if not fm:
                    skipped_reason_counts["m"] += 1
                    _report_nonfinite(m_cpu, "monthly labels (m)")
                if not fa:
                    skipped_reason_counts["a"] += 1
                    _report_nonfinite(a_cpu, "annual labels (a)")
                if is_main:
                    log.info(
                        "[test] Skipping batch %d due to non-finite tensors "
                        "(ignoring %s): finite(x)=%s finite(m)=%s finite(a)=%s",
                        batch_idx, ", ".join(sorted(IGNORE_VARS)),
                        fx, fm, fa,
                    )
                continue
            # ------------------------------------------------------------------------

            inputs   = x_cpu.squeeze(0).float().to(device, non_blocking=True)
            labels_m = m_cpu.squeeze(0).float().to(device, non_blocking=True)
            labels_a = a_cpu.squeeze(0).float().to(device, non_blocking=True)

            if carry_mode == "full_sequence":
                Y = int(labels_a.shape[1])
                cfg["carry_horizon"] = max(0, Y - 1)

            s, n, _step, _mb = rollout(
                model=model,
                loss_func=loss_func,
                inputs=inputs,
                labels_m=labels_m,
                labels_a=labels_a,
                device=device,
                rollout_cfg=cfg,
                training=False,
                mb_size=mb_size_eval,
            )

            if n > 0 and np.isfinite(s):
                sum_loss += float(s)
                count += int(n)
                num_batches += 1

    # --- DDP reduction for skip counters and aggregates ---
    if dist.is_available() and dist.is_initialized():
        t_main = torch.tensor(
            [
                sum_loss,
                float(count),
                float(num_batches),
                float(skipped_batches),
                float(skipped_reason_counts["x"]),
                float(skipped_reason_counts["m"]),
                float(skipped_reason_counts["a"]),
                float(skipped_reason_counts["short"]),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(t_main, op=dist.ReduceOp.SUM)
        (
            sum_loss,
            count_f,
            num_batches_f,
            sk_b_f,
            sk_x_f,
            sk_m_f,
            sk_a_f,
            sk_short_f,
        ) = t_main.tolist()
        count = int(count_f)
        num_batches = int(num_batches_f)
        skipped_batches = int(sk_b_f)
        skipped_reason_counts = {
            "x": int(sk_x_f),
            "m": int(sk_m_f),
            "a": int(sk_a_f),
            "short": int(sk_short_f),
        }

    avg = sum_loss / max(1.0, float(count))
    return {
        "sum_loss": float(sum_loss),
        "count": int(count),
        "num_batches": int(num_batches),
        "avg_loss": float(avg),
        "skipped_batches": int(skipped_batches),
        "skipped_reason_counts": dict(skipped_reason_counts),
    }