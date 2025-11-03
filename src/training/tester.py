#!/usr/bin/env python3
"""
Simplified tester: wrappers around trainer.test() + optional diagnostics.

Public entry points:
  - evaluate_all_modes(...)
  - run_diagnostics(...)

Usage pattern:
  results = evaluate_all_modes(
      model, loss_func, test_dl, device,
      rollout_cfg=rollout_cfg, args=args, run_dir=run_dir,
      logger=log, eval_mb_size=args.eval_mb_size,
      full_sequence_horizon=123,    # or override as needed
  )

  # Optional diagnostics (plots/CSVs/NPZs)
  run_diagnostics(
      model, test_dl, device, rollout_cfg, run_dir,
      logger=log,
      do_teacher_forced=True, do_full_sequence=True, do_tail_only=True
  )
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple
import numpy as np
import torch
import sys

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.training.trainer import test_once, model_mode, unwrap
from src.dataset.variables import output_attributes
from src.analysis.vis_modular import scatter_grid_from_pairs

# =============================================================================
# Constants
# =============================================================================

MONTH_LENGTHS_FALLBACK = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# =============================================================================
# Small helpers (metrics, denorm, I/O)
# =============================================================================

def _finite_mask_np(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return np.isfinite(y) & np.isfinite(yhat)

def metric_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    m = _finite_mask_np(y, yhat)
    if not np.any(m):
        return float("nan")
    y, yh = y[m].astype(np.float64, copy=False), yhat[m].astype(np.float64, copy=False)
    denom = np.sum((y - y.mean())**2)
    if denom <= 0.0:
        return float("nan")
    return 1.0 - np.sum((y - yh)**2) / denom

def metric_nrmse(y: np.ndarray, yhat: np.ndarray) -> float:
    m = _finite_mask_np(y, yhat)
    if not np.any(m):
        return float("nan")
    y, yh = y[m].astype(np.float64, copy=False), yhat[m].astype(np.float64, copy=False)
    sd = y.std(ddof=0)
    if sd <= 0.0:
        return float("nan")
    return float(np.sqrt(np.mean((y - yh)**2)) / sd)

def metric_acc(y: np.ndarray, yhat: np.ndarray) -> float:
    m = _finite_mask_np(y, yhat)
    if not np.any(m):
        return float("nan")
    y, yh = y[m].astype(np.float64, copy=False), yhat[m].astype(np.float64, copy=False)
    yc, yhc = y - y.mean(), yh - yh.mean()
    num = np.sum(yc * yhc)
    den = np.sqrt(np.sum(yc**2)) * np.sqrt(np.sum(yhc**2))
    return float(num / den) if den > 0 else float("nan")

def _compute_metrics_per_variable(
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[list[dict], dict]:
    rows = []
    for var, (y, yhat) in pairs.items():
        m = _finite_mask_np(y, yhat)
        n = int(np.count_nonzero(m))
        rows.append({
            "variable": var,
            "n": n,
            "R2":   metric_r2(y, yhat),
            "nRMSE":metric_nrmse(y, yhat),
            "ACC":  metric_acc(y, yhat),
        })
    def _mean_ignore_nan(vals: Iterable[float]) -> float:
        arr = np.asarray(list(vals), dtype=float)
        m = np.isfinite(arr)
        return float(np.mean(arr[m])) if np.any(m) else float("nan")
    global_row = {
        "variable": "GLOBAL(equal_weight)",
        "n": int(np.sum([r["n"] for r in rows])),
        "R2":    _mean_ignore_nan([r["R2"]    for r in rows]),
        "nRMSE": _mean_ignore_nan([r["nRMSE"] for r in rows]),
        "ACC":   _mean_ignore_nan([r["ACC"]   for r in rows]),
    }
    return rows, global_row

def write_metrics_csv(out_csv_path: Path, pairs: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    rows, global_row = _compute_metrics_per_variable(pairs)
    rows_sorted = sorted(rows, key=lambda r: r["variable"])
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    def _fmt(x: float) -> str:
        return f"{x:.6f}" if np.isfinite(x) else "nan"
    with out_csv_path.open("w") as f:
        f.write("variable,n,R2,nRMSE,ACC\n")
        f.write(f'{global_row["variable"]},{global_row["n"]},{_fmt(global_row["R2"])},'
                f'{_fmt(global_row["nRMSE"])},{_fmt(global_row["ACC"])}\n')
        for r in rows_sorted:
            f.write(f'{r["variable"]},{r["n"]},{_fmt(r["R2"])},{_fmt(r["nRMSE"])},{_fmt(r["ACC"])}\n')
    return global_row

def _to_physical_pairs(
    pairs_norm: Mapping[str, tuple[np.ndarray, np.ndarray]],
    std_map: Mapping[str, Mapping[str, float]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for var, (y, yhat) in pairs_norm.items():
        stats = std_map.get(var, {})
        mu = float(stats.get("mean", 0.0))
        sd = float(stats.get("std",  1.0)) or 1.0
        out[var] = (y * sd + mu, yhat * sd + mu)
    return out

def _save_pairs_npz(out_path: Path, pairs_phys: Mapping[str, tuple[np.ndarray, np.ndarray]], meta: Optional[Mapping[str, dict]] = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, (y, yhat) in pairs_phys.items():
        arrays[f"{k}__y"] = y
        arrays[f"{k}__yhat"] = yhat
    np.savez_compressed(out_path, **arrays)
    if meta:
        out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

def _name_unit_maps_for_pairs(
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    name_map, xlabel_map, ylabel_map = {}, {}, {}
    for var in pairs.keys():
        meta = output_attributes.get(var, {})
        disp = meta.get("long_name", var)
        unit = meta.get("units", "")
        name_map[var] = disp
        suffix = f" ({unit})" if unit else ""
        xlabel_map[disp] = f"Observed{suffix}"
        ylabel_map[disp] = f"Predicted{suffix}"
    return name_map, xlabel_map, ylabel_map

def _title_with_r2_base(
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    base: str
) -> str:
    try:
        _, g = _compute_metrics_per_variable(pairs)
        r2 = g["R2"]
        if np.isfinite(r2):
            return f"{base} • R²={r2:.3f}"
    except Exception:
        pass
    return base

# =============================================================================
# Pair gathering (three modes)
# =============================================================================

def _year_bounds(y0: int, y1: int) -> tuple[int, int]:
    return (y0 * 365, y1 * 365)

def _slice_last_year_bounds(t: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (t * 12, (t + 1) * 12), (t, t + 1)

@torch.no_grad()
def gather_pred_label_pairs(
    *,
    model: torch.nn.Module,
    test_dl,
    device: torch.device,
    rollout_cfg: dict,
    eval_mode: str,  # {"teacher_forced","full_sequence","tail_only","sequential_no_carry"}
    max_points_per_var: int | None = 2_000_000,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Returns {var: (y_norm, yhat_norm)} in NORMALISED units, matching runtime semantics.
    - teacher_forced:     H=0, evaluate all years independently; monthly_mode = 'batch_months' (pass via rollout_cfg)
    - sequential_no_carry:H=0, evaluate all years independently; monthly_mode = 'sequential_months' (pass via rollout_cfg)
    - full_sequence:      warm-up y=0, score 1..Y-1; monthly_mode = 'sequential_months'.
    - tail_only:          windowed with H=rollout_cfg['carry_horizon']; score tail year; monthly_mode forced sequential if H>0 else user.
    """
    assert eval_mode in {"teacher_forced", "full_sequence", "tail_only", "sequential_no_carry"}
    model.eval()

    monthly_names = list(rollout_cfg.get("out_monthly_names", []))
    annual_names  = list(rollout_cfg.get("out_annual_names", []))
    buf = {n: ([], []) for n in (monthly_names + annual_names)}

    month_lengths = rollout_cfg.get("month_lengths", MONTH_LENGTHS_FALLBACK)
    bounds = [0]; 
    for m in month_lengths: bounds.append(bounds[-1] + m)
    first_month_len = int(month_lengths[0])

    # Index helpers (sanitise per batch)
    def _san_idx(idx_list, size):
        if not idx_list: return []
        good = [int(i) for i in idx_list if 0 <= int(i) < size]
        return good

    # Pool daily absolutes -> (monthly, annual)
    def _pool_from_daily_abs(y_daily_abs: torch.Tensor, nm: int, na: int) -> tuple[torch.Tensor, torch.Tensor]:
        y_m_daily = y_daily_abs[..., :nm]
        y_a_daily = y_daily_abs[..., nm:nm+na]
        pieces = [y_m_daily[:, bounds[i]:bounds[i+1], :].mean(dim=1, keepdim=True) for i in range(12)]
        preds_m = torch.cat(pieces, dim=1)            # [B,12,nm]
        preds_a = y_a_daily.mean(dim=1, keepdim=True) # [B,1, na]
        return preds_m, preds_a

    # Carry helpers
    def _prev_monthly_from_preds(preds_m: torch.Tensor, out_m_idx: list[int]) -> Optional[torch.Tensor]:
        if (preds_m is None) or (not out_m_idx): return None
        return preds_m[:, -1, out_m_idx]

    def _inject_monthly_carry(x_year: torch.Tensor, carry_vec: torch.Tensor, in_m_idx: list[int]):
        if (carry_vec is None) or (not in_m_idx): return
        B = x_year.size(0)
        idx = torch.as_tensor(in_m_idx, device=x_year.device, dtype=torch.long)
        # January inject
        Ljan = first_month_len
        x_year[:, :Ljan, :].index_copy_(
            2, idx, carry_vec.unsqueeze(1).expand(B, Ljan, len(in_m_idx))
        )
        # Poison Feb–Dec for those features
        if Ljan < 365:
            x_year[:, Ljan:, :].index_fill_(2, idx, float("nan"))

    def _isfinite_except_monthly_carry_mask(x_year: torch.Tensor, in_m_idx: list[int]) -> bool:
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

    for batch_inputs, batch_monthly, batch_annual in test_dl:
        inputs   = batch_inputs.squeeze(0).float().to(device, non_blocking=True)   # [nin,365*Y,L]
        labels_m = batch_monthly.squeeze(0).float().to(device, non_blocking=True)  # [nm,12*Y,L]
        labels_a = batch_annual.squeeze(0).float().to(device, non_blocking=True)   # [na,Y,L]

        nin, Ttot, L = int(inputs.shape[0]), int(inputs.shape[1]), int(inputs.shape[2])
        Y = int(labels_a.shape[1])
        if L <= 0 or Y <= 0:
            continue

        nm = int(labels_m.shape[0]); na = int(labels_a.shape[0])

        in_m_idx  = _san_idx(rollout_cfg.get("in_monthly_state_idx", []) or [], nin)
        in_a_idx  = _san_idx(rollout_cfg.get("in_annual_state_idx", [])  or [], nin)
        out_m_idx = _san_idx(rollout_cfg.get("out_monthly_state_idx", []) or [], nm)
        out_a_idx = _san_idx(rollout_cfg.get("out_annual_state_idx", [])  or [], na)

        # ----------- MODE A: teacher-forced (H=0, forced monthly mode from cfg) -----------
        if eval_mode in {"teacher_forced", "sequential_no_carry"}:
            forced_mode = rollout_cfg.get("monthly_mode", "batch_months")
            with model_mode(model, forced_mode):
                for y in range(Y):
                    ds, de = _year_bounds(y, y + 1)
                    xb = torch.stack([inputs[:, ds:de, loc].T for loc in range(L)], dim=0)  # [L,365,nin]
                    if not torch.isfinite(xb).all():  # strict finiteness
                        continue
                    preds_abs = model(xb)  # [L,365,nm+na]
                    if not torch.isfinite(preds_abs).all():
                        continue
                    pm, pa = _pool_from_daily_abs(preds_abs, nm, na)
                    (ms, me), (ys, ye) = _slice_last_year_bounds(y)
                    ybm = torch.stack([labels_m[:, ms:me, loc].T for loc in range(L)], dim=0)  # [L,12,nm]
                    yba = torch.stack([labels_a[:, ys:ye, loc].T for loc in range(L)], dim=0)  # [L, 1,na]
                    if not (torch.isfinite(ybm).all() and torch.isfinite(yba).all()):
                        continue
                    for j, name in enumerate(monthly_names):
                        buf[name][0].append(ybm[..., j].reshape(-1).cpu().numpy())
                        buf[name][1].append(pm[..., j].reshape(-1).cpu().numpy())
                    for j, name in enumerate(annual_names):
                        buf[name][0].append(yba[..., j].reshape(-1).cpu().numpy())
                        buf[name][1].append(pa[..., j].reshape(-1).cpu().numpy())
            continue

        # ----------- MODE B: full-sequence (warm-up + carry, forced sequential) ----------
        if eval_mode == "full_sequence":
            with model_mode(model, "sequential_months"):
                for loc in range(L):
                    ds0, de0 = _year_bounds(0, 1)
                    xb0 = inputs[:, ds0:de0, loc].T.unsqueeze(0)  # [1,365,nin]
                    if not torch.isfinite(xb0).all():
                        continue
                    preds0 = model(xb0)
                    if not torch.isfinite(preds0).all():
                        continue
                    pm0, pa0 = _pool_from_daily_abs(preds0, nm, na)
                    prev_m = _prev_monthly_from_preds(pm0, out_m_idx)
                    prev_a = pa0[:, 0, out_a_idx] if out_a_idx else None

                    for y in range(1, Y):
                        ds, de = _year_bounds(y, y+1)
                        xb = inputs[:, ds:de, loc].T.unsqueeze(0)
                        if (prev_m is not None) and in_m_idx:
                            _inject_monthly_carry(xb, prev_m, in_m_idx)
                        if (prev_a is not None) and in_a_idx:
                            idx = torch.as_tensor(in_a_idx, device=xb.device, dtype=torch.long)
                            xb.index_copy_(2, idx, prev_a.unsqueeze(1).expand(1, 365, len(in_a_idx)))
                        if not _isfinite_except_monthly_carry_mask(xb, in_m_idx):
                            continue
                        preds = model(xb)
                        if not torch.isfinite(preds).all():
                            continue
                        pm, pa = _pool_from_daily_abs(preds, nm, na)
                        (ms, me), (ys, ye) = _slice_last_year_bounds(y)
                        ybm = labels_m[:, ms:me, loc].T.unsqueeze(0)
                        yba = labels_a[:, ys:ye, loc].T.unsqueeze(0)
                        if torch.isfinite(ybm).all() and torch.isfinite(yba).all():
                            for j, name in enumerate(monthly_names):
                                buf[name][0].append(ybm[..., j].reshape(-1).cpu().numpy())
                                buf[name][1].append(pm[..., j].reshape(-1).cpu().numpy())
                            for j, name in enumerate(annual_names):
                                buf[name][0].append(yba[..., j].reshape(-1).cpu().numpy())
                                buf[name][1].append(pa[..., j].reshape(-1).cpu().numpy())
                        prev_m = _prev_monthly_from_preds(pm, out_m_idx)
                        prev_a = pa[:, 0, out_a_idx] if out_a_idx else None
            continue

        # ----------- MODE C: tail-only (windowed; H = cfg['carry_horizon']) -----------
        H = int(rollout_cfg.get("carry_horizon", 0) or 0)
        target_mode = "sequential_months" if H > 0 else rollout_cfg.get("monthly_mode", "batch_months")
        windows = [(loc, t) for loc in range(L) for t in range(H, Y)]
        if not windows:
            continue

        with model_mode(model, target_mode):
            # Simple streaming loop over windows
            for loc, t in windows:
                prev_m = None
                prev_a = None
                for y_off in range(H + 1):
                    y_abs = t - H + y_off
                    ds, de = _year_bounds(y_abs, y_abs + 1)
                    xb = inputs[:, ds:de, loc].T.unsqueeze(0)  # [1,365,nin]
                    if y_off > 0:
                        if (prev_m is not None) and in_m_idx:
                            _inject_monthly_carry(xb, prev_m, in_m_idx)
                        if (prev_a is not None) and in_a_idx:
                            idx = torch.as_tensor(in_a_idx, device=xb.device, dtype=torch.long)
                            xb.index_copy_(2, idx, prev_a.unsqueeze(1).expand(1, 365, len(in_a_idx)))
                    if not _isfinite_except_monthly_carry_mask(xb, in_m_idx):
                        continue
                    preds_abs = model(xb)
                    if not torch.isfinite(preds_abs).all():
                        continue
                    pm, pa = _pool_from_daily_abs(preds_abs, nm, na)
                    prev_m = _prev_monthly_from_preds(pm, out_m_idx)
                    prev_a = pa[:, 0, out_a_idx] if out_a_idx else None

                    # collect only tail
                    if y_off == H:
                        (ms, me), (ys, ye) = _slice_last_year_bounds(t)
                        ybm = labels_m[:, ms:me, loc].T.unsqueeze(0)
                        yba = labels_a[:, ys:ye, loc].T.unsqueeze(0)
                        if torch.isfinite(ybm).all() and torch.isfinite(yba).all():
                            for j, name in enumerate(monthly_names):
                                buf[name][0].append(ybm[..., j].reshape(-1).cpu().numpy())
                                buf[name][1].append(pm[..., j].reshape(-1).cpu().numpy())
                            for j, name in enumerate(annual_names):
                                buf[name][0].append(yba[..., j].reshape(-1).cpu().numpy())
                                buf[name][1].append(pa[..., j].reshape(-1).cpu().numpy())

    # Concatenate & optional downsample
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

# =============================================================================
# Wrappers around trainer.test()
# =============================================================================

def _merge_monthly_mode(cfg: dict, args, carry_horizon: int, *, force_monthly_mode: str | None = None) -> dict:
    """
    Decide monthly_mode for a given carry_horizon, with optional override.
    Default rule: if H==0 ⇒ use user mode; if H>0 ⇒ force 'sequential_months'.
    If force_monthly_mode is provided, it takes precedence.
    """
    out = dict(cfg or {})
    out["carry_horizon"] = int(carry_horizon)
    if force_monthly_mode is not None:
        out["monthly_mode"] = str(force_monthly_mode)
        return out
    user_mode = getattr(args, "model_monthly_mode", "batch_months")
    out["monthly_mode"] = ("sequential_months" if carry_horizon > 0 else user_mode)
    return out

def evaluate_all_modes(
    *,
    model: torch.nn.Module,
    loss_func,
    test_dl,
    device: torch.device,
    rollout_cfg: Optional[dict],
    args: object,
    run_dir: Path,
    logger: Optional[logging.Logger] = None,
    eval_mb_size: Optional[int] = None,   # kept for API parity; mb size comes from args.eval_mb_size
    full_sequence_horizon: int = 123,     # label only; actual H is Y-1 inside test_once(full_sequence)
) -> Dict[str, dict]:
    """
    Runs the loss-only tests via trainer.test() in the requested modes:
      - Teacher-forced (H=0, monthly_mode=FORCED 'batch_months')
      - Sequential, no-carry (H=0, monthly_mode=FORCED 'sequential_months')
      - Full sequence (H=Y-1 per batch, monthly_mode=FORCED 'sequential_months')

    Returns a dict with the raw trainer.test() outputs for each mode and writes JSONs.
    """
    log = logger or logging.getLogger("evaluate_all_modes")
    test_root = run_dir / "test"
    test_root.mkdir(parents=True, exist_ok=True)

    # -------------------- Teacher-forced (H=0, FORCE batch_months) --------------------
    tf_cfg = _merge_monthly_mode(
        rollout_cfg or {}, args, carry_horizon=0, force_monthly_mode="batch_months"
    )
    log.info(
        "[eval] teacher_forced: cfg.monthly_mode=%s",
        tf_cfg.get("monthly_mode")
    )
    tf_out = test_once(
        model=model, loss_func=loss_func, test_dl=test_dl, device=device,
        base_cfg=tf_cfg,
        loss_policy="tail_only",
        carry_mode="fixed",        # use H from base_cfg (0)
        args=args,
        logger=log,
    )
    (test_root / "test_teacher_forced.json").write_text(json.dumps(tf_out, indent=2))
    if log:
        log.info("[Test] Teacher-forced avg=%.6f", tf_out["sum_loss"] / max(1, tf_out["count"]))

    # ---------------- Sequential, no-carry (H=0, FORCE sequential_months) -------------
    seq0_cfg = _merge_monthly_mode(
        rollout_cfg or {}, args, carry_horizon=0, force_monthly_mode="sequential_months"
    )
    log.info(
        "[eval] sequential_no_carry(H=0): cfg.monthly_mode=%s",
        seq0_cfg.get("monthly_mode")
    )
    seq0_out = test_once(
        model=model, loss_func=loss_func, test_dl=test_dl, device=device,
        base_cfg=seq0_cfg,
        loss_policy="tail_only",
        carry_mode="fixed",          # H=0
        args=args,
        logger=log,
    )
    (test_root / "test_sequential_no_carry.json").write_text(json.dumps(seq0_out, indent=2))
    if log:
        log.info("[Test] Sequential-no-carry avg=%.6f", seq0_out["sum_loss"] / max(1, seq0_out["count"]))

    # ---------------- Full-sequence (H = Y-1 per batch, FORCE sequential) -------------
    fs_cfg = _merge_monthly_mode(
        rollout_cfg or {}, args, carry_horizon=int(full_sequence_horizon),
        force_monthly_mode="sequential_months"
    )
    log.info(
        "[eval] full_sequence: cfg.monthly_mode=%s",
        fs_cfg.get("monthly_mode")
    )
    fs_out = test_once(
        model=model, loss_func=loss_func, test_dl=test_dl, device=device,
        base_cfg=fs_cfg,
        loss_policy="tail_only",
        carry_mode="full_sequence",  # derive H = Y-1 per batch
        args=args,
        logger=log,
    )
    (test_root / "test_carry_full_sequence.json").write_text(json.dumps(fs_out, indent=2))
    if log:
        log.info("[Test] Carry-%dy avg=%.6f", int(full_sequence_horizon), fs_out["sum_loss"] / max(1, fs_out["count"]))

    out = {
        "teacher_forced": tf_out,
        "sequential_no_carry": seq0_out,
        "carry_full_sequence": fs_out,
    }

    # ---------------- Tail-only (user H>0) (forced sequential by carry) ----------------
    H_user = int((rollout_cfg or {}).get("carry_horizon", 0) or 0)
    tail_cfg = _merge_monthly_mode(rollout_cfg or {}, args, carry_horizon=H_user)
    log.info(
        "[eval] tail_only(H=%d): cfg.monthly_mode=%s",
        H_user, tail_cfg.get("monthly_mode")
    )
    if H_user > 0:
        tail_out = test_once(
            model=model, loss_func=loss_func, test_dl=test_dl, device=device,
            base_cfg=tail_cfg,           # contains user H
            loss_policy="tail_only",
            carry_mode="fixed",          # use H from base_cfg (user's H)
            args=args,
            logger=log,
        )
        (test_root / "test_carry_tail_only.json").write_text(json.dumps(tail_out, indent=2))
        if log:
            log.info("[Test] Tail-only (H=%d) avg=%.6f", H_user, tail_out["sum_loss"] / max(1, tail_out["count"]))
        out["carry_tail_only"] = tail_out

    return out

# =============================================================================
# optional diagnostics (pairs → CSV/NPZ; scatter plots)
# =============================================================================

def run_diagnostics(
    *,
    model: torch.nn.Module,
    test_dl,
    device: torch.device,
    rollout_cfg: dict,
    run_dir: Path,
    logger: Optional[logging.Logger] = None,
    subsample_points_pairs: int = 2_000_000,
    subsample_points_plots: int = 200_000,
    do_teacher_forced: bool = True,
    do_full_sequence: bool = True,
    do_tail_only: bool = True,
    do_sequential_no_carry: bool = True,
) -> None:
    """
    Produces metrics CSVs and scatter grids in PHYSICAL units for selected modes.
    Files written under run_dir / "test".
    """
    log = logger or logging.getLogger("diagnostics")
    test_root = run_dir / "test"
    plots_dir = test_root / "plots"
    test_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    std_map = (rollout_cfg or {}).get("std_out", {})

    def _prep_and_write(tag: str, pairs_norm: Mapping[str, tuple[np.ndarray, np.ndarray]]):
        pairs_phys = _to_physical_pairs(pairs_norm, std_map)

        # metrics CSV
        csv_path = test_root / f"metrics_{tag}_physical.csv"
        global_row = write_metrics_csv(out_csv_path=csv_path, pairs=pairs_phys)
        if log: log.info("[Metrics] %s (R2=%.3f, nRMSE=%.3f, ACC=%.3f)", tag,
                         global_row["R2"], global_row["nRMSE"], global_row["ACC"])

        # npz
        meta = {
            var: {
                "units": output_attributes.get(var, {}).get("units", ""),
                "long_name": output_attributes.get(var, {}).get("long_name", var),
            } for var in pairs_phys.keys()
        }
        npz_path = test_root / f"pairs_{tag}.npz"
        _save_pairs_npz(npz_path, pairs_phys, meta)

        # scatter plot (drop tiny vars)
        cleaned = {}
        for k, (y, p) in pairs_phys.items():
            m = np.isfinite(y) & np.isfinite(p)
            if np.count_nonzero(m) >= 50:
                cleaned[k] = (y[m], p[m])

        if not cleaned:
            if log: log.warning("[Plots] No valid pairs for %s; skipping plot.", tag)
            return

        name_map, xlabel_map, ylabel_map = _name_unit_maps_for_pairs(cleaned)
        pretty = {name_map[k]: v for k, v in cleaned.items()}

        title = _title_with_r2_base(pretty, f"Observed vs Predicted ({tag.replace('_',' ').title()})")
        out_img = plots_dir / f"scatter_{tag}.png"
        scatter_grid_from_pairs(
            pretty, ncols=3, suptitle=title, out_path=out_img,
            subsample=subsample_points_plots, density_alpha=True,
            xlabel_by_name=xlabel_map, ylabel_by_name=ylabel_map, dpi=700,
        )
        if log: log.info("[Plots] Saved → %s", out_img)

    # Teacher-forced (force batch)
    if do_teacher_forced:
        tf_pairs = gather_pred_label_pairs(
            model=model, test_dl=test_dl, device=device,
            rollout_cfg={**rollout_cfg, "monthly_mode": "batch_months"},
            eval_mode="teacher_forced",
            max_points_per_var=subsample_points_pairs,
        )
        _prep_and_write("teacher_forced", tf_pairs)

    # Decide based on user carry horizon and user monthly_mode
    H = int((rollout_cfg or {}).get("carry_horizon", 0) or 0)
    user_mode = str((rollout_cfg or {}).get("monthly_mode", "batch_months"))

    # Sequential but H=0 — only if the USER selected sequential mode
    if do_sequential_no_carry and H == 0 and user_mode == "sequential_months":
        seq0_pairs = gather_pred_label_pairs(
            model=model, test_dl=test_dl, device=device,
            rollout_cfg={**rollout_cfg, "monthly_mode": "sequential_months"},
            eval_mode="sequential_no_carry",
            max_points_per_var=subsample_points_pairs,
        )
        _prep_and_write("sequential_no_carry", seq0_pairs)
    elif do_sequential_no_carry and H == 0 and user_mode != "sequential_months":
        if log: log.info("[Diagnostics] Skipping sequential_no_carry: user monthly_mode is %s", user_mode)
        
    
    # Tail-only (only if H>0)
    if do_tail_only and H > 0:
        tail_pairs = gather_pred_label_pairs(
            model=model, test_dl=test_dl, device=device,
            rollout_cfg={**rollout_cfg, "monthly_mode": "sequential_months"},
            eval_mode="tail_only",
            max_points_per_var=subsample_points_pairs,
        )
        _prep_and_write(f"carry_tail_only_H{H}", tail_pairs)
    elif do_tail_only and H == 0:
        if log: log.info("[Diagnostics] Skipping tail_only: carry_horizon (H) is 0")

    # Full sequence
    if do_full_sequence:
        fs_pairs = gather_pred_label_pairs(
            model=model, test_dl=test_dl, device=device,
            rollout_cfg={**rollout_cfg, "monthly_mode": "sequential_months"},
            eval_mode="full_sequence",
            max_points_per_var=subsample_points_pairs,
        )
        _prep_and_write("carry_full_sequence", fs_pairs)