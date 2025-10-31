from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import csv
import json
import logging
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------
from src.analysis.vis_modular import scatter_grid_from_pairs
from src.training.carry import gather_pred_label_pairs, rollout_outer_batch
from src.dataset.variables import output_attributes
# =============================================================================
# Utilities: finite checks & small DDP helper
# =============================================================================

def _finite_mask_np(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Elementwise finiteness mask for paired arrays (True where both finite)."""
    return np.isfinite(y) & np.isfinite(yhat)


def _is_finite_batch_torch(x: torch.Tensor, m: torch.Tensor, a: torch.Tensor) -> bool:
    """True iff x, m, a are all finite (no NaN/Inf)."""
    return torch.isfinite(x).all() and torch.isfinite(m).all() and torch.isfinite(a).all()


def _ddp_sum_int(v: int, device: torch.device) -> int:
    """
    All-reduce a single integer across DDP ranks and return the summed int.
    If not in DDP, returns v unchanged.
    """
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([int(v)], dtype=torch.int64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())
    return int(v)

# ========= Physical-space helpers =========

def _to_physical_pairs(
    pairs_norm: Mapping[str, tuple[np.ndarray, np.ndarray]],
    std_map: Mapping[str, Mapping[str, float]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Convert normalized (z-scored) pairs back to physical units using
    std_map[var] = {"mean": μ, "std": σ}. Falls back to identity if missing.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for var, (y, yhat) in pairs_norm.items():
        stats = std_map.get(var, {})
        mu = float(stats.get("mean", 0.0))
        sd = float(stats.get("std", 1.0))
        if not np.isfinite(sd) or sd == 0.0:
            sd = 1.0
        out[var] = (y * sd + mu, yhat * sd + mu)
    return out


def _save_pairs_npz(
    out_path: Path,
    pairs_phys: Mapping[str, tuple[np.ndarray, np.ndarray]],
    meta: Optional[Mapping[str, dict]] = None,
) -> None:
    """
    Save pairs as an .npz with keys like '<var>__y' and '<var>__yhat'.
    Also writes a '<file>.meta.json' with units/long names if provided.
    """
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
    """
    Build:
      - name_map:   var -> display name (long_name or var)
      - xlabel_map: display name -> 'Observed (unit)'
      - ylabel_map: display name -> 'Predicted (unit)'
    """
    name_map: dict[str, str] = {}
    xlabel_map: dict[str, str] = {}
    ylabel_map: dict[str, str] = {}

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
    """Compose a title with global R² on the given arrays."""
    try:
        _, g = _compute_metrics_per_variable(pairs)
        r2 = g["R2"]
        if math.isfinite(r2):
            return f"{base} • R²={r2:.3f}"
    except Exception:
        pass
    return base

# =============================================================================
# Metrics (computed in physical space)
# =============================================================================

def metric_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Coefficient of determination R^2:
      R^2 = 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)
    Returns NaN if variance(y) == 0 or there are no finite pairs.
    """
    m = _finite_mask_np(y, yhat)
    if not np.any(m):
        return float("nan")
    y  = y[m].astype(np.float64, copy=False)
    yh = yhat[m].astype(np.float64, copy=False)
    denom = np.sum((y - y.mean()) ** 2)
    if denom <= 0.0:
        return float("nan")
    num = np.sum((y - yh) ** 2)
    return 1.0 - (num / denom)


def metric_nrmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Normalized RMSE:
      nRMSE = RMSE / std(y)  (on normalized data, std≈1, so ~RMSE)
    Returns NaN if std(y) == 0 or no finite pairs.
    """
    m = _finite_mask_np(y, yhat)
    if not np.any(m):
        return float("nan")
    y  = y[m].astype(np.float64, copy=False)
    yh = yhat[m].astype(np.float64, copy=False)
    denom = y.std(ddof=0)
    if denom <= 0.0:
        return float("nan")
    rmse = np.sqrt(np.mean((y - yh) ** 2))
    return rmse / denom


def metric_acc(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Anomaly Correlation Coefficient:
      Pearson correlation of mean-removed series.
    Returns NaN if either centered variance is 0 or no finite pairs.
    """
    m = _finite_mask_np(y, yhat)
    if not np.any(m):
        return float("nan")
    y  = y[m].astype(np.float64, copy=False)
    yh = yhat[m].astype(np.float64, copy=False)
    yc, yhc = y - y.mean(), yh - yh.mean()
    num   = np.sum(yc * yhc)
    den_y = np.sqrt(np.sum(yc ** 2))
    den_h = np.sqrt(np.sum(yhc ** 2))
    den = den_y * den_h
    if den <= 0.0:
        return float("nan")
    return float(num / den)


# =============================================================================
# CSV metrics: compute per-variable + a global equal-weight row (no NSE)
# =============================================================================

def _compute_metrics_per_variable(
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[list[dict], dict]:
    """
    Args:
      pairs: {var_name: (y, yhat)} arrays in physical units.

    Returns:
      rows: list of dicts, one per variable
      global_row: dict of equal-weight mean across variables (ignores NaNs)
    """
    rows: list[dict] = []
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
        vals = np.asarray(list(vals), dtype=float)
        m = np.isfinite(vals)
        return float(np.mean(vals[m])) if np.any(m) else float("nan")

    global_row = {
        "variable": "GLOBAL(equal_weight)",
        "n": int(np.sum([r["n"] for r in rows])),  # informative only
        "R2":    _mean_ignore_nan([r["R2"]    for r in rows]),
        "nRMSE": _mean_ignore_nan([r["nRMSE"] for r in rows]),
        "ACC":   _mean_ignore_nan([r["ACC"]   for r in rows]),
    }
    return rows, global_row


def write_metrics_csv(
    *,
    out_csv_path: Path,
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Write CSV with first row = GLOBAL(equal_weight), then one row per variable.
    Columns: variable, n, R2, nRMSE, ACC
    """
    var_rows, global_row = _compute_metrics_per_variable(pairs)
    var_rows_sorted = sorted(var_rows, key=lambda r: r["variable"])  # stable order

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variable", "n", "R2", "nRMSE", "ACC"])

        def _fmt(x: float) -> str:
            return f"{x:.6f}" if math.isfinite(x) else "nan"

        # Global row
        w.writerow([
            global_row["variable"], global_row["n"],
            _fmt(global_row["R2"]),
            _fmt(global_row["nRMSE"]),
            _fmt(global_row["ACC"]),
        ])

        # Per-variable rows
        for r in var_rows_sorted:
            w.writerow([r["variable"], r["n"], _fmt(r["R2"]), _fmt(r["nRMSE"]), _fmt(r["ACC"])])


def run_and_save_metrics_csv(
    *,
    out_dir_name: str = "test",
    model: torch.nn.Module,
    test_dl: DataLoader,
    device: torch.device,
    rollout_cfg: dict,
    run_dir: Path,
    logger: Optional[logging.Logger] = None,
    subsample_points: int = 2_000_000,
    eval_mb_size: Optional[int] = None,  # used for tail-only windows
) -> dict:
    """
    Write metrics in PHYSICAL space ONLY, and save NPZs (also physical):
      - metrics_teacher_forced_physical.csv   + pairs_teacher_forced.npz
      - metrics_carry_full_sequence_physical.csv + pairs_carry_full_sequence.npz
      - metrics_carry_tail_only_physical.csv + pairs_carry_tail_only.npz   (when H>0)

    Returns a dict of GLOBAL(equal_weight) rows computed on PHYSICAL units.
    """
    test_root = run_dir / out_dir_name
    test_root.mkdir(parents=True, exist_ok=True)

    std_map = (rollout_cfg or {}).get("std_out", {})

    # ---------- teacher-forced ----------
    tf_pairs_norm = gather_pred_label_pairs(
        model=model, test_dl=test_dl, device=device,
        rollout_cfg=rollout_cfg,
        eval_mode="teacher_forced",
        mb_size=None,
        max_points_per_var=subsample_points,
    )
    tf_pairs_phys = _to_physical_pairs(tf_pairs_norm, std_map)
    tf_csv_phys = test_root / "metrics_teacher_forced_physical.csv"
    write_metrics_csv(out_csv_path=tf_csv_phys, pairs=tf_pairs_phys)
    if logger: logger.info(f"[Metrics] Wrote {tf_csv_phys}")

    tf_npz = test_root / "pairs_teacher_forced.npz"
    tf_meta = {
        var: {
            "units": output_attributes.get(var, {}).get("units", ""),
            "long_name": output_attributes.get(var, {}).get("long_name", var),
        } for var in tf_pairs_phys.keys()
    }
    _save_pairs_npz(tf_npz, tf_pairs_phys, tf_meta)
    if logger: logger.info(f"[Pairs] Saved physical pairs → {tf_npz}")

    # ---------- full-sequence carry ----------
    fs_pairs_norm = gather_pred_label_pairs(
        model=model, test_dl=test_dl, device=device,
        rollout_cfg=rollout_cfg,
        eval_mode="full_sequence",
        mb_size=None,
        max_points_per_var=subsample_points,
    )
    fs_pairs_phys = _to_physical_pairs(fs_pairs_norm, std_map)
    fs_csv_phys = test_root / "metrics_carry_full_sequence_physical.csv"
    write_metrics_csv(out_csv_path=fs_csv_phys, pairs=fs_pairs_phys)
    if logger: logger.info(f"[Metrics] Wrote {fs_csv_phys}")

    fs_npz = test_root / "pairs_carry_full_sequence.npz"
    fs_meta = {
        var: {
            "units": output_attributes.get(var, {}).get("units", ""),
            "long_name": output_attributes.get(var, {}).get("long_name", var),
        } for var in fs_pairs_phys.keys()
    }
    _save_pairs_npz(fs_npz, fs_pairs_phys, fs_meta)
    if logger: logger.info(f"[Pairs] Saved physical pairs → {fs_npz}")

    # ---------- tail-only carry (auto when H>0) ----------
    out_dict = {
        "teacher_forced": _compute_metrics_per_variable(tf_pairs_phys)[1],
        "carry_full_sequence": _compute_metrics_per_variable(fs_pairs_phys)[1],
    }

    H = float(rollout_cfg.get("carry_horizon", 0.0) or 0.0)
    if H > 0.0:
        tail_pairs_norm = gather_pred_label_pairs(
            model=model, test_dl=test_dl, device=device,
            rollout_cfg=rollout_cfg,
            eval_mode="windowed_tail_only",
            mb_size=eval_mb_size,
            max_points_per_var=subsample_points,
        )
        tail_pairs_phys = _to_physical_pairs(tail_pairs_norm, std_map)
        tail_csv_phys = test_root / "metrics_carry_tail_only_physical.csv"
        write_metrics_csv(out_csv_path=tail_csv_phys, pairs=tail_pairs_phys)
        if logger: logger.info(f"[Metrics] Wrote {tail_csv_phys}")

        tail_npz = test_root / "pairs_carry_tail_only.npz"
        tail_meta = {
            var: {
                "units": output_attributes.get(var, {}).get("units", ""),
                "long_name": output_attributes.get(var, {}).get("long_name", var),
            } for var in tail_pairs_phys.keys()
        }
        _save_pairs_npz(tail_npz, tail_pairs_phys, tail_meta)
        if logger: logger.info(f"[Pairs] Saved physical pairs → {tail_npz}")

        out_dict["carry_tail_only"] = _compute_metrics_per_variable(tail_pairs_phys)[1]

    return out_dict

# =============================================================================
# Scatter grids: teacher-forced vs carry (titles now R² only)
# =============================================================================

def run_and_save_scatter_grids(
    *,
    model,
    test_dl,
    device,
    rollout_cfg,
    run_dir: Path,
    logger=None,
    subsample_points: int = 200_000,
    eval_mb_size: Optional[int] = None,   # not used here
) -> None:
    """
    Generates two scatter grids in PHYSICAL units:
      1) Teacher-forced
      2) Full sequence carry
    Titles include global R² (computed in physical space). Axes show units.
    """
    test_root = run_dir / "test"
    plots_dir = test_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    std_map = (rollout_cfg or {}).get("std_out", {})

    def _prep_for_plot(pairs_phys: Mapping[str, tuple[np.ndarray, np.ndarray]]):
        # drop vars with too few finite points
        cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for k, (y, p) in (pairs_phys or {}).items():
            m = np.isfinite(y) & np.isfinite(p)
            if np.count_nonzero(m) >= 50:
                cleaned[k] = (y[m], p[m])
        if not cleaned:
            return {}, {}, {}, {}
        name_map, xlabel_map, ylabel_map = _name_unit_maps_for_pairs(cleaned)
        pairs_pretty = {name_map[k]: v for k, v in cleaned.items()}
        return cleaned, pairs_pretty, xlabel_map, ylabel_map

    # Teacher-forced
    tf_pairs_norm = gather_pred_label_pairs(
        model=model, test_dl=test_dl, device=device,
        rollout_cfg=rollout_cfg, eval_mode="teacher_forced",
        mb_size=None, max_points_per_var=subsample_points,
    )
    tf_pairs_phys = _to_physical_pairs(tf_pairs_norm, std_map)
    tf_clean, tf_pretty, tf_xlab, tf_ylab = _prep_for_plot(tf_pairs_phys)
    if tf_pretty:
        tf_title = _title_with_r2_base(tf_pretty, "Observed vs Predicted (Teacher-Forced)")
        scatter_grid_from_pairs(
            tf_pretty, ncols=3, suptitle=tf_title,
            out_path=plots_dir / "scatter_teacher_forced.png",
            subsample=subsample_points, density_alpha=True,
            xlabel_by_name=tf_xlab, ylabel_by_name=tf_ylab, dpi=700,
        )
        if logger: logger.info(f"[Plots] Saved → {plots_dir / 'scatter_teacher_forced.png'}")
    else:
        if logger: logger.warning("[Plots] No valid TF pairs for plotting; skipping.")

    # Full-sequence carry
    fs_pairs_norm = gather_pred_label_pairs(
        model=model, test_dl=test_dl, device=device,
        rollout_cfg=rollout_cfg, eval_mode="full_sequence",
        mb_size=None, max_points_per_var=subsample_points,
    )
    fs_pairs_phys = _to_physical_pairs(fs_pairs_norm, std_map)
    fs_clean, fs_pretty, fs_xlab, fs_ylab = _prep_for_plot(fs_pairs_phys)
    if fs_pretty:
        fs_title = _title_with_r2_base(fs_pretty, "Observed vs Predicted (Full-Sequence Carry)")
        scatter_grid_from_pairs(
            fs_pretty, ncols=3, suptitle=fs_title,
            out_path=plots_dir / "scatter_carry_full_sequence.png",
            subsample=subsample_points, density_alpha=True,
            xlabel_by_name=fs_xlab, ylabel_by_name=fs_ylab, dpi=700,
        )
        if logger: logger.info(f"[Plots] Saved → {plots_dir / 'scatter_carry_full_sequence.png'}")
    else:
        if logger: logger.warning("[Plots] No valid FS pairs for plotting; skipping.")
        
# =============================================================================
# Loss-only testing (teacher-forced and carry=123y)
# =============================================================================

def test(
    model, loss_func, test_dl, device,
    logger: Optional[logging.Logger] = None,
    rollout_cfg: Optional[dict] = None,
    test_mb_size: Optional[int] = None,   # << add
) -> Dict[str, Any]:
    if logger is None:
        logger = logging.getLogger("test")

    H = int((rollout_cfg or {}).get("carry_horizon", 0) or 0)

    model.eval()
    sum_loss = 0.0
    n_windows = 0
    n_batches = 0
    t0 = time.time()
    skipped_batches = 0
    skipped_windows = 0

    with torch.no_grad():
        total_batches = len(test_dl)
        if total_batches == 0:
            logger.warning("Test dataloader is empty; returning zeros.")
            return {"sum_loss": 0.0, "count": 0, "num_batches": 0,
                    "skipped_batches": 0, "skipped_windows": 0}

        every_n = max(1, total_batches // 10)

        for batch_inputs, batch_monthly, batch_annual in test_dl:
            inputs   = batch_inputs.squeeze(0).float().to(device, non_blocking=True)
            labels_m = batch_monthly.squeeze(0).float().to(device, non_blocking=True)
            labels_a = batch_annual.squeeze(0).float().to(device, non_blocking=True)

            if (not torch.isfinite(inputs).all() or
                not torch.isfinite(labels_m).all() or
                not torch.isfinite(labels_a).all()):
                Y = int(labels_a.shape[1]) if labels_a.ndim >= 2 else 0
                L = int(inputs.shape[2])    if inputs.ndim    >= 3 else 0
                
                expected_windows = max(0, (Y - H) * L)  # tail-only semantics; H=0 → Y*L
                skipped_batches += 1
                skipped_windows += expected_windows
                logger.warning("[test] Skipping batch %d: non-finite values", n_batches)
                continue

            # Use the unified outer-batch (eval mode)
            s, n, _ = rollout_outer_batch(
                model=model,
                inputs=inputs,
                labels_m=labels_m,
                labels_a=labels_a,
                device=device,
                rollout_cfg=rollout_cfg or {},
                training=False,
                loss_func=loss_func,
                mb_size=test_mb_size,  # external control here
            )

            if not (isinstance(s, (float, int)) and math.isfinite(float(s))) or int(n) <= 0:
                skipped_batches += 1
                skipped_windows += max(int(n), 0)
                continue

            sum_loss += float(s)
            n_windows += int(n)
            n_batches += 1

            if (n_batches % every_n == 0) or (n_batches == total_batches):
                logger.info("[Test] %5.1f%% — elapsed=%.1fs",
                            100.0 * n_batches / total_batches, time.time() - t0)

    return {
        "sum_loss": sum_loss,
        "count": n_windows,
        "num_batches": n_batches,
        "skipped_batches": skipped_batches,
        "skipped_windows": skipped_windows,
    }

def test_with_and_without_carry(
    model, loss_func, test_dl, device,
    logger: Optional[logging.Logger] = None,
    rollout_cfg: Optional[dict] = None,
    test_mb_size: Optional[int] = None,     
) -> Dict[str, Any]:
    
    """Runs the tests for carry = 0 and carry = 123"""
    if logger is None:
        logger = logging.getLogger("test_with_and_without_carry")

    logger.info("=== Starting test: teacher-forced (no carry) ===")
    tf_cfg = deepcopy(rollout_cfg) if rollout_cfg is not None else {}
    tf_cfg["carry_horizon"] = 0
    tf_out = test(model, loss_func, test_dl, device, logger, tf_cfg, test_mb_size=test_mb_size)
    tf_out["carry_horizon_forced"] = 0.0

    logger.info("=== Starting test: full carry (123-year horizon) ===")
    carry_cfg = deepcopy(rollout_cfg) if rollout_cfg is not None else {}
    carry_cfg["carry_horizon"] = 123
    carry_out = test(model, loss_func, test_dl, device, logger, carry_cfg, test_mb_size=test_mb_size)
    carry_out["carry_horizon_forced"] = 123.0

    logger.info("=== Test complete ===")
    logger.info("Teacher-forced avg loss: %.6f", tf_out["sum_loss"]/max(1, tf_out["count"]))
    logger.info("Carry(123y)    avg loss: %.6f", carry_out["sum_loss"]/max(1, carry_out["count"]))
    return {"teacher_forced": tf_out, "carry_123y": carry_out}

@torch.no_grad()
def test_tail_only_carry(
    *, model, loss_func, test_dl, device, rollout_cfg: dict,
    logger: Optional[logging.Logger] = None,
    eval_mb_size: Optional[int] = None,
) -> Dict[str, Any]:
    if logger is None:
        logger = logging.getLogger("test_tail_only_carry")

    H = int(rollout_cfg.get("carry_horizon", 0) or 0)
    if H <= 0:
        logger.info("[tail_only] carry_horizon <= 0; nothing to do.")
        return {"sum_loss": 0.0, "count": 0, "num_batches": 0}

    total_loss = 0.0
    total_windows = 0
    batches = 0

    for bi, bm, ba in test_dl:
        s, n, _ = rollout_outer_batch(
            model=model,
            inputs=bi.squeeze(0).float().to(device, non_blocking=True),
            labels_m=bm.squeeze(0).float().to(device, non_blocking=True),
            labels_a=ba.squeeze(0).float().to(device, non_blocking=True),
            device=device,
            rollout_cfg=rollout_cfg,
            training=False,
            loss_func=loss_func,
            mb_size=eval_mb_size,        # << full external control here (tail-only)
            logger=logger,
        )
        if math.isfinite(s) and n > 0:
            total_loss    += float(s)
            total_windows += int(n)
            batches       += 1

    return {"sum_loss": total_loss, "count": total_windows, "num_batches": batches}

def run_and_save_scatter_tail_only(
    *,
    model: torch.nn.Module,
    test_dl: DataLoader,
    device: torch.device,
    rollout_cfg: dict,
    run_dir: Path,
    logger: Optional[logging.Logger] = None,
    subsample_points: int = 200_000,
    eval_mb_size: Optional[int] = None,
) -> Optional[Path]:
    """
    Tail-year-only scatter in PHYSICAL units when carry_horizon>0,
    using windowed microbatch semantics as training. Saves one PNG.
    """
    H = float(rollout_cfg.get("carry_horizon", 0.0) or 0.0)
    if H <= 0.0:
        if logger: logger.info("[tail_only] carry_horizon <= 0; skipping tail-only scatter.")
        return None

    test_root = run_dir / "test"
    plots_dir = test_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    std_map = (rollout_cfg or {}).get("std_out", {})

    # Tail-year pairs (normalized → physical)
    pairs_norm = gather_pred_label_pairs(
        model=model, test_dl=test_dl, device=device,
        rollout_cfg=rollout_cfg, eval_mode="windowed_tail_only",
        mb_size=eval_mb_size, max_points_per_var=subsample_points,
    )
    pairs_phys = _to_physical_pairs(pairs_norm, std_map)

    # Clean + pretty labels
    cleaned = {}
    for k, (y, p) in pairs_phys.items():
        m = np.isfinite(y) & np.isfinite(p)
        if np.count_nonzero(m) >= 50:
            cleaned[k] = (y[m], p[m])

    if not cleaned:
        if logger: logger.warning("[tail_only] no valid pairs for tail-only scatter; skipping.")
        return None

    name_map, xlabel_map, ylabel_map = _name_unit_maps_for_pairs(cleaned)
    pretty = {name_map[k]: v for k, v in cleaned.items()}
    title = _title_with_r2_base(pretty, f"Observed vs Predicted (Tail-Only Carry, H={H:g}y)")

    out_path = plots_dir / "scatter_carry_tail_only.png"
    scatter_grid_from_pairs(
        pretty, ncols=3, suptitle=title, out_path=out_path,
        subsample=subsample_points, density_alpha=True,
        xlabel_by_name=xlabel_map, ylabel_by_name=ylabel_map, dpi=700,
    )
    if logger: logger.info(f"[tail_only] Saved scatter grid → {out_path}")
    return out_path

# =============================================================================
# Full suite wrapper: run both tests, DDP-reduce, log, and save JSONs
# =============================================================================

def run_and_save_test_suite(
    *,
    model: torch.nn.Module,
    loss_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    test_dl: DataLoader,
    device: torch.device,
    logger: Optional[logging.Logger],
    rollout_cfg: Optional[dict],
    run_dir: Path,
    is_main: bool,
    ddp: bool,
    world_size: int,  # kept for API parity (not directly used)
    eval_mb_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute the evaluation suite:
      - Teacher-forced (carry_horizon=0.0)
      - Carry with horizon=123y
      - Tail-only carry (same tail-year-only loss placement as training) IF rollout_cfg.horizon > 0

    Reduces (sum,count) across DDP, logs global means, and writes JSON outputs.
    Returns a dict with local + reduced summaries for all modes.
    """
    if logger is None:
        logger = logging.getLogger("run_and_save_test_suite")

    # ---- Run legacy two-pass tests ----
    suite = test_with_and_without_carry(
        model=model,
        loss_func=loss_func,
        test_dl=test_dl,
        device=device,
        logger=logger,
        rollout_cfg=rollout_cfg,
        test_mb_size=eval_mb_size,
    )

    # ------------- helpers for DDP reduction -------------
    def _ddp_sum_field(out: dict, key: str) -> int:
        val = int(out.get(key, 0))
        if ddp and dist.is_initialized():
            t = torch.tensor([val], dtype=torch.int64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return int(t.item())
        return val

    def _reduce_sum_count(local: dict) -> tuple[float, float, float]:
        local_sum = float(local.get("sum_loss", 0.0))
        local_cnt = float(local.get("count", 0.0))
        if ddp and dist.is_initialized():
            t = torch.tensor([local_sum, local_cnt], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            global_sum, global_cnt = t.tolist()
        else:
            global_sum, global_cnt = local_sum, local_cnt
        global_avg = (global_sum / max(1.0, global_cnt))
        return global_sum, global_cnt, global_avg

    # ------------- reduce + persist TF & C123 -------------
    tf_sum, tf_cnt, tf_avg       = _reduce_sum_count(suite["teacher_forced"])
    c123_sum, c123_cnt, c123_avg = _reduce_sum_count(suite["carry_123y"])

    tf_skipped_batches = _ddp_sum_field(suite["teacher_forced"], "skipped_batches")
    tf_skipped_windows = _ddp_sum_field(suite["teacher_forced"], "skipped_windows")
    c_skipped_batches  = _ddp_sum_field(suite["carry_123y"],      "skipped_batches")
    c_skipped_windows  = _ddp_sum_field(suite["carry_123y"],      "skipped_windows")

    if is_main:
        logger.info("[Test/Teacher-Forced] skipped %d batch(es), %d window(s) due to non-finite data",
                    tf_skipped_batches, tf_skipped_windows)
        logger.info("[Test/Carry-123y]    skipped %d batch(es), %d window(s) due to non-finite data",
                    c_skipped_batches, c_skipped_windows)
        logger.info("[Test/Teacher-Forced] avg_loss=%.6f (sum=%.6f, count=%d)",
                    tf_avg, tf_sum, int(tf_cnt))
        logger.info("[Test/Carry-123y]    avg_loss=%.6f (sum=%.6f, count=%d)",
                    c123_avg, c123_sum, int(c123_cnt))

        info_dir = run_dir / "test"
        info_dir.mkdir(parents=True, exist_ok=True)

        tf_payload = {
            **suite["teacher_forced"],
            "global_sum_loss": tf_sum,
            "global_count": tf_cnt,
            "global_avg_loss": tf_avg,
            "global_skipped_batches": tf_skipped_batches,
            "global_skipped_windows": tf_skipped_windows,
        }
        c123_payload = {
            **suite["carry_123y"],
            "global_sum_loss": c123_sum,
            "global_count": c123_cnt,
            "global_avg_loss": c123_avg,
            "global_skipped_batches": c_skipped_batches,
            "global_skipped_windows": c_skipped_windows,
        }

        (info_dir / "test_teacher_forced.json").write_text(json.dumps(tf_payload, indent=2))
        (info_dir / "test_carry_123y.json").write_text(json.dumps(c123_payload, indent=2))

    # ------------- NEW: tail-only carry when H>0 -------------
    tail_payload_reduced = None
    H = float((rollout_cfg or {}).get("carry_horizon", 0.0) or 0.0)
    if H > 0.0:
        logger.info("=== Starting test: tail-only carry eval (training semantics) ===")
        tail_local = test_tail_only_carry(
            model=model,
            loss_func=loss_func,
            test_dl=test_dl,
            device=device,
            rollout_cfg=rollout_cfg or {},
            logger=logger,
            eval_mb_size=eval_mb_size,
        )
        t_sum, t_cnt, t_avg = _reduce_sum_count(tail_local)

        if is_main:
            info_dir = run_dir / "test"
            info_dir.mkdir(parents=True, exist_ok=True)
            tail_payload = {
                **tail_local,
                "global_sum_loss": t_sum,
                "global_count": t_cnt,
                "global_avg_loss": t_avg,
            }
            (info_dir / "test_carry_tail_only.json").write_text(json.dumps(tail_payload, indent=2))
            logger.info("[Test/Carry-tail_only] avg_loss=%.6f (sum=%.6f, count=%d)",
                        t_avg, t_sum, int(t_cnt))

        tail_payload_reduced = {
            "global_sum_loss": t_sum,
            "global_count": t_cnt,
            "global_avg_loss": t_avg,
        }

    # ------------- assemble return dict -------------
    suite["reduced"] = {
        "teacher_forced": {
            "global_sum_loss": tf_sum,
            "global_count": tf_cnt,
            "global_avg_loss": tf_avg,
            "global_skipped_batches": tf_skipped_batches,
            "global_skipped_windows": tf_skipped_windows,
        },
        "carry_123y": {
            "global_sum_loss": c123_sum,
            "global_count": c123_cnt,
            "global_avg_loss": c123_avg,
            "global_skipped_batches": c_skipped_batches,
            "global_skipped_windows": c_skipped_windows,
        },
    }
    if tail_payload_reduced is not None:
        suite["reduced"]["carry_tail_only"] = tail_payload_reduced

    return suite