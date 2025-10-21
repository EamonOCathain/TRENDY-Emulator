from torch.utils.data import DataLoader
from typing import Callable, Optional, Dict, Any
import time
import random
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
import sys
import logging
from copy import deepcopy
import json
import csv
import math
from typing import Mapping

# set project root
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.analysis.vis_modular import scatter_grid_from_pairs
from src.training.carry import gather_pred_label_pairs, _rollout_core

def _finite_mask_np(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Finite mask for paired arrays."""
    return np.isfinite(y) & np.isfinite(yhat)

def _is_finite_batch_torch(x: torch.Tensor, m: torch.Tensor, a: torch.Tensor) -> bool:
    """True if x, m, a are all finite (no NaN/Inf)."""
    return torch.isfinite(x).all() and torch.isfinite(m).all() and torch.isfinite(a).all()

def _ddp_sum_int(v: int, device: torch.device) -> int:
    """All-reduce a single integer across DDP ranks; return the summed int."""
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([int(v)], dtype=torch.int64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())
    return int(v)


# ------------------------------ metrics on normalized-space arrays ------------------------------
def metric_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Coefficient of determination R^2 on NORMALIZED values.
    R^2 = 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)
    Returns NaN if variance(y) == 0 or no finite pairs.
    """
    m = _finite_mask_np(y, yhat)
    if not np.any(m):
        return float("nan")
    y  = y[m].astype(np.float64, copy=False)
    yh = yhat[m].astype(np.float64, copy=False)
    denom = np.sum((y - y.mean())**2)
    if denom <= 0.0:
        return float("nan")
    num = np.sum((y - yh)**2)
    return 1.0 - (num / denom)

def metric_nse(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Nash–Sutcliffe Efficiency on NORMALIZED values.
    E = 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)
    Returns NaN if variance(y) == 0 or no finite pairs.
    """
    m = _finite_mask_np(y, yhat)
    if not np.any(m):
        return float("nan")
    y  = y[m].astype(np.float64, copy=False)
    yh = yhat[m].astype(np.float64, copy=False)
    denom = np.sum((y - y.mean())**2)
    if denom <= 0.0:
        return float("nan")
    num = np.sum((y - yh)**2)
    return 1.0 - (num / denom)

def metric_nrmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Normalized RMSE on NORMALIZED values.
    nRMSE = RMSE / std(y). On normalized data (std≈1), this ~ RMSE.
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
    rmse = np.sqrt(np.mean((y - yh)**2))
    return rmse / denom

def metric_acc(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Anomaly Correlation Coefficient on NORMALIZED values.
    Pearson correlation of anomalies (mean-removed series).
    Returns NaN if either centered variance is 0 or no finite pairs.
    """
    m = _finite_mask_np(y, yhat)
    if not np.any(m):
        return float("nan")
    y  = y[m].astype(np.float64, copy=False)
    yh = yhat[m].astype(np.float64, copy=False)
    yc  = y  - y.mean()
    yhc = yh - yh.mean()
    num   = np.sum(yc * yhc)
    den_y = np.sqrt(np.sum(yc**2))
    den_h = np.sqrt(np.sum(yhc**2))
    den = den_y * den_h
    if den <= 0.0:
        return float("nan")
    return float(num / den)

# ------------------------------ CSV writing ------------------------------

def _compute_metrics_per_variable(pairs: Mapping[str, tuple[np.ndarray, np.ndarray]]):
    """
    pairs: {var_name: (y, yhat)} with arrays in NORMALIZED units.
    Returns: list of dict rows (one per var) and a dict of global (equal-weight mean across vars).
    """
    rows = []
    for var, (y, yhat) in pairs.items():
        m = _finite_mask_np(y, yhat)
        n = int(np.count_nonzero(m))
        r2   = metric_r2(y, yhat)
        nse  = metric_nse(y, yhat)
        nrm  = metric_nrmse(y, yhat)
        acc  = metric_acc(y, yhat)
        rows.append({
            "variable": var,
            "n": n,
            "R2": r2,
            "NSE": nse,
            "nRMSE": nrm,
            "ACC": acc,
        })

    # equal-weight GLOBAL: mean of per-variable metrics (ignore NaNs)
    def _mean_ignore_nan(vals):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            return float("nan")
        m = np.isfinite(vals)
        if not np.any(m):
            return float("nan")
        return float(np.mean(vals[m]))

    global_row = {
        "variable": "GLOBAL(equal_weight)",
        "n": int(np.sum([r["n"] for r in rows])),  # informative only
        "R2":   _mean_ignore_nan([r["R2"]   for r in rows]),
        "NSE":  _mean_ignore_nan([r["NSE"]  for r in rows]),
        "nRMSE":_mean_ignore_nan([r["nRMSE"] for r in rows]),
        "ACC":  _mean_ignore_nan([r["ACC"]  for r in rows]),
    }
    return rows, global_row

def write_metrics_csv(
    *,
    out_csv_path: Path,
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Write CSV with first row = GLOBAL(equal_weight), then one row per variable.
    Columns: variable, n, R2, NSE, nRMSE, ACC
    """
    var_rows, global_row = _compute_metrics_per_variable(pairs)
    # order variables alphabetically for stability
    var_rows_sorted = sorted(var_rows, key=lambda r: r["variable"])

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variable", "n", "R2", "NSE", "nRMSE", "ACC"])
        w.writerow([
            global_row["variable"], global_row["n"],
            f"{global_row['R2']:.6f}"   if math.isfinite(global_row["R2"])   else "nan",
            f"{global_row['NSE']:.6f}"  if math.isfinite(global_row["NSE"])  else "nan",
            f"{global_row['nRMSE']:.6f}"if math.isfinite(global_row["nRMSE"])else "nan",
            f"{global_row['ACC']:.6f}"  if math.isfinite(global_row["ACC"])  else "nan",
        ])
        for r in var_rows_sorted:
            w.writerow([
                r["variable"], r["n"],
                f"{r['R2']:.6f}"   if math.isfinite(r["R2"])   else "nan",
                f"{r['NSE']:.6f}"  if math.isfinite(r["NSE"])  else "nan",
                f"{r['nRMSE']:.6f}"if math.isfinite(r["nRMSE"])else "nan",
                f"{r['ACC']:.6f}"  if math.isfinite(r["ACC"])  else "nan",
            ])

def run_and_save_metrics_csv(
    *,
    model: torch.nn.Module,
    test_dl: DataLoader,
    device: torch.device,
    rollout_cfg: dict,
    run_dir: Path,
    logger: Optional[logging.Logger] = None,
    subsample_points: int = 2_000_000,  # same default as your plotting
) -> dict:
    """
    Produces two CSVs (teacher-forced and carry=123y) with columns:
      variable, n, R2, NSE, nRMSE, ACC
    All metrics are computed in NORMALIZED space so they can be averaged across variables.
    """
    test_root = run_dir / "test"
    test_root.mkdir(parents=True, exist_ok=True)

    # Teacher-forced pairs
    tf_pairs = gather_pred_label_pairs(
        model=model, test_dl=test_dl, device=device,
        rollout_cfg=rollout_cfg, carry_horizon=0.0,
        max_points_per_var=subsample_points,
    )
    tf_csv = test_root / "metrics_teacher_forced.csv"
    write_metrics_csv(out_csv_path=tf_csv, pairs=tf_pairs)
    if logger: logger.info(f"[Metrics] Wrote {tf_csv}")

    # Carry 123y pairs
    c_pairs = gather_pred_label_pairs(
        model=model, test_dl=test_dl, device=device,
        rollout_cfg=rollout_cfg, carry_horizon=123.0,
        max_points_per_var=subsample_points,
    )
    c_csv = test_root / "metrics_carry_123y.csv"
    write_metrics_csv(out_csv_path=c_csv, pairs=c_pairs)
    if logger: logger.info(f"[Metrics] Wrote {c_csv}")

    # Also return global metrics (handy for logging or titles)
    _, tf_global = _compute_metrics_per_variable(tf_pairs)
    _, c_global  = _compute_metrics_per_variable(c_pairs)
    return {"teacher_forced": tf_global, "carry_123y": c_global}

# ------------------------------ Plotting ------------------------------

def run_and_save_scatter_grids(
    *,
    model,
    test_dl,
    device,
    rollout_cfg,
    run_dir: Path,
    logger=None,
    subsample_points: int = 200_000,  # safer default; override from main if you like
) -> None:
    """
    Generates two scatter grids:
      1) Teacher-forced (no carry)
      2) Full carry (123y)
    Now robust to partial failures, non-finite data, and empty pairs.
    """
    test_root  = run_dir / "test"
    plots_dir  = test_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _clean_pairs(pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
                     min_points: int = 50) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Mask non-finite, drop vars with too few points."""
        cleaned = {}
        for k, (y, p) in (pairs or {}).items():
            try:
                m = np.isfinite(y) & np.isfinite(p)
                if np.count_nonzero(m) >= min_points:
                    cleaned[k] = (y[m], p[m])
            except Exception:
                # skip corrupt variable silently but keep going
                if logger:
                    logger.warning(f"[Plots] Skipping variable {k} due to malformed arrays.", exc_info=True)
        return cleaned

    def _safe_gather(carry_horizon: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        try:
            return gather_pred_label_pairs(
                model=model, test_dl=test_dl, device=device,
                rollout_cfg=rollout_cfg, carry_horizon=carry_horizon,
                max_points_per_var=subsample_points,
            )
        except Exception as e:
            if logger:
                which = "teacher-forced" if carry_horizon == 0.0 else f"carry({carry_horizon}y)"
                logger.warning(f"[Plots] Pair gathering failed for {which}: {e}", exc_info=True)
            return {}

    def _title_for(pairs: Mapping[str, tuple[np.ndarray, np.ndarray]], base: str) -> str:
        # Compute global metrics for the title (ignore NaNs)
        try:
            _, g = _compute_metrics_per_variable(pairs)
            nse = g["NSE"]; r2 = g["R2"]
            if math.isfinite(nse) and math.isfinite(r2):
                return f"{base} • NSE={nse:.3f}, R²={r2:.3f}"
        except Exception:
            pass
        return base

    def _safe_plot(pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
                   title: str, out_path: Path) -> None:
        if not pairs:
            if logger: logger.warning(f"[Plots] No valid pairs to plot for {out_path.name}; skipping.")
            return
        try:
            scatter_grid_from_pairs(
                pairs,
                ncols=3,
                suptitle=title,
                out_path=out_path,
                subsample=subsample_points,  # main controls; this function honors it
                density_alpha=True,
            )
            if logger: logger.info(f"[Plots] Saved scatter grid → {out_path}")
        except Exception as e:
            if logger:
                logger.warning(f"[Plots] Failed to render {out_path.name}: {e}. "
                               f"Writing counts snapshot instead.", exc_info=True)
            # Fallback: write per-variable point counts so you have a quick sanity artifact
            counts = {k: int(len(v[0])) for k, v in pairs.items()}
            out_path.with_suffix(".counts.json").write_text(json.dumps(counts, indent=2))

    # 1) Teacher-forced
    tf_pairs_raw = _safe_gather(0.0)
    tf_pairs = _clean_pairs(tf_pairs_raw)
    tf_title = _title_for(tf_pairs, "Observed vs Predicted (Teacher-Forced)")
    _safe_plot(tf_pairs, tf_title, plots_dir / "scatter_teacher_forced.png")

    # 2) Carry=123y
    c_pairs_raw = _safe_gather(123.0)
    c_pairs = _clean_pairs(c_pairs_raw)
    c_title = _title_for(c_pairs, "Observed vs Predicted (Carry = 123 years)")
    _safe_plot(c_pairs, c_title, plots_dir / "scatter_carry_123y.png")
    
# ------------------------------ test -----------------------------------
# Testing function
def test(
    model: torch.nn.Module,
    loss_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    test_dl: DataLoader,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    rollout_cfg: Optional[dict] = None,
) -> Dict[str, Any]:
    if logger is None:
        logger = logging.getLogger("test")

    carry_on = False
    if rollout_cfg is not None:
        try:
            carry_on = float(rollout_cfg.get("carry_horizon", 0.0)) > 0.0
        except Exception:
            carry_on = False

    model.eval()
    sum_loss = 0.0
    n_windows = 0
    n_batches = 0
    t0 = time.time()

    with torch.no_grad():
        total_batches = len(test_dl)
        if total_batches == 0:
            logger.warning("Test dataloader is empty; returning zeros.")
            return {"sum_loss": 0.0, "count": 0, "num_batches": 0, "skipped_batches": 0, "skipped_windows": 0}

        every_n = max(1, total_batches // 10)

        skipped_batches = 0
        skipped_windows = 0

        for batch_inputs, batch_monthly, batch_annual in test_dl:
            # --- hard-coded finite check on CPU tensors straight from DataLoader
            if not _is_finite_batch_torch(batch_inputs, batch_monthly, batch_annual):
                skipped_batches += 1
                # best-effort estimate of windows in this skipped batch
                try:
                    skipped_windows += int(batch_inputs.shape[0])
                except Exception:
                    skipped_windows += 0
                continue

            s, n = _rollout_core(
                model=model,
                inputs=batch_inputs.squeeze(0).float(),
                labels_m=batch_monthly.squeeze(0).float(),
                labels_a=batch_annual.squeeze(0).float(),
                device=device,
                rollout_cfg=rollout_cfg or {},
                training=False,
                loss_func=loss_func,
                carry_on=carry_on,
                return_pairs=False,
                logger=logger,
            )

            # Guard against NaN/Inf loss coming back from the core
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
    model: torch.nn.Module,
    loss_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    test_dl: DataLoader,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    rollout_cfg: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Runs test() twice:
      1) Teacher-forced (no carry)
      2) Full carry with horizon=123y
    Returns loss summaries only (metrics are handled elsewhere via gather_pred_label_pairs).
    """
    if logger is None:
        logger = logging.getLogger("test_with_and_without_carry")

    logger.info("=== Starting test: teacher-forced (no carry) ===")
    tf_cfg = deepcopy(rollout_cfg) if rollout_cfg is not None else {}
    tf_cfg["carry_horizon"] = 0.0
    tf_out = test(
        model=model,
        loss_func=loss_func,
        test_dl=test_dl,
        device=device,
        logger=logger,
        rollout_cfg=tf_cfg,
    )
    tf_out["carry_horizon_forced"] = 0.0

    logger.info("=== Starting test: full carry (123-year horizon) ===")
    carry_cfg = deepcopy(rollout_cfg) if rollout_cfg is not None else {}
    carry_cfg["carry_horizon"] = 123.0
    carry_out = test(
        model=model,
        loss_func=loss_func,
        test_dl=test_dl,
        device=device,
        logger=logger,
        rollout_cfg=carry_cfg,
    )
    carry_out["carry_horizon_forced"] = 123.0

    logger.info("=== Test complete: both teacher-forced and full-carry runs done ===")
    logger.info(f"Teacher-forced avg loss: {tf_out['sum_loss']/max(1,tf_out['count']):.6f}")
    logger.info(f"Carry(123y) avg loss:    {carry_out['sum_loss']/max(1,carry_out['count']):.6f}")

    return {
        "teacher_forced": tf_out,
        "carry_123y": carry_out,
    }

# Full final wrapper
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
    world_size: int,
) -> Dict[str, Any]:
    """
    Runs the two-pass test (teacher-forced and carry=123y), DDP-reduces
    the losses/counts for each pass, logs and writes both JSON outputs.
    """
    suite = test_with_and_without_carry(
        model=model,
        loss_func=loss_func,
        test_dl=test_dl,
        device=device,
        logger=logger,
        rollout_cfg=rollout_cfg,
    )
    
    def _ddp_sum_field(out: dict, key: str) -> int:
        val = int(out.get(key, 0))
        return _ddp_sum_int(val, device)

    def _ddp_reduce_sum_count(out: dict) -> tuple[float, float, float]:
        local_sum = float(out["sum_loss"])
        local_cnt = float(out["count"])
        if ddp and dist.is_initialized():
            t = torch.tensor([local_sum, local_cnt], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            global_sum, global_cnt = t.tolist()
        else:
            global_sum, global_cnt = local_sum, local_cnt
        global_avg = (global_sum / max(1.0, global_cnt))
        return global_sum, global_cnt, global_avg

    tf_sum, tf_cnt, tf_avg = _ddp_reduce_sum_count(suite["teacher_forced"])
    c123_sum, c123_cnt, c123_avg = _ddp_reduce_sum_count(suite["carry_123y"])
    
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
    return suite