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
from src.training.carry import gather_pred_label_pairs, _rollout_core
from src.dataset.variables import output_attributes


# =============================================================================
# Helpers (top-level)
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


def clean_pairs(
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    min_points: int = 50
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mask non-finite values and drop variables with too few points."""
    cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k, (y, p) in (pairs or {}).items():
        m = np.isfinite(y) & np.isfinite(p)
        if np.count_nonzero(m) >= min_points:
            cleaned[k] = (y[m], p[m])
    return cleaned


def name_unit_maps(
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


def title_for_r2(
    pairs_display_named: Mapping[str, tuple[np.ndarray, np.ndarray]],
    base: str
) -> str:
    """Compose a title with global R² if computable (metrics run on the given arrays)."""
    try:
        _, g = _compute_metrics_per_variable(pairs_display_named)
        r2 = g["R2"]
        if math.isfinite(r2):
            return f"{base} • R²={r2:.3f}"
    except Exception:
        pass
    return base


def save_scatter_grid(
    *,
    pairs_phys: Mapping[str, tuple[np.ndarray, np.ndarray]],
    base_title: str,
    out_png: Path,
    logger: Optional[logging.Logger] = None,
    subsample_points: int = 200_000,
) -> None:
    """Clean, relabel with long names/units, and save a combined scatter grid."""
    if not pairs_phys:
        if logger:
            logger.warning(f"[Plots] No valid pairs to plot for {out_png.name}; skipping.")
        return

    pairs_clean = clean_pairs(pairs_phys)
    if not pairs_clean:
        if logger:
            logger.warning(f"[Plots] Insufficient finite pairs for {out_png.name}; skipping.")
        return

    # Build display names + axis labels
    name_map, xlabel_map, ylabel_map = name_unit_maps(pairs_clean)
    pairs_pretty = {name_map[k]: v for k, v in pairs_clean.items()}

    title = title_for_r2(pairs_pretty, base_title)

    scatter_grid_from_pairs(
        pairs_pretty,
        ncols=3,
        suptitle=title,
        out_path=out_png,
        subsample=subsample_points,
        density_alpha=True,
        xlabel_by_name=xlabel_map,
        ylabel_by_name=ylabel_map,
        dpi=700,
    )
    if logger:
        logger.info(f"[Plots] Saved → {out_png}")
        
def gather_pairs_phys(
    *,
    model,
    test_dl,
    device,
    rollout_cfg,
    carry_horizon: float,
    std_map: Mapping[str, Mapping[str, float]],
    max_points_per_var: int,
    logger: Optional[logging.Logger] = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Gather normalized pairs → convert to physical; on error return {}."""
    try:
        pairs_norm = gather_pred_label_pairs(
            model=model,
            test_dl=test_dl,
            device=device,
            rollout_cfg=rollout_cfg,
            carry_horizon=carry_horizon,
            max_points_per_var=max_points_per_var,
        )
        return _to_physical_pairs(pairs_norm, std_map)
    except Exception as e:
        if logger:
            which = "teacher-forced" if carry_horizon == 0.0 else f"carry({carry_horizon}y)"
            logger.warning(f"[Plots] Pair gathering failed for {which}: {e}", exc_info=True)
        return {}


# =============================================================================
# Metrics (computed on arrays passed in; for CSV we pass PHYSICAL pairs)
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
      nRMSE = RMSE / std(y)
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


def _compute_metrics_per_variable(
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[list[dict], dict]:
    """
    Args:
      pairs: {var_name: (y, yhat)} arrays (assumed already in the target unit space).

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
            

# =============================================================================
# Main functions (bottom)
# =============================================================================

def run_and_save_metrics_csv(
    *,
    model: torch.nn.Module,
    test_dl: DataLoader,
    device: torch.device,
    rollout_cfg: dict,
    run_dir: Path,
    logger: Optional[logging.Logger] = None,
    subsample_points: int = 2_000_000,  # matches plotting default
) -> dict:
    """
    Produce two CSVs (teacher-forced and carry=123y), with columns:
      variable, n, R2, nRMSE, ACC

    - Pairs are gathered from the model in NORMALIZED space, then converted
      back to PHYSICAL units using rollout_cfg["std_out"] (per-variable mean/std).
    - Metrics are computed on PHYSICAL units.
    - The physical pairs are also saved to NPZ for later reuse, alongside a
      meta JSON containing units and long names from output_attributes.
    """
    test_root = run_dir / "test"
    test_root.mkdir(parents=True, exist_ok=True)

    # Map of output variable -> {"mean": μ, "std": σ}
    std_map = (rollout_cfg or {}).get("std_out", {})

    # =========================
    # Teacher-forced (physical)
    # =========================
    tf_pairs_norm = gather_pred_label_pairs(
        model=model, test_dl=test_dl, device=device,
        rollout_cfg=rollout_cfg, carry_horizon=0.0,
        max_points_per_var=subsample_points,
    )
    tf_pairs = _to_physical_pairs(tf_pairs_norm, std_map)

    # Save metrics CSV
    tf_csv = test_root / "metrics_teacher_forced.csv"
    write_metrics_csv(out_csv_path=tf_csv, pairs=tf_pairs)
    if logger:
        logger.info(f"[Metrics] Wrote {tf_csv}")

    # Save physical pairs for reuse
    tf_npz = test_root / "pairs_teacher_forced.npz"
    tf_meta = {
        var: {
            "units": output_attributes.get(var, {}).get("units", ""),
            "long_name": output_attributes.get(var, {}).get("long_name", var),
        }
        for var in tf_pairs.keys()
    }
    _save_pairs_npz(tf_npz, tf_pairs, tf_meta)
    if logger:
        logger.info(f"[Pairs] Saved physical pairs → {tf_npz}")

    # =========================
    # Carry = 123y (physical)
    # =========================
    c_pairs_norm = gather_pred_label_pairs(
        model=model, test_dl=test_dl, device=device,
        rollout_cfg=rollout_cfg, carry_horizon=123.0,
        max_points_per_var=subsample_points,
    )
    c_pairs = _to_physical_pairs(c_pairs_norm, std_map)

    c_csv = test_root / "metrics_carry_123y.csv"
    write_metrics_csv(out_csv_path=c_csv, pairs=c_pairs)
    if logger:
        logger.info(f"[Metrics] Wrote {c_csv}")

    c_npz = test_root / "pairs_carry_123y.npz"
    c_meta = {
        var: {
            "units": output_attributes.get(var, {}).get("units", ""),
            "long_name": output_attributes.get(var, {}).get("long_name", var),
        }
        for var in c_pairs.keys()
    }
    _save_pairs_npz(c_npz, c_pairs, c_meta)
    if logger:
        logger.info(f"[Pairs] Saved physical pairs → {c_npz}")

    # Return global rows (computed on PHYSICAL units)
    _, tf_global = _compute_metrics_per_variable(tf_pairs)
    _, c_global  = _compute_metrics_per_variable(c_pairs)
    return {"teacher_forced": tf_global, "carry_123y": c_global}


def run_and_save_scatter_grids(
    *,
    model,
    test_dl,
    device,
    rollout_cfg,
    run_dir: Path,
    logger=None,
    subsample_points: int = 200_000,
) -> None:
    """
    Generates two scatter grids in PHYSICAL units:
      1) Teacher-forced (carry_horizon=0.0)
      2) Full carry (carry_horizon=123.0)

    Uses `output_attributes` for pretty names and axis units.
    Saves combined figures under <run_dir>/test/plots/{teacher,carry}.png.
    """
    test_root = run_dir / "test"
    plots_dir = test_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    std_map = (rollout_cfg or {}).get("std_out", {})

    # Teacher-forced
    tf_pairs_phys = gather_pairs_phys(
        model=model,
        test_dl=test_dl,
        device=device,
        rollout_cfg=rollout_cfg,
        carry_horizon=0.0,
        std_map=std_map,
        max_points_per_var=subsample_points,
        logger=logger,
    )
    save_scatter_grid(
        pairs_phys=tf_pairs_phys,
        base_title="Observed vs Predicted (Teacher-Forced)",
        out_png=plots_dir / "scatter_teacher_forced.png",
        logger=logger,
        subsample_points=subsample_points,
    )

    # Carry = 123y
    c_pairs_phys = gather_pairs_phys(
        model=model,
        test_dl=test_dl,
        device=device,
        rollout_cfg=rollout_cfg,
        carry_horizon=123.0,
        std_map=std_map,
        max_points_per_var=subsample_points,
        logger=logger,
    )
    save_scatter_grid(
        pairs_phys=c_pairs_phys,
        base_title="Observed vs Predicted (Carry = 123 years)",
        out_png=plots_dir / "scatter_carry_123y.png",
        logger=logger,
        subsample_points=subsample_points,
    )

# Main testing function
def test(
    model: torch.nn.Module,
    loss_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    test_dl: DataLoader,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    rollout_cfg: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Run loss evaluation over a test dataloader. If rollout_cfg['carry_horizon'] > 0 → carry path,
    else teacher-forced path. Skips non-finite batches; logs progress.
    Returns:
      dict(sum_loss, count, num_batches, skipped_batches, skipped_windows)
    """
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
            return {
                "sum_loss": 0.0, "count": 0, "num_batches": 0,
                "skipped_batches": 0, "skipped_windows": 0
            }

        every_n = max(1, total_batches // 10)
        skipped_batches = 0
        skipped_windows = 0

        for batch_inputs, batch_monthly, batch_annual in test_dl:
            # Non-finite raw batch guard (on CPU tensors from DataLoader)
            if not _is_finite_batch_torch(batch_inputs, batch_monthly, batch_annual):
                skipped_batches += 1
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

            # Guard against NaN/Inf loss or invalid window count
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
    Run `test()` twice:
      1) Teacher-forced (carry_horizon=0.0)
      2) Carry with horizon=123y
    Returns only loss summaries; scatter/metrics are handled elsewhere.
    """
    if logger is None:
        logger = logging.getLogger("test_with_and_without_carry")

    logger.info("=== Starting test: teacher-forced (no carry) ===")
    tf_cfg = deepcopy(rollout_cfg) if rollout_cfg is not None else {}
    tf_cfg["carry_horizon"] = 0.0
    tf_out = test(model, loss_func, test_dl, device, logger, tf_cfg)
    tf_out["carry_horizon_forced"] = 0.0

    logger.info("=== Starting test: full carry (123-year horizon) ===")
    carry_cfg = deepcopy(rollout_cfg) if rollout_cfg is not None else {}
    carry_cfg["carry_horizon"] = 123.0
    carry_out = test(model, loss_func, test_dl, device, logger, carry_cfg)
    carry_out["carry_horizon_forced"] = 123.0

    logger.info("=== Test complete: both teacher-forced and full-carry runs done ===")
    logger.info(f"Teacher-forced avg loss: {tf_out['sum_loss']/max(1, tf_out['count']):.6f}")
    logger.info(f"Carry(123y) avg loss:    {carry_out['sum_loss']/max(1, carry_out['count']):.6f}")

    return {"teacher_forced": tf_out, "carry_123y": carry_out}


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
) -> Dict[str, Any]:
    """
    Execute the two-pass test (teacher-forced & carry=123y), DDP-reduce
    loss sums/counts, log results, and write both JSON outputs.
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
        """All-reduce (sum, count) to (global_sum, global_cnt, global_avg)."""
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

    tf_sum, tf_cnt, tf_avg       = _ddp_reduce_sum_count(suite["teacher_forced"])
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