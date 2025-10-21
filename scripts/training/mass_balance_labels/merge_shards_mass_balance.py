#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, List, Tuple, Any, Optional

def _np_isfinite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]

def _load_npz_and_meta(npz_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    data = np.load(npz_path)
    meta_path = npz_path.with_suffix(".json")
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    arrays = {k: data[k] for k in data.files}
    return arrays, meta

def load_split(dir_for_split: Path) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """
    Load all residual arrays and metadata JSONs for a split directory.
    Returns:
      residuals_by_key: dict[name] -> concatenated np.ndarray (across shards)
      shard_meta: list of JSON dicts per shard (if present)
    """
    files = sorted(dir_for_split.glob("residuals_*.npz"))
    if not files:
        raise SystemExit(f"No shard npz files found in {dir_for_split}")
    accum: Dict[str, List[np.ndarray]] = {}
    shard_meta: List[Dict[str, Any]] = []
    for f in files:
        arrays, meta = _load_npz_and_meta(f)
        # concat residual arrays by balance name
        for k, arr in arrays.items():
            if arr.ndim == 0:
                continue
            accum.setdefault(k, []).append(arr)
        if meta:
            shard_meta.append(meta)
    residuals = {k: np.concatenate(v, axis=0) for k, v in accum.items()}
    return residuals, shard_meta

def compute_residual_stats(residuals_np: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Per-balance descriptive stats + MSE (mean of squared residual).
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for name, arr in residuals_np.items():
        a = _np_isfinite(arr).astype(np.float64)
        if a.size == 0:
            out[name] = {"count": 0, "mean": None, "std": None, "mean_abs": None, "mse": None}
            continue
        out[name] = {
            "count": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std(ddof=0)),
            "mean_abs": float(np.mean(np.abs(a))),
            "mse": float(np.mean(a*a)),
        }
    return out

def combine_weighted_shard_means(shard_meta: List[Dict[str, Any]],
                                 key_chain: List[str]) -> Optional[float]:
    """
    Combine shard-level mean_penalty using weights = year_windows.
    key_chain lets us fetch nested per-balance means if present, e.g.:
      key_chain=["per_balance","water_balance","mean_penalty"]
    Returns None if insufficient metadata.
    """
    num, den = 0.0, 0.0
    for m in shard_meta:
        # required weight
        yw = m.get("year_windows", None)
        if yw in (None, 0):
            continue

        # navigate to value
        cur = m
        for k in key_chain:
            if not isinstance(cur, dict) or k not in cur:
                cur = None
                break
            cur = cur[k]
        if cur is None:
            continue
        try:
            mp = float(cur)
        except Exception:
            continue

        num += mp * float(yw)
        den += float(yw)
    return (num / den) if den > 0 else None

def extract_per_window_totals(shard_meta: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """
    If shards store per-window total penalties (e.g., meta["per_window"]["total"]),
    merge them into a single big array. Returns None if not found.
    """
    chunks: List[np.ndarray] = []
    for m in shard_meta:
        perw = m.get("per_window")
        if not isinstance(perw, dict):
            continue
        tot = perw.get("total")
        if tot is None:
            continue
        try:
            arr = np.asarray(tot, dtype=np.float64).reshape(-1)
            chunks.append(arr)
        except Exception:
            pass
    if chunks:
        return np.concatenate(chunks, axis=0)
    return None

def make_boxplot_single(residuals_np: Dict[str, np.ndarray],
                        total_series: np.ndarray,
                        out_png: Path,
                        title: str):
    labels = list(residuals_np.keys()) + ["TOTAL"]
    data   = [_np_isfinite(residuals_np[k]) for k in residuals_np.keys()]
    data.append(_np_isfinite(total_series))

    plt.figure(figsize=(max(8, 1.2*len(labels)), 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def make_hist_grid(residuals_np: Dict[str, np.ndarray],
                   total_series: np.ndarray,
                   out_png: Path,
                   title_prefix: str):
    keys = list(residuals_np.keys())
    keys_plus_total = keys + ["TOTAL"]

    # grid size
    n = len(keys_plus_total)
    cols = min(4, max(2, math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4.5, rows*3.5), squeeze=False)
    fig.suptitle(f"{title_prefix} – histograms", fontsize=14)

    def _plot(ax, name, arr):
        a = _np_isfinite(arr)
        ax.hist(a, bins=60)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("residual")
        ax.set_ylabel("count")

    idx = 0
    for k in keys:
        r = idx // cols; c = idx % cols; idx += 1
        _plot(axes[r][c], k, residuals_np[k])
    # TOTAL
    r = idx // cols; c = idx % cols
    _plot(axes[r][c], "TOTAL", total_series)
    idx += 1

    # turn off unused axes
    while idx < rows * cols:
        r = idx // cols; c = idx % cols; idx += 1
        axes[r][c].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def save_summary(outdir: Path,
                 split: str,
                 stats: Dict[str, Dict[str, Optional[float]]],
                 overall_weighted: Optional[float],
                 per_balance_weighted: Dict[str, Optional[float]],
                 total_series_is_true_total: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "per_balance_stats_from_residuals": stats,  # mean, std, mean_abs, mse
        "overall_mean_penalty_weighted": overall_weighted,  # from JSON, weighted by year_windows (if available)
        "per_balance_mean_penalty_weighted": per_balance_weighted,  # from JSON (if available)
        "total_series_is_true_per_window_total": bool(total_series_is_true_total),
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2))

    # CSV
    with (outdir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["balance", "count", "mean", "std", "mean_abs", "mse",
                    "weighted_mean_from_json"])
        for k, s in stats.items():
            w.writerow([
                k, s.get("count"),
                s.get("mean"), s.get("std"),
                s.get("mean_abs"), s.get("mse"),
                per_balance_weighted.get(k)
            ])
        w.writerow([])
        w.writerow(["overall_mean_penalty_weighted", overall_weighted])
        w.writerow(["total_series_is_true_per_window_total", total_series_is_true_total])

def process_split(split: str, in_dir: Path, out_dir: Path):
    split_dir = in_dir / split
    residuals_np, shard_meta = load_split(split_dir)

    # Per-balance residual stats (from arrays)
    stats = compute_residual_stats(residuals_np)

    # Weighted overall mean (from shard JSON, if present)
    overall_weighted = combine_weighted_shard_means(shard_meta, ["mean_penalty"])

    # Per-balance weighted means (from shard JSON, if present)
    per_balance_weighted: Dict[str, Optional[float]] = {}
    for k in residuals_np.keys():
        per_balance_weighted[k] = combine_weighted_shard_means(
            shard_meta, ["per_balance", k, "mean_penalty"]
        )

    # TOTAL series
    per_window_total = extract_per_window_totals(shard_meta)  # preferred
    if per_window_total is not None and per_window_total.size > 0:
        total_series = per_window_total
        total_is_true = True
    else:
        # Fallback: pooled residuals across all balances (clearly labeled)
        total_series = np.concatenate(
            [residuals_np[k].reshape(-1) for k in residuals_np.keys()], axis=0
        )
        total_is_true = False

    # Plots
    out_split = out_dir / split
    out_split.mkdir(parents=True, exist_ok=True)
    make_boxplot_single(
        residuals_np, total_series,
        out_split / "boxplot_balances_plus_total.png",
        title=f"{split} — residual distributions"
    )
    make_hist_grid(
        residuals_np, total_series,
        out_split / "histograms_balances_plus_total.png",
        title_prefix=split
    )

    # Summary
    save_summary(out_split, split, stats, overall_weighted, per_balance_weighted, total_is_true)

    # Return combined for ALL
    return residuals_np, shard_meta, total_series, total_is_true

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, required=True,
                    help="Directory with per-split folders (each holds residuals_*.npz + .json)")
    ap.add_argument("--out_dir", type=Path, required=True,
                    help="Where to write merged plots & summaries")
    ap.add_argument("--splits", nargs="+",
                    default=["train","val","test"],
                    choices=["train","val","test"])
    args = ap.parse_args()

    # Per-split
    overall_residuals_accum: Dict[str, List[np.ndarray]] = {}
    overall_meta: List[Dict[str, Any]] = []
    overall_total_chunks: List[np.ndarray] = []
    total_is_true_flags: List[bool] = []

    for split in args.splits:
        res_np, shard_meta, total_series, total_is_true = process_split(
            split, args.in_dir, args.out_dir
        )
        for k, arr in res_np.items():
            overall_residuals_accum.setdefault(k, []).append(arr)
        overall_meta.extend(shard_meta)
        overall_total_chunks.append(total_series)
        total_is_true_flags.append(total_is_true)

    # ALL combined
    if overall_residuals_accum:
        all_np = {k: np.concatenate(v, axis=0) for k, v in overall_residuals_accum.items()}
        all_stats = compute_residual_stats(all_np)

        # Weighted overall and per-balance means from JSON
        overall_weighted = combine_weighted_shard_means(overall_meta, ["mean_penalty"])
        per_balance_weighted: Dict[str, Optional[float]] = {}
        for k in all_np.keys():
            per_balance_weighted[k] = combine_weighted_shard_means(
                overall_meta, ["per_balance", k, "mean_penalty"]
            )

        # TOTAL series for ALL: only “true” if every split had true per-window totals
        all_total_series = np.concatenate(overall_total_chunks, axis=0)
        all_total_true = all(total_is_true_flags) and (all_total_series.size > 0)

        out_all = args.out_dir / "all"
        out_all.mkdir(parents=True, exist_ok=True)
        make_boxplot_single(
            all_np, all_total_series,
            out_all / "boxplot_balances_plus_total.png",
            title="ALL — residual distributions"
        )
        make_hist_grid(
            all_np, all_total_series,
            out_all / "histograms_balances_plus_total.png",
            title_prefix="ALL"
        )
        save_summary(out_all, "ALL", all_stats, overall_weighted, per_balance_weighted, all_total_true)

if __name__ == "__main__":
    main()