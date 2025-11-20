#!/usr/bin/env python3
"""
Merge per-task standardisation JSONs into a single standardisation dictionary.

- Scans:   data/standardisation_dict/per_task_data/*.json
- Merges:  sums / sumsq / count / min / max across files per variable
- Outputs: data/standardisation_dict/standardisation.json  (std_dict_path)
           (override with --out)

Each per-task JSON is expected to look like:
{
  "cLitter_annual": {
    "mean": 0.65,
    "std": 0.54,
    "count": 10908240,
    "sum": 7165412.01,
    "sumsq": 7936653.32,
    "min": -3.12,
    "max": 15.87
  },
  ...
}
Only sum/sumsq/count/min/max are used for merging; mean/std are recomputed.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, Any

# --- Repo paths ---
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.paths.paths import std_dict_path, training_dir

# --------------------------- helpers ---------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0

def merge_dicts(files: list[Path]) -> Dict[str, Dict[str, float]]:
    """
    Merge per-task dicts by summing 'sum', 'sumsq', 'count' and taking
    global 'min'/'max' for each variable key. Recompute mean/std at the end
    (population std).
    """
    acc: Dict[str, Dict[str, float]] = {}

    for fp in files:
        try:
            d = load_json(fp)
        except Exception as e:
            print(f"[WARN] Skipping {fp} (could not parse JSON): {e}", flush=True)
            continue

        if not isinstance(d, dict):
            print(f"[WARN] Skipping {fp} (root is not a dict)", flush=True)
            continue

        for var, stats in d.items():
            if not isinstance(stats, dict):
                print(f"[WARN] {fp}: key '{var}' value is not an object; skipping", flush=True)
                continue

            s   = safe_float(stats.get("sum", 0.0))
            s2  = safe_float(stats.get("sumsq", 0.0))
            cnt = safe_int(stats.get("count", 0))
            vmin = safe_float(stats.get("min", float("nan")))
            vmax = safe_float(stats.get("max", float("nan")))

            if cnt <= 0:
                # Nothing to merge from this shard for this var
                continue

            a = acc.setdefault(
                var,
                {
                    "sum":   0.0,
                    "sumsq": 0.0,
                    "count": 0,
                    "min":   float("inf"),
                    "max":   float("-inf"),
                },
            )

            a["sum"]   += s
            a["sumsq"] += s2
            a["count"] += cnt

            # Update global min/max if shard values are finite
            if not math.isnan(vmin):
                a["min"] = min(a["min"], vmin)
            if not math.isnan(vmax):
                a["max"] = max(a["max"], vmax)

    # Recompute mean/std and final min/max
    out: Dict[str, Dict[str, float]] = {}
    for var, a in acc.items():
        n = int(a["count"])
        if n <= 0:
            continue

        s  = float(a["sum"])
        s2 = float(a["sumsq"])

        mean = s / n
        # population variance = E[x^2] - (E[x])^2
        var_pop = (s2 / n) - (mean * mean)
        if var_pop < 0:
            # numeric floor for tiny negatives due to FP error
            var_pop = 0.0
        std = math.sqrt(var_pop)

        vmin = a.get("min", float("inf"))
        vmax = a.get("max", float("-inf"))
        # If min/max were never updated, store None
        if vmin == float("inf"):
            vmin = None
        if vmax == float("-inf"):
            vmax = None

        out[var] = {
            "mean":  float(mean),
            "std":   float(std),
            "count": n,
            "sum":   float(s),
            "sumsq": float(s2),
            "min":   vmin,
            "max":   vmax,
        }

    return out

# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Merge per-task standardisation JSONs.")
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=(training_dir / "standardisation" / "per_task_data"),
        help="Directory containing per-task JSONs (default: data/standardisation_dict/per_task_data)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=std_dict_path,
        help="Output JSON path (default: std_dict_path from repo)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report counts but do not write output JSON.",
    )
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    out_path: Path = args.out

    if not in_dir.exists():
        print(f"[ERROR] Input directory not found: {in_dir}", flush=True)
        sys.exit(2)

    files = sorted(p for p in in_dir.glob("*.json") if p.is_file())
    if not files:
        print(f"[ERROR] No JSON files found in: {in_dir}", flush=True)
        sys.exit(3)

    print(f"[INFO] Found {len(files)} per-task file(s) under: {in_dir}", flush=True)
    for p in files[:10]:
        print(f"  - {p.name}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more", flush=True)

    merged = merge_dicts(files)

    print(f"[INFO] Merged variables: {len(merged)}", flush=True)

    if args.dry_run:
        print("[DRY-RUN] Not writing output.", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(merged, f, indent=2)
    print(f"[OK] Wrote merged standardisation dict → {out_path}", flush=True)

if __name__ == "__main__":
    main()