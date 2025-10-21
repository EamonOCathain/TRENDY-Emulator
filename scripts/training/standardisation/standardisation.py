#!/usr/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from pathlib import Path
import sys
import dask
from multiprocessing.pool import ThreadPool

# ---------------- Project imports ----------------
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.paths.paths import zarr_dir, training_dir
from src.dataset.variables import var_names
from src.utils.tools import slurm_shard

# ---------------- CPU / Dask pool ----------------
cpus = max(1, int(os.getenv("SLURM_CPUS_PER_TASK", "1")))
pool = ThreadPool(processes=cpus)
# keep pool pinned so dask won't oversubscribe memory
dask.config.set(pool=pool, scheduler="threads")

# --- Inputs / discovery ---
zarr_root = zarr_dir / "training_new"
train_dir = zarr_root / "train"
val_dir   = zarr_root / "val"   # we exclude 'test' on purpose
current_dir = training_dir / "standardisation"

train_zarrs = sorted(train_dir.rglob("*.zarr"))
val_zarrs   = sorted(val_dir.rglob("*.zarr"))
all_zarrs   = train_zarrs + val_zarrs

annual_zarrs  = [p for p in all_zarrs if "annual"  in str(p)]
monthly_zarrs = [p for p in all_zarrs if "monthly" in str(p)]
daily_zarrs   = [p for p in all_zarrs if "daily"   in str(p)]

# Output directory for per-shard JSONs (merge later)
per_task_output_dir = current_dir / "per_task_data"
per_task_output_dir.mkdir(parents=True, exist_ok=True)

# --- Shard over variables only ---
all_vars = var_names["all"]
tasks = slurm_shard(all_vars)
if tasks is None:
    print("[ERROR] No SLURM task ID found in environment", flush=True)

# --- Strict consolidated open (fail if not consolidated) ---
def _open_zarr_consolidated(path: Path):
    # Require consolidated metadata; error if missing/corrupt
    return xr.open_zarr(path, consolidated=True, decode_times=False, chunks={})

# ---- Process this shard’s variables ----
results = {}  # var -> stats dict
EXCLUDE = {"time", "lat", "lon", "location", "scenario"}  # safety: coords not data

for var in tasks:
    print(f"[INFO] Processing var='{var}' (cpus={cpus})", flush=True)

    # Pick zarrs by cadence
    if var in var_names["annual"]:
        relevant_zarrs = annual_zarrs
    elif var in var_names["monthly"]:
        relevant_zarrs = monthly_zarrs
    elif var in var_names["daily"]:
        relevant_zarrs = daily_zarrs
    else:
        print(f"[WARN] Variable '{var}' not in daily/monthly/annual lists; skipping", flush=True)
        continue

    s_sum = 0.0
    s_sumsq = 0.0
    s_count = 0

    for z in relevant_zarrs:
        # Open store (consolidated only)
        try:
            ds = _open_zarr_consolidated(z)
        except Exception as e:
            print(f"[ERROR] Open (consolidated) failed: {z} ({e})", flush=True)
            continue

        # Skip if var not present
        if var not in ds.data_vars or var in EXCLUDE:
            try: ds.close()
            except Exception: pass
            continue

        # Use dask-native reduction: sum, sumsq, count (ignore NaNs)
        da = ds[var].astype("float64")
        if not hasattr(da.data, "to_delayed"):
            da = da.chunk()  # ensure dask-backed; keep native zarr chunks

        total    = da.sum(skipna=True)
        total_sq = (da * da).sum(skipna=True)
        count    = da.count()

        try:
            s, s2, n = dask.compute(total, total_sq, count)
            s = float(s); s2 = float(s2); n = int(n)
        except Exception as e:
            print(f"[ERROR] Reduction failed for {z.name} var={var}: {e}", flush=True)
            try: ds.close()
            except Exception: pass
            continue
        finally:
            try: ds.close()
            except Exception: pass

        if n == 0:
            continue

        s_sum   += s
        s_sumsq += s2
        s_count += n

        print(f"[INFO] {z.name}: +count={n} +sum={s:.6g} +sumsq={s2:.6g}", flush=True)

    # Finalize stats for this var
    if s_count > 0:
        mean = s_sum / s_count
        var_pop = (s_sumsq / s_count) - (mean * mean)
        std = float(np.sqrt(max(var_pop, 0.0)))
        results[var] = {
            "mean": float(mean),
            "std":  std,
            "count": int(s_count),
            "sum":  float(s_sum),
            "sumsq": float(s_sumsq),
        }
    else:
        results[var] = {
            "mean": None,
            "std":  None,
            "count": 0,
            "sum":  0.0,
            "sumsq": 0.0,
        }
        print(f"[WARN] No data accumulated for {var}", flush=True)

# --- Write one JSON per shard ---
task_id = os.getenv("SLURM_ARRAY_TASK_ID") or os.getenv("SLURM_JOB_ID") or "single"
out_file = per_task_output_dir / f"std_shard_{task_id}.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2, allow_nan=False)
print(f"[WRITE] {out_file} ({len(results)} variables)", flush=True)

# Tidy up Dask pool on shutdown
try:
    dask.config.set(pool=None)
    pool.terminate()
except Exception:
    pass