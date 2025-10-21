#!/usr/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from pathlib import Path
import sys
import dask
from dask import delayed
from multiprocessing.pool import ThreadPool

# ---------------- Project imports ----------------
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.paths.paths import zarr_dir, std_dict_path, data_dir, training_dir
from src.dataset.variables import var_names
from src.utils.tools import slurm_shard

# ---------------- CPU / Dask pool ----------------
cpus = max(1, int(os.getenv("SLURM_CPUS_PER_TASK", "1")))
pool = ThreadPool(processes=cpus)
dask.config.set(pool=pool)   # thread scheduler using exactly `cpus` threads

# --- Inputs / discovery ---
zarr_root = zarr_dir / "training_new"
train_dir = zarr_root / "train"
val_dir   = zarr_root / "val"
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

# --- Helper: robust open_zarr (consolidated then fallback) ---
def _open_zarr(path: Path):
    try:
        return xr.open_zarr(path, consolidated=True, decode_times=False, chunks=None)
    except Exception:
        return xr.open_zarr(path, consolidated=False, decode_times=False, chunks=None)

# --- Heuristic rechunking to keep blocks small ---
TIME_NAMES = ("time", "day", "days", "t")
Y_NAMES    = ("lat", "latitude", "y")
X_NAMES    = ("lon", "longitude", "x")

def _small_chunks(da: xr.DataArray) -> dict:
    """
    Build a small-chunk spec to keep per-block memory modest.
    - time: ~90 days
    - spatial: ~128x128
    """
    chunks = {}
    dims = list(da.dims)
    # pick names if present
    tdim = next((d for d in dims if d in TIME_NAMES), None)
    ydim = next((d for d in dims if d in Y_NAMES), None)
    xdim = next((d for d in dims if d in X_NAMES), None)

    if tdim is not None:
        chunks[tdim] = 90
    if ydim is not None:
        chunks[ydim] = 128
    if xdim is not None:
        chunks[xdim] = 128
    return chunks

# --- Blockwise stats to minimize peak memory ---
def _block_stats(np_block):
    # np_block is a NumPy view for one dask chunk
    arr = np.asarray(np_block, dtype=np.float64)  # local cast; freed after return
    n = arr.size - np.isnan(arr).sum()
    s = np.nansum(arr)
    s2 = np.nansum(arr * arr)
    return float(s), float(s2), int(n)

def compute_stats_blockwise(da: xr.DataArray, batch: int = None):
    """
    Compute (sum, sumsq, count) with low memory by reducing per *native* Dask block.
    No rechunking — we honor the zarr’s original chunking.

    Args:
      da: xarray.DataArray (dask-backed)
      batch: number of blocks to compute concurrently (default: 2 * cpus)
    """
    if batch is None:
        batch = max(1, cpus) 

    # One delayed object per native chunk
    blocks = da.data.to_delayed().ravel()
    delayed_stats = [delayed(_block_stats)(blk) for blk in blocks]

    tot_s = 0.0
    tot_s2 = 0.0
    tot_n = 0

    # Compute in small batches so we never hold many chunks at once
    for i in range(0, len(delayed_stats), batch):
        results = dask.compute(*delayed_stats[i:i+batch])  # threads = SLURM_CPUS_PER_TASK
        for s, s2, n in results:
            tot_s  += s
            tot_s2 += s2
            tot_n  += n

    return tot_s, tot_s2, tot_n

# ---- Process this shard’s variables ----
results = {}  # var -> stats dict

for var in tasks:
    print(f"[INFO] Processing var='{var}' (cpus={cpus})", flush=True)

    # Pick zarrs by this var’s cadence
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
        # Open store
        try:
            ds = _open_zarr(z)
        except Exception as e:
            print(f"[ERROR] Open failed: {z} ({e})", flush=True)
            continue

        # Skip if var not present in this store
        if var not in ds.data_vars:
            try: ds.close()
            except Exception: pass
            continue

        da = ds[var]
        # ensure dask-backed, adopting native zarr chunks (no rechunk)
        if not hasattr(da.data, "to_delayed"):
            da = da.chunk()

        try:
            s, s2, n = compute_stats_blockwise(da)
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
        # Use None instead of NaN for strict JSON
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