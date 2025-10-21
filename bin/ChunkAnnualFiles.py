#!/usr/bin/env python3
"""
Recursive chunking of all NetCDF files using nccopy (lat=1, lon=1).
- Scans TWO fixed source roots (historical & preindustrial) recursively.
- Preserves folder structure under corresponding *_1x1 output roots.
- Array-friendly via src.utils.tools.slurm_shard (uses SLURM_ARRAY_TASK_ID if set).
- Times each file, prints running average after every successful write, and
  prints a final summary at the end.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
import shutil

# --- Project imports / paths (hardcoded) --------------------------------------
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.utils.preprocessing import nccopy_chunk  # calls 'nccopy' CLI
from src.utils.tools import slurm_shard

# Fixed sources/targets
HIST_SRC = PROJECT_ROOT / "data/preprocessed/historical/annual_files"
HIST_DST = PROJECT_ROOT / "data/preprocessed/historical/annual_files_1x1"
PI_SRC   = PROJECT_ROOT / "data/preprocessed/preindustrial/annual_files"
PI_DST   = PROJECT_ROOT / "data/preprocessed/preindustrial/annual_files_1x1"

# nccopy settings
CLEVEL     = 4     # deflate level
LAT_CHUNK  = 1
LON_CHUNK  = 1
OVERWRITE  = True  # set False to skip existing outputs

# ------------------------------------------------------------------------------

def find_nc_files(root: Path) -> list[Path]:
    if not root.exists():
        print(f"[WARN] Missing source root: {root}")
        return []
    return sorted(root.rglob("*.nc"))

def plan_jobs(src_root: Path, dst_root: Path) -> list[tuple[Path, Path]]:
    jobs = []
    files = find_nc_files(src_root)
    for in_path in files:
        rel = in_path.relative_to(src_root)
        out_path = (dst_root / rel).with_suffix(".nc")
        jobs.append((in_path, out_path))
    return jobs

def main():
    # Ensure nccopy availability
    if shutil.which("nccopy") is None:
        print("[WARN] 'nccopy' not found in PATH. Jobs may fail.")

    # Build full job list across both roots
    jobs = []
    jobs += plan_jobs(HIST_SRC, HIST_DST)
    jobs += plan_jobs(PI_SRC,   PI_DST)

    if not jobs:
        print("[INFO] No .nc files found to process.")
        return

    # Shard across SLURM array (round-robin); if not under SLURM, returns full list
    shard = slurm_shard(jobs)
    print(f"[INFO] Total files: {len(jobs)} | This shard: {len(shard)}")

    shard_start = time.perf_counter()
    n_done = 0
    total_secs = 0.0
    n_errors = 0
    n_skipped = 0  # Only increments if OVERWRITE=False and output exists (handled below)

    for in_path, out_path in shard:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # If we're not overwriting and file exists, skip and record as skipped
        if not OVERWRITE and out_path.exists():
            print(f"[SKIP] Exists (overwrite=False): {out_path}")
            n_skipped += 1
            continue

        print(f"[TASK] {in_path}  ->  {out_path}")
        t0 = time.perf_counter()
        try:
            nccopy_chunk(
                in_path=in_path,
                out_path=out_path,
                clevel=CLEVEL,
                lat_chunk=LAT_CHUNK,
                lon_chunk=LON_CHUNK,
                overwrite=OVERWRITE,
            )
            dt = time.perf_counter() - t0
            n_done += 1
            total_secs += dt
            avg = total_secs / n_done if n_done else 0.0
            print(f"[TIME] {out_path.name}: {dt:.2f}s | running avg: {avg:.2f}s over {n_done} file(s)")
        except Exception as e:
            dt = time.perf_counter() - t0
            n_errors += 1
            print(f"[ERROR] ({dt:.2f}s) {in_path} -> {out_path}: {e}")

    shard_total = time.perf_counter() - shard_start
    avg = (total_secs / n_done) if n_done else 0.0
    print("\n[SUMMARY]")
    print(f"  Processed (this shard) : {n_done}")
    print(f"  Skipped (exists)        : {n_skipped}")
    print(f"  Errors                  : {n_errors}")
    print(f"  Total shard wall time   : {shard_total:.2f}s")
    print(f"  Sum of file times       : {total_secs:.2f}s")
    print(f"  Average per successful  : {avg:.2f}s")

if __name__ == "__main__":
    main()