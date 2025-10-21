#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys
import shutil

import zarr
import xarray as xr

# add your project root if needed for slurm_shard
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.utils.tools import slurm_shard  # SLURM sharding helper

try:
    from rechunker import rechunk
except ImportError:
    print("ERROR: Please `pip install rechunker` in this environment.", file=sys.stderr)
    sys.exit(1)


def is_zarr_dir(p: Path) -> bool:
    return p.is_dir() and ((p / ".zgroup").exists() or (p / ".zarray").exists())


def detect_resolution(zarr_path: Path) -> str:
    # infer by name / parents
    candidates = [zarr_path.name] + list(zarr_path.parts[::-1])
    s = " ".join(c.lower() for c in candidates)
    if "daily" in s:
        return "daily"
    if "monthly" in s:
        return "monthly"
    if "annual" in s or "yearly" in s:
        return "annual"
    raise ValueError(f"Could not infer resolution from path: {zarr_path}")


def dim_target_chunks(
    ds: xr.Dataset,
    resolution: str,
    loc_chunk: int,
    scen_chunk: int,
    time_daily: int,
    time_monthly: int,
    time_annual: int,
) -> dict:
    """
    Build a **dimension-based** chunk mapping (simpler & matches Rechunker 0.5 docs).
    Only include dims that exist in the dataset to avoid typos.
    """
    time_target = {"daily": time_daily, "monthly": time_monthly, "annual": time_annual}[resolution]

    dims = set(ds.dims)  # present dims only
    chunks = {}
    if "time" in dims:
        chunks["time"] = min(int(ds.sizes["time"]), int(time_target))
    if "location" in dims:
        chunks["location"] = min(int(ds.sizes["location"]), int(loc_chunk))
    if "scenario" in dims:
        chunks["scenario"] = min(int(ds.sizes["scenario"]), int(scen_chunk))
    # Any other dims are left unchanged (rechunker keeps existing chunks)
    return chunks


def mirror_out_path(in_root: Path, out_root: Path, zarr_path: Path) -> Path:
    rel = zarr_path.relative_to(in_root)
    return out_root / rel


def find_zarrs(in_dir: Path) -> list[Path]:
    zarr_paths: list[Path] = []
    for root, dirs, _ in os.walk(in_dir):
        rp = Path(root)
        for d in dirs:
            if d.endswith(".zarr"):
                p = rp / d
                if is_zarr_dir(p):
                    zarr_paths.append(p)
    return zarr_paths


def rechunk_store(
    in_store: Path,
    out_store: Path,
    tmp_store_root: Path,
    resolution: str,
    loc_chunk: int,
    scen_chunk: int,
    max_mem: str,
    use_dask: bool,
    threads: int,
    time_daily: int,
    time_monthly: int,
    time_annual: int,
) -> None:
    out_store.parent.mkdir(parents=True, exist_ok=True)

    # Open (consolidated if possible)
    ds = xr.open_zarr(in_store, consolidated=True, decode_times=False)

    # Dimension-based target chunks (as per Rechunker 0.5 tutorial)
    tchunks = dim_target_chunks(
        ds,
        resolution=resolution,
        loc_chunk=loc_chunk,
        scen_chunk=scen_chunk,
        time_daily=time_daily,
        time_monthly=time_monthly,
        time_annual=time_annual,
    )

    # Prepare stores
    target_store = zarr.DirectoryStore(str(out_store))
    tmp_dir = tmp_store_root / (out_store.name + ".__tmp__")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    temp_store = zarr.DirectoryStore(str(tmp_dir))

    # Optional local Dask cluster
    client = cluster = None
    if use_dask:
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(
            n_workers=max(1, int(threads)),
            threads_per_worker=1,  # good for IO/compression
            processes=True,
            memory_limit=None,
            dashboard_address=None,
        )
        client = Client(cluster)

    try:
        # Make plan (group/dataset-level)
        plan = rechunk(
            ds,               # source Dataset/Group
            tchunks,          # dict mapping dims to chunk sizes
            max_mem,          # e.g. "8GB"
            target_store,     # final Zarr store
            temp_store=temp_store,
        )
        # Execute (uses Dask; if no client, defaults to threaded scheduler)
        plan.execute()
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()

    # Consolidate metadata for fast future opens
    zarr.consolidate_metadata(str(out_store))
    # Clean temp
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Rechunk Zarr stores for carry-friendly training (Rechunker 0.5).")
    ap.add_argument("in_dir", type=Path, help="Input directory to scan for *.zarr")
    ap.add_argument("out_dir", type=Path, help="Output directory (mirror structure will be created)")
    ap.add_argument("--tmp", type=Path, default=Path("./_rechunk_tmp"), help="Temporary directory root")
    ap.add_argument("--loc", type=int, default=512, help="Target location chunk size")
    ap.add_argument("--scenario", type=int, default=1, help="Target scenario chunk size")
    ap.add_argument("--time-daily", type=int, default=365, help="Target daily time chunk")
    ap.add_argument("--time-monthly", type=int, default=12, help="Target monthly time chunk")
    ap.add_argument("--time-annual", type=int, default=1, help="Target annual time chunk")
    ap.add_argument("--max-mem", type=str, default="8GB", help="Max memory budget for rechunker (e.g. 8GB)")
    ap.add_argument("--threads", type=int, default=8, help="Threads/workers for local Dask cluster")
    ap.add_argument("--dask", action="store_true", help="Start a local Dask cluster (recommended on SLURM)")
    ap.add_argument("--dry-run", action="store_true", help="List assigned stores (after sharding) and exit")
    args = ap.parse_args()

    in_dir = args.in_dir.resolve()
    out_dir = args.out_dir.resolve()
    tmp_root = args.tmp.resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(find_zarrs(in_dir))
    if not all_paths:
        print("No .zarr stores found.", file=sys.stderr)
        return

    # SLURM sharding
    shard_paths = slurm_shard(all_paths)
    if not shard_paths:
        print("[INFO] Nothing assigned to this shard.")
        return

    print(f"[INFO] Total stores: {len(all_paths)} | This shard: {len(shard_paths)}")

    for zpath in shard_paths:
        try:
            res = detect_resolution(zpath)
        except ValueError as e:
            print(f"[SKIP] {zpath}: {e}", file=sys.stderr)
            continue

        out_path = mirror_out_path(in_dir, out_dir, zpath)
        if out_path.exists():
            print(f"[SKIP] exists: {out_path}")
            continue

        print(f"[{res:7}] {zpath}  →  {out_path}")
        if args.dry_run:
            continue

        # unique per-task tmp root (good for arrays)
        ta = os.getenv("SLURM_ARRAY_TASK_ID", "0")
        jid = os.getenv("SLURM_JOB_ID", "local")
        per_task_tmp = tmp_root / f"task_{jid}_{ta}"
        per_task_tmp.mkdir(parents=True, exist_ok=True)

        try:
            rechunk_store(
                in_store=zpath,
                out_store=out_path,
                tmp_store_root=per_task_tmp,
                resolution=res,
                loc_chunk=args.loc,
                scen_chunk=args.scenario,
                max_mem=args.max_mem,
                use_dask=args.dask,
                threads=args.threads,
                time_daily=args.time_daily,
                time_monthly=args.time_monthly,
                time_annual=args.time_annual,
            )
        except Exception as e:
            print(f"[ERROR] Failed rechunking {zpath}: {e}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()