#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path

import zarr

'''python consolidate_zarrs_in_dir.py /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/base_model_new_loss/no_carry/zarr/S3 --force

Can be run on a directory containing many .zarr stores under several subdirs'''

# (optional) use your project helper if available; otherwise no-op sharding
def _slurm_shard(paths):
    try:
        from src.utils.tools import slurm_shard  # your helper
        return slurm_shard(paths)
    except Exception:
        # simple fallback: this rank does all paths
        return paths

def is_zarr_dir(p: Path) -> bool:
    return p.is_dir() and ((p / ".zgroup").exists() or (p / ".zarray").exists())

def find_zarrs(root: Path):
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.endswith(".zarr"):
                p = Path(dirpath) / d
                if is_zarr_dir(p):
                    out.append(p)
    return sorted(out)

def consolidate(path: Path, force: bool = False, quiet: bool = False):
    zmeta = path / ".zmetadata"
    if zmeta.exists() and not force:
        if not quiet:
            print(f"[SKIP] already consolidated: {path}")
        return
    if not quiet:
        print(f"[DO] consolidating: {path}")
    # This scans the store and writes a single .zmetadata at the root
    zarr.consolidate_metadata(str(path))

def main():
    ap = argparse.ArgumentParser(description="Consolidate Zarr stores (write .zmetadata).")
    ap.add_argument("root", type=Path, help="Directory to scan for *.zarr")
    ap.add_argument("--force", action="store_true", help="Re-write .zmetadata even if it exists")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging")
    ap.add_argument("--dry-run", action="store_true", help="List targets without writing")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    all_stores = find_zarrs(root)
    if not all_stores:
        print("No .zarr stores found.", file=sys.stderr)
        return

    # Optional SLURM sharding
    shard = _slurm_shard(all_stores)
    if not shard:
        print("[INFO] Nothing assigned to this shard.")
        return

    print(f"[INFO] total={len(all_stores)} | this shard={len(shard)}")
    for p in shard:
        if args.dry_run:
            print(f"[DRY] would consolidate: {p}")
            continue
        try:
            consolidate(p, force=args.force, quiet=args.quiet)
        except Exception as e:
            print(f"[ERROR] {p}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()