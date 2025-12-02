#!/usr/bin/env python3
import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np

try:
    import zarr
except ImportError:
    print("This script requires `zarr` (pip install zarr).", file=sys.stderr)
    sys.exit(1)


# --------------------------- utils ---------------------------

def is_zarr_dir(p: Path) -> bool:
    """Heuristic: directory that ends with .zarr OR contains .zarray/.zmetadata somewhere at top."""
    if not p.is_dir():
        return False
    if p.name.endswith(".zarr"):
        return True
    # quick top-level check
    for name in (".zarray", ".zmetadata"):
        if (p / name).exists():
            return True
    return False


def find_zarr_groups(root: Path, max_depth: int | None, include_glob: str | None) -> list[Path]:
    """
    Recursively discover Zarr groups.
    - max_depth: 0 means only root; 1 means root/*; None means unlimited.
    - include_glob: optional glob that must match the directory name (e.g. '*.zarr').
    """
    root = root.resolve()
    zarrs = []
    root_depth = len(root.parts)

    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        depth = len(p.parts) - root_depth
        if max_depth is not None and depth > max_depth:
            # prune deeper descent
            dirnames[:] = []
            continue

        if is_zarr_dir(p):
            if include_glob is None or p.match(include_glob):
                zarrs.append(p)
            # do not descend into a zarr group
            dirnames[:] = []
            continue

    zarrs.sort()
    return zarrs


def open_zarr_group(path: str):
    """Open a Zarr group, attempting consolidated first."""
    try:
        return zarr.open_consolidated(path, mode='r')
    except Exception:
        return zarr.open_group(path, mode='r')


def parse_range(s: str | None):
    if not s:
        return None
    if ":" in s:
        a, b = s.split(":", 1)
        a = int(a) if a else None
        b = int(b) if b else None
        return (a, b)
    idx = int(s)
    return (idx, idx + 1)


def partition(total: int, tasks: int, task_id: int) -> tuple[int, int]:
    """Even partition [0,total) into tasks parts, return slice for task_id."""
    if tasks <= 0:
        return (0, total)
    if task_id < 0 or task_id >= tasks:
        raise SystemExit(f"--task-id {task_id} must be in [0, {tasks-1}] for --tasks {tasks}")
    base = total // tasks
    rem = total % tasks
    start = task_id * base + min(task_id, rem)
    end = start + base + (1 if task_id < rem else 0)
    return (start, end)


def check_block(arr: zarr.Array, l_start: int, l_end: int):
    """Return set of relative location indices with any non-finite in block [l_start:l_end)."""
    sl = [slice(None)] * (arr.ndim - 1) + [slice(l_start, l_end)]
    chunk = arr.oindex[tuple(sl)]
    finite_mask = np.isfinite(chunk)
    all_finite_per_loc = np.all(finite_mask, axis=tuple(range(chunk.ndim - 1)))
    bad_rel = np.nonzero(~all_finite_per_loc)[0]
    return set(map(int, bad_rel))


def scan_one_group(
    group_path: Path,
    array_names: list[str],
    loc_range: tuple[int, int] | None,
    block_size: int,
    csv_fh
) -> dict:
    """Scan a single Zarr group; print one line per offending location; return summary dict."""
    grp = open_zarr_group(str(group_path))
    arrays = {}
    for nm in array_names:
        if nm not in grp:
            print(f"[WARN] {group_path}: array '{nm}' not found; skipping it.", file=sys.stderr)
            continue
        arr = grp[nm]
        if arr.ndim < 1:
            print(f"[WARN] {group_path}: array '{nm}' has ndim={arr.ndim}; skipping.", file=sys.stderr)
            continue
        arrays[nm] = arr

    if not arrays:
        print(f"[WARN] {group_path}: no valid arrays; skipping group.", file=sys.stderr)
        return {
            "zarr_path": str(group_path),
            "arrays_scanned": [],
            "L": 0,
            "range_start": 0,
            "range_end": 0,
            "num_bad_locations": 0,
            "bad_locations_unique": [],
            "bad_counts_by_array": {},
        }

    # ensure all arrays share same L
    first = next(iter(arrays.values()))
    L = int(first.shape[-1])
    for nm, arr in arrays.items():
        if int(arr.shape[-1]) != L:
            raise SystemExit(f"{group_path}: array '{nm}' last-dim L={arr.shape[-1]} "
                             f"!= {L} from first array")

    if loc_range is None:
        start, end = 0, L
    else:
        start, end = loc_range
        start = 0 if start is None else start
        end = L if end is None else end
        if not (0 <= start < end <= L):
            raise SystemExit(f"{group_path}: --loc-range out of bounds for L={L}: {start}:{end}")

    print(f"[scan] group={group_path} arrays={list(arrays.keys())} L={L} range={start}:{end} block={block_size}")

    bad_locs_union = set()
    bad_counts_by_array = {k: 0 for k in arrays.keys()}

    for bstart in range(start, end, block_size):
        bend = min(bstart + block_size, end)

        block_reports = {}  # loc -> set(array_names_bad)
        for nm, arr in arrays.items():
            bad_rel = check_block(arr, bstart, bend)
            if not bad_rel:
                continue
            for rel in bad_rel:
                gl = bstart + rel
                block_reports.setdefault(gl, set()).add(nm)
            bad_counts_by_array[nm] += len(bad_rel)

        # Emit one line per bad location in this block
        for gl in sorted(block_reports.keys()):
            if gl not in bad_locs_union:
                which = ",".join(sorted(block_reports[gl]))
                print(f"[bad] group={group_path} loc={gl} arrays={which}")
            bad_locs_union.add(gl)
            if csv_fh:
                for nm in block_reports[gl]:
                    csv_fh.write(f"{group_path},{nm},{gl}\n")

    summary = {
        "zarr_path": str(group_path),
        "arrays_scanned": list(arrays.keys()),
        "L": L,
        "range_start": start,
        "range_end": end,
        "bad_locations_unique": sorted(map(int, bad_locs_union)),
        "num_bad_locations": len(bad_locs_union),
        "bad_counts_by_array": bad_counts_by_array,
    }
    print(f"[scan] summary for {group_path}: {json.dumps(summary, indent=2)}")
    return summary


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Scan a Zarr group OR a directory tree of many Zarrs for NaNs/Infs along the location axis."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zarr-path", type=str, help="Single Zarr group path/URL")
    src.add_argument("--root", type=str, help="Root directory containing many Zarrs under subdirectories")

    ap.add_argument("--arrays", nargs="+", default=["x", "m", "a"],
                    help="Array names to scan (default: x m a)")
    ap.add_argument("--loc-range", default=None,
                    help="Location range 'start:end' or single 'k' (only for --zarr-path)")
    ap.add_argument("--block-size", type=int, default=2048,
                    help="Locations per I/O block (default: 2048)")

    # Discovery options for --root
    ap.add_argument("--include-glob", type=str, default=None,
                    help="Only include Zarr dirs whose name matches this glob (e.g. '*.zarr')")
    ap.add_argument("--max-depth", type=int, default=None,
                    help="Max directory depth to descend from --root (0=root only, None=unlimited)")

    # Parallelization *across Zarr groups*
    ap.add_argument("--tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "0")),
                    help="Total tasks (for SLURM array). If >0, Zarr list is partitioned across tasks.")
    ap.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")),
                    help="This task id (0-based) when using --tasks>0")

    # Outputs
    ap.add_argument("--csv", type=str, default=None,
                    help="CSV path to append lines: zarr_path,array,loc")
    ap.add_argument("--summary-json", type=str, default=None,
                    help="Write a combined JSON summary across processed Zarrs")

    args = ap.parse_args()

    # Optional CSV
    csv_fh = None
    if args.csv:
        first_write = not Path(args.csv).exists()
        csv_fh = open(args.csv, "a", buffering=1)
        if first_write:
            csv_fh.write("zarr_path,array,loc\n")

    summaries = []

    if args.zarr_path:
        # Single group mode
        lr = parse_range(args.loc_range)
        summaries.append(
            scan_one_group(Path(args.zarr_path), args.arrays, lr, args.block_size, csv_fh)
        )

    else:
        # Root discovery mode
        root = Path(args.root)
        if not root.exists():
            raise SystemExit(f"--root does not exist: {root}")

        all_groups = find_zarr_groups(root, args.max_depth, args.include_glob)
        if not all_groups:
            print(f"[INFO] No Zarr groups found under {root}.", file=sys.stderr)
            if csv_fh:
                csv_fh.close()
            # still write an empty summary if requested
            if args.summary_json:
                with open(args.summary_json, "w") as f:
                    json.dump({"groups": [], "processed": [], "skipped": []}, f, indent=2)
            return

        # Partition across groups (SLURM arrays)
        start, end = partition(len(all_groups), args.tasks, args.task_id) if args.tasks and args.tasks > 0 else (0, len(all_groups))
        groups = all_groups[start:end]

        print(f"[discover] root={root} total_groups={len(all_groups)} scanning_slice={start}:{end} count={len(groups)}")
        for gp in groups:
            try:
                summaries.append(
                    scan_one_group(gp, args.arrays, None, args.block_size, csv_fh)
                )
            except Exception as e:
                print(f"[ERROR] scanning {gp}: {e}", file=sys.stderr)

    if csv_fh:
        csv_fh.close()

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump({"groups": summaries}, f, indent=2)


if __name__ == "__main__":
    main()