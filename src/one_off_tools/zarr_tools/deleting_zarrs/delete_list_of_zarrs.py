#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple

# Path to your existing Slurm delete script
DELETE_SCRIPT = Path(
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/src/one_off_tools/delete_single_zarr.sh"
)

# The list of Zarr stores you want to delete
ZARR_TARGETS: List[str] = [
    # --- inference_orig S3
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_orig/S3/annual.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_orig/S3/daily.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_orig/S3/monthly.zarr",

    # --- inference_seperate S0/S3
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S0/daily/tmp.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S0/daily/ugrd.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S0/daily.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily/cld.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily/dlwrf.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily/fd.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily/potential_radiation.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily/pres.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily/pre.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily/spfh.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily/tmax.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily/tmin.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily/tmp.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily/tswrf.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily/ugrd.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily/vgrd.zarr",
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference_seperate/S3/daily.zarr",

    # --- training zarrs
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/test/test_location_whole_period/daily.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/test/test_location_whole_period/monthly.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_test_period_early/annual.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_test_period_early/daily.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_test_period_early/monthly.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_test_period_late/annual.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_test_period_late/daily.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_test_period_late/monthly.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_train_period/annual.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_train_period/daily.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_train_period/monthly.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_val_period_early/annual.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_val_period_early/daily.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_val_period_early/monthly.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_val_period_late/annual.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_val_period_late/daily.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/train/train_location_val_period_late/monthly.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/val/val_location_whole_period/annual.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/val/val_location_whole_period/daily.zarr",
    "/Net/Groups/BGI/work_2/TRENDY-Emulator-Data/Zarr/training/val/val_location_whole_period/monthly.zarr",

    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/training_tiles/train/train_location_train_period/daily.zarr/daily.zarr",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shard a list of Zarr stores across a Slurm array and submit per-item delete jobs."
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be submitted without calling sbatch.")
    p.add_argument("--skip-missing", action="store_true",
                   help="Skip items that do not exist instead of failing.")
    p.add_argument("--delete-script", type=Path, default=DELETE_SCRIPT,
                   help="Path to the delete_single_zarr.sh script.")
    p.add_argument("--chdir", type=Path, default=DELETE_SCRIPT.parent,
                   help="Working directory for sbatch (controls where log dir is created).")
    return p.parse_args()

def get_array_context() -> Tuple[int, int]:
    """Read SLURM array env; default to a single shard (0/1) if not set."""
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    task_cnt = int(os.environ.get("SLURM_ARRAY_TASK_COUNT",
                                  os.environ.get("SLURM_ARRAY_TASK_MAX", "1")))
    # Some clusters expose MIN/MAX, some COUNT; fall back to 1 if unknown.
    if task_cnt <= 0:
        task_cnt = 1
    return task_id, task_cnt

def pick_assigned(items: List[str], task_id: int, task_cnt: int) -> List[str]:
    return [p for i, p in enumerate(items) if i % task_cnt == task_id]

def submit(delete_script: Path, target: Path, chdir: Path, dry_run: bool) -> str:
    cmd = ["sbatch", str(delete_script), str(target)]
    if dry_run:
        print("[DRY] ", " ".join(cmd))
        return "DRY-RUN"
    # Ensure working dir (for script's logs/ path) exists
    chdir.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(cmd, cwd=str(chdir), capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"sbatch failed for {target} (rc={res.returncode}):\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    # sbatch prints: "Submitted batch job <jobid>"
    line = (res.stdout or "").strip().splitlines()[-1] if res.stdout else ""
    print(f"[OK] {target} -> {line}")
    return line

def main() -> None:
    args = parse_args()

    # Validate delete script
    if not args.delete_script.exists():
        print(f"[FATAL] Delete script not found: {args.delete_script}", file=sys.stderr)
        sys.exit(2)

    task_id, task_cnt = get_array_context()
    print(f"[INFO] Array context: task_id={task_id} / task_cnt={task_cnt}")
    assigned = pick_assigned(ZARR_TARGETS, task_id, task_cnt)
    print(f"[INFO] Assigned {len(assigned)} of {len(ZARR_TARGETS)} targets.")

    if not assigned:
        print("[INFO] Nothing to do for this shard.")
        return

    # Submit jobs
    failures = []
    for raw in assigned:
        tgt = Path(raw)
        # Sanity checks; mirror the shell script's checks for earlier failure visibility
        if not tgt.name.endswith(".zarr"):
            msg = f"[SKIP] Not a .zarr path: {tgt}"
            if args.skip-missing:
                print(msg)
                continue
            print(msg, file=sys.stderr)
            failures.append(raw)
            continue
        if not tgt.exists():
            msg = f"[SKIP] Missing path: {tgt}"
            if args.skip-missing:
                print(msg)
                continue
            print(msg, file=sys.stderr)
            failures.append(raw)
            continue
        if not tgt.is_dir():
            print(f"[SKIP] Not a directory: {tgt}", file=sys.stderr)
            failures.append(raw)
            continue

        try:
            submit(args.delete_script, tgt, args.chdir, args.dry_run)
        except Exception as e:
            print(f"[ERR] Submission failed for {tgt}: {e}", file=sys.stderr)
            failures.append(raw)

    if failures:
        print(f"[DONE] Completed with {len(failures)} submission failure(s).", file=sys.stderr)
        sys.exit(1)
    print("[DONE] All submissions dispatched.")

if __name__ == "__main__":
    main()