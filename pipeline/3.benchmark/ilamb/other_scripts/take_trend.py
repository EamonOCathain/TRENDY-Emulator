#!/usr/bin/env python3
import argparse, subprocess, sys, tempfile
from pathlib import Path

PATTERNS = ("cveg", "clitter", "csoil", "lai")  # case-insensitive match on filename

def has_cdo() -> bool:
    from shutil import which
    return which("cdo") is not None

def want(file: Path) -> bool:
    name = file.name.lower()
    return any(tok in name for tok in PATTERNS)

def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def out_paths(in_nc: Path, out_dir: Path|None):
    base = in_nc.with_suffix("")  # strip .nc
    stem = base.name
    if out_dir is None:
        out_slope = in_nc.parent / f"{stem}_trend.nc"
        out_inter = in_nc.parent / f"{stem}_trend_intercept.nc"
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_slope = out_dir / f"{stem}_trend.nc"
        out_inter = out_dir / f"{stem}_trend_intercept.nc"
    return out_slope, out_inter

def process_one(nc_path: Path, *, overwrite: bool, out_dir: Path|None):
    if not want(nc_path):
        print(f"[SKIP] {nc_path.name} (not in {PATTERNS})")
        return

    out_slope, out_inter = out_paths(nc_path, out_dir)
    if not overwrite and out_slope.exists() and out_inter.exists():
        print(f"[SKIP] {nc_path.name} (outputs exist)")
        return

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        clim = td / "clim.nc"
        anom = td / "anom.nc"

        # 1) monthly climatology
        run(["cdo", "-L", "-O", "ymonmean", str(nc_path), str(clim)])
        # 2) anomalies
        run(["cdo", "-L", "-O", "ymonsub", str(nc_path), str(clim), str(anom)])
        # 3) trend (compressed NetCDF4)
        run(["cdo", "-L", "-O", "-f", "nc4c", "-z", "zip_4",
             "trend", str(anom), str(out_slope), str(out_inter)])

    print(f"[OK] {nc_path.name} → {out_slope.name}, {out_inter.name}")

def main():
    ap = argparse.ArgumentParser(description="Compute deseasonalised trends for cVeg/cLitter/cSoil/lai files under a directory.")
    ap.add_argument("--in_dir", required=True, type=Path, help="Directory to search (recursively) for .nc files")
    ap.add_argument("--out_dir", type=Path, default=None, help="If set, write trends here (flat). Else write beside inputs.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    if not has_cdo():
        sys.exit("CDO not found in PATH.")

    in_dir = args.in_dir.resolve()
    if not in_dir.is_dir():
        sys.exit(f"--in_dir is not a directory: {in_dir}")

    files = sorted(p for p in in_dir.rglob("*.nc"))
    targets = [p for p in files if want(p)]

    print(f"[INFO] Found {len(files)} .nc; matching targets: {len(targets)} under {in_dir}")
    if not targets:
        print("[INFO] Nothing to do.")
        return

    # Optional SLURM striding
    tid = os.getenv("SLURM_ARRAY_TASK_ID")
    tct = os.getenv("SLURM_ARRAY_TASK_COUNT")
    if tid is not None and tct is not None:
        tid, tct = int(tid), max(1, int(tct))
        targets = [p for i, p in enumerate(targets) if (i % tct) == tid]
        print(f"[INFO] SLURM shard {tid}/{tct}: {len(targets)} files")

    for p in targets:
        try:
            process_one(p, overwrite=args.overwrite, out_dir=args.out_dir)
        except subprocess.CalledProcessError as e:
            print(f"[ERR] CDO failed for {p}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[ERR] Unexpected error for {p}: {e}", file=sys.stderr)

if __name__ == "__main__":
    import os
    main()