#!/usr/bin/env python3
import os
import re
from pathlib import Path
from typing import List
import xarray as xr
import numpy as np

# --- Project config ---
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
import sys; sys.path.append(str(project_root))
from src.paths.paths import preprocessed_dir
from src.dataset.variables import var_names

OVERWRITE = False

# Where the yearly files live
HIST_ROOT = preprocessed_dir / "1x1" / "historical"    / "annual_files"
PREI_ROOT = preprocessed_dir / "1x1" / "preindustrial" / "annual_files"

# Where to write 10-year files
HIST_OUT = preprocessed_dir / "1x1" / "historical"    / "decade_files"
PREI_OUT = preprocessed_dir / "1x1" / "preindustrial" / "decade_files"

# Variables to process (your daily forcings)
VARS = list(var_names["daily"])

YEAR_RE = re.compile(r"(\d{4})")

def list_year_files(var_dir: Path, var: str) -> List[Path]:
    """Return sorted list of {var}_YYYY.nc files."""
    files = sorted(var_dir.glob(f"{var}_*.nc"))
    return [p for p in files if YEAR_RE.search(p.stem)]

def year_of(path: Path) -> int:
    m = YEAR_RE.search(path.stem)
    if not m:
        raise ValueError(f"Cannot parse year from {path.name}")
    return int(m.group(1))

def split_into_blocks(files: List[Path], block_len: int = 10) -> List[List[Path]]:
    """Split sorted yearly files into consecutive blocks of size block_len (last block may be shorter)."""
    files = sorted(files, key=year_of)
    blocks, block, prev_y = [], [], None
    for f in files:
        y = year_of(f)
        if prev_y is not None and y != prev_y + 1:
            if block:
                blocks.append(block)
            block = [f]
        else:
            block.append(f)
        prev_y = y
        if len(block) == block_len:
            blocks.append(block)
            block = []
    if block:
        blocks.append(block)
    return blocks

def concat_and_write_block(files: List[Path], var: str, out_dir: Path):
    y0, y1 = year_of(files[0]), year_of(files[-1])
    out_dir_var = out_dir / var
    out_dir_var.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir_var / f"{var}_{y0}-{y1}.nc"
    if out_fp.exists() and not OVERWRITE:
        print(f"[SKIP] {out_fp} exists")
        return

    # Preserve _FillValue from first file (if present)
    try:
        with xr.open_dataset(files[0], decode_times=False) as ds0:
            fv = ds0[var].encoding.get("_FillValue", None)
    except Exception:
        fv = None

    # Uncompressed, keep (365,1,1) chunking, f4 dtype
    var_encoding = {
        "zlib": False,
        "complevel": 0,
        "shuffle": False,
        "chunksizes": (365, 1, 1),
        "_FillValue": fv,
        "dtype": "f4",
    }

    print(f"[BUILD] {var} {y0}-{y1} -> {out_fp.name}")

    ds = xr.open_mfdataset(
        files,
        concat_dim="time",
        combine="nested",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        decode_times=False,
        parallel=False,              # single worker per SLURM task
        engine="netcdf4",
        chunks={"time": 365, "lat": 1, "lon": 1},
    )

    try:
        ds_out = ds[[var]].astype("float32")
        ds_out.to_netcdf(
            out_fp,
            mode="w",
            format="NETCDF4",
            engine="netcdf4",
            encoding={var: var_encoding},
            unlimited_dims=("time",),
            compute=True,
        )
    finally:
        ds.close()

    print(f"[DONE] {out_fp}")

def build_tasks() -> list[tuple[Path, Path, str, list[Path]]]:
    """
    Create a flat list of (src_root, dst_root, var, block_files_list) tasks
    across BOTH historical and preindustrial, for all daily vars.
    """
    tasks = []
    for (src_root, dst_root) in [
        (HIST_ROOT, HIST_OUT),
        (PREI_ROOT, PREI_OUT),
    ]:
        # Drop potential_radiation for preindustrial
        vars_to_use = VARS if src_root != PREI_ROOT else [v for v in VARS if v != "potential_radiation"]

        for var in vars_to_use:
            var_dir = src_root / var
            if not var_dir.is_dir():
                print(f"[WARN] Missing dir for {var}: {var_dir}")
                continue
            files = list_year_files(var_dir, var)
            if not files:
                print(f"[WARN] No files for {var} in {var_dir}")
                continue
            blocks = split_into_blocks(files, block_len=10)
            for block in blocks:
                tasks.append((src_root, dst_root, var, block))

    # Stable sort for reproducibility
    def task_key(t):
        _, _, v, block = t
        return (v, year_of(block[0]), year_of(block[-1]))
    tasks.sort(key=task_key)
    return tasks

def shard_tasks_by_array(tasks: list, *, count: int, idx: int) -> list:
    """Modulo sharding by output file across the full task list."""
    if count <= 1:
        return tasks
    return [t for k, t in enumerate(tasks) if k % count == idx]

def process_task(task: tuple[Path, Path, str, list[Path]]):
    _, dst_root, var, block = task
    concat_and_write_block(block, var, dst_root)

if __name__ == "__main__":
    all_tasks = build_tasks()
    total = len(all_tasks)
    print(total)  # lets you script dynamic --array sizing
    array_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
    array_id    = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    my_tasks = shard_tasks_by_array(all_tasks, count=array_count, idx=array_id)

    print(f"[INFO] Total tasks: {total} | This array task ({array_id}/{array_count}) will run: {len(my_tasks)}")
    for task in my_tasks:
        process_task(task)