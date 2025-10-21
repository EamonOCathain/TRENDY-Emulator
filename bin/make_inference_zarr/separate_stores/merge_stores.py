#!/usr/bin/env python3
from pathlib import Path
import zarr
import xarray as xr
from numcodecs import Blosc
import sys

# Project paths & utils
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.paths.paths import zarr_dir
from src.dataset.variables import var_names
from src.utils.tools import slurm_shard

TIME_RES = "daily"
PER_VAR_ROOT_NAME = "inference_seperate"  # folder where per-var stores live
SCENARIOS = ["S0", "S1","S2","S3"]                 # fixed set; no CLI args

def make_skeleton(dst_store: Path, like_store: Path):
    """Create coords-only skeleton at dst_store by copying coords from like_store."""
    ds_like = xr.open_zarr(like_store, consolidated=False)

    coords_only = xr.Dataset(coords=dict(
        time=("time", ds_like["time"].values),
        lat =("lat",  ds_like["lat"].values),
        lon =("lon",  ds_like["lon"].values),
    ))

    compressor = Blosc(cname="zstd", clevel=8, shuffle=Blosc.BITSHUFFLE)
    enc = {
        "time": {"compressor": compressor, "chunks": (-1,)},
        "lat":  {"compressor": compressor, "chunks": (ds_like.sizes["lat"],)},
        "lon":  {"compressor": compressor, "chunks": (ds_like.sizes["lon"],)},
    }
    dst_store.parent.mkdir(parents=True, exist_ok=True)
    coords_only.to_zarr(str(dst_store), mode="w", encoding=enc)

def copy_var(src_var_store: Path, dst_store: Path, var_name: str):
    """Copy raw array from src var-store into dst combined store (fast, no coords)."""
    src = zarr.open_group(str(src_var_store), mode="r")
    dst = zarr.open_group(str(dst_store), mode="a")

    if var_name not in src:
        raise FileNotFoundError(f"{var_name} not found in {src_var_store}")

    if var_name in dst:
        del dst[var_name]

    zarr.copy(src[var_name], dst, name=var_name, if_exists="replace")
    dst[var_name].attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]

def merge_one_scenario(scenario: str):
    var_root = zarr_dir / f"{PER_VAR_ROOT_NAME}/{scenario}/{TIME_RES}"
    dst_store = zarr_dir / f"{PER_VAR_ROOT_NAME}/{scenario}/{TIME_RES}.zarr"

    daily_vars = list(var_names[TIME_RES])

    # pick a "like" store (first existing var) to seed coords
    like = None
    for v in daily_vars:
        pth = var_root / f"{v}.zarr"
        if pth.exists():
            like = pth
            break
    if like is None:
        raise FileNotFoundError(f"No per-var stores found in {var_root}")

    print(f"[INFO] ({scenario}) Creating skeleton from: {like}")
    make_skeleton(dst_store, like)

    missing = 0
    for v in daily_vars:
        src_store = var_root / f"{v}.zarr"
        if not src_store.exists():
            print(f"[SKIP] ({scenario}) missing {src_store}")
            missing += 1
            continue
        print(f"[MERGE] ({scenario}) {v}")
        copy_var(src_store, dst_store, v)

    zarr.consolidate_metadata(str(dst_store))
    print(f"[OK] ({scenario}) merged -> {dst_store} (missing: {missing})")

def main():
    # Build tasks and shard via SLURM
    tasks = SCENARIOS[:]  # ["S0","S3"]
    tasks = slurm_shard(tasks)
    print(f"[INFO] total tasks={len(SCENARIOS)}, shard has {len(tasks)}")

    for scenario in tasks:
        merge_one_scenario(scenario)

if __name__ == "__main__":
    main()