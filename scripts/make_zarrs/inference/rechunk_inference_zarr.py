import zarr, xarray as xr
from rechunker import rechunk
from pathlib import Path
import fsspec
import sys, os

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.paths.paths import (masks_dir, zarr_dir)
from src.utils.tools import slurm_shard

inference_dir = zarr_dir / "inference"

SCENARIOS = ("S0", "S1", "S2", "S3")
TIME_RESES = ("annual", "monthly", "daily")

# Slurm logic 
tasks = []
for scenario in SCENARIOS:
    for time_res in TIME_RESES:
        src = inference_dir / f"/{scenario}/{time_res}.zarr"
        dst = zarr_dir / f"inference_6x6/{scenario}/{time_res}.zarr"
        tmp = zarr_dir / f"inference_rechunked/tmp/{scenario}_{time_res}"
        tasks.append((src, dst, tmp, scenario, time_res))

task_to_process = slurm_shard(tasks)

src = task_to_process[0]
dst = task_to_process[1]
tmp = task_to_process[2]
scenario = task_to_process[3]
time_res = task_to_process[4]

ds = xr.open_zarr(src, consolidated=True)

# choose good per-res chunks
if time_res == 'daily':
    target_chunks = {"time": 44895, "lat": 6, "lon": 6}
if time_res == 'monthly':
    target_chunks = {"time": 1476, "lat": 6, "lon": 6}
if time_res == "annual":
    target_chunks = {"time": 123, "lat": 6, "lon": 6}
    
# you can loop over ds.data_vars if you want to rechunk everything
store_in  = zarr.open_group(str(src), mode="r")
store_out = zarr.open_group(str(dst), mode="w")
store_tmp = zarr.DirectoryStore(str(tmp))

for v in ds.data_vars:
    arr_in  = store_in[v]
    arr_out = zarr.create(
        shape=arr_in.shape, chunks=(target_chunks["time"], target_chunks["lat"], target_chunks["lon"]),
        dtype=arr_in.dtype, store=store_out, path=v, overwrite=True, compressor=arr_in.compressor
    )
    plan = rechunk(arr_in, target_chunks,  # target
                   max_mem="1GB",          # worker memory during rechunk
                   target=arr_out, temp_store=store_tmp)
    plan.execute()

# copy coords & attrs once
ds[[]].to_zarr(dst, mode="a", consolidated=False)  # creates group/coords if needed
zarr.consolidate_metadata(str(dst))