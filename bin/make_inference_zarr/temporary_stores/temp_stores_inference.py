#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import argparse
import zarr

# ------------------ USER CONFIG ------------------
OVERWRITE = False
LAT_CHUNK = 30
LON_CHUNK = 30
YEARS = np.arange(1901, 2024, 1)

# ------------------ Project paths & imports ------------------
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.paths.paths import zarr_dir, preindustrial_dir, historical_dir, data_dir
from src.dataset.variables import var_names, climate_vars, land_use_vars
from src.utils.tools import slurm_shard
from src.utils.zarr_tools import (
    netcdf_to_zarr_var,
    annual_netcdf_to_zarr,
)

# ------------------ Helpers ------------------
def choose_scenario_dir(scenario: str, var: str) -> Path:
    S0_vars = set(climate_vars + land_use_vars + ["co2"])
    S1_vars = set(climate_vars + land_use_vars)
    S2_vars = set(land_use_vars)
    if scenario == "S0" and var in S0_vars: return preindustrial_dir
    if scenario == "S1" and var in S1_vars: return preindustrial_dir
    if scenario == "S2" and var in S2_vars: return preindustrial_dir
    return historical_dir

def parse_args():
    p = argparse.ArgumentParser(description="Per-variable writer to temporary Zarr stores.")
    p.add_argument("--time-res", required=True, choices=["annual", "monthly", "daily"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--tmp-root", default="inference_tmp")
    return p.parse_args()

def time_chunk_for(time_res: str) -> int:
    if time_res == "annual":  return 1
    if time_res == "monthly": return 12
    if time_res == "daily":   return 365  
    raise ValueError(time_res)

# ------------------ Main ------------------
def main():
    args = parse_args()
    time_res = args.time_res
    overwrite = args.overwrite or OVERWRITE

    SCENARIOS = ["S0", "S1", "S2", "S3"]
    vars_for_tr = list(var_names[time_res])
    tasks = [(scen, var) for scen in SCENARIOS for var in vars_for_tr]
    tasks = slurm_shard(tasks)

    preproc_outputs_dir = data_dir / "preprocessed" / "model_outputs"
    tchunk = time_chunk_for(time_res)

    for scenario, var in tasks:
        tmp_store = zarr_dir / f"{args.tmp_root}/{scenario}/{time_res}/{var}.zarr"
        tmp_store.parent.mkdir(parents=True, exist_ok=True)
        print(f"[TASK] {scenario}/{time_res}:{var} -> {tmp_store}", flush=True)

        if var in var_names["outputs"]:
            if time_res == "daily":
                raise RuntimeError(f"Daily outputs not supported for {var} (expected only monthly/annual).")
            in_path = preproc_outputs_dir / f"ENSMEAN_{scenario}_{var}.nc"
            if not in_path.exists():
                raise FileNotFoundError(f"Missing output source: {in_path}")

            netcdf_to_zarr_var(
                nc_path=in_path,
                zarr_store=tmp_store,
                var_name=var,
                overwrite=overwrite,
                lat_chunk=LAT_CHUNK,
                lon_chunk=LON_CHUNK,
                time_chunk=tchunk,         
                consolidate=False,
            )
            print(f"[DONE] outputs {scenario}/{time_res}:{var} -> {tmp_store.name}")
            continue

        # Forcings
        src_root = choose_scenario_dir(scenario, var)

        if time_res in ("monthly", "annual"):
            in_path = src_root / f"full_time/{var}.nc"
            if not in_path.exists():
                raise FileNotFoundError(f"Missing source: {in_path}")

            netcdf_to_zarr_var(
                nc_path=in_path,
                zarr_store=tmp_store,
                var_name=var,
                overwrite=overwrite,
                lat_chunk=LAT_CHUNK,
                lon_chunk=LON_CHUNK,
                time_chunk=tchunk,          
                consolidate=False,
            )
            print(f"[DONE] forcing {scenario}/{time_res}:{var} -> {tmp_store.name}")

        elif time_res == "daily":
            year_files = [src_root / f"annual_files/{var}/{var}_{y}.nc" for y in YEARS]
            missing = [p for p in year_files if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing daily source files (first 3): {missing[:3]} ...")

            annual_netcdf_to_zarr(
                year_files=year_files,
                store=tmp_store,
                var_name=var,
                lat_chunks=LAT_CHUNK,
                lon_chunks=LON_CHUNK,
                time_chunks=tchunk,        
                overwrite=overwrite,
                mark_complete_when_done=True,
                consolidated=False,
            )
            print(f"[DONE] forcing {scenario}/{time_res}:{var} -> {tmp_store.name}")

        else:
            raise ValueError(f"Unsupported time_res: {time_res}")

if __name__ == "__main__":
    main()