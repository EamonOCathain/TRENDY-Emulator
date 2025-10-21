#!/usr/bin/env python3
"""
Per-variable writer for inference Zarr (SLURM-array ready, no consolidation, no skeleton creation).

Run:
  python write_inference_var.py --time-res annual
  python write_inference_var.py --time-res monthly
  python write_inference_var.py --time-res daily
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import argparse
import zarr

# ------------------ USER CONFIG ------------------
OVERWRITE = False
LAT_CHUNK = 5
LON_CHUNK = 5
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
    """Return historical vs preindustrial for forcing vars."""
    S0_vars = set(climate_vars + land_use_vars + ["co2"])
    S1_vars = set(climate_vars + land_use_vars)
    S2_vars = set(land_use_vars)

    if scenario == "S0" and var in S0_vars:
        return preindustrial_dir
    if scenario == "S1" and var in S1_vars:
        return preindustrial_dir
    if scenario == "S2" and var in S2_vars:
        return preindustrial_dir
    return historical_dir

def parse_args():
    p = argparse.ArgumentParser(description="Per-variable writer for inference Zarr.")
    p.add_argument("--time-res", required=True, choices=["annual", "monthly", "daily"],
                   help="time resolution of variables to write")
    p.add_argument("--overwrite", action="store_true",
                   help="force overwrite existing data for each variable")
    return p.parse_args()

# ------------------ Main ------------------
def main():
    args = parse_args()
    time_res = args.time_res
    overwrite = args.overwrite or OVERWRITE

    SCENARIOS = ["S0", "S1", "S2", "S3"]
    vars_for_tr = list(var_names[time_res])     # e.g. var_names["annual"]
    tasks = [(scen, var) for scen in SCENARIOS for var in vars_for_tr]
    tasks = slurm_shard(tasks)                   # array sharding
    
    preproc_outputs_dir = data_dir / "preprocessed" / "model_outputs"

    for scenario, var in tasks:
        store = zarr_dir / f"inference_new/{scenario}/{time_res}.zarr"
        if not store.exists():
            raise FileNotFoundError(
                f"Destination store missing: {store}\n"
                f"(Create skeletons in a separate setup script before running this writer.)"
            )

        print(f"[TASK] {scenario}/{time_res}:{var} -> {store}", flush=True)
        
        # ---------- Outputs live in preprocessed/model_outputs ----------
        if var in var_names["outputs"]:
            if time_res == "daily":
                raise RuntimeError(f"Daily outputs not supported for {var} (expected only monthly/annual).")

            in_path = preproc_outputs_dir / f"ENSMEAN_{scenario}_{var}.nc"
            if not in_path.exists():
                raise FileNotFoundError(f"Missing output source: {in_path}")

            netcdf_to_zarr_var(
                nc_path=in_path,
                zarr_store=store,
                var_name=var,
                overwrite=overwrite,
                lat_chunk=LAT_CHUNK,
                lon_chunk=LON_CHUNK,
                consolidate=False,
            )
            print(f"[DONE] outputs {scenario}/{time_res}:{var} -> {store.name}")
            continue

        # ---------- Forcings (historical/preindustrial split) ----------
        src_root = choose_scenario_dir(scenario, var)

        if time_res in ("monthly", "annual"):
            in_path = src_root / f"full_time/{var}.nc"
            if not in_path.exists():
                raise FileNotFoundError(f"Missing source: {in_path}")

            netcdf_to_zarr_var(
                nc_path=in_path,
                zarr_store=store,
                var_name=var,
                overwrite=overwrite,
                lat_chunk=LAT_CHUNK,
                lon_chunk=LON_CHUNK,
                consolidate=False,
            )
            print(f"[DONE] forcing {scenario}/{time_res}:{var} -> {store.name}")

        elif time_res == "daily":
            year_files = [src_root / f"annual_files/{var}/{var}_{y}.nc" for y in YEARS]
            missing = [p for p in year_files if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing daily source files (first 3): {missing[:3]} ...")

            annual_netcdf_to_zarr(
                year_files=year_files,
                store=store,
                var_name=var,
                lat_chunks=LAT_CHUNK,
                lon_chunks=LON_CHUNK,
                overwrite=overwrite,
                mark_complete_when_done=True,
                consolidated=False,
            )
            print(f"[DONE] forcing {scenario}/{time_res}:{var} -> {store.name}")

        else:
            raise ValueError(f"Unsupported time_res: {time_res}")

if __name__ == "__main__":
    main()