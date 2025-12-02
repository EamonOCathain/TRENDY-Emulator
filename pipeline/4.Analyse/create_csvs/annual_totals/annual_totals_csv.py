#!/usr/bin/env python3
"""
compute_annual_means.py
-----------------------

For each (scenario, var, model), compute spatially averaged *annual totals*
(time and space integrated, assuming monthly data on a 365-day calendar)
and write a CSV per (scenario, var):

  /.../annual_means/<SCENARIO>/<VAR>_annual_means.csv

Each CSV:
  - rows: years (union of all years across models for that scenario+var)
  - columns: models (keys of the 'MODEL_PATHS' dict below)
  - values: annual totals in Gt per year (Gt of the variable, e.g. Gt C yr-1)
  - missing years for a given model are NaN

Parallelisation
---------------

We build a list of (scenario, var) tasks and shard them across a SLURM array
using src.utils.tools.slurm_shard.

Example SLURM job:

  #SBATCH --array=0-15
  python compute_annual_means.py

Each array task will process a disjoint subset of (scenario, var) pairs.

You can optionally restrict scenarios / vars:

  python compute_annual_means.py --scenario S0 S1 --var gpp npp
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import xarray as xr
import pandas as pd
import numpy as np

# --- Project paths & imports -------------------------------------------------
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.utils.tools import slurm_shard

# -----------------------------------------------------------------------------


ROOT = Path(
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preds_for_analysis"
)

OUTPUT_ROOT = Path(
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/analysis/CSVs/annual_totals"
)

# Models / sources
MODEL_PATHS: Dict[str, Path] = {
    "Base-Emulator_No_Carry": ROOT / "base/no_carry",
    "Stable-Emulator_No_Carry": ROOT / "stable/no_carry",
    "Stable-Emulator_With_Carry": ROOT / "stable/carry",
    "Stable-Emulator_Carry_2000_2020": ROOT / "stable/2000_2020_carry",
    "TRENDY-Ensemble-Mean": ROOT / "ensmean",
    "ORCHIDEE": ROOT / "DGVMs/ORCHIDEE",
    "CLM": ROOT / "DGVMs/CLM",
    "ELM": ROOT / "DGVMs/ELM",
    "JSBACH": ROOT / "DGVMs/JSBACH",
    "CLASSIC": ROOT / "DGVMs/CLASSIC",
    "SDGVM": ROOT / "DGVMs/SDGVM",
    "VISIT": ROOT / "DGVMs/VISIT",
    "VISIT-UT": ROOT / "DGVMs/VISIT-UT",
    "TL-Emulator": ROOT / "TL_1982_2018",
}

VARS: List[str] = [
    "gpp",
    "rh",
    "ra",
    "nee",
    "npp",
    "cVeg",
    "cSoil",
    "cLitter",
    "mrro",
    "mrso",
    "nbp",
    "lai",
    "fLuc",
    "fFire",
    "evapotrans",
    "cTotal",
]

SCENARIOS: List[str] = ["S0", "S1", "S2", "S3"]

# ----------------------------------------------------------------------------- 
# Fast global tvt_mask (values in {0,1,2} are kept)
# ----------------------------------------------------------------------------- 

TVT_MASK_PATH = PROJECT_ROOT / "data/masks/tvt_mask.nc"

try:
    _tvt = xr.open_dataset(TVT_MASK_PATH)["tvt_mask"]
    # Drop time dimension if present
    if "time" in _tvt.dims:
        _tvt = _tvt.isel(time=0)
    # Boolean mask: True where tvt_mask ∈ {0,1,2}
    LAND_MASK = _tvt.isin([0, 1, 2]).load()  # load once into memory
    del _tvt
    print(f"[INFO] Loaded tvt_mask from {TVT_MASK_PATH}")
except Exception as e:
    LAND_MASK = None
    print(f"[WARN] Could not load tvt_mask from {TVT_MASK_PATH}: {e}")


def find_nc_file_for_var(scenario_dir: Path, var: str) -> Optional[Path]:
    """
    Find the NetCDF file for a variable in a scenario directory.

    Special rules:
      - For cTotal: accept either '*cTotal.nc' or '*cTotal_monthly.nc'
      - Otherwise: require exactly one '*<var>.nc' match
    """
    if not scenario_dir.is_dir():
        return None

    # Special case for cTotal → accept two naming conventions
    if var == "cTotal":
        candidates = (
            list(scenario_dir.glob("*cTotal_monthly.nc")) +
            list(scenario_dir.glob("*cTotal.nc"))
        )
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) == 0:
            print(f"[WARN] No file for var='cTotal' in {scenario_dir}")
            return None
        else:
            names = ", ".join(p.name for p in candidates)
            print(f"[WARN] Multiple cTotal files in {scenario_dir}: {names}")
            return None

    # Default rule for all other variables
    candidates = list(scenario_dir.glob(f"*{var}.nc"))
    if len(candidates) != 1:
        if len(candidates) == 0:
            print(f"[WARN] No file for var='{var}' in {scenario_dir}")
        else:
            names = ", ".join(p.name for p in candidates)
            print(f"[WARN] Multiple files for var='{var}' in {scenario_dir}: {names}")
        return None

    return candidates[0]


# ----------------------------------------------------------------------------- 
# Global cell-area field (m² per cell), built from ENSMEAN S0 nbp grid
# ----------------------------------------------------------------------------- 

CELL_AREA = None
DT_MONTH_SECONDS = (365.0 / 12.0) * 86400.0  # fixed TRENDY 365-day calendar

try:
    tmpl_dir = ROOT / "ensmean" / "S0"
    tmpl_nc = find_nc_file_for_var(tmpl_dir, "nbp")
    if tmpl_nc is not None:
        _ds_tmpl = xr.open_dataset(tmpl_nc)
        if ("lat" in _ds_tmpl.dims) and ("lon" in _ds_tmpl.dims):
            lat = _ds_tmpl["lat"]
            lon = _ds_tmpl["lon"]

            R = 6_371_000.0  # Earth radius (m)

            dlat = np.deg2rad(lat.diff("lat").mean())
            dlon = np.deg2rad(lon.diff("lon").mean())

            # Half-grid assumption for 0.5° (as in reference script): ±0.25°
            lat_bounds = (np.deg2rad(lat + 0.25), np.deg2rad(lat - 0.25))

            cell_area_1d = (R ** 2) * dlon * (np.sin(lat_bounds[0]) - np.sin(lat_bounds[1]))
            # Broadcast to 2D grid using a reference variable's first time slice
            ref_var_name = list(_ds_tmpl.data_vars)[0]
            CELL_AREA = cell_area_1d.broadcast_like(_ds_tmpl[ref_var_name].isel(time=0)).load()

            print(f"[INFO] Built CELL_AREA from {tmpl_nc}")
        else:
            print(f"[WARN] Template dataset {tmpl_nc} has no lat/lon dims.")
        _ds_tmpl.close()
    else:
        print("[WARN] No template nbp file found to build CELL_AREA.")
except Exception as e:
    CELL_AREA = None
    print(f"[WARN] Failed to build CELL_AREA: {e}")


def load_annual_totals(
    nc_path: Path,
    var: str,
) -> Optional[pd.Series]:
    """
    Load time+space integrated annual totals for one variable from one file.

    Assumes:
      - time axis is monthly (TRENDY-style 365-day calendar)
      - data are on the same 0.5° grid as ENSMEAN_S0_nbp (for CELL_AREA)
      - var has units like kg m-2 s-1 (flux) or similar

    Steps:
      - open with decode_times=True and use_cftime
      - apply tvt_mask land mask (values in {0,1,2})
      - multiply by cell area → kg s-1 per cell
      - sum over lat, lon → kg s-1 (global/land total)
      - multiply by fixed seconds-per-month → kg per month
      - group by year, sum over months → kg per year
      - convert to Gt per year (divide by 1e12)
      - return a pandas Series indexed by year
    """
    if CELL_AREA is None:
        print(f"[ERROR] CELL_AREA not available; cannot integrate {nc_path}")
        return None

    try:
        ds = xr.open_dataset(nc_path, decode_times=True, use_cftime=True)
    except Exception as e:
        print(f"[ERROR] Failed to open {nc_path}: {e}")
        return None

    if var not in ds:
        print(f"[WARN] Variable '{var}' not found in {nc_path.name}, skipping")
        ds.close()
        return None

    da = ds[var]

    # Spatial dims check
    dims = set(da.dims)
    lat_dim = "lat" if "lat" in dims else None
    lon_dim = "lon" if "lon" in dims else None

    if not lat_dim or not lon_dim:
        print(
            f"[WARN] Var '{var}' in {nc_path.name} does not have lat/lon dims, dims={da.dims}"
        )
        ds.close()
        return None

    # Apply tvt_mask where ∈ {0,1,2}, if available
    if LAND_MASK is not None:
        try:
            mask = LAND_MASK
            if "lat" in mask.dims and "lon" in mask.dims:
                mask = mask.sel(lat=da[lat_dim], lon=da[lon_dim])
            da = da.where(mask)
        except Exception as e:
            print(f"[WARN] Failed to apply tvt_mask for {nc_path.name}: {e}")

    # Align cell area to this dataset's grid
    try:
        ca = CELL_AREA
        if "lat" in ca.dims and "lon" in ca.dims:
            ca = ca.sel(lat=da[lat_dim], lon=da[lon_dim])
    except Exception as e:
        print(f"[WARN] Failed to align CELL_AREA for {nc_path.name}: {e}")
        ds.close()
        return None

    if "time" not in da.dims:
        print(
            f"[WARN] Var '{var}' in {nc_path.name} has no time dimension; cannot integrate."
        )
        ds.close()
        return None

    # Multiply by area → kg s-1 per cell
    try:
        area_flux = da * ca  # kg s-1
    except Exception as e:
        print(f"[ERROR] Failed to multiply by CELL_AREA for {nc_path.name}: {e}")
        ds.close()
        return None

    # Monthly totals: sum over lat/lon, multiply by seconds per "month"
    try:
        monthly_kg = area_flux.sum(dim=[lat_dim, lon_dim], skipna=True) * DT_MONTH_SECONDS
    except Exception as e:
        print(f"[ERROR] Failed monthly integration for {nc_path.name}: {e}")
        ds.close()
        return None

    # Annual totals (kg/yr), then convert to Gt/yr
    try:
        annual_kg = monthly_kg.groupby("time.year").sum("time", skipna=True)
    except Exception as e:
        print(f"[ERROR] Failed year grouping for {nc_path.name}: {e}")
        ds.close()
        return None

    years = annual_kg["year"].values
    values_gt = annual_kg.values / 1e12  # Gt per year

    ds.close()
    return pd.Series(values_gt, index=pd.Index(years, name="year"))


def process_scenario_var(scenario: str, var: str) -> None:
    """
    For one (scenario, var), compute a CSV with:

      rows   = years (union over all models)
      cols   = models
      values = annual totals (Gt per year) for that var / model / year
    """
    print(f"[INFO] Processing scenario={scenario}, var={var}")

    out_dir = OUTPUT_ROOT / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    series_by_model: Dict[str, pd.Series] = {}

    for model_name, base_dir in MODEL_PATHS.items():
        scenario_dir = base_dir / scenario

        if not scenario_dir.is_dir():
            # Some models may not have all scenarios
            print(f"[INFO]    Model {model_name}: no dir {scenario_dir}, skipping")
            continue

        nc_path = find_nc_file_for_var(scenario_dir, var)
        if nc_path is None:
            print(
                f"[INFO]    Model {model_name}: no usable file for var={var}, skipping"
            )
            continue

        print(f"[INFO]    Model {model_name}: using {nc_path.name}")
        s = load_annual_totals(nc_path, var)
        if s is None:
            print(
                f"[WARN]    Model {model_name}: failed to compute annual totals for {var}"
            )
            continue

        series_by_model[model_name] = s

    if not series_by_model:
        print(f"[WARN]  No data found for var={var} in scenario={scenario}")
        return

    # Build union of all years across models
    all_years = sorted(
        {int(y) for s in series_by_model.values() for y in s.index.values}
    )
    index = pd.Index(all_years, name="year")

    # Construct DataFrame with NaN where missing
    df = pd.DataFrame(index=index)
    for model_name, s in series_by_model.items():
        df[model_name] = s.reindex(index)

    out_path = out_dir / f"{var}_annual_means.csv"
    df.to_csv(out_path)
    print(f"[INFO]  Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compute time+space integrated annual totals per (scenario, variable, model), "
            "sharded over tasks via slurm_shard."
        )
    )
    ap.add_argument(
        "--scenario",
        choices=SCENARIOS,
        nargs="*",
        help="Optional subset of scenarios to include (default: all).",
    )
    ap.add_argument(
        "--var",
        choices=VARS,
        nargs="*",
        help="Optional subset of variables to include (default: all).",
    )
    args = ap.parse_args()

    scenarios_to_use = args.scenario if args.scenario else SCENARIOS
    vars_to_use = args.var if args.var else VARS

    # Build all (scenario, var) tasks
    all_tasks: List[Tuple[str, str]] = [
        (scen, var) for scen in scenarios_to_use for var in vars_to_use
    ]

    print(f"[INFO] Total (scenario, var) tasks: {len(all_tasks)}")

    # Shard across SLURM array (or return all if not under SLURM)
    tasks_to_process = slurm_shard(all_tasks)
    print(f"[INFO] This rank will process {len(tasks_to_process)} tasks.")

    for scenario, var in tasks_to_process:
        process_scenario_var(scenario, var)


if __name__ == "__main__":
    main()