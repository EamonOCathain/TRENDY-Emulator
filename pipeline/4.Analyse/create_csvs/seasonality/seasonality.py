#!/usr/bin/env python3
"""
compute_monthly_seasonality.py
------------------------------

For each (scenario, var, model), compute spatially averaged mean seasonal
cycles (monthly climatology) and write a CSV per (scenario, var):

  /.../monthly_seasonality/<SCENARIO>/<VAR>_monthly_seasonality.csv

Each CSV:
  - rows: months 1..12
  - columns: models (keys of the 'MODEL_PATHS' dict below)
  - values: spatial mean of the variable, averaged over all years
           for each calendar month

Parallelisation
---------------

We build a list of (scenario, var) tasks and shard them across a SLURM array
using src.utils.tools.slurm_shard.

Example SLURM job:

  #SBATCH --array=0-15
  python compute_monthly_seasonality.py

You can optionally restrict scenarios / vars:

  python compute_monthly_seasonality.py --scenario S0 S1 --var gpp npp
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import xarray as xr
import pandas as pd
args = None

# --- Project paths & imports -------------------------------------------------
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.utils.tools import slurm_shard

# -----------------------------------------------------------------------------


ROOT = Path(
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preds_for_analysis"
)

OUTPUT_ROOT = Path(
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/analysis/CSVs/monthly_seasonality"
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
    "JSBACH": ROOT / "DGVMs/JSBACH",
    "CLASSIC": ROOT / "DGVMs/CLASSIC",
    "ELM": ROOT / "DGVMs/ELM",
    "SDGVM": ROOT / "DGVMs/SDGVM",
    "VISIT": ROOT / "DGVMs/VISIT",
    "VISIT-UT": ROOT / "DGVMs/VISIT-UT",
    "TL-Emulator": ROOT / "TL/TL_1982_2018",
    "AVH15C1": ROOT / "AVH15C1"
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
# Load tvt mask
# -----------------------------------------------------------------------------
TVT_MASK_PATH = PROJECT_ROOT / "data/masks/tvt_mask.nc"

LAND_MASK = None
TEST_MASK = None

try:
    _tvt = xr.open_dataset(TVT_MASK_PATH)["tvt_mask"]

    # Drop time dimension if present
    if "time" in _tvt.dims:
        _tvt = _tvt.isel(time=0)

    # full land mask (0,1,2)
    LAND_MASK = _tvt.isin([0, 1, 2]).load()

    # test-only mask (2)
    TEST_MASK = (_tvt == 2).load()

    del _tvt
    print(f"[INFO] Loaded tvt_mask from {TVT_MASK_PATH}")

except Exception as e:
    LAND_MASK = None
    TEST_MASK = None
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
    
    # Special case for cTotal → accept two naming conventions
    if var == "lai":
        candidates = (
            list(scenario_dir.glob("*lai_avh15c1.nc")) +
            list(scenario_dir.glob("*lai.nc"))
        )
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) == 0:
            print(f"[WARN] No file for var='lai' in {scenario_dir}")
            return None
        else:
            names = ", ".join(p.name for p in candidates)
            print(f"[WARN] Multiple lai files in {scenario_dir}: {names}")
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


def load_monthly_seasonality(
    nc_path: Path,
    var: str,
) -> Optional[pd.Series]:
    """
    Load spatially averaged mean seasonal cycle (monthly climatology)
    for one variable from one file.

    Steps:
      - open with decode_times=True and use_cftime
      - mask to tvt_mask ∈ {0,1,2} (if mask loaded)
      - take mean over lat, lon
      - group by calendar month and average over time dimension
      - return a pandas Series indexed by month (1..12)
    """
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
            mask = TEST_MASK if args.mask_test_only else LAND_MASK

            if "lat" in mask.dims and "lon" in mask.dims:
                mask = mask.sel(lat=da[lat_dim], lon=da[lon_dim])

            da = da.where(mask)

        except Exception as e:
            print(f"[WARN] Failed to apply tvt_mask for {nc_path.name}: {e}")

    # Spatial mean over masked data
    da_space = da.mean(dim=[lat_dim, lon_dim], skipna=True)

    if "time" not in da_space.dims:
        print(
            f"[WARN] Var '{var}' in {nc_path.name} has no time dimension after spatial mean"
        )
        ds.close()
        return None

    # Monthly climatology: group by calendar month and average over all years
    try:
        da_mon = da_space.groupby("time.month").mean("time", skipna=True)
    except Exception as e:
        print(f"[ERROR] Failed month grouping for {nc_path.name}: {e}")
        ds.close()
        return None

    months = da_mon["month"].values
    values = da_mon.values

    ds.close()
    return pd.Series(values, index=pd.Index(months, name="month"))


def process_scenario_var(scenario: str, var: str) -> None:
    """
    For one (scenario, var), compute a CSV with:

      rows   = months 1..12 (union over all models, though typically all 1..12)
      cols   = models
      values = mean seasonal cycle (monthly climatology) for that var / model
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
        s = load_monthly_seasonality(nc_path, var)
        if s is None:
            print(
                f"[WARN]    Model {model_name}: failed to compute monthly seasonality for {var}"
            )
            continue

        series_by_model[model_name] = s

    if not series_by_model:
        print(f"[WARN]  No data found for var={var} in scenario={scenario}")
        return

    # Build union of all months across models (should be 1..12, but be robust)
    all_months = sorted(
        {int(m) for s in series_by_model.values() for m in s.index.values}
    )
    index = pd.Index(all_months, name="month")

    # Construct DataFrame with NaN where missing
    df = pd.DataFrame(index=index)
    for model_name, s in series_by_model.items():
        df[model_name] = s.reindex(index)

    out_path = out_dir / f"{var}_monthly_seasonality.csv"
    df.to_csv(out_path)
    print(f"[INFO]  Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compute spatially averaged monthly climatology (seasonality) per "
            "(scenario, variable, model), sharded over tasks via slurm_shard."
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
    ap.add_argument(
        "--mask_test_only",
        action="store_true",
        help="If set, mask to tvt_mask == 2 only (test locations).",
    )

    global args
    args = ap.parse_args()

    scenarios_to_use = args.scenario if args.scenario else SCENARIOS
    vars_to_use = args.var if args.var else VARS

    all_tasks = [(scen, var) for scen in scenarios_to_use for var in vars_to_use]

    print(f"[INFO] Total (scenario, var) tasks: {len(all_tasks)}")

    tasks_to_process = slurm_shard(all_tasks)
    print(f"[INFO] This rank will process {len(tasks_to_process)} tasks.")

    for scenario, var in tasks_to_process:
        process_scenario_var(scenario, var)


if __name__ == "__main__":
    main()