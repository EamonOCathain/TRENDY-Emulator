#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import os

# ----------------------------
# Basic paths & imports
# ----------------------------
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))
from src.utils.tools import slurm_shard

# Mask
tvt_mask = xr.open_dataarray(
    project_root / "data/masks/tvt_mask.nc"
)
if "time" in tvt_mask.dims:
    tvt_mask_2d = tvt_mask.isel(time=0)
else:
    tvt_mask_2d = tvt_mask

mask = tvt_mask_2d.isin([0, 1, 2])

# ----------------------------
# Models & scenarios
# ----------------------------
models_all = [
    "ENSMEAN",
    "Stable-Emulator",
    "CLASSIC",
    "CLM",
    "ELM",
    "JSBACH",
    "ORCHIDEE",
    "SDGVM",
    "VISIT",
    "VISIT-UT",
    "Stable-Emulator-No-Carry",
]

# Remove any accidental duplicates but keep order
models_all = list(dict.fromkeys(models_all))

scenarios = ["S0", "S1", "S2", "S3"]

# Split models across SLURM array
models_chunk = slurm_shard(models_all)
print(f"[INFO] Models handled by this task: {models_chunk}")

# ----------------------------
# Directories
# ----------------------------
ensmean_root   = project_root / "pipeline/3.benchmark/ilamb/ensmean_files"
stable_root    = project_root / "pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios"
dgvms_root     = project_root / "scripts/preprocessing/model_outputs/data/4.misc"
no_carry_root  = project_root / "pipeline/3.benchmark/ilamb/base_no_carry_files"

# ----------------------------
# Helper: find a single file by substrings
# ----------------------------
def find_file(base_dir: Path, substrings):
    """
    Search recursively under base_dir for a single .nc file whose *name*
    contains all substrings in the list `substrings`.
    Returns the first sorted match.
    """
    base_dir = Path(base_dir)
    matches = []
    for p in base_dir.rglob("*.nc"):
        name = p.name
        if all(sub in name for sub in substrings):
            matches.append(p)
    if not matches:
        raise FileNotFoundError(
            f"No .nc file in {base_dir} with substrings {substrings}"
        )
    matches = sorted(matches)
    return matches[0]

# ----------------------------
# Loaders for each model type
# ----------------------------

def load_ensmean_fields(scenario: str):
    """Load ENSMEAN NBP, GPP, NPP and compute NEE = GPP - (RA + RH)."""
    scen_dir = ensmean_root / scenario

    nbp_da = xr.open_dataarray(find_file(scen_dir, ["nbp"])).where(mask)
    gpp_da = xr.open_dataarray(find_file(scen_dir, ["gpp"])).where(mask)
    npp_da = xr.open_dataarray(find_file(scen_dir, ["npp"])).where(mask)
    ra_da  = xr.open_dataarray(find_file(scen_dir, ["ra"])).where(mask)
    rh_da  = xr.open_dataarray(find_file(scen_dir, ["rh"])).where(mask)

    nee_da = gpp_da - (ra_da + rh_da)
    return {
        "nbp": nbp_da,
        "nee": nee_da,
        "gpp": gpp_da,
        "npp": npp_da,
    }


def load_stable_fields(scenario: str):
    """Load Stable-Emulator NBP, GPP, NPP and compute NEE = GPP − (RA + RH)."""
    scen_dir = stable_root / scenario / "MODELS" / "32_year"

    nbp_da = xr.open_dataarray(find_file(scen_dir, ["nbp"])).where(mask)
    gpp_da = xr.open_dataarray(find_file(scen_dir, ["gpp"])).where(mask)
    npp_da = xr.open_dataarray(find_file(scen_dir, ["npp"])).where(mask)
    ra_da  = xr.open_dataarray(find_file(scen_dir, ["ra"])).where(mask)
    rh_da  = xr.open_dataarray(find_file(scen_dir, ["rh"])).where(mask)

    nee_da = gpp_da - (ra_da + rh_da)
    return {
        "nbp": nbp_da,
        "nee": nee_da,
        "gpp": gpp_da,
        "npp": npp_da,
    }


def load_no_carry_fields(scenario: str):
    """
    Load Stable-Emulator-No-Carry NBP, GPP, NPP and compute
    NEE = GPP − (RA + RH) from no_carry_root/{scenario},
    searching by variable name.
    """
    scen_dir = no_carry_root / scenario

    nbp_da = xr.open_dataarray(find_file(scen_dir, ["nbp"])).where(mask)
    gpp_da = xr.open_dataarray(find_file(scen_dir, ["gpp"])).where(mask)
    npp_da = xr.open_dataarray(find_file(scen_dir, ["npp"])).where(mask)
    ra_da  = xr.open_dataarray(find_file(scen_dir, ["ra"])).where(mask)
    rh_da  = xr.open_dataarray(find_file(scen_dir, ["rh"])).where(mask)

    nee_da = gpp_da - (ra_da + rh_da)
    return {
        "nbp": nbp_da,
        "nee": nee_da,
        "gpp": gpp_da,
        "npp": npp_da,
    }


def load_dgvm_fields(model: str, scenario: str):
    """Load DGVM NBP, GPP, NPP, RA, RH and compute NEE = GPP − (RA + RH)."""
    model_dir = dgvms_root / model
    search_base = model_dir if model_dir.is_dir() else dgvms_root

    nbp_da = xr.open_dataarray(
        find_file(search_base, [model, scenario, "nbp"])
    ).where(mask)
    gpp_da = xr.open_dataarray(
        find_file(search_base, [model, scenario, "gpp"])
    ).where(mask)
    npp_da = xr.open_dataarray(
        find_file(search_base, [model, scenario, "npp"])
    ).where(mask)
    ra_da  = xr.open_dataarray(
        find_file(search_base, [model, scenario, "ra"])
    ).where(mask)
    rh_da  = xr.open_dataarray(
        find_file(search_base, [model, scenario, "rh"])
    ).where(mask)

    nee_da = gpp_da - (ra_da + rh_da)
    return {
        "nbp": nbp_da,
        "nee": nee_da,
        "gpp": gpp_da,
        "npp": npp_da,
    }


def load_model_fields(model: str, scenario: str):
    """Dispatch to ENSMEAN / Stable-Emulator / Stable-Emulator-No-Carry / DGVM loaders."""
    if model == "ENSMEAN":
        return load_ensmean_fields(scenario)
    elif model == "Stable-Emulator":
        return load_stable_fields(scenario)
    elif model == "Stable-Emulator-No-Carry":
        return load_no_carry_fields(scenario)
    else:
        return load_dgvm_fields(model, scenario)

# ----------------------------
# Cell-area mask
# ----------------------------
ds_example = xr.open_dataset(
    ensmean_root / "S0" / "ENSMEAN_S0_nbp.nc"
)
nbp_example = ds_example["nbp"]

R = 6_371_000  # Earth radius (m)

lat = ds_example["lat"]
lon = ds_example["lon"]

dlat = np.deg2rad(lat.diff("lat").mean())
dlon = np.deg2rad(lon.diff("lon").mean())

lat_bounds = np.deg2rad(lat + 0.25), np.deg2rad(lat - 0.25)

cell_area = (R ** 2) * dlon * (np.sin(lat_bounds[0]) - np.sin(lat_bounds[1]))
cell_area = cell_area.broadcast_like(nbp_example.isel(time=0))

# ----------------------------
# Time assumptions
# ----------------------------
time_monthly = pd.date_range("1901-01-01", "2023-12-01", freq="MS")
assert len(time_monthly) == 1476

dt_month_seconds = (365.0 / 12.0) * 86400.0  # TRENDY 365-day calendar

# ----------------------------
# Integration helpers
# ----------------------------
def integrate_total(field: xr.DataArray, cell_area: xr.DataArray) -> float:
    """
    Integrate (kg m-2 s-1) field over 1901–2023 monthly data (1476 steps),
    using fixed month length 365/12 days.
    Returns total uptake in GtC.
    """
    area_flux = field * cell_area  # kg s-1
    total_kg = (area_flux.sum(dim=["lat", "lon"]) * dt_month_seconds).sum().item()
    return total_kg / 1e12  # GtC


def integrate_annual(field: xr.DataArray, cell_area: xr.DataArray) -> xr.DataArray:
    """
    Compute annual total for each year (GtC yr-1).
    Assumes monthly data, 1476 steps (1901–2023).
    Returns DataArray(year).
    """
    area_flux = field * cell_area
    monthly_total = area_flux.sum(dim=["lat", "lon"]) * dt_month_seconds  # kg / month

    vals = monthly_total.values.reshape(123, 12)
    annual_gtc = vals.sum(axis=1) / 1e12  # GtC yr-1

    years = np.arange(1901, 2024)
    return xr.DataArray(
        annual_gtc,
        coords={"year": years},
        dims=["year"],
        name="annual_total",
    )

# ----------------------------
# Output directories
# ----------------------------
out_dir = project_root / "data/analysis/S_land"
(out_dir / "nbp_annual_1d").mkdir(parents=True, exist_ok=True)
(out_dir / "nee_annual_1d").mkdir(parents=True, exist_ok=True)
(out_dir / "csv").mkdir(parents=True, exist_ok=True)

# Label for CSVs written by this array chunk
if len(models_chunk) == 1:
    chunk_label = models_chunk[0]
else:
    chunk_label = f"{models_chunk[0]}_to_{models_chunk[-1]}"

# ----------------------------
# Main loop: integrate & save
# ----------------------------
nbp_totals = {}   # {scenario: {model: total_GtC}}
nee_totals = {}   # {scenario: {model: total_GtC}}
gpp_totals = {}   # {scenario: {model: total_GtC}}
npp_totals = {}   # {scenario: {model: total_GtC}}

# annual series containers
nbp_annual_series = {}  # {scenario: {model: DataArray(year)}}
nee_annual_series = {}  # {scenario: {model: DataArray(year)}}

for model in models_chunk:
    for scenario in scenarios:
        try:
            fields = load_model_fields(model, scenario)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping {model} {scenario}: {e}")
            continue

        nbp_da = fields["nbp"]
        nee_da = fields["nee"]
        gpp_da = fields["gpp"]
        npp_da = fields["npp"]

        # --- NBP ---
        nbp_total = integrate_total(nbp_da, cell_area)
        nbp_annual = integrate_annual(nbp_da, cell_area)

        print(f"NBP {model} {scenario}: {nbp_total:.2f} Gt C "
              f"({nbp_total/123.0:.2f} Gt C yr-1)")

        nbp_annual.to_netcdf(
            out_dir / "nbp_annual_1d" / f"NBP_{model}_{scenario}_annual.nc"
        )
        nbp_totals.setdefault(scenario, {})[model] = float(nbp_total)
        nbp_annual_series.setdefault(scenario, {})[model] = nbp_annual

        # --- NEE ---
        nee_total = integrate_total(nee_da, cell_area)
        nee_annual = integrate_annual(nee_da, cell_area)

        print(f"NEE {model} {scenario}: {nee_total:.2f} Gt C "
              f"({nee_total/123.0:.2f} Gt C yr-1)")

        nee_annual.to_netcdf(
            out_dir / "nee_annual_1d" / f"NEE_{model}_{scenario}_annual.nc"
        )
        nee_totals.setdefault(scenario, {})[model] = float(nee_total)
        nee_annual_series.setdefault(scenario, {})[model] = nee_annual

        # --- GPP total (no annual series requested) ---
        gpp_total = integrate_total(gpp_da, cell_area)
        gpp_totals.setdefault(scenario, {})[model] = float(gpp_total)

        # --- NPP total (no annual series requested) ---
        npp_total = integrate_total(npp_da, cell_area)
        npp_totals.setdefault(scenario, {})[model] = float(npp_total)

# ----------------------------
# Write per-chunk CSVs (totals)
# ----------------------------
if nbp_totals:
    df_nbp = pd.DataFrame.from_dict(nbp_totals, orient="index").sort_index()
    df_nbp.index.name = "Scenario"
    df_nbp.to_csv(
        out_dir / "csv" / f"NBP_totals_1901_2023_GtC_{chunk_label}.csv"
    )

if nee_totals:
    df_nee = pd.DataFrame.from_dict(nee_totals, orient="index").sort_index()
    df_nee.index.name = "Scenario"
    df_nee.to_csv(
        out_dir / "csv" / f"NEE_totals_1901_2023_GtC_{chunk_label}.csv"
    )

if gpp_totals:
    df_gpp = pd.DataFrame.from_dict(gpp_totals, orient="index").sort_index()
    df_gpp.index.name = "Scenario"
    df_gpp.to_csv(
        out_dir / "csv" / f"GPP_totals_1901_2023_GtC_{chunk_label}.csv"
    )

if npp_totals:
    df_npp = pd.DataFrame.from_dict(npp_totals, orient="index").sort_index()
    df_npp.index.name = "Scenario"
    df_npp.to_csv(
        out_dir / "csv" / f"NPP_totals_1901_2023_GtC_{chunk_label}.csv"
    )

# ----------------------------
# Per-shard label for filenames
# ----------------------------
shard_number = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

# ----------------------------
# Per-scenario annual CSVs
# ----------------------------
# NBP annual: rows = Year, cols = models (only models in this shard)
for scen in scenarios:
    if scen in nbp_annual_series and nbp_annual_series[scen]:
        scen_dict = nbp_annual_series[scen]
        any_da = next(iter(scen_dict.values()))
        years = any_da["year"].values

        df_nbp_annual = pd.DataFrame(
            {model: da.values for model, da in scen_dict.items()},
            index=years,
        )
        df_nbp_annual.index.name = "Year"

        out_csv = out_dir / "csv" / f"{scen}_NBP_annual_{shard_number}.csv"
        df_nbp_annual.to_csv(out_csv)
        print(f"[INFO] Wrote {out_csv}")

# NEE annual: rows = Year, cols = models (only models in this shard)
for scen in scenarios:
    if scen in nee_annual_series and nee_annual_series[scen]:
        scen_dict = nee_annual_series[scen]
        any_da = next(iter(scen_dict.values()))
        years = any_da["year"].values

        df_nee_annual = pd.DataFrame(
            {model: da.values for model, da in scen_dict.items()},
            index=years,
        )
        df_nee_annual.index.name = "Year"

        out_csv = out_dir / "csv" / f"{scen}_NEE_annual_{shard_number}.csv"
        df_nee_annual.to_csv(out_csv)
        print(f"[INFO] Wrote {out_csv}")