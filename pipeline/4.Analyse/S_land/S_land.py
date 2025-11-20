import xarray as xr
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Import mask
tvt_mask = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/tvt_mask.nc")
# Build 2D boolean mask once
if "time" in tvt_mask.dims:
    tvt_mask_2d = tvt_mask.isel(time=0)
else:
    tvt_mask_2d = tvt_mask

# Set project root (for slurm_shard)
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))
from src.utils.tools import slurm_shard

tasks = ['ens_nbp', 'ens_nee', 'stable_nbp', 'stable_nee']

task = slurm_shard(tasks)  

mask = tvt_mask_2d.isin([0, 1, 2])

nbp = {}
nee = {}
# Ensmean nbp
if 'ens_nbp' in task:
    ens_nbp_S0 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S0/ENSMEAN_S0_nbp.nc").where(mask)
    ens_nbp_s1 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S1/ENSMEAN_S1_nbp.nc").where(mask)
    ens_nbp_s2 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S2/ENSMEAN_S2_nbp.nc").where(mask)
    ens_nbp_s3 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S3/ENSMEAN_S3_nbp.nc").where(mask)
    ens_nbp = {"S0": ens_nbp_S0, "S1": ens_nbp_s1, "S2": ens_nbp_s2, "S3": ens_nbp_s3}
    nbp['ens']=ens_nbp

# Ensmean nbp
if 'ens_nee' in task:
    ens_nee_S0 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S0/ENSMEAN_S0_nee.nc").where(mask)
    ens_nee_s1 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S1/ENSMEAN_S1_nee.nc").where(mask)
    ens_nee_s2 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S2/ENSMEAN_S2_nee.nc").where(mask)
    ens_nee_s3 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S3/ENSMEAN_S3_nee.nc").where(mask)
    ens_nee = {"S0": ens_nee_S0, "S1": ens_nee_s1, "S2": ens_nee_s2, "S3": ens_nee_s3}
    nee['ens']=ens_nee

#stable nbp
if 'stable_nbp' in task:
    stable_nbp_s0 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S0/MODELS/32_year/nbp.nc").where(mask)
    stable_nbp_s1 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S1/MODELS/32_year/nbp.nc").where(mask)
    stable_nbp_s2 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S2/MODELS/32_year/nbp.nc").where(mask)
    stable_nbp_s3 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S3/MODELS/32_year/nbp.nc").where(mask)
    stable_nbp = {"S0": stable_nbp_s0, "S1": stable_nbp_s1, "S2": stable_nbp_s2, "S3": stable_nbp_s3}
    nbp['stable']=stable_nbp

# Bits needed for stable NEE
if 'stable_nee' in task:
    stable_gpp_s0 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S0/MODELS/32_year/gpp.nc").where(mask)
    stable_gpp_s1 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S1/MODELS/32_year/gpp.nc").where(mask)
    stable_gpp_s2 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S2/MODELS/32_year/gpp.nc").where(mask)
    stable_gpp_s3 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S3/MODELS/32_year/gpp.nc").where(mask)

    stable_ra_S0 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S0/MODELS/32_year/ra.nc").where(mask)
    stable_ra_S1 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S1/MODELS/32_year/ra.nc").where(mask)
    stable_ra_S2 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S2/MODELS/32_year/ra.nc").where(mask)
    stable_ra_S3 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S3/MODELS/32_year/ra.nc").where(mask)

    stable_rh_S0 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S0/MODELS/32_year/rh.nc").where(mask)
    stable_rh_S1 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S1/MODELS/32_year/rh.nc").where(mask)
    stable_rh_S2 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S2/MODELS/32_year/rh.nc").where(mask)
    stable_rh_S3 = xr.open_dataarray("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/stabilised_scenarios/S3/MODELS/32_year/rh.nc").where(mask)

    # CAlculate stable NEE
    stable_nee_s0 = stable_gpp_s0 - (stable_ra_S0 + stable_rh_S0)
    stable_nee_s1 = stable_gpp_s1 - (stable_ra_S1 + stable_rh_S1)
    stable_nee_s2 = stable_gpp_s2 - (stable_ra_S2 + stable_rh_S2)
    stable_nee_s3 = stable_gpp_s3 - (stable_ra_S3 + stable_rh_S3)  

    stable_nee = {"S0": stable_nee_s0, "S1": stable_nee_s1, "S2": stable_nee_s2, "S3": stable_nee_s3}
    nee['stable']=stable_nee
    
# ----------------------
# Construct Cell Area Mask
# ----------------------
ds = xr.open_dataset(
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S0/ENSMEAN_S0_nbp.nc"
)
nbp_da = ds["nbp"]   # <-- renamed to avoid clashing with `nbp` dict above

R = 6_371_000  # Earth radius in meters

# lat/lon in degrees
lat = ds.lat
lon = ds.lon

# Convert spacing to radians
dlat = np.deg2rad(lat.diff("lat").mean())
dlon = np.deg2rad(lon.diff("lon").mean())

# Bounds of each latitude cell (±0.25° around cell center)
lat_bounds = np.deg2rad(lat + 0.25), np.deg2rad(lat - 0.25)

# Compute latitude-dependent cell area (m²)
cell_area = (
    (R**2) * dlon *
    (np.sin(lat_bounds[0]) - np.sin(lat_bounds[1]))
)

# Broadcast to lon dimension
cell_area = cell_area.broadcast_like(nbp_da.isel(time=0))

# ----------------------
# Time / dt assumptions
# ----------------------
# 1476 monthly timesteps from Jan 1901 to Dec 2023
time_monthly = pd.date_range("1901-01-01", "2023-12-01", freq="MS")
assert len(time_monthly) == 1476

# TRENDY 365-day calendar: each month is 365/12 days
dt_month_seconds = (365.0 / 12.0) * 86400.0  # ≈ 2,629,440 s per month

# ----------------------
# Integration helpers
# ----------------------
def integrate_total(field, cell_area):
    """
    Integrate (kg m-2 s-1) over 1901–2023 monthly data (1476 steps),
    using a fixed TRENDY 365-day calendar (dt per month = 365/12 days).
    Returns total uptake in GtC.
    """
    area_flux = field * cell_area  # kg s-1

    dt = (365.0 / 12.0) * 86400.0  # seconds per month

    total_kg = (area_flux.sum(dim=["lat", "lon"]) * dt).sum().item()

    return total_kg / 1e12  # GtC


def integrate_annual(field, cell_area):
    """
    Compute annual total for each year (GtC yr-1).
    Assumes monthly data, 1476 steps (1901–2023).
    Returns xarray.DataArray(year).
    """
    area_flux = field * cell_area  # kg s-1

    dt = (365.0 / 12.0) * 86400.0  # seconds per month

    monthly_total = area_flux.sum(dim=["lat", "lon"]) * dt  # kg / month

    # reshape 1476 → (123 years, 12 months)
    monthly_vals = monthly_total.values.reshape(123, 12)

    annual_gtc = monthly_vals.sum(axis=1) / 1e12  # GtC yr-1

    years = np.arange(1901, 2024)
    annual_da = xr.DataArray(
        annual_gtc,
        coords={"year": years},
        dims=["year"],
        name="annual_total"
    )
    return annual_da

# ----------------------
# Output dir
# ----------------------
out_dir = Path(
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/analysis/S_land"
)
(out_dir / "nbp_annual_1d").mkdir(parents=True, exist_ok=True)
(out_dir / "nee_annual_1d").mkdir(parents=True, exist_ok=True)

# ----------------------
# Do the calculations
# ----------------------
nbp_totals = {}  # nested dict: {scenario: {model: total_GtC}}
for model, scen_dict in nbp.items():
    for scenario, da in scen_dict.items():
        total = integrate_total(da, cell_area)
        print(f"NBP {model} {scenario}: {total:.2f} Gt C")
        yearly_avg = total / 123
        print(f"  Yearly: {yearly_avg:.2f} Gt C yr-1")

        annual_1d = integrate_annual(da, cell_area)
        print(annual_1d.values[:10])

        # Save annual 1D series
        out_nc = out_dir / "nbp_annual_1d" / f"NBP_{model}_{scenario}_annual.nc"
        annual_1d.to_netcdf(out_nc)

        # Store total for CSV
        nbp_totals.setdefault(scenario, {})[model] = float(total)

nee_totals = {}  # nested dict: {scenario: {model: total_GtC}}
for model, scen_dict in nee.items():
    for scenario, da in scen_dict.items():
        total = integrate_total(da, cell_area)
        print(f"NEE {model} {scenario}: {total:.2f} Gt C")
        yearly_avg = total / 123
        print(f"  Yearly: {yearly_avg:.2f} Gt C yr-1")

        annual_1d = integrate_annual(da, cell_area)
        print(annual_1d.values[:10])

        out_nc = out_dir / "nee_annual_1d" / f"NEE_{model}_{scenario}_annual.nc"
        annual_1d.to_netcdf(out_nc)

        nee_totals.setdefault(scenario, {})[model] = float(total)

# ----------------------
# Write summary CSVs
# ----------------------
# NBP totals: rows = scenarios, columns = models
df_nbp = pd.DataFrame.from_dict(nbp_totals, orient="index").sort_index()
df_nbp.index.name = "Scenario"
df_nbp.to_csv(out_dir / "NBP_totals_1901_2023_GtC.csv")

# NEE totals: rows = scenarios, columns = models
df_nee = pd.DataFrame.from_dict(nee_totals, orient="index").sort_index()
df_nee.index.name = "Scenario"
df_nee.to_csv(out_dir / "NEE_totals_1901_2023_GtC.csv")