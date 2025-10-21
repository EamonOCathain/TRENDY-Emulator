import xarray as xr
from pathlib import Path
import subprocess
import os
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import sys
import shutil
import cftime
import pandas as pd
import sys
from typing import Union
OVERWRITE = False

"""
Files come mostly with fill value -99999.0 but VISIT has None. It instead has 
"""

# Some Paths
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.paths.paths import (
    scripts_dir, 
    preprocessing_dir,
    raw_data_dir,
    historical_dir,
    raw_outputs_dir
)

# Some Imports
from src.utils.visualisation import (
    finite_mask, 
    first_timestep, 
    plot_mean_seasonal_cycle,
    plot_timeseries, 
    plot_mean_seasonal_cycle_grid)

from src.utils.preprocessing import (
    regrid_file,
    trim_time_xarray,
    yearmean_cdo,
    cdo_add3,
    zero_out_netcdf,
    cdo_subtract,
    standardise_vars,
    make_mask_from_paths_df,
    reconstruct_monthly_cTotal_dec_anchor,
    set_negative_to_zero 
)
from src.utils.tools import slurm_shard
from src.dataset.variables import models as models_pre_slurm
from src.dataset.variables import var_names 

# Lists of vars
fluxes = var_names['fluxes']
states = var_names['states']
carbon_states = ['cVeg', 'cLitter', 'cSoil']
all_vars = fluxes + states
scenarios_pre_slurm = ['S0','S1','S2','S3']

'''
File comes as annual from 1850-2023 (171 timestamps), 720x360 grid with lon(-179.75, 179.75) and lon(-89.75, 89.75). Steps were:
1. Trim - Trim monthly to 1476 and annual to 123.
2. Regrid - All regridded the same.
3. Time axis - Monthly and annual treated differently. 
4. Time avg the monthly states to annual.
5. Fill gaps.
6. cTotal.
4. Ensmean.
5. Chunk.
'''

# more paths
current_dir = preprocessing_dir / "model_outputs"
std_dir = current_dir / "data/1.standard_vars"
trim_dir = current_dir / "data/2.trim"
regrid_dir = current_dir / "data/3.regrid"
misc_dir = current_dir / "data/4.misc"
ensmean_dir = current_dir / "data/5.ensmean"
final_dir = project_root / "data/preprocessed/model_outputs/"

plot_dir = current_dir / "val_plots"

dirs = [std_dir, trim_dir, regrid_dir, misc_dir, ensmean_dir, final_dir, plot_dir]
for d in dirs:
    d.mkdir(exist_ok=True, parents=True)

# =========================== Set up Slurm ===========================
# Build (model, scenario) pairs from your full lists
pairs = [(m, s) for m in models_pre_slurm for s in scenarios_pre_slurm]
pairs_to_process = slurm_shard(pairs)

# Derive filtered lists WITHOUT overwriting your originals
models    = sorted({m for (m, _) in pairs_to_process})
scenarios = sorted({s for (_, s) in pairs_to_process})

print(f"[INFO] Selected models: {models}")
print(f"[INFO] Selected scenarios: {scenarios}")

# Subset models based on time res of carbon states
models_with_monthly_states    = list(set(models) & {"ELM", "VISIT-UT", "VISIT"})
models_without_monthly_states = list(set(models) - set(models_with_monthly_states))

# Open paths df
paths_df = pd.read_csv(current_dir / "paths_dfs/paths_raw_outputs.csv")

# Filter to selected (model, scenario) pairs before any work
paths_df = paths_df[
    paths_df["model"].isin(models) &
    paths_df["scenario"].isin(scenarios)
].copy()

print(f"[INFO] After SLURM filter: {len(paths_df)} rows")

# Add time_res column
if 'time_res' not in paths_df.columns:
    paths_df['time_res'] = pd.NA

# ------------------------------- Functions -------------------------------

def _get_path(df, model, scenario, variable):
    """
    Return the path from df where model, scenario and variable match.
    Raises an error if no match or multiple matches.
    """
    mask = (
        (df["model"] == model) &
        (df["scenario"] == scenario) &
        (df["variable"] == variable)
    )
    matches = df.loc[mask, "path"]

    if matches.empty:
        raise ValueError(f"No match found for {model=}, {scenario=}, {variable=}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches found: {list(matches)}")

    return Path(matches.iloc[0])

def append_rows(df, model, scenario, variable, path, time_res):
    
    row = {"variable": variable, 
           "model": model,
           "scenario": scenario,
           "path": str(path),
           "time_res": time_res}
    
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

def get_row(row: pd.Series):
    """
    Return (variable, model, scenario, path (Path|None), time_res (str|None))
    from a DataFrame row, tolerating missing columns.
    """
    var      = row.get("variable")
    model    = row.get("model")
    scenario = row.get("scenario")
    path_val = row.get("path")
    time_res = row.get("time_res", None)   # safe if column missing
    path     = Path(path_val) if (path_val is not None and pd.notna(path_val)) else None
    return var, model, scenario, path, time_res

def update_row(
    df: pd.DataFrame,
    idx: int,
    model: str | None = None,
    path: Path | None = None,
    time_res: str | None = None,
):
    """
    Update model, path, and/or time_res for row `idx` in df.
    Any argument left as None will be skipped.
    """
    if model is not None:
        df.loc[idx, "model"] = model
    if path is not None:
        df.loc[idx, "path"] = str(path)
    if time_res is not None:
        df.loc[idx, "time_res"] = time_res

def get_fill(path: Union[str, Path], var: str):
    """
    Open a NetCDF file, inspect a variable, and return its _FillValue (if any).
    """
    path = Path(path)
    with xr.open_dataset(path, decode_times=False) as ds:
        if var not in ds.variables:
            raise KeyError(f"Variable '{var}' not found in {path}. Available: {list(ds.variables)}")

        da = ds[var]
        # Prefer backend-provided encoding
        fv = da.encoding.get("_FillValue", None)
        if fv is None:
            # Fall back to attrs if present
            fv = da.attrs.get("_FillValue", da.attrs.get("missing_value", None))

    return fv

def has_hard_nans(path: Union[str, Path], var: str, sample: bool = False) -> bool:
    """
    Check if a NetCDF variable contains *hard stored* IEEE NaNs 
    (ignoring _FillValue / missing_value placeholders).
    """
    path = Path(path)
    with xr.open_dataset(path, decode_times=False, mask_and_scale=False) as ds:
        if var not in ds:
            raise KeyError(f"Variable '{var}' not found in {path}. "
                           f"Available: {list(ds.variables)}")
        da = ds[var]

        if sample and "time" in da.dims:
            arr = da.isel(time=0).values
        else:
            arr = da.values

        return np.isnan(arr).any()

# Detect Nan or Fill Value
'''for idx, row in paths_df.iterrows():
    var, model, scenario, path, time_res = get_row(row)
    
    if path is not None and scenario == 'S3' and var == 'gpp':
        fill = get_fill(path, var)
        print(f"fill for {model} - {scenario} - {var} = {fill}")

        has_nans = has_hard_nans(path, var)
        print(f"Hard nans for {model} - {scenario} - {var} = {has_nans}")'''

# ------------------------------- Standardise Vars, Dims, Fill -------------------------------
to_drop = []
for idx, row in paths_df.iterrows():
    var, model, scenario, path, time_res = get_row(row)
    
    if path == None:
        print(f"removing row as path is NA {var}")
        to_drop.append(idx)
        continue
    elif not path.exists():
        print(f"Skipping file as it doesn't exist - {path}")
        continue
    
    out_path = std_dir / path.name 
        
    if out_path.exists() and not OVERWRITE:
        print(f"File exists, skipping standardising - {out_path}")
    else:
        standardise_vars(path, out_path)
        
    update_row(paths_df, idx, path=out_path)

paths_df = paths_df.drop(to_drop)

# ------------------------------- Trim Time Vars -------------------------------
for idx, row in paths_df.iterrows():
    var, model, scenario, path, time_res = get_row(row)

    # Build a clean output name from columns 
    out_name = f"{model}_{scenario}_{var}.nc"
    out_path = trim_dir / out_name

    if model in models_with_monthly_states or var not in carbon_states:
        time_res = "monthly"
        start_idx = -1476
    else:
        time_res = "annual"
        start_idx = -123
        
    end_idx = -1
    
    if out_path.exists() and not OVERWRITE:
        print(f"[INFO] Skipping standardising vars (exists): {out_path}")
    else:
        # Use the real input path that exists
        trim_time_xarray(path, out_path, start_idx, end_idx, time_res)

    # Update row
    update_row(paths_df, idx, path=out_path, time_res=time_res)

# ------------------------------- Regrid -------------------------------
for idx, row in paths_df.iterrows():
    var, model, scenario, path, time_res = get_row(row)

    # Build a clean output name from columns 
    out_name = f"{model}_{scenario}_{var}.nc"
    out_path = regrid_dir / out_name

    if out_path.exists() and not OVERWRITE:
        print(f"[INFO] Skipping regridding (exists): {out_path}")
    else:
        # Use the real input path that exists
        regrid_file(path, out_path)  

    # Update row
    update_row(paths_df, idx, path=out_path, time_res=time_res)

# ------------------------------- Set Negative to 0 -------------------------------
vars_which_cant_be_neg = ['gpp', 'ra', 'rh', 
                          'evapotrans', 'fFire',
                          'mrro', 'mrso', 
                          'lai', 'cVeg', 'cLitter', 'cSoil']

for idx, row in paths_df.iterrows():
    var, model, scenario, path, time_res = get_row(row)

    # Build a clean output name from columns 
    
    if model in models_with_monthly_states and var in ['cVeg', 'cLitter', 'cSoil']:
        out_name = f"{model}_{scenario}_{var}_monthly.nc"
    else:
        out_name = f"{model}_{scenario}_{var}.nc"
        
    out_path = misc_dir / out_name

    if out_path.exists() and not OVERWRITE:
        print(f"[INFO] Skipping negative clipping (exists): {out_path}")
    else:
        if var in vars_which_cant_be_neg:
            set_negative_to_zero(path, out_path)  
        else:
            shutil.copy2(path, out_path) 

    # Update row
    update_row(paths_df, idx, path=out_path, time_res=time_res)

# ------------------------------- Monthly C States -> cTotal_monthly + Annual vars + cTotal_annual -------------------------------
for model in models_with_monthly_states:
    for scenario in scenarios:
        # 1) Grab monthly carbon states
        cVeg    = _get_path(paths_df, model, scenario, "cVeg")
        cLitter = _get_path(paths_df, model, scenario, "cLitter")
        cSoil   = _get_path(paths_df, model, scenario, "cSoil")

        # 2) Compute monthly cTotal = cVeg + cLitter + cSoil  (unchanged)
        cTotal_monthly_out = misc_dir / f"{model}_{scenario}_cTotal_monthly.nc"
        cdo_add3(str(cVeg), str(cLitter), str(cSoil),
                 str(cTotal_monthly_out), var_name='cTotal_monthly', overwrite=OVERWRITE)

        # Append the new monthly cTotal row (unchanged)
        paths_df = append_rows(paths_df, model, scenario, 'cTotal_monthly', str(cTotal_monthly_out), 'monthly')

        # 3) NEW: cTotal_annual = yearmean(cTotal_monthly) and append as a new row
        cTotal_annual_out = misc_dir / f"{model}_{scenario}_cTotal_annual.nc"
        yearmean_cdo(str(cTotal_monthly_out), str(cTotal_annual_out), overwrite=OVERWRITE)
        paths_df = append_rows(paths_df, model, scenario, 'cTotal_annual', str(cTotal_annual_out), 'annual')

        # 4) Convert cVeg, cSoil, cLitter to annual and overwrite their rows in paths_df (in place)
        for var, in_path in zip(['cVeg', 'cSoil', 'cLitter'], [cVeg, cSoil, cLitter]):
            out_ann = misc_dir / f"{model}_{scenario}_{var}.nc"
            yearmean_cdo(str(in_path), str(out_ann), overwrite=OVERWRITE)

            mask_var = make_mask_from_paths_df(paths_df, model=model, scenario=scenario, variable=var)
            paths_df.loc[mask_var, "time_res"] = "annual"
            paths_df.loc[mask_var, "path"]     = str(out_ann)
        
OVERWRITE=False
# ------------------------------- Annual C States -> Annual cTotal and Monthly cTotal -------------------------------
# Now reconstruct the monthly carbon states from the Annual - NBP
for model in models_without_monthly_states:
    for scenario in scenarios:
        # Get current path
        cVeg = _get_path(paths_df, model, scenario, 'cVeg')
        cLitter = _get_path(paths_df, model, scenario, 'cLitter')
        cSoil = _get_path(paths_df, model, scenario, 'cSoil')
        nbp = _get_path(paths_df, model, scenario, 'nbp')
        
        # Calculate cTotal Annual
        cTotal_annual_out = misc_dir / f"{model}_{scenario}_cTotal_annual.nc"
        cdo_add3(cVeg, cLitter, cSoil, cTotal_annual_out, overwrite=OVERWRITE)
        paths_df = append_rows(paths_df, model, scenario, 'cTotal_annual', cTotal_annual_out, 'annual')
        
        # Calculate cTotal Monthly
        cTotal_monthly_out = misc_dir / f"{model}_{scenario}_cTotal_monthly.nc"
        reconstruct_monthly_cTotal_dec_anchor(cTotal_annual_out, nbp, cTotal_monthly_out, overwrite=OVERWRITE)
        paths_df = append_rows(paths_df, model, scenario, 'cTotal_monthly', cTotal_monthly_out, 'monthly')

OVERWRITE=False

# ------------------------------- fLuc of CLASSIC to 0 -------------------------------
fFire_mask = make_mask_from_paths_df(paths_df, model='CLASSIC', scenario=["S0", "S1", "S2"], variable = 'fFire')
fFire_sub_df = paths_df[fFire_mask]

for _, row in fFire_sub_df.iterrows():
    scenario = row["scenario"]
    fFire_path = Path(row["path"])
    fLuc_out_path = misc_dir / f"CLASSIC_{scenario}_fLuc.nc"
    time_res = row["time_res"]

    zero_out_netcdf(fFire_path, fLuc_out_path, 'fLuc', overwrite=OVERWRITE)
    paths_df = append_rows(paths_df, 'CLASSIC', scenario, 'fLuc', str(fLuc_out_path), time_res)

# ------------------------------- fFire of ELM to 0 -------------------------------
fLuc_mask = make_mask_from_paths_df(paths_df, model='ELM', scenario=["S0", "S1", "S2"], variable = 'fLuc')
fLuc_sub_df = paths_df[fLuc_mask]
new_rows = []

for _, row in fLuc_sub_df.iterrows():
    scenario = row["scenario"]
    fLuc_in = Path(row["path"])
    fFire_out = misc_dir / f"ELM_{scenario}_fFire.nc"
    time_res      = row["time_res"]
    
    zero_out_netcdf(fLuc_in, fFire_out, 'fFire', overwrite=OVERWRITE)
    paths_df = append_rows(paths_df, 'ELM', scenario, 'fFire', str(fFire_out), time_res)
 
# ------------------------------- GPP of VISIT: NPP = GPP - Ra -------------------------------
GPP_mask = make_mask_from_paths_df(paths_df, model = 'VISIT', variable = 'gpp') 
GPP_sub_df = paths_df[GPP_mask]

Ra_mask = make_mask_from_paths_df(paths_df, model = 'VISIT', variable = 'ra') 
Ra_subdf = paths_df[Ra_mask]

for _, row in GPP_sub_df.iterrows():
    scenario = row["scenario"]
    gpp_path = Path(row["path"])
    time_res = row['time_res']
    
    # Get path of Ra file for this scenario 
    ra_mask = make_mask_from_paths_df(paths_df, model = 'VISIT', scenario= scenario, variable = 'ra') 
    ra_matches = paths_df.loc[ra_mask, "path"]
    if ra_matches.empty:
        raise ValueError(f"No Ra file found for VISIT {scenario}")
    elif len(ra_matches) > 1:
        raise ValueError(f"Multiple Ra files found for VISIT {scenario}: {list(ra_matches)}")
    ra_path = ra_matches.iloc[0]
    
    # Store the out path for the npp
    out_path = misc_dir / f"VISIT_{scenario}_npp.nc"

    cdo_subtract(gpp_path, ra_path, out_path, var_name = 'npp', overwrite=OVERWRITE)
    
    paths_df = append_rows(paths_df, 'VISIT', scenario, 'npp', str(out_path), time_res)
    
# Save the paths df
# Sanity check on the DF
counter =0
good =0
for model in models:
    for scenario in scenarios:
        for var in var_names['outputs']:
            sanity_mask = make_mask_from_paths_df(paths_df, model, scenario, var)
            sanity_df = paths_df[sanity_mask]
            n = len(sanity_df)
            if n > 1:
                print(f"Error: There is more than 1 row matching {model}:{scenario}:{var}")
            elif n < 1:
                print(f"Error: There are no rows matching {model}:{scenario}:{var}")
            else:
                good+=1
            counter+=1
if good == counter:
    print("Paths df is looking good. Huzzah.")
    
paths_df.to_csv(current_dir / "paths_dfs/paths_models_mid_processing.csv")

# ------------------------------- Validation Plotting -------------------------------
fresh_paths_df = pd.read_csv(current_dir / "paths_dfs/paths_raw_outputs.csv")

# ---------- RAW (fresh_paths_df) ---------
'''OVERWRITE=False
for _, row in fresh_paths_df.iterrows():
    var, model, scenario, path, time_res = get_row(row)
    if path is None or not path.exists():
        continue
    if scenario != "S3":
        continue

    # Targets
    first_out  = plot_dir / f"raw/first_timestep/{model}/{var}.png"
    finite_out = plot_dir / f"raw/finite_mask/{model}/{var}.png"

    # Ensure directories
    first_out.parent.mkdir(parents=True, exist_ok=True)
    finite_out.parent.mkdir(parents=True, exist_ok=True)

    # first_timestep
    if first_out.exists():
        if OVERWRITE:
            try: first_out.unlink()
            except FileNotFoundError: pass
        else:
            print(f"[SKIP] Exists: {first_out}")
            # fall through to still create finite_mask if needed
    if not first_out.exists():
        first_timestep(path, output_path=first_out, title=path.stem, overwrite=True)

    # finite_mask
    if finite_out.exists():
        if OVERWRITE:
            try: finite_out.unlink()
            except FileNotFoundError: pass
        else:
            print(f"[SKIP] Exists: {finite_out}")
            continue
    finite_mask(path, output_path=finite_out, title=path.stem, overwrite=True)'''

# ---------- Check Each Step for Finite Mask ----------

'''OVERWRITE=True
ctotal_monthly_paths = []

dirs = [std_dir, trim_dir, regrid_dir, misc_dir]

for dir in dirs:
    files = dir.glob("*.nc")
    for path in files:
        path = Path(path)
        # Make plots for gpp, S3
        if "S3" in path.name and ("gpp" in path.name or "fLuc" in path.name):
            finite_out = plot_dir / f"preprocessed_models/{dir.name}/finite_mask/{path.stem}.png"
            finite_out.parent.mkdir(parents=True, exist_ok=True)
            finite_mask(path, output_path=finite_out, title=path.stem, overwrite=OVERWRITE, ntimesteps = 10)'''

# ---------- PREPROCESSED (paths_df) ----------
ctotal_monthly_paths = []

OVERWRITE_PLOTS = False 

for _, row in paths_df.iterrows():
    var, model, scenario, path, time_res = get_row(row)
    if path is None or not path.exists() or scenario != "S3":
        continue

    first_out  = plot_dir / f"preprocessed_models/first_timestep/{model}/{var}.png"
    finite_out = plot_dir / f"preprocessed_models/finite_mask/{model}/{var}.png"
    first_out.parent.mkdir(parents=True, exist_ok=True)
    finite_out.parent.mkdir(parents=True, exist_ok=True)

    if OVERWRITE_PLOTS or not first_out.exists():
        first_timestep(path, output_path=first_out, title=path.stem, overwrite=True)

    if OVERWRITE_PLOTS or not finite_out.exists():
        finite_mask(path, output_path=finite_out, title=path.stem, overwrite=True)

    # Collect for the seasonal-cycle grid
    if var == "cTotal_monthly" and scenario == "S3":
        ctotal_monthly_paths.append(path)
        
OVERWRITE = True
seasonal_avg_out_path=plot_dir / f"preprocessed_models/seasonal_avgs/ctotal_monthly_seasonal_avgs.png"
plot_mean_seasonal_cycle_grid(ctotal_monthly_paths, 
                              output_path=seasonal_avg_out_path, 
                              overwrite=OVERWRITE)
