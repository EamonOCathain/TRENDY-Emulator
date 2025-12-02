#!/usr/bin/env python3
"""
make_lai_S3_iav_and_seasonality_panel.py
----------------------------------------

Create a 2-panel figure for scenario S3:

  Left panel : LAI annual means vs year (1982–2018 only)
  Right panel: LAI monthly climatology vs month

In both panels:
  - DGVM spread is shown as shaded bands (min / 75% / 50%)
  - TRENDY ensemble mean is a purple line
  - Stable-Emulator, TL-Emulator, and AVH15C1 are drawn as individual lines
    (if those columns exist in the CSVs, or are derived from NetCDF files).

Fallback:
  If TL-Emulator or AVH15C1 are missing from the CSVs, they are derived from:

    TL-Emulator:
      PROJECT_ROOT/data/preds_for_analysis/TL/TL_1982_2018/lai.nc

    AVH15C1:
      PROJECT_ROOT/data/preds_for_analysis/AVH15C1/S3/lai_avh15c1_filled.nc

  Both are masked with tvt_mask.nc where tvt_mask ∈ {0,1,2} and then
  converted to spatially averaged annual means (1982–2018) and
  monthly climatology (computed over 1982–2018).

Inputs
------

Annual means:
  PROJECT_ROOT/data/analysis/CSVs/annual_means/S3/lai_annual_means.csv

Monthly climatology (seasonality):
  PROJECT_ROOT/data/analysis/CSVs/monthly_seasonality/S3/lai_monthly_seasonality.csv

Each CSV:
  - rows: time index (years for annual, months for climatology)
  - columns: models (DGVMs, TRENDY-Ensemble-Mean, Stable-Emulator, TL-Emulator,
    AVH15C1, etc.)

Output
------

  PROJECT_ROOT/data/analysis/CSVs/plots/lai_S3/
      lai_S3_iav_and_seasonality.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Tuple

import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import xarray as xr

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")

CSV_ANNUAL_ROOT = PROJECT_ROOT / "data/analysis/CSVs/annual_means"
CSV_SEASONAL_ROOT = PROJECT_ROOT / "data/analysis/CSVs/monthly_seasonality"

SCENARIO = "S3"
VAR = "lai"

PLOTS_ROOT = PROJECT_ROOT / "data/analysis/CSVs/plots/lai_S3"
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

OUT_FIG = PLOTS_ROOT / "lai_S3_iav_and_seasonality.png"

# DGVMs
DGVM_MODELS: List[str] = [
    "CLASSIC",
    "CLM",
    "ELM",
    "JSBACH",
    "ORCHIDEE",
    "SDGVM",
    "VISIT",
    "VISIT-UT",
]

# Key columns
ENS_COLUMN = "TRENDY-Ensemble-Mean"
STABLE_CANDIDATES = ["Stable-Emulator_With_Carry", "Stable-Emulator"]
TL_COLUMN = "TL-Emulator"
OBS_COLUMN = "AVH15C1"  # observational LAI reference

# External NetCDF sources (fallback)
TL_NC_PATH = PROJECT_ROOT / "data/preds_for_analysis/TL/TL_1982_2018/lai.nc"
OBS_NC_PATH = PROJECT_ROOT / "data/preds_for_analysis/obs/lai/AVH15C1/lai_avh15c1_filled.nc"
TVT_MASK_PATH = PROJECT_ROOT / "data/masks/tvt_mask.nc"

# Time window for plotting / climatology
START_YEAR = 1982
END_YEAR = 2018
START_DATE = f"{START_YEAR}-01-01"
END_DATE = f"{END_YEAR}-12-31"

# Colours
COLOR_ENS = "#410253"   # TRENDY ensemble (purple)
COLOR_STABLE = "#EF8F40"  # Stable-Emulator (orange)
COLOR_TL = "#D81B60"      # TL-Emulator (bright red)
COLOR_OBS = "#000000"     # AVH15C1 (black; stands out, not green)

COLOR_50 = "#3A528B"
COLOR_75 = "#218F8C"
COLOR_100 = "#5DC762"

# -----------------------------------------------------------------------------
# Mask loading
# -----------------------------------------------------------------------------

LAND_MASK: Optional[xr.DataArray] = None

def load_land_mask() -> Optional[xr.DataArray]:
    """Load tvt_mask and create a land mask where tvt_mask ∈ {0,1,2}."""
    global LAND_MASK
    if LAND_MASK is not None:
        return LAND_MASK
    try:
        tvt = xr.open_dataset(TVT_MASK_PATH)["tvt_mask"]
        if "time" in tvt.dims:
            tvt = tvt.isel(time=0)
        LAND_MASK = tvt.isin([0, 1, 2])
        LAND_MASK.load()
        print(f"[INFO] Loaded land mask from {TVT_MASK_PATH}")
    except Exception as e:
        print(f"[WARN] Could not load land mask from {TVT_MASK_PATH}: {e}")
        LAND_MASK = None
    return LAND_MASK

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_csv(path: Path, index_name: str) -> Optional[pd.DataFrame]:
    """Load a CSV with given index name, stripping column names."""
    if not path.is_file():
        print(f"[WARN] Missing CSV {path}")
        return None
    df = pd.read_csv(path, index_col=0)
    df.index.name = index_name
    df.columns = df.columns.astype(str).str.strip()
    return df


def collapse_to_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a model-by-time table (index: time; columns: models),
    compute DGVM bands + keep all original columns.

    Returns a DataFrame with original columns +:
      DGVM_min, DGVM_max, DGVM_p50_low/high, DGVM_p75_low/high
    """
    dgvm_cols = [c for c in DGVM_MODELS if c in df.columns]
    if not dgvm_cols:
        print("[WARN] No DGVM columns found when collapsing to bands.")
        out = df.copy()
        return out

    d_dgvm = df[dgvm_cols]
    out = df.copy()

    out["DGVM_min"] = d_dgvm.min(axis=1)
    out["DGVM_max"] = d_dgvm.max(axis=1)
    out["DGVM_p50_low"] = d_dgvm.quantile(0.25, axis=1)
    out["DGVM_p50_high"] = d_dgvm.quantile(0.75, axis=1)
    out["DGVM_p75_low"] = d_dgvm.quantile(0.125, axis=1)
    out["DGVM_p75_high"] = d_dgvm.quantile(0.875, axis=1)

    return out


def pick_stable_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first stable-emulator column that exists in df.columns."""
    for cand in STABLE_CANDIDATES:
        if cand in df.columns:
            return cand
    return None


def compute_spatial_masked_timeseries(
    nc_path: Path,
    var: str,
) -> Optional[xr.DataArray]:
    """
    Open a NetCDF file, apply land mask (tvt ∈ {0,1,2}), and compute
    spatial mean over lat/lon, returning a 1D DataArray over time.
    Allows AVH15C1 variable name 'lai_avh15c1' when 'lai' is missing.
    """
    if not nc_path.is_file():
        print(f"[WARN] NetCDF file not found: {nc_path}")
        return None

    try:
        ds = xr.open_dataset(nc_path, decode_times=True, use_cftime=True)
    except Exception as e:
        print(f"[ERROR] Failed to open {nc_path}: {e}")
        return None

    var_name = var
    if var_name not in ds.variables:
        # Special case: AVH15C1 file with 'lai_avh15c1'
        if "lai_avh15c1" in ds.variables:
            var_name = "lai_avh15c1"
        else:
            print(f"[WARN] Variable '{var}' not found in {nc_path.name}")
            ds.close()
            return None

    da = ds[var_name]

    # Check for lat/lon
    dims = set(da.dims)
    lat_dim = "lat" if "lat" in dims else None
    lon_dim = "lon" if "lon" in dims else None

    if not lat_dim or not lon_dim:
        print(f"[WARN] {nc_path.name}: var '{var_name}' lacks lat/lon dims; dims={da.dims}")
        ds.close()
        return None

    # Apply land mask if available
    mask = load_land_mask()
    if mask is not None:
        try:
            m = mask
            if "lat" in m.dims and "lon" in m.dims:
                m = m.sel(lat=da[lat_dim], lon=da[lon_dim])
            da = da.where(m)
        except Exception as e:
            print(f"[WARN] Failed to apply land mask to {nc_path.name}: {e}")

    # Spatial mean
    da_space = da.mean(dim=[lat_dim, lon_dim], skipna=True)

    if "time" not in da_space.dims:
        print(f"[WARN] {nc_path.name}: no time dimension after spatial mean.")
        ds.close()
        return None

    # Restrict to 1982–2018 for any subsequent aggregation
    try:
        da_space = da_space.sel(time=slice(START_DATE, END_DATE))
    except Exception as e:
        print(f"[WARN] Failed to time-slice {nc_path.name} to {START_DATE}–{END_DATE}: {e}")

    ds.close()
    return da_space


def compute_annual_means_from_nc(nc_path: Path, var: str) -> Optional[pd.Series]:
    ts = compute_spatial_masked_timeseries(nc_path, var)
    if ts is None or ts.size == 0:
        return None
    try:
        da_ann = ts.groupby("time.year").mean("time", skipna=True)
    except Exception as e:
        print(f"[ERROR] Failed annual grouping for {nc_path}: {e}")
        return None

    years = da_ann["year"].values
    vals = da_ann.values
    s = pd.Series(vals, index=pd.Index(years, name="year"))

    # Restrict to 1982–2018
    s = s[(s.index >= START_YEAR) & (s.index <= END_YEAR)]
    return s


def compute_monthly_climatology_from_nc(nc_path: Path, var: str) -> Optional[pd.Series]:
    """
    Compute monthly climatology over 1982–2018 from a spatially averaged
    timeseries.
    """
    ts = compute_spatial_masked_timeseries(nc_path, var)
    if ts is None or ts.size == 0:
        return None

    # ts already sliced to 1982–2018 in compute_spatial_masked_timeseries
    try:
        da_mon = ts.groupby("time.month").mean("time", skipna=True)
    except Exception as e:
        print(f"[ERROR] Failed monthly climatology for {nc_path}: {e}")
        return None

    months = da_mon["month"].values
    vals = da_mon.values
    return pd.Series(vals, index=pd.Index(months, name="month"))


def maybe_add_external_model(
    df_annual: pd.DataFrame,
    df_seasonal: pd.DataFrame,
    col_name: str,
    nc_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    If col_name is missing from annual/seasonal DataFrames, compute
    it from the given NetCDF (annual means + monthly climatology)
    and insert as new columns (aligned by index intersection).
    """
    # Annual
    if col_name not in df_annual.columns:
        s_ann = compute_annual_means_from_nc(nc_path, VAR)
        if s_ann is not None and not s_ann.empty:
            df_annual.index = df_annual.index.astype(int)
            s_ann.index = s_ann.index.astype(int)

            common_years = df_annual.index.intersection(s_ann.index)
            if len(common_years) == 0:
                print(f"[WARN] No overlapping years for {col_name} from {nc_path}")
            else:
                df_annual[col_name] = np.nan
                df_annual.loc[common_years, col_name] = s_ann.loc[common_years].values
                print(f"[INFO] Added annual series for {col_name} from {nc_path}")
        else:
            print(f"[WARN] Could not derive annual series for {col_name} from {nc_path}")

    # Monthly climatology
    if col_name not in df_seasonal.columns:
        s_mon = compute_monthly_climatology_from_nc(nc_path, VAR)
        if s_mon is not None and not s_mon.empty:
            df_seasonal.index = df_seasonal.index.astype(int)
            s_mon.index = s_mon.index.astype(int)

            common_months = df_seasonal.index.intersection(s_mon.index)
            if len(common_months) == 0:
                print(f"[WARN] No overlapping months for {col_name} from {nc_path}")
            else:
                df_seasonal[col_name] = np.nan
                df_seasonal.loc[common_months, col_name] = s_mon.loc[common_months].values
                print(f"[INFO] Added monthly climatology for {col_name} from {nc_path}")
        else:
            print(f"[WARN] Could not derive monthly climatology for {col_name} from {nc_path}")

    return df_annual, df_seasonal


def plot_time_series_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_label: str,
    title: str,
    is_monthly: bool = False,
) -> None:
    """
    Plot DGVM bands + TRENDY-Ensemble-Mean + Stable + TL + AVH15C1
    on the given axis.

    df index: time (years or months)
    """
    df_plot = df.sort_index().copy()
    x = df_plot.index.values

    # DGVM bands
    if all(k in df_plot.columns for k in ["DGVM_min", "DGVM_max",
                                          "DGVM_p50_low", "DGVM_p50_high",
                                          "DGVM_p75_low", "DGVM_p75_high"]):
        ax.fill_between(
            x,
            df_plot["DGVM_min"],
            df_plot["DGVM_max"],
            color=COLOR_100,
            alpha=0.25,
            linewidth=0,
        )
        ax.fill_between(
            x,
            df_plot["DGVM_p75_low"],
            df_plot["DGVM_p75_high"],
            color=COLOR_75,
            alpha=0.35,
            linewidth=0,
        )
        ax.fill_between(
            x,
            df_plot["DGVM_p50_low"],
            df_plot["DGVM_p50_high"],
            color=COLOR_50,
            alpha=0.45,
            linewidth=0,
        )

    # TRENDY ensemble
    if ENS_COLUMN in df_plot.columns:
        ax.plot(
            x, df_plot[ENS_COLUMN],
            color=COLOR_ENS,
            linewidth=1.8,
            label="TRENDY Ensemble Mean",
        )

    # Stable-Emulator
    stable_col = pick_stable_column(df_plot)
    if stable_col is not None:
        ax.plot(
            x, df_plot[stable_col],
            color=COLOR_STABLE,
            linewidth=1.5,
            label="Stable-Emulator",
        )
    else:
        print("[WARN] No Stable-Emulator column found for this panel.")

    # TL-Emulator (bright red)
    if TL_COLUMN in df_plot.columns:
        ax.plot(
            x, df_plot[TL_COLUMN],
            linewidth=1.5,
            color=COLOR_TL,
            label="TL-Emulator",
        )
    else:
        print("[WARN] TL-Emulator column not found for this panel.")

    # AVH15C1 (black)
    if OBS_COLUMN in df_plot.columns:
        ax.plot(
            x, df_plot[OBS_COLUMN],
            linewidth=1.5,
            color=COLOR_OBS,
            linestyle="--",
            label="AVH15C1",
        )
    else:
        print("[WARN] AVH15C1 column not found for this panel.")

    # Styling
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if is_monthly:
        ax.set_xlabel("Month")
        try:
            ax.set_xticks(range(1, 13))
        except Exception:
            pass
    else:
        ax.set_xlabel(x_label)

    ax.set_ylabel("LAI (-)")  # dimensionless


def build_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    """Create a shared legend based on line styles from ax."""
    handles: List[Line2D] = []
    labels: List[str] = []

    for h, l in zip(*ax.get_legend_handles_labels()):
        if l and l not in labels:
            handles.append(h)
            labels.append(l)

    # Add DGVM band patches manually
    band_handles = [
        Patch(facecolor=COLOR_50, alpha=0.45, label="DGVM 50th Percentile"),
        Patch(facecolor=COLOR_75, alpha=0.35, label="DGVM 75th Percentile"),
        Patch(facecolor=COLOR_100, alpha=0.25, label="DGVM 100th Percentile"),
    ]

    handles = band_handles + handles
    labels = [h.get_label() for h in band_handles] + labels

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=11,
        columnspacing=1.5,
        handletextpad=0.8,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # --- Load annual means for S3 / LAI ---
    annual_csv = CSV_ANNUAL_ROOT / SCENARIO / f"{VAR}_annual_means.csv"
    df_annual_raw = load_csv(annual_csv, index_name="year")
    if df_annual_raw is None or df_annual_raw.empty:
        print("[ERROR] No annual means data; aborting.")
        return

    # Restrict CSV annual data to 1982–2018
    df_annual_raw.index = df_annual_raw.index.astype(int)
    df_annual_raw = df_annual_raw.loc[
        (df_annual_raw.index >= START_YEAR) & (df_annual_raw.index <= END_YEAR)
    ]

    # --- Load monthly climatology for S3 / LAI ---
    seasonal_csv = CSV_SEASONAL_ROOT / SCENARIO / f"{VAR}_monthly_seasonality.csv"
    df_seasonal_raw = load_csv(seasonal_csv, index_name="month")
    if df_seasonal_raw is None or df_seasonal_raw.empty:
        print("[ERROR] No monthly seasonality data; aborting.")
        return

    # --- Possibly inject TL-Emulator and AVH15C1 from NetCDF if missing ---
    df_annual_raw, df_seasonal_raw = maybe_add_external_model(
        df_annual_raw, df_seasonal_raw, TL_COLUMN, TL_NC_PATH
    )
    df_annual_raw, df_seasonal_raw = maybe_add_external_model(
        df_annual_raw, df_seasonal_raw, OBS_COLUMN, OBS_NC_PATH
    )

    # --- Collapse to DGVM bands etc. ---
    df_annual = collapse_to_bands(df_annual_raw)
    df_seasonal = collapse_to_bands(df_seasonal_raw)

    # --- Create figure: 1 row, 2 columns ---
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5),
        sharey=True,
    )

    ax_left, ax_right = axes
    
    # --- Add subplot letters ---
    ax_left.text(
        0.02, 0.95, "A", transform=ax_left.transAxes,
        fontsize=14, fontweight="bold", va="top"
    )
    ax_right.text(
        0.02, 0.95, "B", transform=ax_right.transAxes,
        fontsize=14, fontweight="bold", va="top"
    )

    # Left panel: annual means (1982–2018)
    plot_time_series_panel(
        ax=ax_left,
        df=df_annual,
        x_label="Year",
        title=f"LAI Annual Means",
        is_monthly=False,
    )

    # Right panel: monthly climatology (computed over 1982–2018 for TL/AVH)
    plot_time_series_panel(
        ax=ax_right,
        df=df_seasonal,
        x_label="Month",
        title="LAI Monthly Climatology",
        is_monthly=True,
    )

    # Shared legend
    build_legend(fig, ax_left)

    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        bottom=0.25,
        top=0.92,
        wspace=0.25,
    )

    fig.savefig(OUT_FIG, dpi=400)
    plt.close(fig)
    print(f"[INFO] Saved figure to {OUT_FIG}")


if __name__ == "__main__":
    sys.path.append(str(PROJECT_ROOT))
    main()