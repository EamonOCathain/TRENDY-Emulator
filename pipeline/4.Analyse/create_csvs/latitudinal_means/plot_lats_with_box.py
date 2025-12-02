#!/usr/bin/env python3
"""
make_latitudinal_line_box_panel.py
----------------------------------

Create a single multi-panel figure of *time-averaged* latitudinal means,
averaged across scenarios, with a small global-mean boxplot inset per variable.

For each variable:
  - Read lat_means CSVs for scenarios (default: S0,S1,S2,S3)
  - Average model latitudinal means across scenarios (for lines)
  - Compute global-mean (over latitude and scenarios) per model (for box)
  - Collapse DGVMs into distribution (box) + ENSMEAN + Stable + extras

Final figure:
  - 3 columns × 5 rows = 15 subplots
  - Variables (top-left → bottom-right):

      gpp, ra, npp,
      rh, nee, fFire,
      fLuc, nbp, mrro,
      evapotrans, mrso, lai,
      cVeg, cLitter, cSoil

  - cTotal excluded.

Input CSVs (latitudinal means):

  lat_means:
    PROJECT_ROOT/data/analysis/CSVs/lat_means/<SCENARIO>/<var>_lat_means.csv

Each CSV:
  - rows: latitudes (e.g.  -89.5, -88.5, ..., 89.5)
  - columns: models

Output:

  PROJECT_ROOT/data/analysis/CSVs/plots/lines_and_boxes/
      lat_means_with_boxes/lat_means_all_vars_panel.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xarray as xr  

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")

CSV_LAT_MEANS_ROOT = PROJECT_ROOT / "data/analysis/CSVs/lat_means"

PLOTS_ROOT = PROJECT_ROOT / "data/analysis/CSVs/plots/lat_means"
PLOTS_LAT_BOX_DIR = PLOTS_ROOT / "lat_means_avg_scenario/with_boxes/nee_nbp"

PLOTS_LAT_BOX_DIR.mkdir(parents=True, exist_ok=True)

# NEW: tvt_mask path + cache for area weights
TVT_MASK_PATH = PROJECT_ROOT / "data/masks/tvt_mask.nc"
_LAT_BAND_WEIGHTS: Optional[pd.Series] = None

SCENARIOS: List[str] = ["S0", "S1", "S2", "S3"]

# Desired plotting order (no cTotal)
VARS_ORDER: List[str] = [
    "gpp",
    "ra",
    "npp",
    "rh",
    "nee",
    "fFire",
    "fLuc",
    "nbp",
    "mrro",
    "evapotrans",
    "mrso",
    "lai",
    "cVeg",
    "cLitter",
    "cSoil",
]

DGVM_MODELS = [
    "CLASSIC",
    "CLM",
    "ELM",
    "JSBACH",
    "ORCHIDEE",
    "SDGVM",
    "VISIT",
    "VISIT-UT",
]

ENS_COLUMN = "TRENDY-Ensemble-Mean"
STABLE_CANDIDATES = [
    "Stable-Emulator_With_Carry",
]

# Default colours for extra overlay models (order matters)
DEFAULT_EXTRA_COLORS = [
    "#cc4678",  # 1st extra model
    "#f0f922",  # 2nd extra model
    "#3b528b",  # 3rd extra model
]

# Units
VAR_UNITS = {
    "gpp": ("kg m$^{-2}$ s$^{-1}$", "kg m$^{-2}$ s$^{-1}$"),
    "npp": ("kg m$^{-2}$ s$^{-1}$", "kg m$^{-2}$ s$^{-1}$"),
    "ra": ("kg m$^{-2}$ s$^{-1}$", "kg m$^{-2}$ s$^{-1}$"),
    "rh": ("kg m$^{-2}$ s$^{-1}$", "kg m$^{-2}$ s$^{-1}$"),
    "nee": ("kg m$^{-2}$ s$^{-1}$", "kg m$^{-2}$ s$^{-1}$"),
    "nbp": ("kg m$^{-2}$ s$^{-1}$", "kg m$^{-2}$ s$^{-1}$"),
    "fLuc": ("kg m$^{-2}$ s$^{-1}$", "kg m$^{-2}$ s$^{-1}$"),
    "fFire": ("kg m$^{-2}$ s$^{-1}$", "kg m$^{-2}$ s$^{-1}$"),
    "evapotrans": ("kg m$^{-2}$ s$^{-1}$", "kg m$^{-2}$ s$^{-1}$"),
    "cVeg": ("kg m$^{-2}$", "kg m$^{-2}$"),
    "cSoil": ("kg m$^{-2}$", "kg m$^{-2}$"),
    "cLitter": ("kg m$^{-2}$", "kg m$^{-2}$"),
    "mrro": ("kg m$^{-2}$ s$^{-1}$", "kg m$^{-2}$ s$^{-1}$"),
    "mrso": ("kg m$^{-2}$", "kg m$^{-2}$"),
    "lai": ("-", "-"),
}

COLOR_ENS = "#410253"
COLOR_STABLE = "#EF8F40"

COLOR_50 = "#3A528B"
COLOR_75 = "#218F8C"
COLOR_100 = "#5DC762"

# -----------------------------------------------------------------------------
# Helpers to read and summarise CSVs
# -----------------------------------------------------------------------------

def get_lat_band_area_weights() -> pd.Series:
    """
    Compute area weights per latitude row using tvt_mask == 2 on the native grid.

    Steps:
      - open tvt_mask.nc
      - restrict to first time slice if there is a time dim
      - for each latitude row, count number of pixels where tvt_mask == 2
      - compute approximate area per cell on a lon/lat grid (∝ cos(lat))
      - multiply by counts to get total test-area per latitude row

    Returns:
      pd.Series indexed by latitude value (same grid as tvt_mask),
      giving area weight per latitude row.
    """
    global _LAT_BAND_WEIGHTS
    if _LAT_BAND_WEIGHTS is not None:
        return _LAT_BAND_WEIGHTS

    try:
        ds_mask = xr.open_dataset(TVT_MASK_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to open tvt_mask at {TVT_MASK_PATH}: {e}")

    tvt = ds_mask["tvt_mask"]
    if "time" in tvt.dims:
        tvt = tvt.isel(time=0)

    dims = list(tvt.dims)
    lat_dim = "lat" if "lat" in dims else dims[0]
    lon_dim = "lon" if "lon" in dims else dims[1]

    # test-only cells (value == 2)
    mask2 = (tvt == 2)

    # number of test pixels per latitude row
    counts_per_lat = mask2.sum(dim=lon_dim)
    lat_values = counts_per_lat[lat_dim].values.astype(float)

    # grid spacing (native)
    if len(lat_values) > 1:
        dlat_deg = float(abs(lat_values[1] - lat_values[0]))
    else:
        dlat_deg = 0.5  # fallback
    n_lon = mask2.sizes[lon_dim]
    dlon_deg = 360.0 / float(n_lon)

    # approximate cell area on sphere (constant × cos(lat))
    R = 6_371_000.0  # m
    dlat_rad = np.deg2rad(dlat_deg)
    dlon_rad = np.deg2rad(dlon_deg)
    lat_rad = np.deg2rad(lat_values)

    area_per_cell = (R ** 2) * dlat_rad * dlon_rad * np.cos(lat_rad)
    row_area = counts_per_lat.values * area_per_cell  # total area of test cells in row

    # weights per latitude row, index matches native lat grid
    lat_row_weights = pd.Series(
        row_area,
        index=pd.Index(lat_values, name="lat"),
        name="row_area",
    )

    _LAT_BAND_WEIGHTS = lat_row_weights
    ds_mask.close()
    return lat_row_weights

def load_lat_means_for_scenario(var: str, scenario: str) -> Optional[pd.DataFrame]:
    """
    Load raw lat_means for a single (var, scenario):

      - index: lat
      - columns: models (DGVMs, ENSMEAN, Stable, extras,...)
    """
    csv_path = CSV_LAT_MEANS_ROOT / scenario / f"{var}_lat_means.csv"
    if not csv_path.is_file():
        print(f"[WARN] Missing CSV {csv_path}")
        return None

    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = "lat"
    df.columns = df.columns.astype(str).str.strip()
    return df


def build_scenario_averaged_summary(
    var: str,
    scenarios: List[str],
    extra_models: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    For a given variable, load lat_means for all selected scenarios and
    compute a scenario-averaged latitudinal mean per model (for line plots),
    then collapse into DGVM bands + ENSMEAN + Stable-Emulator + extras.

    Returns a DataFrame indexed by latitude with columns:
      - ENSMEAN, Stable-Emulator
      - DGVM_min, DGVM_max
      - DGVM_p50_low, DGVM_p50_high
      - DGVM_p75_low, DGVM_p75_high
      - extra model columns (if requested)
    """
    if extra_models is None:
        extra_models = []

    raw_dfs = []
    used_scenarios = []
    for scen in scenarios:
        df = load_lat_means_for_scenario(var, scen)
        if df is None or df.empty:
            continue
        raw_dfs.append(df)
        used_scenarios.append(scen)

    if not raw_dfs:
        print(f"[WARN] No lat_means data for var={var} in any scenario; skipping.")
        return None

    # Stack along a scenario level, then average over scenarios for each lat, model
    stacked = pd.concat(raw_dfs, keys=used_scenarios, names=["scenario", "lat"])
    df_mean = stacked.groupby("lat").mean()  # average across scenarios
    df_mean.index.name = "lat"

    # Now collapse across DGVMs to bands, and keep ENSMEAN, Stable etc
    dgvm_cols = [m for m in DGVM_MODELS if m in df_mean.columns]
    if not dgvm_cols:
        print(f"[WARN] No DGVM columns for var={var} after scenario averaging.")
        return None

    d_dgvm = df_mean[dgvm_cols]
    out = pd.DataFrame(index=df_mean.index)

    # ENSMEAN
    if ENS_COLUMN in df_mean.columns:
        out["ENSMEAN"] = df_mean[ENS_COLUMN]
    else:
        print(f"[WARN] {ENS_COLUMN} not found for var={var} after averaging.")

    # Stable-Emulator
    stable_col = None
    for cand in STABLE_CANDIDATES:
        if cand in df_mean.columns:
            stable_col = cand
            break
    if stable_col is not None:
        out["Stable-Emulator"] = df_mean[stable_col]
    else:
        print(f"[WARN] No Stable-Emulator column found for var={var} after averaging.")

    # DGVM bands across models at each latitude
    out["DGVM_min"] = d_dgvm.min(axis=1)
    out["DGVM_max"] = d_dgvm.max(axis=1)
    out["DGVM_p50_low"] = d_dgvm.quantile(0.25, axis=1)
    out["DGVM_p50_high"] = d_dgvm.quantile(0.75, axis=1)
    out["DGVM_p75_low"] = d_dgvm.quantile(0.125, axis=1)
    out["DGVM_p75_high"] = d_dgvm.quantile(0.875, axis=1)

    # Extra overlay models (lines)
    for name in extra_models:
        if name in df_mean.columns:
            out[name] = df_mean[name]
        else:
            print(f"[WARN] extra model '{name}' not found for var={var} after averaging.")

    return out


def compute_global_mean_stats_across_scenarios(
    var: str,
    scenarios: List[str],
    extra_models: Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    From lat_means CSVs for all scenarios, compute a single global mean
    per model, averaged across latitude and scenarios.

    Returns stats needed for a boxplot:
      - dgvm_vals (array)
      - q25, q75, q125, q875, median
      - ensmean, stable
      - extras: dict name -> value
    """
    if extra_models is None:
        extra_models = []

    raw_dfs = []
    used_scenarios = []
    for scen in scenarios:
        df = load_lat_means_for_scenario(var, scen)
        if df is None or df.empty:
            continue
        raw_dfs.append(df)
        used_scenarios.append(scen)

    if not raw_dfs:
        print(f"[WARN] No lat_means data for global stats var={var}; skipping.")
        return None

    stacked = pd.concat(raw_dfs, keys=used_scenarios, names=["scenario", "lat"])

    dgvm_cols = [m for m in DGVM_MODELS if m in stacked.columns]
    if not dgvm_cols:
        print(f"[WARN] No DGVM columns for global stats var={var}.")
        return None

        # ------------------------------------------------------------------
    # Global mean = area-weighted mean over latitude (using tvt_mask==2)
    # and averaged over scenarios (implicitly, since weights are per lat)
    # ------------------------------------------------------------------
    lat_band_weights = get_lat_band_area_weights()

    # Build a weight per row of "stacked" based on its latitude band
    lat_index = stacked.index.get_level_values("lat").astype(float)
    try:
        w = lat_band_weights.loc[lat_index].values
    except KeyError:
        # In case of tiny float mismatches, round to 1 decimal place
        lat_band_weights_rounded = lat_band_weights.copy()
        lat_band_weights_rounded.index = np.round(
            lat_band_weights_rounded.index.values, 1
        )
        w = lat_band_weights_rounded.loc[np.round(lat_index, 1)].values

    def _weighted_mean(col_values: np.ndarray, weights: np.ndarray) -> float:
        mask = np.isfinite(col_values) & np.isfinite(weights) & (weights > 0)
        if not np.any(mask):
            return np.nan
        w_eff = weights[mask]
        v_eff = col_values[mask]
        return float(np.sum(v_eff * w_eff) / np.sum(w_eff))

    # Weighted global mean per DGVM
    global_means: Dict[str, float] = {}
    for col in dgvm_cols:
        global_means[col] = _weighted_mean(stacked[col].values, w)

    vals = np.array(
        [v for v in global_means.values() if np.isfinite(v)]
    )
    if vals.size == 0:
        print(f"[WARN] All DGVM global means NaN for var={var}.")
        return None

    q25 = np.quantile(vals, 0.25)
    q75 = np.quantile(vals, 0.75)
    q125 = np.quantile(vals, 0.125)
    q875 = np.quantile(vals, 0.875)
    median = np.median(vals)

    # ENSMEAN global mean (area-weighted)
    ens_val = None
    if ENS_COLUMN in stacked.columns:
        ens_val = _weighted_mean(stacked[ENS_COLUMN].values, w)

    # Stable-Emulator global mean (area-weighted)
    stable_val = None
    for cand in STABLE_CANDIDATES:
        if cand in stacked.columns:
            stable_val = _weighted_mean(stacked[cand].values, w)
            break

    extras_vals: Dict[str, float] = {}
    for name in extra_models:
        if name in stacked.columns:
            extras_vals[name] = _weighted_mean(stacked[name].values, w)
        else:
            print(f"[WARN] extra model '{name}' not found for global stats var={var}.")

    return {
        "dgvm_vals": vals,
        "q25": q25,
        "q75": q75,
        "q125": q125,
        "q875": q875,
        "median": median,
        "ensmean": ens_val,
        "stable": stable_val,
        "extras": extras_vals,
    }

# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def _plot_latitudinal_profile(
    ax: plt.Axes,
    df_s: pd.DataFrame,
    var_name: str,
    line_units: str,
    title: str,
    show_ylabel: bool,   # kept for API compatibility, but no longer used
    show_xlabel: bool,
    extra_models: Optional[List[str]] = None,
    extra_colors: Optional[List[str]] = None,
) -> None:
    if extra_models is None:
        extra_models = []
    if extra_colors is None:
        extra_colors = DEFAULT_EXTRA_COLORS

    df_plot = df_s.sort_index().copy()
    lats = df_plot.index.values

    # Bands across DGVMs
    ax.fill_between(
        lats,
        df_plot["DGVM_min"],
        df_plot["DGVM_max"],
        color=COLOR_100,
        alpha=0.25,
        linewidth=0,
    )
    ax.fill_between(
        lats,
        df_plot["DGVM_p75_low"],
        df_plot["DGVM_p75_high"],
        color=COLOR_75,
        alpha=0.35,
        linewidth=0,
    )
    ax.fill_between(
        lats,
        df_plot["DGVM_p50_low"],
        df_plot["DGVM_p50_high"],
        color=COLOR_50,
        alpha=0.45,
        linewidth=0,
    )

    # Lines
    if "ENSMEAN" in df_plot:
        ax.plot(lats, df_plot["ENSMEAN"], color=COLOR_ENS, linewidth=1.8)
    if "Stable-Emulator" in df_plot:
        ax.plot(lats, df_plot["Stable-Emulator"], color=COLOR_STABLE, linewidth=1.8)

    for j, model_name in enumerate(extra_models):
        if model_name not in df_plot.columns:
            continue
        color = extra_colors[j] if j < len(extra_colors) else "k"
        ax.plot(lats, df_plot[model_name], color=color, linewidth=1.5)

    ax.set_title(title, fontweight="bold", fontsize=11, pad=6)
    ax.grid(True, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # NEW: every subplot gets its own y-label with just the units
    if line_units and line_units != "-":
        ax.set_ylabel(line_units, fontsize=10)
    else:
        # fall back to variable name if units are missing or "-"
        ax.set_ylabel(var_name, fontsize=10)

    if show_xlabel:
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xlabel("Latitude (°)", fontsize=10)
    else:
        ax.tick_params(axis="x", labelbottom=False)

def _plot_box_inset(
    ax: plt.Axes,
    stats: Dict,
) -> None:
    """
    Draw a small boxplot (global mean stats) as an inset within the given axis.
    Positioned so that the 'Global Mean' title stays below the top of
    the main plot's y-axis.
    """
    if stats is None:
        return

    vals = stats["dgvm_vals"]
    if vals.size == 0:
        return

    q25 = stats["q25"]
    q75 = stats["q75"]
    q125 = stats["q125"]
    q875 = stats["q875"]
    median = stats["median"]

    # Inset axis: explicit [x0, y0, width, height] in axes coordinates
    # This keeps the inset well inside the main plot so the title
    # stays below the top of the main y-axis.
    inset_ax = inset_axes(ax, width="30%", height="30%", loc="upper right", borderpad=0.7)

    # Small title inside inset
    inset_ax.set_title(
        "Global Mean",
        fontsize=8,
        fontweight="bold",
        pad=1.5,
        loc="center",
    )

    # Basic x-position arrangement
    x_base = 0.0
    box_width = 0.18   # narrow box
    dx_extra = 0.1    # models close to DGVMs

    # DGVM points
    rng = np.random.default_rng(42)
    jitter = rng.normal(loc=0.0, scale=0.02, size=vals.size)
    inset_ax.scatter(
        x_base + jitter,
        vals,
        facecolors="white",
        edgecolors="0.6",
        linewidths=0.6,
        s=8,
        zorder=3,
    )

    # Box (IQR)
    rect = Rectangle(
        (x_base - box_width / 2.0, q25),
        box_width,
        q75 - q25,
        facecolor="none",
        edgecolor="k",
        linewidth=0.8,
        zorder=4,
    )
    inset_ax.add_patch(rect)

    # Whiskers
    inset_ax.vlines(
        x_base,
        q125,
        q875,
        colors="k",
        linewidth=0.7,
        zorder=4,
    )

    # Median
    inset_ax.hlines(
        median,
        x_base - box_width / 2.0,
        x_base + box_width / 2.0,
        colors="k",
        linewidth=1.0,
        zorder=5,
    )

    # ENSMEAN
    if stats.get("ensmean") is not None:
        inset_ax.scatter(
            [x_base],
            [stats["ensmean"]],
            marker="o",
            s=18,
            facecolors=COLOR_ENS,
            edgecolors=COLOR_ENS,
            linewidths=0.5,
            zorder=6,
        )

    # Stable-Emulator
    if stats.get("stable") is not None:
        x_stable = x_base + dx_extra
        inset_ax.scatter(
            [x_stable],
            [stats["stable"]],
            marker="o",
            s=16,
            facecolors=COLOR_STABLE,
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
        )

    # Extra models
    extras = stats.get("extras", {})
    for i, (name, val) in enumerate(extras.items()):
        x = x_base + dx_extra * (i + 2)
        color = DEFAULT_EXTRA_COLORS[i] if i < len(DEFAULT_EXTRA_COLORS) else "k"
        inset_ax.scatter(
            [x],
            [val],
            marker="o",
            s=16,
            facecolors=color,
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
        )

    # Tight x-limits so points/models sit close together
    max_extra_index = (len(extras) + 2)
    x_min = x_base - 0.25
    x_max = x_base + dx_extra * max_extra_index + 0.15
    inset_ax.set_xlim(x_min, x_max)

    # Smaller scientific-notation offset text
    inset_ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))
    offset_text = inset_ax.yaxis.get_offset_text()
    offset_text.set_fontsize(6)

    # Clean inset
    inset_ax.set_xticks([])
    inset_ax.tick_params(axis="y", labelsize=6)
    for spine in ("top", "right", "bottom"):
        inset_ax.spines[spine].set_visible(False)


# -----------------------------------------------------------------------------
# Multi-panel figure
# -----------------------------------------------------------------------------

def make_multi_variable_panel(
    vars_to_use: List[str],
    scenarios: List[str],
    df_by_var: Dict[str, pd.DataFrame],
    stats_by_var: Dict[str, Dict],
    units_by_var: Dict[str, str],
    out_path: Path,
    extra_models: Optional[List[str]] = None,
    extra_colors: Optional[List[str]] = None,
) -> None:
    """
    Create a 3×5 panel of latitudinal mean profiles, one per variable,
    with a small global-mean boxplot inset in each subplot.
    """
    if extra_models is None:
        extra_models = []
    if extra_colors is None:
        extra_colors = DEFAULT_EXTRA_COLORS

    n_rows, n_cols = 5, 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 14),
        sharex=True,
    )

    axes = axes.reshape(n_rows, n_cols)
    
    # NEW: last row that actually has a variable
    last_row_used = (len(vars_to_use) - 1) // n_cols

    # NEW: labels for subplots
    subplot_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Plot each variable in the requested order
    for idx, var in enumerate(vars_to_use):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # --- NEW: add subplot letter in top-left ---
        if idx < len(subplot_labels):
            ax.text(
                0.02, 0.96,
                subplot_labels[idx],
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="top",
                ha="left",
            )

        df_s = df_by_var.get(var)
        if df_s is None or df_s.empty:
            ax.text(
                0.5,
                0.5,
                f"No data\n{var}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        mean_units = units_by_var.get(var, "arbitrary units")
        title = var.upper()

        show_ylabel = (col == 0)
        # NEW: last row that actually has a variable
        show_xlabel = (row == last_row_used)

        _plot_latitudinal_profile(
            ax=ax,
            df_s=df_s,
            var_name=var.upper(),
            line_units=mean_units,
            title=title,
            show_ylabel=show_ylabel,
            show_xlabel=show_xlabel,
            extra_models=extra_models,
            extra_colors=extra_colors,
        )

        # Box inset with global stats
        stats = stats_by_var.get(var)
        if stats is not None:
            _plot_box_inset(ax, stats)

    # Turn off any unused axes (in case vars_to_use < 15)
    total_axes = n_rows * n_cols
    if len(vars_to_use) < total_axes:
        for idx in range(len(vars_to_use), total_axes):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_axis_off()

    # Legend: global at bottom
    dgvm_handle = Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor="0.6",
        label="Contributing DGVMs",
    )

    legend_handles = [
        dgvm_handle,
        Patch(facecolor=COLOR_50, alpha=0.45, label="DGVM 50th percentile"),
        Patch(facecolor=COLOR_75, alpha=0.35, label="DGVM 75th percentile"),
        Patch(facecolor=COLOR_100, alpha=0.25, label="DGVM range"),
        Line2D(
            [0], [0],
            linestyle="-",
            color=COLOR_ENS,
            label="TRENDY Ensemble Mean",
        ),
        Line2D(
            [0], [0],
            linestyle="-",
            color=COLOR_STABLE,
            label="Stable-Emulator (autoregressive)",
        ),
    ]

    for j, model_name in enumerate(extra_models):
        color = extra_colors[j] if j < len(extra_colors) else "k"
        if model_name == "Base-Emulator_No_Carry":
            label = "Base-Emulator (non-autoregressive)"
        elif model_name == "Stable-Emulator_No_Carry":
            label = "Stable-Emulator (non-autoregressive)"
        else:
            label = model_name
        legend_handles.append(
            Line2D(
                [0], [0],
                linestyle="-",
                color=color,
                label=label,
            )
        )

    ncols_legend = min(3, len(legend_handles))

    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=ncols_legend,
        frameon=False,
        fontsize=11,
        columnspacing=1.5,
        handletextpad=0.8,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        bottom=0.10,
        top=0.96,
        wspace=0.22,
        hspace=0.40,
    )

    fig.savefig(out_path, dpi=500)
    plt.close(fig)
    print(f"[INFO] Saved latitudinal multi-variable panel (with box insets) to {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Make a single multi-panel figure of scenario-averaged latitudinal "
            "means for multiple variables, with global-mean boxplot insets."
        )
    )
    ap.add_argument(
        "--extra-model",
        dest="extra_models",
        nargs="*",
        default=[],
        help=(
            "Non-DGVM model columns to overlay as extra lines/points "
            "(e.g. Base-Emulator_No_Carry TL-Emulator). "
            "Order determines colours."
        ),
    )
    ap.add_argument(
        "--scenario",
        choices=SCENARIOS,
        nargs="*",
        help="Optional subset of scenarios to include in the average (default: all).",
    )
    ap.add_argument(
        "--var",
        choices=VARS_ORDER,
        nargs="*",
        help=(
            "Optional subset of variables to include. "
            "Default: fixed 15-variable panel in a predefined order."
        ),
    )

    args = ap.parse_args()

    extra_models = args.extra_models or []
    scenarios_to_use = args.scenario if args.scenario else SCENARIOS

    # Variables to use, respecting the requested panel order
    if args.var:
        vars_to_use = [v for v in VARS_ORDER if v in args.var]
    else:
        vars_to_use = VARS_ORDER.copy()

    # Build scenario-averaged summaries and global stats per variable
    df_by_var: Dict[str, pd.DataFrame] = {}
    stats_by_var: Dict[str, Dict] = {}
    units_by_var: Dict[str, str] = {}

    for var in vars_to_use:
        print(f"[INFO] Processing variable: {var}")

        mean_units, _ = VAR_UNITS.get(
            var,
            ("arbitrary units", "arbitrary units"),
        )

        df_summary = build_scenario_averaged_summary(
            var, scenarios_to_use, extra_models=extra_models
        )
        if df_summary is None or df_summary.empty:
            print(f"[WARN] No usable latitudinal data for var={var}; skipping in panel.")
            continue

        stats = compute_global_mean_stats_across_scenarios(
            var, scenarios_to_use, extra_models=extra_models
        )
        if stats is None:
            print(f"[WARN] No usable global stats for var={var}; box inset will be empty.")

        df_by_var[var] = df_summary
        stats_by_var[var] = stats
        units_by_var[var] = mean_units

    if not df_by_var:
        print("[WARN] No variables with data; nothing to plot.")
        return

    out_path = PLOTS_LAT_BOX_DIR / "lat_means_all_vars_panel.png"
    make_multi_variable_panel(
        vars_to_use=list(df_by_var.keys()),
        scenarios=scenarios_to_use,
        df_by_var=df_by_var,
        stats_by_var=stats_by_var,
        units_by_var=units_by_var,
        out_path=out_path,
        extra_models=extra_models,
        extra_colors=DEFAULT_EXTRA_COLORS,
    )


if __name__ == "__main__":
    sys.path.append(str(PROJECT_ROOT))
    main()