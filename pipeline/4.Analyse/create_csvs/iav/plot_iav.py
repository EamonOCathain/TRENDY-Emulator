#!/usr/bin/env python3
"""
make_iav_line_panel.py
----------------------

Create multi-panel figures of *interannual variability* time series.

For each variable:
  - Read inter_annual_variability CSVs for each scenario (default: S0,S1,S2,S3)
  - For each scenario separately:
      * Collapse models into DGVM bands + ENSMEAN + Stable-Emulator + extras
      * Plot time series (year on x-axis)
  - Also compute an average across scenarios and plot that as a fifth figure.

Each scenario figure:
  - 3 columns × 5 rows = 15 subplots
  - Variables in this order (top-left → bottom-right):

      1)  gpp
      2)  ra
      3)  npp
      4)  rh
      5)  nee
      6)  fFire
      7)  fLuc
      8)  nbp
      9)  mrro
      10) evapotrans
      11) mrso
      12) lai
      13) cVeg
      14) cLitter
      15) cSoil

  - cTotal is excluded.

Input CSVs (interannual variability):

  inter_annual_variability:
    PROJECT_ROOT/data/analysis/CSVs/inter_annual_variability/<SCENARIO>/<var>_annual_means.csv

Each CSV:
  - rows: years (e.g. 1901, 1902, ..., 2023)
  - columns: models

Outputs:

  PROJECT_ROOT/data/analysis/CSVs/plots/inter_annual_variability/
      iav_S0_all_vars_panel.png
      iav_S1_all_vars_panel.png
      iav_S2_all_vars_panel.png
      iav_S3_all_vars_panel.png
      iav_avg_all_vars_panel.png   (average over selected scenarios)
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
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")

CSV_IAV_ROOT = PROJECT_ROOT / "data/analysis/CSVs/iav"

PLOTS_ROOT = PROJECT_ROOT / "data/analysis/CSVs/plots/iav"
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

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

# Units (same as latitudinal script)
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

def load_iav_for_scenario(var: str, scenario: str) -> Optional[pd.DataFrame]:
    """
    Load interannual variability data for a single (var, scenario):

      - index: year
      - columns: models (DGVMs, ENSMEAN, Stable, extras,...)
    """
    csv_path = CSV_IAV_ROOT / scenario / f"{var}_annual_means.csv"
    if not csv_path.is_file():
        print(f"[WARN] Missing CSV {csv_path}")
        return None

    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = "year"
    df.columns = df.columns.astype(str).str.strip()
    return df


def build_summary_for_scenario(
    var: str,
    scenario: str,
    extra_models: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    For a given variable and scenario, load interannual variability and
    collapse models into DGVM bands + ENSMEAN + Stable-Emulator + extras.

    Returns a DataFrame indexed by year with columns:
      - ENSMEAN, Stable-Emulator
      - DGVM_min, DGVM_max
      - DGVM_p50_low, DGVM_p50_high
      - DGVM_p75_low, DGVM_p75_high
      - extra model columns (if requested)
    """
    if extra_models is None:
        extra_models = []

    df = load_iav_for_scenario(var, scenario)
    if df is None or df.empty:
        return None

    dgvm_cols = [m for m in DGVM_MODELS if m in df.columns]
    if not dgvm_cols:
        print(f"[WARN] No DGVM columns for var={var} scenario={scenario}")
        return None

    d_dgvm = df[dgvm_cols]
    out = pd.DataFrame(index=df.index)

    # ENSMEAN
    if ENS_COLUMN in df.columns:
        out["ENSMEAN"] = df[ENS_COLUMN]
    else:
        print(f"[WARN] {ENS_COLUMN} not found for var={var} scenario={scenario}.")

    # Stable-Emulator
    stable_col = None
    for cand in STABLE_CANDIDATES:
        if cand in df.columns:
            stable_col = cand
            break
    if stable_col is not None:
        out["Stable-Emulator"] = df[stable_col]
    else:
        print(f"[WARN] No Stable-Emulator column for var={var} scenario={scenario}.")

    # DGVM bands across models at each year
    out["DGVM_min"] = d_dgvm.min(axis=1)
    out["DGVM_max"] = d_dgvm.max(axis=1)
    out["DGVM_p50_low"] = d_dgvm.quantile(0.25, axis=1)
    out["DGVM_p50_high"] = d_dgvm.quantile(0.75, axis=1)
    out["DGVM_p75_low"] = d_dgvm.quantile(0.125, axis=1)
    out["DGVM_p75_high"] = d_dgvm.quantile(0.875, axis=1)

    # Extra overlay models (lines)
    for name in extra_models:
        if name in df.columns:
            out[name] = df[name]
        else:
            print(f"[WARN] extra model '{name}' not found for var={var} scenario={scenario}.")

    return out


def build_scenario_averaged_summary(
    var: str,
    scenarios: List[str],
    extra_models: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Build a summary averaged across scenarios, for one variable.

    Steps:
      - Load per-scenario interannual variability (year × model)
      - Stack with a "scenario" axis
      - Average across scenarios per year and model
      - Collapse averaged models into DGVM bands + ENSMEAN + Stable-Emulator + extras.
    """
    if extra_models is None:
        extra_models = []

    raw_dfs = []
    used_scenarios = []
    for scen in scenarios:
        df = load_iav_for_scenario(var, scen)
        if df is None or df.empty:
            continue
        raw_dfs.append(df)
        used_scenarios.append(scen)

    if not raw_dfs:
        print(f"[WARN] No IAV data for var={var} in any scenario; skipping average.")
        return None

    # Stack along a scenario level, then average across scenarios by year & model
    stacked = pd.concat(raw_dfs, keys=used_scenarios, names=["scenario", "year"])
    df_mean = stacked.groupby("year").mean()
    df_mean.index.name = "year"

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
        print(f"[WARN] No Stable-Emulator column for var={var} after averaging.")

    # DGVM bands across models at each year
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


# -----------------------------------------------------------------------------
# Line panel helper (year on x-axis)
# -----------------------------------------------------------------------------

def _plot_time_profile(
    ax: plt.Axes,
    df_s: pd.DataFrame,
    var_name: str,
    line_units: str,
    title: str,
    show_ylabel: bool,
    show_xlabel: bool,
    extra_models: Optional[List[str]] = None,
    extra_colors: Optional[List[str]] = None,
) -> None:
    if extra_models is None:
        extra_models = []
    if extra_colors is None:
        extra_colors = DEFAULT_EXTRA_COLORS

    df_plot = df_s.sort_index().copy()
    years = df_plot.index.values

    # Bands across DGVMs
    ax.fill_between(
        years,
        df_plot["DGVM_min"],
        df_plot["DGVM_max"],
        color=COLOR_100,
        alpha=0.25,
        linewidth=0,
    )
    ax.fill_between(
        years,
        df_plot["DGVM_p75_low"],
        df_plot["DGVM_p75_high"],
        color=COLOR_75,
        alpha=0.35,
        linewidth=0,
    )
    ax.fill_between(
        years,
        df_plot["DGVM_p50_low"],
        df_plot["DGVM_p50_high"],
        color=COLOR_50,
        alpha=0.45,
        linewidth=0,
    )

    # Lines
    if "ENSMEAN" in df_plot:
        ax.plot(years, df_plot["ENSMEAN"], color=COLOR_ENS, linewidth=1.8)
    if "Stable-Emulator" in df_plot:
        ax.plot(years, df_plot["Stable-Emulator"], color=COLOR_STABLE, linewidth=1.8)

    for j, model_name in enumerate(extra_models):
        if model_name not in df_plot.columns:
            continue
        color = extra_colors[j] if j < len(extra_colors) else "k"
        ax.plot(years, df_plot[model_name], color=color, linewidth=1.5)

    ax.set_title(title, fontweight="bold", fontsize=11, pad=6)
    ax.grid(True, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if show_ylabel:
        label = f"{var_name} ({line_units})" if line_units != "-" else var_name
        ax.set_ylabel(label, fontsize=10)

    if show_xlabel:
        ax.set_xlabel("Year", fontsize=10)


# -----------------------------------------------------------------------------
# Multi-panel figure
# -----------------------------------------------------------------------------

def make_multi_variable_panel(
    vars_to_use: List[str],
    df_by_var: Dict[str, pd.DataFrame],
    units_by_var: Dict[str, str],
    out_path: Path,
    figure_label: str,
    extra_models: Optional[List[str]] = None,
    extra_colors: Optional[List[str]] = None,
) -> None:
    """
    Create a 3×5 panel of interannual variability time series, one per variable.
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

    # Plot each variable in the requested order
    for idx, var in enumerate(vars_to_use):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # ---- Add subplot panel letter (A, B, C...) ----
        letter = chr(ord('A') + idx)
        ax.text(
            0.025, 0.98, letter,
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=12, fontweight='bold'
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
        show_xlabel = (row == n_rows - 1)

        _plot_time_profile(
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
        Patch(facecolor=COLOR_50, alpha=0.45, label="DGVM 50th Percentile"),
        Patch(facecolor=COLOR_75, alpha=0.35, label="DGVM 75th Percentile"),
        Patch(facecolor=COLOR_100, alpha=0.25, label="DGVM 100th Percentile"),
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
            label="Stable-Emulator (Autoregressive)",
        ),
    ]

    for j, model_name in enumerate(extra_models):
        color = extra_colors[j] if j < len(extra_colors) else "k"
        if model_name == "Base-Emulator_No_Carry":
            label = "Base-Emulator (Non-Autoregressive)"
        elif model_name == "Stable-Emulator_No_Carry":
            label = "Stable-Emulator (Non-Autoregressive)"
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
        fontsize=9,
        columnspacing=1.5,
        handletextpad=0.8,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        bottom=0.10,
        top=0.93,
        wspace=0.30,
        hspace=0.40,
    )

    fig.savefig(out_path, dpi=500)
    plt.close(fig)
    print(f"[INFO] Saved IAV multi-variable panel to {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Make multi-panel figures of interannual variability time series "
            "for multiple variables, per scenario and averaged across scenarios."
        )
    )
    ap.add_argument(
        "--extra-model",
        dest="extra_models",
        nargs="*",
        help=(
            "Non-DGVM model columns to overlay as extra lines "
            "(e.g. Base-Emulator_No_Carry TL-Emulator). "
            "Order determines colours."
        ),
    )
    ap.add_argument(
        "--scenario",
        choices=SCENARIOS,
        nargs="*",
        help=(
            "Optional subset of scenarios to include. "
            "Used both for per-scenario plots and for the scenario-average."
        ),
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

    # --------- Per-scenario panels ---------
    for scen in scenarios_to_use:
        print(f"[INFO] Building per-scenario panel for {scen}")
        df_by_var: Dict[str, pd.DataFrame] = {}
        units_by_var: Dict[str, str] = {}

        for var in vars_to_use:
            print(f"[INFO]  Scenario {scen} – variable {var}")
            mean_units, _ = VAR_UNITS.get(
                var,
                ("arbitrary units", "arbitrary units"),
            )
            df_summary = build_summary_for_scenario(
                var, scen, extra_models=extra_models
            )
            if df_summary is None or df_summary.empty:
                print(f"[WARN]  No usable data for var={var} in {scen}; skipping.")
                continue

            df_by_var[var] = df_summary
            units_by_var[var] = mean_units

        if not df_by_var:
            print(f"[WARN] No variables with data for scenario {scen}; skipping panel.")
            continue

        out_path = PLOTS_ROOT / f"iav_{scen}_all_vars_panel.png"
        make_multi_variable_panel(
            vars_to_use=list(df_by_var.keys()),
            df_by_var=df_by_var,
            units_by_var=units_by_var,
            out_path=out_path,
            figure_label=scen,
            extra_models=extra_models,
            extra_colors=DEFAULT_EXTRA_COLORS,
        )

    # --------- Scenario-averaged panel ---------
    print("[INFO] Building scenario-averaged panel")
    df_avg_by_var: Dict[str, pd.DataFrame] = {}
    units_avg_by_var: Dict[str, str] = {}

    for var in vars_to_use:
        print(f"[INFO]  Scenario-average – variable {var}")
        mean_units, _ = VAR_UNITS.get(
            var,
            ("arbitrary units", "arbitrary units"),
        )
        df_summary = build_scenario_averaged_summary(
            var, scenarios_to_use, extra_models=extra_models
        )
        if df_summary is None or df_summary.empty:
            print(f"[WARN]  No usable averaged data for var={var}; skipping.")
            continue

        df_avg_by_var[var] = df_summary
        units_avg_by_var[var] = mean_units

    if df_avg_by_var:
        out_path_avg = PLOTS_ROOT / "iav_avg_all_vars_panel.png"
        make_multi_variable_panel(
            vars_to_use=list(df_avg_by_var.keys()),
            df_by_var=df_avg_by_var,
            units_by_var=units_avg_by_var,
            out_path=out_path_avg,
            figure_label="Average across scenarios",
            extra_models=extra_models,
            extra_colors=DEFAULT_EXTRA_COLORS,
        )
    else:
        print("[WARN] No variables with averaged data; no avg panel produced.")


if __name__ == "__main__":
    # Ensure the project root is on the path (if you need imports later)
    sys.path.append(str(PROJECT_ROOT))
    main()