#!/usr/bin/env python3
"""
make_summary_plots.py
---------------------

Read annual_means and annual_totals CSVs (per scenario/variable/model),
collapse them into:
  - TRENDY ensemble mean line ("ENSMEAN")
  - Stable-Emulator line ("Stable-Emulator")
  - DGVM uncertainty bands (min/max, 75%, 50%)

Then plot, for each variable:

  1) Annual means (4-panel, one per scenario)
  2) Annual totals (4-panel, one per scenario)
  3) Cumulative totals (4-panel, one per scenario)

Plots are saved under:

  /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/analysis/CSVs/plots/
      annual_means/<var>_annual_means.png
      annual_totals/<var>_annual_totals.png
      cumulative_totals/<var>_cumulative_totals.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe on cluster
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")

CSV_MEANS_ROOT = PROJECT_ROOT / "data/analysis/CSVs/annual_means"
CSV_TOTALS_ROOT = PROJECT_ROOT / "data/analysis/CSVs/annual_totals"

PLOTS_ROOT = PROJECT_ROOT / "data/analysis/CSVs/plots/with_base_and_stable"
PLOTS_MEANS_DIR = PLOTS_ROOT / "annual_means"
PLOTS_TOTALS_DIR = PLOTS_ROOT / "annual_totals"
PLOTS_CUM_DIR = PLOTS_ROOT / "cumulative_totals"

for d in (PLOTS_MEANS_DIR, PLOTS_TOTALS_DIR, PLOTS_CUM_DIR):
    d.mkdir(parents=True, exist_ok=True)

SCENARIOS: List[str] = ["S0", "S1", "S2", "S3"]

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

# DGVM models (columns in the CSVs)
DGVM_MODELS = [
    "CLASSIC",
    "CLM",
    "JSBACH",
    "ORCHIDEE",
    "SDGVM",
    "VISIT",
    "VISIT-UT",
]

# Column names in CSVs that should map to plot lines
ENS_COLUMN = "TRENDY-Ensemble-Mean"
STABLE_CANDIDATES = [
    "Stable-Emulator_With_Carry"
]

# Default colours for extra (non-DGVM) overlay models.
# 1st extra: arbitrary but distinct; 2nd & 3rd fixed as requested.
DEFAULT_EXTRA_COLORS = [
    "#cc4678",  # 1st extra model (e.g. Base-Emulator_No_Carry)
    "#f0f922",  # 2nd extra model (e.g. TL-Emulator)
    "#3b528b",  # any 3rd extra model, if present
]

# Units for pretty labels
# (first entry: annual means, second: annual totals)
VAR_UNITS = {
    "gpp":       ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "npp":       ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "ra":        ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "rh":        ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "nee":       ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "nbp":       ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "fLuc":      ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "fFire":     ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "evapotrans":("kg m$^{-2}$ s$^{-1}$", "Gt yr$^{-1}$"),  # rough
    "cVeg":      ("kg m$^{-2}$", "Gt"),
    "cSoil":     ("kg m$^{-2}$", "Gt"),
    "cLitter":   ("kg m$^{-2}$", "Gt"),
    "cTotal":    ("kg m$^{-2}$", "Gt"),
    "mrro":      ("kg m$^{-2}$ s$^{-1}$", "Gt yr$^{-1}$"),
    "mrso":      ("kg m$^{-2}$", "Gt"),
    "lai":       ("-", "-"),
}

# -----------------------------------------------------------------------------
# Plot functions
# -----------------------------------------------------------------------------

def plot_four_scenarios_bands(
    df_by_scenario: Dict[str, pd.DataFrame],
    var_name: str = "NEE",
    units: str = "Gt C yr$^{-1}$",
    scen_order=("S0", "S1", "S2", "S3"),
    figsize=(12, 8),
    ylim=None,
    extra_models: Optional[List[str]] = None,
    extra_colors: Optional[List[str]] = None,
):
    """
    Plot 4 subplots (2x2), one per scenario, with DGVM uncertainty bands.

    extra_models : list of column names (non-DGVM models) to overlay as lines.
    extra_colors : list of colours for extra_models (same length or longer).
    """
    if extra_models is None:
        extra_models = []
    if extra_colors is None:
        extra_colors = DEFAULT_EXTRA_COLORS

    # Colours
    color_ens_line    = "#410253"  # dark navy
    color_stable_line = "#EF8F40"  # Stable-Emulator line

    color_50  = "#3A528B"  # 50% band
    color_75  = "#218F8C"  # 75% band
    color_100 = "#5DC762"  # 100% band

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    panel_letters = ["A", "B", "C", "D"]

    for i, scen in enumerate(scen_order):
        ax = axes[i]

        if scen not in df_by_scenario:
            ax.text(
                0.5,
                0.5,
                f"{scen}\n(no data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        df_s = df_by_scenario[scen]
        years = df_s.index.values

        # ---- DGVM bands ----
        ax.fill_between(
            years,
            df_s["DGVM_min"],
            df_s["DGVM_max"],
            color=color_100,
            alpha=0.25,
            linewidth=0,
        )
        ax.fill_between(
            years,
            df_s["DGVM_p75_low"],
            df_s["DGVM_p75_high"],
            color=color_75,
            alpha=0.35,
            linewidth=0,
        )
        ax.fill_between(
            years,
            df_s["DGVM_p50_low"],
            df_s["DGVM_p50_high"],
            color=color_50,
            alpha=0.45,
            linewidth=0,
        )

        # ---- ENSMEAN + Stable lines ----
        if "ENSMEAN" in df_s:
            ax.plot(years, df_s["ENSMEAN"], color=color_ens_line, linewidth=1.8)
        if "Stable-Emulator" in df_s:
            ax.plot(years, df_s["Stable-Emulator"], color=color_stable_line, linewidth=1.8)

        # ---- Extra model lines ----
        for j, model_name in enumerate(extra_models):
            if model_name not in df_s.columns:
                continue
            color = extra_colors[j] if j < len(extra_colors) else "k"
            ax.plot(
                years,
                df_s[model_name],
                color=color,
                linewidth=1.5,
            )

        # ---- Panel annotation ----
        ax.text(
            0.02,
            0.95,
            panel_letters[i],
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
        )
        ax.set_title(f"{scen}", fontweight="bold", fontsize=12)

        ax.grid(True, alpha=0.3)

        # Apply global y-limits if provided
        if ylim is not None:
            ax.set_ylim(ylim)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Axis labels
    fig.text(0.5, 0.10, "Year", ha="center", fontsize=12)
    fig.text(
        0.06,
        0.5,
        f"{var_name} ({units})",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    fig.suptitle(
        f"{var_name} Annual Means (1901–2023)",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    # Shared legend
    legend_handles = [
        Patch(facecolor=color_50,  alpha=0.45, label="DGVM 50th Percentile"),
        Patch(facecolor=color_75,  alpha=0.35, label="DGVM 75th Percentile"),
        Patch(facecolor=color_100, alpha=0.25, label="DGVM 100th Percentile"),
        Line2D([0], [0], color=color_ens_line,    linewidth=1.8, label="TRENDY Ensemble Mean"),
        Line2D([0], [0], color=color_stable_line, linewidth=1.8, label="Stable-Emulator"),
    ]

    # Extra models in legend
    for j, model_name in enumerate(extra_models):
        color = extra_colors[j] if j < len(extra_colors) else "k"
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.5, label=model_name)
        )

    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(rect=[0.08, 0.12, 0.98, 0.95])
    return fig, axes


def plot_four_scenarios_bands_cumulative(
    df_by_scenario: Dict[str, pd.DataFrame],
    var_name: str = "NEE",
    units: str = "Gt C yr$^{-1}$",
    scen_order=("S0", "S1", "S2", "S3"),
    figsize=(12, 8),
    ylim=None,
    extra_models: Optional[List[str]] = None,
    extra_colors: Optional[List[str]] = None,
):
    """
    Plot 4 subplots (2x2), one per scenario, with DGVM uncertainty bands,
    but using CUMULATIVE sums over time instead of annual values.
    """
    if extra_models is None:
        extra_models = []
    if extra_colors is None:
        extra_colors = DEFAULT_EXTRA_COLORS

    # Colours (same as original)
    color_ens_line    = "#410253"  # dark navy
    color_stable_line = "#EF8F40"  # Stable-Emulator line

    color_50  = "#3A528B"  # 50% band
    color_75  = "#218F8C"  # 75% band
    color_100 = "#5DC762"  # 100% band

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    panel_letters = ["A", "B", "C", "D"]

    for i, scen in enumerate(scen_order):
        ax = axes[i]

        if scen not in df_by_scenario:
            ax.text(
                0.5,
                0.5,
                f"{scen}\n(no data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        df_s = df_by_scenario[scen].sort_index()
        years = df_s.index.values

        cols_to_cumsum = [
            "ENSMEAN",
            "Stable-Emulator",
            "DGVM_p50_low",
            "DGVM_p50_high",
            "DGVM_p75_low",
            "DGVM_p75_high",
            "DGVM_min",
            "DGVM_max",
        ] + extra_models  # also cumsum extra model columns
        cols_to_cumsum = [c for c in cols_to_cumsum if c in df_s.columns]

        df_cum = df_s.copy()
        for c in cols_to_cumsum:
            df_cum[c] = df_s[c].cumsum()

        # ---- DGVM bands (cumulative) ----
        ax.fill_between(
            years,
            df_cum["DGVM_min"],
            df_cum["DGVM_max"],
            color=color_100,
            alpha=0.25,
            linewidth=0,
        )
        ax.fill_between(
            years,
            df_cum["DGVM_p75_low"],
            df_cum["DGVM_p75_high"],
            color=color_75,
            alpha=0.35,
            linewidth=0,
        )
        ax.fill_between(
            years,
            df_cum["DGVM_p50_low"],
            df_cum["DGVM_p50_high"],
            color=color_50,
            alpha=0.45,
            linewidth=0,
        )

        # ---- ENSMEAN + Stable-Emulator lines ----
        if "ENSMEAN" in df_cum:
            ax.plot(
                years,
                df_cum["ENSMEAN"],
                color=color_ens_line,
                linewidth=1.8,
            )
        if "Stable-Emulator" in df_cum:
            ax.plot(
                years,
                df_cum["Stable-Emulator"],
                color=color_stable_line,
                linewidth=1.8,
            )

        # ---- Extra model lines (cumulative) ----
        for j, model_name in enumerate(extra_models):
            if model_name not in df_cum.columns:
                continue
            color = extra_colors[j] if j < len(extra_colors) else "k"
            ax.plot(
                years,
                df_cum[model_name],
                color=color,
                linewidth=1.5,
            )

        # Panel letter (A–D)
        ax.text(
            0.02,
            0.95,
            panel_letters[i],
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
        )

        # Scenario title
        ax.set_title(f"{scen}", fontweight="bold", fontsize=12)

        ax.grid(True, alpha=0.3)

        if ylim is not None:
            ax.set_ylim(ylim)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Axis labels
    fig.text(0.5, 0.10, "Year", ha="center", fontsize=12)
    fig.text(
        0.06,
        0.5,
        f"Cumulative {var_name} ({units})",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    fig.suptitle(
        f"Cumulative {var_name} (1901–2023)",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    # Shared legend
    legend_handles = [
        Patch(facecolor=color_50,  alpha=0.45, label="DGVM 50% band"),
        Patch(facecolor=color_75,  alpha=0.35, label="DGVM 75% band"),
        Patch(facecolor=color_100, alpha=0.25, label="DGVM 100% band"),
        Line2D([0], [0], color=color_ens_line,    linewidth=1.8, label="TRENDY Ensemble Mean"),
        Line2D([0], [0], color=color_stable_line, linewidth=1.8, label="Stable-Emulator"),
    ]

    for j, model_name in enumerate(extra_models):
        color = extra_colors[j] if j < len(extra_colors) else "k"
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.5, label=model_name)
        )

    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(rect=[0.08, 0.12, 0.98, 0.95])
    return fig, axes

# -----------------------------------------------------------------------------
# Helpers to build df_by_scenario from CSVs
# -----------------------------------------------------------------------------

def build_summary_for_scenario_var(
    root: Path,
    var: str,
    scenario: str,
    extra_models: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Read CSV for (scenario, var) and collapse model columns into:

      ENSMEAN, Stable-Emulator,
      DGVM_min/max, DGVM_p50_low/high, DGVM_p75_low/high,
      plus any requested extra_models columns.
    """
    csv_path = root / scenario / f"{var}_annual_means.csv"
    if not csv_path.is_file():
        print(f"[WARN] Missing CSV {csv_path}")
        return None

    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = "year"
    df.columns = df.columns.astype(str).str.strip()

    # Collect DGVM columns that exist
    dgvm_cols = [m for m in DGVM_MODELS if m in df.columns]
    if not dgvm_cols:
        print(f"[WARN] No DGVM columns for {var} {scenario} in {csv_path.name}")
        return None

    d_dgvm = df[dgvm_cols]

    out = pd.DataFrame(index=df.index)

    # TRENDY ensemble mean
    if ENS_COLUMN in df.columns:
        out["ENSMEAN"] = df[ENS_COLUMN]
    else:
        print(f"[WARN] {ENS_COLUMN} not found in {csv_path.name}")

    # Stable-Emulator: pick first available candidate
    stable_col = None
    for cand in STABLE_CANDIDATES:
        if cand in df.columns:
            stable_col = cand
            break
    if stable_col is not None:
        out["Stable-Emulator"] = df[stable_col]
    else:
        print(f"[WARN] No Stable-Emulator column found in {csv_path.name}")

    # DGVM bands
    out["DGVM_min"] = d_dgvm.min(axis=1)
    out["DGVM_max"] = d_dgvm.max(axis=1)

    # 50% band: 25–75th percentile
    out["DGVM_p50_low"]  = d_dgvm.quantile(0.25, axis=1)
    out["DGVM_p50_high"] = d_dgvm.quantile(0.75, axis=1)

    # 75% band: 12.5–87.5th percentile
    out["DGVM_p75_low"]  = d_dgvm.quantile(0.125, axis=1)
    out["DGVM_p75_high"] = d_dgvm.quantile(0.875, axis=1)

    # Extra non-DGVM models to overlay
    if extra_models:
        for name in extra_models:
            if name in df.columns:
                out[name] = df[name]
            else:
                print(f"[WARN] extra model '{name}' not found in {csv_path.name}")

    return out


def build_df_by_scenario(
    root: Path,
    var: str,
    extra_models: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    For a given var and a root (annual_means or annual_totals),
    build dict {scenario: summary_df}.
    """
    df_by_scen: Dict[str, pd.DataFrame] = {}
    for scen in SCENARIOS:
        df_s = build_summary_for_scenario_var(root, var, scen, extra_models=extra_models)
        if df_s is not None and not df_s.empty:
            df_by_scen[scen] = df_s
    return df_by_scen

# -----------------------------------------------------------------------------
# Main: loop over variables and make plots
# -----------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Make summary plots (bands + lines) from annual means/totals CSVs."
        )
    )
    ap.add_argument(
        "--extra-model",
        dest="extra_models",
        nargs="*",
        help=(
            "Non-DGVM model columns to overlay as extra lines on top of bands "
            "(e.g. Base-Emulator_No_Carry TL-Emulator). Order determines colours."
        ),
    )
    args = ap.parse_args()

    extra_models = args.extra_models or []

    for var in VARS:
        print(f"[INFO] Processing variable: {var}")

        # Units
        mean_units, total_units = VAR_UNITS.get(
            var,
            ("arbitrary units", "arbitrary units"),
        )

        # ---- Annual means ----
        df_means_by_scen = build_df_by_scenario(CSV_MEANS_ROOT, var, extra_models=extra_models)
        if df_means_by_scen:
            fig, _ = plot_four_scenarios_bands(
                df_means_by_scen,
                var_name=var.upper(),
                units=mean_units,
                scen_order=SCENARIOS,
                extra_models=extra_models,
            )
            out_png = PLOTS_MEANS_DIR / f"{var}_annual_means.png"
            fig.savefig(out_png, dpi=300)
            plt.close(fig)
            print(f"[INFO]  Saved annual means plot to {out_png}")
        else:
            print(f"[WARN]  No data for {var} (annual means); skipping plot.")

        # ---- Annual totals ----
        df_totals_by_scen = build_df_by_scenario(CSV_TOTALS_ROOT, var, extra_models=extra_models)
        if df_totals_by_scen:
            # Annual totals (non-cumulative)
            fig2, _ = plot_four_scenarios_bands(
                df_totals_by_scen,
                var_name=var.upper(),
                units=total_units,
                scen_order=SCENARIOS,
                extra_models=extra_models,
            )
            out_png2 = PLOTS_TOTALS_DIR / f"{var}_annual_totals.png"
            fig2.savefig(out_png2, dpi=300)
            plt.close(fig2)
            print(f"[INFO]  Saved annual totals plot to {out_png2}")

            # Cumulative totals
            fig3, _ = plot_four_scenarios_bands_cumulative(
                df_totals_by_scen,
                var_name=var.upper(),
                units=total_units,
                scen_order=SCENARIOS,
                extra_models=extra_models,
            )
            out_png3 = PLOTS_CUM_DIR / f"{var}_cumulative_totals.png"
            fig3.savefig(out_png3, dpi=300)
            plt.close(fig3)
            print(f"[INFO]  Saved cumulative totals plot to {out_png3}")
        else:
            print(f"[WARN]  No data for {var} (annual totals); skipping totals/cumulative plots.")


if __name__ == "__main__":
    main()