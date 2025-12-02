#!/usr/bin/env python3
"""
plot_state_means_dgvm.py
------------------------

For each scenario, read the annual_means CSVs for four "state" variables:

  - cVeg
  - cSoil
  - cLitter
  - lai

and collapse them into:
  - TRENDY ensemble mean line ("ENSMEAN")
  - Stable-Emulator line ("Stable-Emulator")
  - DGVM uncertainty bands (min/max, 75%, 50%)

Then, for each scenario, make a 2x2 panel figure:

  Panel A: cVeg
  Panel B: cSoil
  Panel C: cLitter
  Panel D: LAI

with the *same* plotting behaviour as make_summary_plots.py
(DGVM percentile bands + ENSMEAN + Stable-Emulator, same colours).

Figures are saved under:

  .../data/analysis/CSVs/plots/states/<SCENARIO>_states.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe on cluster
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Paths & constants (mirroring your other scripts)
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.utils.tools import slurm_shard

CSV_MEANS_ROOT = PROJECT_ROOT / "data/analysis/CSVs/annual_means"

PLOTS_STATES_DIR = PROJECT_ROOT / "data/analysis/CSVs/plots/states/4_vars/with_no_carry"
PLOTS_STATES_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS: List[str] = ["S0", "S1", "S2", "S3"]

STATE_VARS: List[str] = ["cVeg", "cSoil", "cLitter", "lai"]

# Human-readable titles & units
VAR_TITLES: Dict[str, str] = {
    "cVeg": "Vegetation carbon",
    "cSoil": "Soil carbon",
    "cLitter": "Litter carbon",
    "lai": "Leaf area index",
}

VAR_UNITS: Dict[str, str] = {
    "cVeg": "kg C m$^{-2}$",
    "cSoil": "kg C m$^{-2}$",
    "cLitter": "kg C m$^{-2}$",
    "lai": "dimensionless",
}

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
    "Stable-Emulator_With_Carry",
    "Stable-Emulator_Carry_2000_2020",
    "Stable-Emulator_No_Carry",
]

# Colours: keep identical to make_summary_plots.py
color_ens_line    = "#410253"  # TRENDY ensemble
color_stable_line = "#EF8F40"  # Stable-Emulator line

color_50  = "#3A528B"  # 50% band
color_75  = "#218F8C"  # 75% band
color_100 = "#5DC762"  # 100% band

# Colours for extra overlay models (order matters)
EXTRA_MODEL_COLORS = [
    "#cc4678",  # 1st extra model
    "#f0f922",  # 2nd extra model
    "#3b528b",  # 3rd extra model
]

# -----------------------------------------------------------------------------
# Helpers: summary building (copied / adapted from make_summary_plots.py)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# NEW: Build mean across S0–S3 for each variable
# -----------------------------------------------------------------------------

def build_mean_across_scenarios(
    root: Path,
    var: str,
    scenarios: List[str],
    extra_models=None,
):
    """
    For one variable, load per-scenario DataFrames using the same summary
    builder as before, align them on year, and compute the mean across
    scenarios for each model-derived column.

    Returns a DataFrame with the same columns as build_summary_for_scenario_var().
    """
    dfs = []
    for scen in scenarios:
        df = build_summary_for_scenario_var(root, var, scen, extra_models=extra_models)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return None

    # Align years
    base = dfs[0].copy()
    for df in dfs[1:]:
        base = base.join(df, how="inner", lsuffix="", rsuffix=f"_{len(base.columns)}")

    # Now compute the mean scenario-wise
    # Extract the per-scenario columns for each metric by pattern
    out = pd.DataFrame(index=base.index)

    # Identify base column names from one scenario
    example = dfs[0]
    cols = example.columns

    for col in cols:
        # Collect column occurrences across all scenario DataFrames
        matched = [c for c in base.columns if c.startswith(col)]
        if matched:
            out[col] = base[matched].mean(axis=1)

    return out


# -----------------------------------------------------------------------------
# NEW: Plot 2x2 panels for the mean across scenarios
# -----------------------------------------------------------------------------

def plot_states_mean_across_all(
    scenarios: List[str],
    extra_models=None,
):
    """
    Generate a 2×2 panel plot of the *mean across S0–S3* for state variables.
    """

    print("[INFO] Plotting mean across all scenarios")

    # Build mean summaries
    data_by_var = {}
    for var in STATE_VARS:
        df_mean = build_mean_across_scenarios(
            CSV_MEANS_ROOT,
            var,
            scenarios,
            extra_models=extra_models,
        )
        if df_mean is not None and not df_mean.empty:
            data_by_var[var] = df_mean

    if not data_by_var:
        print("[WARN] No data for mean-across-scenarios plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    panel_letters = ["A", "B", "C", "D"]

    for i, var in enumerate(STATE_VARS):
        ax = axes[i]
        if var not in data_by_var:
            ax.text(0.5, 0.5, f"{var}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue

        df_s = data_by_var[var]
        years = df_s.index.values

        # DGVM bands
        ax.fill_between(years, df_s["DGVM_min"], df_s["DGVM_max"],
                        color=color_100, alpha=0.25, linewidth=0)
        ax.fill_between(years, df_s["DGVM_p75_low"], df_s["DGVM_p75_high"],
                        color=color_75, alpha=0.35, linewidth=0)
        ax.fill_between(years, df_s["DGVM_p50_low"], df_s["DGVM_p50_high"],
                        color=color_50, alpha=0.45, linewidth=0)

        if "ENSMEAN" in df_s:
            ax.plot(years, df_s["ENSMEAN"], color=color_ens_line, linewidth=1.8)
        if "Stable-Emulator" in df_s:
            ax.plot(years, df_s["Stable-Emulator"], color=color_stable_line, linewidth=1.8)

        # Extra models
        if extra_models:
            for j, name in enumerate(extra_models):
                if name in df_s.columns:
                    col = EXTRA_MODEL_COLORS[j] if j < len(EXTRA_MODEL_COLORS) else "k"
                    ax.plot(years, df_s[name], color=col, linewidth=1.5)

        # Labels
        ax.text(0.02, 0.95, panel_letters[i],
                transform=ax.transAxes, fontsize=13, fontweight="bold")

        title = VAR_TITLES.get(var, var)
        units = VAR_UNITS.get(var, "")
        ax.set_title(f"{title}", fontsize=11, fontweight="bold")

        ax.grid(True, alpha=0.3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        if i in (0, 2):
            ax.set_ylabel(f"Annual mean ({units})", fontsize=10)

    fig.text(0.5, 0.06, "Year", ha="center", fontsize=12)
    fig.suptitle(
        "Annual Mean Carbon States and LAI — Mean Across S0–S3",
        fontsize=14, fontweight="bold", y=0.97,
    )

    # Legend same as before
    legend_handles = [
        Patch(facecolor=color_50,  alpha=0.45, label="DGVM 50% band"),
        Patch(facecolor=color_75,  alpha=0.35, label="DGVM 75% band"),
        Patch(facecolor=color_100, alpha=0.25, label="DGVM 100% band"),
        Line2D([0], [0], color=color_ens_line, linewidth=1.8, label="TRENDY Ensemble Mean"),
        Line2D([0], [0], color=color_stable_line, linewidth=1.8, label="Stable-Emulator"),
    ]
    if extra_models:
        for j, name in enumerate(extra_models):
            col = EXTRA_MODEL_COLORS[j] if j < len(EXTRA_MODEL_COLORS) else "k"
            legend_handles.append(
                Line2D([0], [0], color=col, linewidth=1.5, label=name)
            )

    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=[0.06, 0.10, 0.98, 0.95])

    out_path = PLOTS_STATES_DIR / "all_scenarios_mean_states.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Wrote {out_path}")

def build_summary_for_scenario_var(
    root: Path,
    var: str,
    scenario: str,
    extra_models: list[str] | None = None,
) -> pd.DataFrame | None:
    """
    Read CSV for (scenario, var) and collapse model columns into:

      ENSMEAN, Stable-Emulator,
      DGVM_min/max, DGVM_p50_low/high, DGVM_p75_low/high,
      plus any requested extra_models columns.

    Returns DataFrame indexed by year, or None if file missing / no DGVMs.
    """
    csv_path = root / scenario / f"{var}_annual_means.csv"
    if not csv_path.is_file():
        print(f"[WARN] Missing CSV {csv_path}")
        return None

    df = pd.read_csv(csv_path, index_col=0)
    if df.empty:
        print(f"[WARN] Empty CSV {csv_path}")
        return None

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

    # Extra overlay models (if requested)
    if extra_models:
        for name in extra_models:
            if name in df.columns:
                out[name] = df[name]
            else:
                print(f"[WARN] extra model '{name}' not found in {csv_path.name}")

    return out


# -----------------------------------------------------------------------------
# Plot one scenario: 2x2 panels (cVeg, cSoil, cLitter, LAI)
# -----------------------------------------------------------------------------

def plot_states_for_scenario(
    scenario: str,
    extra_models: list[str] | None = None,
) -> None:
    """
    For one scenario, create a 2x2 panel figure with DGVM bands and
    ENSMEAN + Stable-Emulator (+ optional extra models) for:

      A: cVeg
      B: cSoil
      C: cLitter
      D: lai

    using the same DGVM percentile plotting style as make_summary_plots.py.
    """
    print(f"[INFO] Plotting state variables for scenario {scenario}")

    # Build summaries per variable
    data_by_var: Dict[str, pd.DataFrame] = {}
    for var in STATE_VARS:
        df = build_summary_for_scenario_var(
            CSV_MEANS_ROOT,
            var,
            scenario,
            extra_models=extra_models,
        )
        if df is not None and not df.empty:
            data_by_var[var] = df

    if not data_by_var:
        print(f"[WARN] No state data found for scenario {scenario}; skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
    axes = axes.flatten()

    panel_letters = ["A", "B", "C", "D"]

    for i, var in enumerate(STATE_VARS):
        ax = axes[i]
        letter = panel_letters[i]

        if var not in data_by_var:
            ax.text(
                0.5,
                0.5,
                f"{var}\n(no data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        df_s = data_by_var[var]
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

        # ---- Extra emulator/model lines (if requested) ----
        if extra_models:
            for j, model_name in enumerate(extra_models):
                if model_name not in df_s.columns:
                    continue
                color = EXTRA_MODEL_COLORS[j] if j < len(EXTRA_MODEL_COLORS) else "k"
                ax.plot(years, df_s[model_name], color=color, linewidth=1.5)

        # Panel letter
        ax.text(
            0.02,
            0.95,
            letter,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
        )

        # Panel title (include units)
        title = VAR_TITLES.get(var, var)
        units = VAR_UNITS.get(var, "units")
        ax.set_title(f"{title} ({units})", fontsize=11, fontweight="bold")

        ax.grid(True, alpha=0.3)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Small y-label on left panels only
        if i in (0, 2):
            ax.set_ylabel("Annual mean", fontsize=10)

    # Shared x-label
    fig.text(0.5, 0.06, "Year", ha="center", fontsize=12)

    # Suptitle
    fig.suptitle(
        f"Annual Mean Carbon States and LAI — Scenario {scenario}",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    # Shared legend (same style as other scripts, with extras)
    legend_handles = [
        Patch(facecolor=color_50,  alpha=0.45, label="DGVM 50% band"),
        Patch(facecolor=color_75,  alpha=0.35, label="DGVM 75% band"),
        Patch(facecolor=color_100, alpha=0.25, label="DGVM 100% band"),
        Line2D(
            [0], [0],
            color=color_ens_line,
            linewidth=1.8,
            label="TRENDY Ensemble Mean",
        ),
        Line2D(
            [0], [0],
            color=color_stable_line,
            linewidth=1.8,
            label="Stable-Emulator",
        ),
    ]

    # Extra models with nice labels
    if extra_models:
        for j, model_name in enumerate(extra_models):
            color = EXTRA_MODEL_COLORS[j] if j < len(EXTRA_MODEL_COLORS) else "k"
            if model_name == "Base-Emulator_No_Carry":
                label = "Base-Emulator (Non-Autoregressive)"
            elif model_name == "Stable-Emulator_No_Carry":
                label = "Stable-Emulator (Non-Autoregressive)"
            else:
                label = model_name
            legend_handles.append(
                Line2D(
                    [0], [0],
                    color=color,
                    linewidth=1.5,
                    label=label,
                )
            )

    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=[0.06, 0.10, 0.98, 0.95])

    out_path = PLOTS_STATES_DIR / f"{scenario}_states.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Wrote {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Plot annual mean carbon states (cVeg, cSoil, cLitter, LAI) "
            "per scenario, using DGVM percentile bands + ENSMEAN + Stable-Emulator."
        )
    )
    ap.add_argument(
        "--scenario",
        choices=SCENARIOS,
        nargs="*",
        help="Optional subset of scenarios to plot (default: all).",
    )
    ap.add_argument(
        "--extra-model",
        dest="extra_models",
        nargs="*",
        help=(
            "Non-DGVM models to overlay as extra lines "
            "(e.g. Base-Emulator_No_Carry TL-Emulator). "
            "Order determines colours."
        ),
    )
    args = ap.parse_args()

    scenarios_to_use = args.scenario if args.scenario else SCENARIOS
    extra_models = args.extra_models or []

    # Shard scenarios across SLURM array (or return all locally)
    all_tasks: List[str] = list(scenarios_to_use)
    scenarios_for_this_task = slurm_shard(all_tasks)

    print(f"[INFO] This task will plot scenarios: {scenarios_for_this_task}")
    if extra_models:
        print(f"[INFO] Extra models to overlay: {extra_models}")

    for scen in scenarios_for_this_task:
        plot_states_for_scenario(scen, extra_models=extra_models)
        # --- NEW: produce the mean-across-scenarios plot ---
        plot_states_mean_across_all(scenarios_to_use, extra_models=extra_models)


if __name__ == "__main__":
    main()