#!/usr/bin/env python3
"""
make_cumulative_boxplots.py
---------------------------

For each variable, read annual_totals CSVs (per scenario/model), compute
FINAL CUMULATIVE TOTALS across time, and plot for each scenario:

  - DGVM distribution as a boxplot:
      * median: DGVM median
      * box: 25–75% (central 50% range)
      * whiskers: 12.5–87.5% (central 75% range)
  - Individual DGVM values as grey circles (thin edge, white fill)
  - TRENDY Ensemble Mean as solid black triangles
  - Stable-Emulator as solid circles in its usual colour
  - Extra models (e.g. Base-Emulator_No_Carry, TL-Emulator) as solid
    circles offset to the right, using the same colours as the line graphs.

Output:

  /.../plots/with_base_and_stable/cumulative_boxplots/<var>_cumulative_boxplots.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe on cluster
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Paths & constants (mirroring make_summary_plots.py)
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")

CSV_TOTALS_ROOT = PROJECT_ROOT / "data/analysis/CSVs/annual_totals"

PLOTS_ROOT = PROJECT_ROOT / "data/analysis/CSVs/plots/boxplots"
PLOTS_BOX_DIR = PLOTS_ROOT / "cumulative_boxplots"
PLOTS_BOX_DIR.mkdir(parents=True, exist_ok=True)

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
    "ELM",
    "JSBACH",
    "ORCHIDEE",
    "SDGVM",
    "VISIT",
    "VISIT-UT",
]

# Column names in CSVs that should map to plot lines / markers
ENS_COLUMN = "TRENDY-Ensemble-Mean"
STABLE_CANDIDATES = [
    "Stable-Emulator_With_Carry",
]

# Colours: match make_summary_plots.py
color_ens_line    = "#410253"  # line colour (not used in marker, marker is black)
color_stable_line = "#EF8F40"  # Stable-Emulator line/marker

# Extra (non-DGVM) overlay models (order determines colour)
DEFAULT_EXTRA_COLORS = [
    "#cc4678",  # 1st extra model (e.g. Base-Emulator_No_Carry)
    "#f0f922",  # 2nd extra model (e.g. TL-Emulator)
    "#3b528b",  # any 3rd extra model
]

# Units for pretty labels (same as your summary script)
VAR_UNITS = {
    "gpp":       ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "npp":       ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "ra":        ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "rh":        ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "nee":       ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "nbp":       ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "fLuc":      ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "fFire":     ("kg m$^{-2}$ s$^{-1}$", "Gt C yr$^{-1}$"),
    "evapotrans":("kg m$^{-2}$ s$^{-1}$", "Gt yr$^{-1}$"),
    "cVeg":      ("kg m$^{-2}$", "Gt"),
    "cSoil":     ("kg m$^{-2}$", "Gt"),
    "cLitter":   ("kg m$^{-2}$", "Gt"),
    "cTotal":    ("kg m$^{-2}$", "Gt"),
    "mrro":      ("kg m$^{-2}$ s$^{-1}$", "Gt yr$^{-1}$"),
    "mrso":      ("kg m$^{-2}$", "Gt"),
    "lai":       ("-", "-"),
}

# -----------------------------------------------------------------------------
# Helpers to read cumulative totals
# -----------------------------------------------------------------------------

def compute_cumulative_for_scenario(
    root: Path,
    var: str,
    scenario: str,
    extra_models: Optional[List[str]] = None,
) -> Optional[Dict[str, object]]:
    """
    For one (var, scenario), read the annual_totals CSV and compute:

      - cumulative DGVM totals (Series indexed by DGVM name)
      - cumulative ENSMEAN total (float or np.nan)
      - cumulative Stable-Emulator total (float or np.nan)
      - cumulative extra model totals (dict name -> float)

    Returns a dict or None if no usable data.
    """
    if extra_models is None:
        extra_models = []

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

    # DGVMs
    dgvm_cols = [m for m in DGVM_MODELS if m in df.columns]
    if not dgvm_cols:
        print(f"[WARN] No DGVM columns for {var} {scenario} in {csv_path.name}")
        return None

    d_dgvm = df[dgvm_cols]

    # Final cumulative total per DGVM: sum over all years
    dgvm_cum = d_dgvm.sum(axis=0)  # Series, index=model_name

    # ENSMEAN
    ens_cum = np.nan
    if ENS_COLUMN in df.columns:
        ens_cum = float(df[ENS_COLUMN].sum())
    else:
        print(f"[WARN] {ENS_COLUMN} not found in {csv_path.name}")

    # Stable-Emulator
    stable_cum = np.nan
    stable_col = None
    for cand in STABLE_CANDIDATES:
        if cand in df.columns:
            stable_col = cand
            break
    if stable_col is not None:
        stable_cum = float(df[stable_col].sum())
    else:
        print(f"[WARN] No Stable-Emulator column found in {csv_path.name}")

    # Extra models
    extras_cum: Dict[str, float] = {}
    for name in extra_models:
        if name in df.columns:
            extras_cum[name] = float(df[name].sum())
        else:
            print(f"[WARN] extra model '{name}' not found in {csv_path.name}")
            extras_cum[name] = np.nan

    return {
        "dgvm_cum": dgvm_cum,
        "ens_cum": ens_cum,
        "stable_cum": stable_cum,
        "extras_cum": extras_cum,
    }


# -----------------------------------------------------------------------------
# Plot cumulative boxplots per variable
# -----------------------------------------------------------------------------

def plot_cumulative_boxplots_for_var(
    var: str,
    extra_models: Optional[List[str]] = None,
    extra_colors: Optional[List[str]] = None,
) -> None:
    """
    For one variable, make a boxplot of final cumulative totals per scenario.

    - x-axis: scenarios S0..S3
    - box/whiskers: DGVM distribution
    - DGVM points: grey circles
    - ENSMEAN: black triangles (offset to the right)
    - Stable-Emulator: solid circles in colour 'color_stable_line'
    - Extra models: solid circles in DEFAULT_EXTRA_COLORS
    """
    if extra_models is None:
        extra_models = []
    if extra_colors is None:
        extra_colors = DEFAULT_EXTRA_COLORS

    print(f"[INFO] Building cumulative boxplots for var={var}")

    # Collect cumulative stats per scenario
    scen_data: Dict[str, Dict[str, object]] = {}
    for scen in SCENARIOS:
        d = compute_cumulative_for_scenario(
            CSV_TOTALS_ROOT,
            var,
            scen,
            extra_models=extra_models,
        )
        if d is not None:
            scen_data[scen] = d

    if not scen_data:
        print(f"[WARN] No data for {var}; skipping boxplot.")
        return

    # Prepare boxplot stats and point arrays
    positions = []
    stats = []  # list of dicts for ax.bxp
    dv_points_x = []
    dv_points_y = []

    ens_x = []
    ens_y = []

    stable_x = []
    stable_y = []

    extras_x: Dict[str, List[float]] = {name: [] for name in extra_models}
    extras_y: Dict[str, List[float]] = {name: [] for name in extra_models}

    # Horizontal layout positions (integer 0..N-1)
    base_positions = []
    for i, scen in enumerate(SCENARIOS):
        if scen not in scen_data:
            continue

        base_positions.append(i)
        positions.append(i)

        dgvm_cum: pd.Series = scen_data[scen]["dgvm_cum"]  # type: ignore
        vals = dgvm_cum.dropna().values
        if len(vals) == 0:
            print(f"[WARN] No DGVM cumulative values for {var} {scen}; skipping in boxplot.")
            continue

        # Percentiles
        q1 = np.quantile(vals, 0.25)
        q3 = np.quantile(vals, 0.75)
        med = np.median(vals)
        whislo = np.quantile(vals, 0.125)
        whishi = np.quantile(vals, 0.875)

        stats.append({
            "med": med,
            "q1": q1,
            "q3": q3,
            "whislo": whislo,
            "whishi": whishi,
            "fliers": [],
        })

        # DGVM points at this x position
        dv_points_x.extend([i] * len(vals))
        dv_points_y.extend(vals.tolist())

        # ENSMEAN
        ens_val = float(scen_data[scen]["ens_cum"])  # type: ignore
        ens_x.append(i)
        ens_y.append(ens_val)

        # Stable-Emulator
        stable_val = float(scen_data[scen]["stable_cum"])  # type: ignore
        stable_x.append(i)
        stable_y.append(stable_val)

        # Extras
        extras_cum: Dict[str, float] = scen_data[scen]["extras_cum"]  # type: ignore
        for name in extra_models:
            extras_x[name].append(i)
            extras_y[name].append(float(extras_cum.get(name, np.nan)))

    if not stats:
        print(f"[WARN] No valid box stats for {var}; skipping.")
        return

    # Make figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Boxplots (DGVM distribution)
    bxp = ax.bxp(
        stats,
        positions=positions,
        widths=0.2,
        showfliers=False,
        patch_artist=True,
    )

    # Style boxes: light fill, darker edges (can tweak if you like)
    for box in bxp['boxes']:
        box.set(facecolor="none", edgecolor="0.4", linewidth=1.2)
    for whisker in bxp['whiskers']:
        whisker.set(color="0.4", linewidth=1.0)
    for cap in bxp['caps']:
        cap.set(color="0.4", linewidth=1.0)
    for median in bxp['medians']:
        median.set(color="0.0", linewidth=1.4)  # black median line

    # DGVM points (grey circles, thin edge, white centre)
    dgvm_scatter = ax.scatter(
        dv_points_x,
        dv_points_y,
        s=25,
        facecolor="white",
        edgecolor="0.6",
        linewidth=0.7,
        zorder=3,
        label="DGVMs",
    )

    # Offsets for model markers to the right of each scenario
    # box width=0.2, so 0.25 is just to the right; then 0.05 steps.
    base_offset = 0.25
    step_offset = 0.05

    # ENSMEAN: black triangle
    ens_positions = np.array(ens_x, dtype=float) + base_offset
    ens_scatter = ax.scatter(
        ens_positions,
        ens_y,
        s=45,
        marker="^",
        facecolor="black",
        edgecolor="black",
        linewidth=0.7,
        zorder=4,
        label="TRENDY Ensemble Mean",
    )

    # Stable-Emulator: solid circle in its usual colour
    stable_positions = np.array(stable_x, dtype=float) + base_offset + step_offset
    stable_scatter = ax.scatter(
        stable_positions,
        stable_y,
        s=40,
        marker="o",
        facecolor=color_stable_line,
        edgecolor="black",
        linewidth=0.7,
        zorder=4,
        label="Stable-Emulator",
    )

    # Extra models: solid circles, coloured as in line plots
    extra_scatters: Dict[str, Line2D] = {}
    for k, name in enumerate(extra_models):
        color = extra_colors[k] if k < len(extra_colors) else "k"
        xpos = np.array(extras_x[name], dtype=float) + base_offset + step_offset * (k + 2)
        extra_scatters[name] = ax.scatter(
            xpos,
            extras_y[name],
            s=35,
            marker="o",
            facecolor=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
            label=name,
        )

    # Axes / labels
    xticks = np.arange(len(SCENARIOS))
    ax.set_xticks(xticks)
    ax.set_xticklabels(SCENARIOS)
    ax.set_xlabel("Scenario", fontsize=12)

    _mean_units, total_units = VAR_UNITS.get(var, ("units", "units"))
    ax.set_ylabel(f"Cumulative {var.upper()} ({total_units})", fontsize=12)

    ax.grid(True, axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_title(
        f"Cumulative {var.upper()} across DGVMs and models (1901–2023)",
        fontsize=14,
        fontweight="bold",
    )

    # Legend: one entry for each element
    handles = [
        dgvm_scatter,
        ens_scatter,
        stable_scatter,
    ] + [extra_scatters[name] for name in extra_models]

    labels = [h.get_label() for h in handles]
    ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()
    out_path = PLOTS_BOX_DIR / f"{var}_cumulative_boxplots.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO]  Saved cumulative boxplot to {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Make cumulative-total boxplots per variable/scenario from "
            "annual_totals CSVs."
        )
    )
    ap.add_argument(
        "--extra-model",
        dest="extra_models",
        nargs="*",
        help=(
            "Non-DGVM model columns to include as extra markers "
            "(e.g. Base-Emulator_No_Carry TL-Emulator). Order "
            "determines colours (see DEFAULT_EXTRA_COLORS)."
        ),
    )
    ap.add_argument(
        "--var",
        choices=VARS,
        nargs="*",
        help="Optional subset of variables (default: all).",
    )
    args = ap.parse_args()

    extra_models = args.extra_models or []
    vars_to_use = args.var if args.var else VARS

    for var in vars_to_use:
        plot_cumulative_boxplots_for_var(var, extra_models=extra_models)


if __name__ == "__main__":
    main()