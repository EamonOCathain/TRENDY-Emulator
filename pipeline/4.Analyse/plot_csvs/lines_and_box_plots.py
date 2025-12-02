#!/usr/bin/env python3
"""
make_summary_line_box_plots.py
------------------------------

Create combined line+box plots for each variable:

  - 2x2 grid over scenarios (S0,S1,S2,S3)
  - For each scenario:
      * Left: DGVM bands + ENSMEAN + Stable-Emulator + optional extras (lines)
      * Right: final cumulative totals boxplot (DGVM distribution + ENSMEAN
               circle + Stable-Emulator + extra models as coloured circles)

Input CSVs:

  annual_means:
    PROJECT_ROOT/data/analysis/CSVs/annual_means/<SCENARIO>/<var>_annual_means.csv

  annual_totals (same filename pattern, different directory):
    PROJECT_ROOT/data/analysis/CSVs/annual_totals/<SCENARIO>/<var>_annual_means.csv

Outputs:

  PROJECT_ROOT/data/analysis/CSVs/plots/lines_and_boxes/
      annual_means_with_boxes/<var>_annual_means_with_boxes.png
      annual_totals_with_boxes/<var>_annual_totals_with_boxes.png
      cumulative_totals_with_boxes/<var>_cumulative_totals_with_boxes.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")

CSV_MEANS_ROOT = PROJECT_ROOT / "data/analysis/CSVs/annual_means"
CSV_TOTALS_ROOT = PROJECT_ROOT / "data/analysis/CSVs/annual_totals"

PLOTS_ROOT = PROJECT_ROOT / "data/analysis/CSVs/plots/isolating_drivers"
PLOTS_MEANS_BOX_DIR = PLOTS_ROOT / "annual_means_with_boxes"
PLOTS_TOTALS_BOX_DIR = PLOTS_ROOT / "annual_totals_with_boxes"
PLOTS_CUM_BOX_DIR = PLOTS_ROOT / "cumulative_totals_with_boxes"

for d in (PLOTS_MEANS_BOX_DIR, PLOTS_TOTALS_BOX_DIR, PLOTS_CUM_BOX_DIR):
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

# Units (annual means, annual/cumulative totals)
# Second element is now "total" units (no per-year).
VAR_UNITS = {
    "gpp": ("kg m$^{-2}$ s$^{-1}$", "Gt C"),
    "npp": ("kg m$^{-2}$ s$^{-1}$", "Gt C"),
    "ra": ("kg m$^{-2}$ s$^{-1}$", "Gt C"),
    "rh": ("kg m$^{-2}$ s$^{-1}$", "Gt C"),
    "nee": ("kg m$^{-2}$ s$^{-1}$", "Gt C"),
    "nbp": ("kg m$^{-2}$ s$^{-1}$", "Gt C"),
    "fLuc": ("kg m$^{-2}$ s$^{-1}$", "Gt C"),
    "fFire": ("kg m$^{-2}$ s$^{-1}$", "Gt C"),
    "evapotrans": ("kg m$^{-2}$ s$^{-1}$", "Gt"),  # total water mass
    "cVeg": ("kg m$^{-2}$", "Gt"),
    "cSoil": ("kg m$^{-2}$", "Gt"),
    "cLitter": ("kg m$^{-2}$", "Gt"),
    "cTotal": ("kg m$^{-2}$", "Gt"),
    "mrro": ("kg m$^{-2}$ s$^{-1}$", "Gt"),
    "mrso": ("kg m$^{-2}$", "Gt"),
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

    NOTE: both annual_means and "annual_totals" live in files named:
          <var>_annual_means.csv, but under different root directories.
    """
    csv_path = root / scenario / f"{var}_annual_means.csv"
    if not csv_path.is_file():
        print(f"[WARN] Missing CSV {csv_path}")
        return None

    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = "year"
    df.columns = df.columns.astype(str).str.strip()

    dgvm_cols = [m for m in DGVM_MODELS if m in df.columns]
    if not dgvm_cols:
        print(f"[WARN] No DGVM columns for {var} {scenario} in {csv_path.name}")
        return None

    d_dgvm = df[dgvm_cols]
    out = pd.DataFrame(index=df.index)

    # Ensemble mean
    if ENS_COLUMN in df.columns:
        out["ENSMEAN"] = df[ENS_COLUMN]
    else:
        print(f"[WARN] {ENS_COLUMN} not found in {csv_path.name}")

    # Stable-Emulator
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
    out["DGVM_p50_low"] = d_dgvm.quantile(0.25, axis=1)
    out["DGVM_p50_high"] = d_dgvm.quantile(0.75, axis=1)
    out["DGVM_p75_low"] = d_dgvm.quantile(0.125, axis=1)
    out["DGVM_p75_high"] = d_dgvm.quantile(0.875, axis=1)

    # Extra overlay models (lines)
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
    scenarios: List[str],
    extra_models: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Return {scenario: summary_df} for a given variable and list of scenarios."""
    df_by_scen: Dict[str, pd.DataFrame] = {}
    for scen in scenarios:
        df_s = build_summary_for_scenario_var(root, var, scen, extra_models=extra_models)
        if df_s is not None and not df_s.empty:
            df_by_scen[scen] = df_s
    return df_by_scen

# -----------------------------------------------------------------------------
# Line panel helper
# -----------------------------------------------------------------------------

def _plot_scenario_bands_on_axis(
    ax: plt.Axes,
    df_s: pd.DataFrame,
    cumulative: bool,
    var_name: str,
    line_units: str,
    title: str,
    show_ylabel: bool,
    extra_models: Optional[List[str]] = None,
    extra_colors: Optional[List[str]] = None,
) -> None:
    if extra_models is None:
        extra_models = []
    if extra_colors is None:
        extra_colors = DEFAULT_EXTRA_COLORS

    df_plot = df_s.sort_index().copy()
    years = df_plot.index.values

    if cumulative:
        cols_to_cumsum = [
            "ENSMEAN",
            "Stable-Emulator",
            "DGVM_p50_low",
            "DGVM_p50_high",
            "DGVM_p75_low",
            "DGVM_p75_high",
            "DGVM_min",
            "DGVM_max",
        ] + extra_models
        cols_to_cumsum = [c for c in cols_to_cumsum if c in df_plot.columns]
        for c in cols_to_cumsum:
            df_plot[c] = df_plot[c].cumsum()

    # Bands
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

    ax.set_title(title, fontweight="bold", fontsize=13, pad=10)
    ax.grid(True, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if show_ylabel:
        if cumulative:
            label = f"Cumulative {var_name} ({line_units})"
        else:
            label = f"{var_name} ({line_units})"
        ax.set_ylabel(label, fontsize=12)

# -----------------------------------------------------------------------------
# Box panel helpers
# -----------------------------------------------------------------------------

def compute_cumulative_stats_for_scenario(
    var: str,
    scenario: str,
    extra_models: Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    From annual_totals CSV (in CSV_TOTALS_ROOT), compute FINAL cumulative totals
    for DGVMs, ENSMEAN, Stable-Emulator, and extra models.
    """
    csv_path = CSV_TOTALS_ROOT / scenario / f"{var}_annual_means.csv"
    if not csv_path.is_file():
        print(f"[WARN] Missing totals CSV for boxplot: {csv_path}")
        return None

    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = "year"
    df.columns = df.columns.astype(str).str.strip()

    dgvm_cols = [m for m in DGVM_MODELS if m in df.columns]
    if not dgvm_cols:
        print(f"[WARN] No DGVM columns for boxplot {var} {scenario} in {csv_path.name}")
        return None

    d_dgvm_cum = df[dgvm_cols].cumsum()
    final_dgvm = d_dgvm_cum.iloc[-1].dropna()
    vals = final_dgvm.values
    if vals.size == 0:
        print(f"[WARN] All DGVM cumulative values NaN for {var} {scenario}")
        return None

    q25 = np.quantile(vals, 0.25)
    q75 = np.quantile(vals, 0.75)
    q125 = np.quantile(vals, 0.125)
    q875 = np.quantile(vals, 0.875)
    median = np.median(vals)

    # ENSMEAN cumulative
    ens_val = None
    if ENS_COLUMN in df.columns:
        ens_cum = df[ENS_COLUMN].cumsum()
        ens_val = float(ens_cum.iloc[-1])

    # Stable-Emulator cumulative
    stable_val = None
    for cand in STABLE_CANDIDATES:
        if cand in df.columns:
            s_cum = df[cand].cumsum()
            stable_val = float(s_cum.iloc[-1])
            break

    extras_vals: Dict[str, float] = {}
    if extra_models:
        for name in extra_models:
            if name in df.columns:
                ecum = df[name].cumsum()
                extras_vals[name] = float(ecum.iloc[-1])
            else:
                print(f"[WARN] extra model '{name}' not found in totals CSV {csv_path.name}")

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


def plot_box_for_scenario(
    ax: plt.Axes,
    stats: Dict,
    box_title: str,
    box_units: str,
    extra_models: Optional[List[str]] = None,
    extra_colors: Optional[List[str]] = None,
    box_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Draw DGVM distribution box + ENSMEAN circle + Stable-Emulator + extra models
    on one axis. The whole cluster is shifted slightly left in the subplot.
    Units are shown as a right-hand y-axis label.
    """
    if extra_models is None:
        extra_models = []
    if extra_colors is None:
        extra_colors = DEFAULT_EXTRA_COLORS

    # Shift cluster slightly left inside the subplot
    x_base = -0.15
    box_width = 0.25
    dx_extra = 0.30

    vals = stats["dgvm_vals"]
    q25 = stats["q25"]
    q75 = stats["q75"]
    q125 = stats["q125"]
    q875 = stats["q875"]
    median = stats["median"]

    # DGVM points
    rng = np.random.default_rng(42)
    jitter = rng.normal(loc=0.0, scale=0.03, size=vals.size)
    ax.scatter(
        x_base + jitter,
        vals,
        facecolors="white",
        edgecolors="0.6",
        linewidths=0.8,
        zorder=3,
    )

    # Box (50th percentile)
    rect = Rectangle(
        (x_base - box_width / 2.0, q25),
        box_width,
        q75 - q25,
        facecolor="none",
        edgecolor="k",
        linewidth=1.2,
        zorder=4,
    )
    ax.add_patch(rect)

    # Whiskers (75th percentile range)
    ax.vlines(
        x_base,
        q125,
        q875,
        colors="k",
        linewidth=1.0,
        zorder=4,
    )

    # Median line
    ax.hlines(
        median,
        x_base - box_width / 2.0,
        x_base + box_width / 2.0,
        colors="k",
        linewidth=1.4,
        zorder=5,
    )

    # ENSMEAN as coloured circle (same colour as line)
    if stats["ensmean"] is not None:
        ax.scatter(
            [x_base],
            [stats["ensmean"]],
            marker="o",
            s=45,
            facecolors=COLOR_ENS,
            edgecolors=COLOR_ENS,
            linewidths=0.7,
            zorder=6,
        )

    # Stable-Emulator as coloured circle
    if stats["stable"] is not None:
        x_stable = x_base + dx_extra
        ax.scatter(
            [x_stable],
            [stats["stable"]],
            marker="o",
            s=40,
            facecolors=COLOR_STABLE,
            edgecolors="black",
            linewidths=0.7,
            zorder=6,
        )

    # Extra models as coloured circles further to the right
    for i, name in enumerate(extra_models):
        if name not in stats["extras"]:
            continue
        x = x_base + dx_extra * (i + 2)
        color = extra_colors[i] if i < len(extra_colors) else "k"
        ax.scatter(
            [x],
            [stats["extras"][name]],
            marker="o",
            s=40,
            facecolors=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=6,
        )

    # X-limits so cluster is left-of-centre
    max_offset = dx_extra * (len(extra_models) + 2)
    ax.set_xlim(x_base - 0.5, x_base + max_offset + 0.2)
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3)

    # Title without units
    ax.set_title(box_title, fontsize=12, fontweight="bold", pad=10)

    # Units as right-hand y-axis label
    if box_units:
        ax.set_ylabel(box_units, fontsize=10)
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()

    # Shared y-limits if provided
    if box_ylim is not None:
        ax.set_ylim(*box_ylim)

# -----------------------------------------------------------------------------
# Combined line+box figure
# -----------------------------------------------------------------------------

def make_combined_line_box_plot(
    var: str,
    scenarios: List[str],
    df_line_by_scen: Dict[str, pd.DataFrame],
    metric_label: str,
    line_units: str,
    box_units: str,
    out_path: Path,
    extra_models: Optional[List[str]] = None,
    extra_colors: Optional[List[str]] = None,
    cumulative_lines: bool = False,
    shared_ylims: bool = True,
) -> None:
    """
    Create combined line+box figure:

      - 2 rows, 4 columns (line + box per scenario, S0..S3)
      - Line panels wider, box panels narrower

    Only scenarios in the provided 'scenarios' list are plotted.
    """
    if extra_models is None:
        extra_models = []
    if extra_colors is None:
        extra_colors = DEFAULT_EXTRA_COLORS

    # Precompute cumulative box stats (from annual totals)
    stats_by_scen: Dict[str, Dict] = {}
    for scen in scenarios:
        stats = compute_cumulative_stats_for_scenario(
            var, scen, extra_models=extra_models
        )
        if stats is not None:
            stats_by_scen[scen] = stats

    if not df_line_by_scen and not stats_by_scen:
        print(f"[WARN] No data for combined plot {var} ({metric_label}); skipping.")
        return

    # ---------------- Shared y-limits for line panels ----------------
    line_ylim: Optional[Tuple[float, float]] = None
    if shared_ylims and df_line_by_scen:
        y_min = np.inf
        y_max = -np.inf

        for df_s in df_line_by_scen.values():
            df_plot = df_s.sort_index().copy()

            cols = [
                "DGVM_min",
                "DGVM_max",
                "DGVM_p75_low",
                "DGVM_p75_high",
                "DGVM_p50_low",
                "DGVM_p50_high",
                "ENSMEAN",
                "Stable-Emulator",
            ] + list(extra_models or [])

            cols = [c for c in cols if c in df_plot.columns]
            if not cols:
                continue

            if cumulative_lines:
                for c in cols:
                    df_plot[c] = df_plot[c].cumsum()

            arr = df_plot[cols].to_numpy().ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue

            y_min = min(y_min, arr.min())
            y_max = max(y_max, arr.max())

        if np.isfinite(y_min) and np.isfinite(y_max):
            line_ylim = (y_min, y_max)

    # ---------------- Shared y-limits for box panels ------------------
    box_ylim: Optional[Tuple[float, float]] = None
    if shared_ylims and stats_by_scen:
        b_min = np.inf
        b_max = -np.inf

        for stats in stats_by_scen.values():
            vals_list = [stats["dgvm_vals"]]

            if stats.get("ensmean") is not None:
                vals_list.append(np.array([stats["ensmean"]]))
            if stats.get("stable") is not None:
                vals_list.append(np.array([stats["stable"]]))
            if stats.get("extras"):
                vals_list.append(np.array(list(stats["extras"].values())))

            arr = np.concatenate(vals_list)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue

            b_min = min(b_min, arr.min())
            b_max = max(b_max, arr.max())

        if np.isfinite(b_min) and np.isfinite(b_max):
            # Add small padding so outermost DGVMs are not clipped
            padding = 0.02 * (b_max - b_min if b_max > b_min else 1.0)
            box_ylim = (b_min - padding, b_max + padding)

    # ---------------- Figure & subplots ----------------
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        width_ratios=[3.5, 0.7, 3.5, 0.7],
        height_ratios=[1.0, 1.0],
        hspace=0.40,
        wspace=0.30,
    )

    # NEW: one letter per line–box pair (scenario)
    panel_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for idx, scen in enumerate(scenarios):
        if idx >= 4:
            # Hard limit: current layout assumes at most 4 scenarios
            break

        row = idx // 2
        col_pair = (idx % 2) * 2

        ax_line = fig.add_subplot(gs[row, col_pair])
        ax_box = fig.add_subplot(gs[row, col_pair + 1])

        # ---- NEW: add single letter on the line panel only ----
        letter = panel_letters[idx]
        ax_line.text(
            0.02,
            0.96,
            letter,
            transform=ax_line.transAxes,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="top",
        )

        # ---- Line panel ----
        if scen in df_line_by_scen:
            df_s = df_line_by_scen[scen]

            if metric_label == "Annual means":
                line_title = f"Annual Mean {var.upper()} {scen}"
            elif metric_label == "Annual totals":
                line_title = f"Annual Total {var.upper()} {scen}"
            elif metric_label == "Cumulative totals":
                line_title = f"Cumulative Total {var.upper()} {scen}"
            else:
                line_title = f"{metric_label} {var.upper()} {scen}"

            _plot_scenario_bands_on_axis(
                ax=ax_line,
                df_s=df_s,
                cumulative=cumulative_lines,
                var_name=var.upper(),
                line_units=line_units,
                title=line_title,
                show_ylabel=(col_pair == 0),
                extra_models=extra_models,
                extra_colors=extra_colors,
            )

            if line_ylim is not None:
                ax_line.set_ylim(*line_ylim)

            if row == 1:
                ax_line.set_xlabel("Year", fontsize=12)
        else:
            ax_line.set_axis_off()

        # ---- Box panel ----
        if scen in stats_by_scen:
            box_title = f"Total {var.upper()} {scen}"
            plot_box_for_scenario(
                ax=ax_box,
                stats=stats_by_scen[scen],
                box_title=box_title,
                box_units=box_units,
                extra_models=extra_models,
                extra_colors=extra_colors,
                box_ylim=box_ylim,
            )
        else:
            ax_box.text(
                0.5,
                0.5,
                f"{scen}\n(no cumulative data)",
                ha="center",
                va="center",
                transform=ax_box.transAxes,
            )
            ax_box.set_axis_off()

    # ----- Legend (no suptitle, no global 'Year') -----
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
            marker="o",
            linestyle="None",
            markersize=6,
            color=COLOR_ENS,
            markeredgecolor="black",
            label="TRENDY Ensemble Mean",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=6,
            color=COLOR_STABLE,
            markeredgecolor="black",
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
                marker="o",
                linestyle="None",
                markersize=6,
                color=color,
                markeredgecolor="black",
                label=label,
            )
        )

    ncols = min(3, len(legend_handles))
    n_scenarios = len(scenarios)

    if n_scenarios <= 2:
        legend_y = 0.38
        bottom_margin = 0.01
    else:
        legend_y = 0
        bottom_margin = 0.2

    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=ncols,
        frameon=False,
        fontsize=11,
        columnspacing=1.5,
        handletextpad=0.8,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        bottom=bottom_margin,
        top=0.93,
        wspace=0.30,
        hspace=0.40,
    )

    fig.savefig(out_path, dpi=500)
    plt.close(fig)
    print(f"[INFO] Saved combined line+box plot to {out_path}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Make combined line+box summary plots from annual means/totals CSVs."
        )
    )
    ap.add_argument(
        "--extra-model",
        dest="extra_models",
        nargs="*",
        default=["Stable-Emulator_No_Carry"],
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
        help="Optional subset of scenarios to include (default: all).",
    )
    ap.add_argument(
        "--var",
        choices=VARS,
        nargs="*",
        help="Optional subset of variables to include (default: all).",
    )
    ap.add_argument(
        "--no-shared-ylims",
        dest="shared_ylims",
        action="store_false",
        help=(
            "Disable sharing y-axis limits across scenarios. "
            "By default, all line panels share a common y-range and all "
            "box panels share a common y-range."
        ),
    )
    ap.set_defaults(shared_ylims=True)

    args = ap.parse_args()

    extra_models = args.extra_models or []
    scenarios_to_use = args.scenario if args.scenario else SCENARIOS
    vars_to_use = args.var if args.var else VARS

    for var in vars_to_use:
        print(f"[INFO] Processing variable: {var}")

        mean_units, total_units = VAR_UNITS.get(
            var,
            ("arbitrary units", "arbitrary units"),
        )

        # Annual means (lines) + cumulative totals (box)
        df_means_by_scen = build_df_by_scenario(
            CSV_MEANS_ROOT, var, scenarios_to_use, extra_models=extra_models
        )
        if df_means_by_scen:
            out_means_box = PLOTS_MEANS_BOX_DIR / f"{var}_annual_means_with_boxes.png"
            make_combined_line_box_plot(
                var=var,
                scenarios=scenarios_to_use,
                df_line_by_scen=df_means_by_scen,
                metric_label="Annual means",
                line_units=mean_units,
                box_units=total_units,
                out_path=out_means_box,
                extra_models=extra_models,
                extra_colors=DEFAULT_EXTRA_COLORS,
                cumulative_lines=False,
                shared_ylims=args.shared_ylims,
            )
        else:
            print(f"[WARN] No annual-means data for {var}; skipping means+box.")

        # Annual totals (lines) + cumulative totals (box)
        df_totals_by_scen = build_df_by_scenario(
            CSV_TOTALS_ROOT, var, scenarios_to_use, extra_models=extra_models
        )
        if df_totals_by_scen:
            out_totals_box = PLOTS_TOTALS_BOX_DIR / f"{var}_annual_totals_with_boxes.png"
            make_combined_line_box_plot(
                var=var,
                scenarios=scenarios_to_use,
                df_line_by_scen=df_totals_by_scen,
                metric_label="Annual totals",
                line_units=total_units,
                box_units=total_units,
                out_path=out_totals_box,
                extra_models=extra_models,
                extra_colors=DEFAULT_EXTRA_COLORS,
                cumulative_lines=False,
                shared_ylims=args.shared_ylims,
            )

            # Cumulative totals (lines) + cumulative totals (box)
            out_cum_box = PLOTS_CUM_BOX_DIR / f"{var}_cumulative_totals_with_boxes.png"
            make_combined_line_box_plot(
                var=var,
                scenarios=scenarios_to_use,
                df_line_by_scen=df_totals_by_scen,
                metric_label="Cumulative totals",
                line_units=total_units,
                box_units=total_units,
                out_path=out_cum_box,
                extra_models=extra_models,
                extra_colors=DEFAULT_EXTRA_COLORS,
                cumulative_lines=True,
                shared_ylims=args.shared_ylims,
            )
        else:
            print(f"[WARN] No annual-totals data for {var}; skipping totals/cumulative+box.")


if __name__ == "__main__":
    main()