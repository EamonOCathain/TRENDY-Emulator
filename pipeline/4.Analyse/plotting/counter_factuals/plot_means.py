#!/usr/bin/env python3
"""
make_forcing_response_plots.py
------------------------------

Load ILAMB scalar_database CSVs for CO2, temperature (tmp), and precipitation
(pre_spfh) counterfactual sensitivity experiments, merge them into a single
DataFrame, normalise the "Model" axis into meaningful forcing coordinates, and
plot response curves for each variable and forcing.

For each forcing experiment ("co2", "tmp", "pre") and each ILAMB Variable:
  - Plot regional model responses (lines with circle markers)
  - Optional benchmark points as "x" markers in the same region colour
  - Optionally draw vertical lines from std_dict.json:
      * CO2:   mean, min, max
      * tmp:   mean, ±1σ, min, max
      * pre:   none
  - Save figures into:
        OUT_ROOT/<forcing>/<variable>.png

Additionally, for each forcing, a combined plot is generated which overlays
selected variables (controlled via --combined-vars) on a single axis.

Usage (examples)
----------------
  python make_forcing_response_plots.py
  python make_forcing_response_plots.py --combined-vars gpp,nbp,nee
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")

# ILAMB scalar_database CSVs
CSV_TMP = PROJECT_ROOT / (
    "pipeline/3.benchmark/ilamb/benchmarks/"
    "counter_factuals_sensitivity/tmp/_build/scalar_database.csv"
)
CSV_CO2 = PROJECT_ROOT / (
    "pipeline/3.benchmark/ilamb/benchmarks/"
    "counter_factuals_sensitivity/CO2/_build/scalar_database.csv"
)
CSV_PRE = PROJECT_ROOT / (
    "pipeline/3.benchmark/ilamb/benchmarks/"
    "counter_factuals_sensitivity/pre_spfh/_build/scalar_database.csv"
)

# std_dict with training forcing stats
STD_JSON_PATH = PROJECT_ROOT / "src/dataset/std_dict.json"

# Output root
OUT_ROOT = PROJECT_ROOT / "data/analysis/CSVs/plots/counterfactual_sensitivity"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Metadata: long names + mapping ILAMB label -> internal key
# -----------------------------------------------------------------------------

output_attributes = {
    "nbp":  "Net Biome Productivity",
    "gpp":  "Gross Primary Production",
    "npp":  "Net Primary Production",
    "ra":   "Autotrophic Respiration",
    "rh":   "Heterotrophic Respiration",
    "fLuc": "Land-Use Change Emissions",
    "fFire": "Fire Emissions",
    "mrro": "Total Runoff",
    "evapotrans": "Evapotranspiration",
    "cLitter": "Carbon in Litter Pool",
    "cSoil":   "Carbon in Soil Pool",
    "cVeg":    "Carbon in Vegetation",
    "cTotal":  "Total Carbon in Ecosystem",
    "cTotal_monthly": "Total Carbon in Ecosystem",
    "mrso": "Total Soil Moisture Content",
    "lai": "Leaf Area Index",
    "lai_avh15c1": "Leaf Area Index",
    "lai_modis":   "Leaf Area Index",
    "nee": "Net Ecosystem Exchange",
}

# Map ILAMB "Variable" labels to internal keys above
VARIABLE_TO_KEY = {
    "NetBiomeProductivity": "nbp",
    "GrossPrimaryProduction": "gpp",
    "NetPrimaryProduction": "npp",
    "AutotrophicRespiration": "ra",
    "HeterotrophicRespiration": "rh",
    "LandUseChangeEmissions": "fLuc",
    "FireEmissions": "fFire",
    "TotalRunoff": "mrro",
    "Evapotranspiration": "evapotrans",
    "CarboninLitterPool": "cLitter",
    "CarboninSoilPool": "cSoil",
    "CarboninVegetation": "cVeg",
    "TotalCarboninEcosystem": "cTotal",
    "SoilMoisture": "mrso",
    "LeafAreaIndex": "lai",
    "NetEcosystemExchange": "nee",
}

# -----------------------------------------------------------------------------
# Load std_dict once
# -----------------------------------------------------------------------------

with open(STD_JSON_PATH, "r") as f:
    STD_DICT = json.load(f)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def resolve_var_label(token: str, df: pd.DataFrame) -> Optional[str]:
    """
    Resolve a token to a df['Variable'] label.

    token can be:
      - an ILAMB Variable label (e.g. 'GrossPrimaryProduction'), or
      - an internal key (e.g. 'gpp', 'nbp', 'nee').
    """
    var_values = set(df["Variable"].unique())
    if token in var_values:
        return token

    key_to_label = {v.lower(): k for k, v in VARIABLE_TO_KEY.items()}
    token_low = token.lower()
    if token_low in key_to_label and key_to_label[token_low] in var_values:
        return key_to_label[token_low]

    return None


# -----------------------------------------------------------------------------
# Core plotting function (single variable)
# -----------------------------------------------------------------------------

def plot_variable_response(
    df: pd.DataFrame,
    variable: str,
    forcing_var: str = "co2",
    std_dict: dict | None = None,
    xlim=None,
    ylim=None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    mean_line: bool = False,
    sd_line: bool = False,
    min_line: bool = False,
    max_line: bool = False,
    trendy_cross_legend: bool = False,
    trendy_cross_points: bool = False,
):
    """
    Plot Data vs Model for a given response variable under a specified forcing.

    - Lines with circle markers for each Region where Type != 'Benchmark'
    - Optional 'x' markers for Benchmark points in the same region colour
      (controlled by `trendy_cross_points`)
    - Y label from output_attributes long_name + Units column (overridden by
      `ylabel` if provided)
    - X label inferred from forcing_var (overridden by `xlabel` if provided)
    - Optional vertical lines from std_dict:
        * mean_line=True   -> black dashed line at mean
        * sd_line=True     -> grey dashed lines at mean ± 1 std
        * min_line=True    -> red dashed line at min
        * max_line=True    -> red dashed line at max
      For 'tmp', mean/min/max are stored in Kelvin and converted to Celsius,
      while std is used unchanged (σ is the same in K and °C).
      If drawn, these lines are also added to the legend.
    - Optional TRENDY ensemble mean cross legend entry controlled by
      `trendy_cross_legend` (legend only, independent of whether points are
      actually drawn).
    """

    if std_dict is None:
        std_dict = {}

    # --------- Resolve variable label used in df ---------
    var_values = df["Variable"].unique()
    if variable in var_values:
        var_label = variable
    else:
        var_key = variable.lower()
        inv_map = {v.lower(): k for k, v in VARIABLE_TO_KEY.items()}
        if var_key in inv_map:
            var_label = inv_map[var_key]
        else:
            raise ValueError(
                f"Could not match variable '{variable}' to any df['Variable'] value "
                f"or VARIABLE_TO_KEY entry."
            )

    sub = df[df["Variable"] == var_label].copy()
    if sub.empty:
        raise ValueError(f"No rows found for Variable == '{var_label}'")

    # --------- Restrict to the chosen experiment (co2/tmp/pre) ---------
    forcing_key = forcing_var.lower()
    sub = sub[sub["Experiment"].str.lower() == forcing_key]
    if sub.empty:
        raise ValueError(
            f"No rows left after filtering for Experiment == '{forcing_var}'. "
            f"Check df['Experiment'] values."
        )

    # --------- Units and long_name ---------
    units_vals = sub["Units"].dropna().unique()
    units = units_vals[0] if len(units_vals) > 0 else ""

    key = VARIABLE_TO_KEY.get(var_label, None)
    if key is None:
        long_name = var_label
    else:
        long_name = output_attributes.get(key, var_label)

    # --------- Figure + aesthetics ---------
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)

    palette = [
        "#4c72b0",  # blue
        "#55a868",  # green
        "#c44e52",  # red
        "#8172b3",  # purple
        "#ccb974",  # ochre
        "#64b5cd",  # light blue
        "#8c8c8c",  # grey
    ]
    ax.set_prop_cycle(cycler(color=palette))

    # Split model and benchmark rows
    model_rows = sub[sub["Type"] != "Benchmark"]
    bench_rows = sub[sub["Type"] == "Benchmark"]

    # --------- Plot regional lines + benchmarks ---------
    regions = sorted(model_rows["Region"].dropna().unique())
    for region in regions:
        r_data = model_rows[model_rows["Region"] == region]
        if r_data.empty:
            continue

        r_data = r_data.sort_values("Model")

        line, = ax.plot(
            r_data["Model"],
            r_data["Data"],
            marker="o",
            markersize=4.5,
            markerfacecolor="none",
            linewidth=1.8,
            linestyle="-",
            label=region.title(),
        )
        color = line.get_color()

        if trendy_cross_points:
            r_bench = bench_rows[bench_rows["Region"] == region]
            if not r_bench.empty:
                ax.plot(
                    r_bench["Model"],
                    r_bench["Data"],
                    marker="x",
                    linestyle="None",
                    markersize=7,
                    markeredgewidth=1.5,
                    color=color,
                )

    # --------- Vertical lines from std_dict.json (optional) ---------
    line_handles: List[Line2D] = []

    if forcing_key in std_dict and any([mean_line, sd_line, min_line, max_line]):
        stats = std_dict[forcing_key]
        mean = stats.get("mean", None)
        std = stats.get("std", None)
        min_val = stats.get("min", None)
        max_val = stats.get("max", None)

        # Convert mean/min/max from K -> °C for tmp; std stays unchanged
        if forcing_key == "tmp":
            if mean is not None:
                mean = mean - 273.15
            if min_val is not None:
                min_val = min_val - 273.15
            if max_val is not None:
                max_val = max_val - 273.15

        def _vline(x_val, color="red", linestyle="--"):
            if x_val is not None:
                ax.axvline(
                    x_val,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.2,
                )

        if mean_line and mean is not None:
            _vline(mean, color="black", linestyle="--")
            line_handles.append(
                Line2D(
                    [], [], color="black", linestyle="--", linewidth=1.2,
                    label="Mean training data",
                )
            )

        if sd_line and (mean is not None) and (std is not None):
            _vline(mean - std, color="grey", linestyle="--")
            _vline(mean + std, color="grey", linestyle="--")
            line_handles.append(
                Line2D(
                    [], [], color="grey", linestyle="--", linewidth=1.2,
                    label="Mean ±1σ",
                )
            )

        if min_line and min_val is not None:
            _vline(min_val, color="red", linestyle="--")
            line_handles.append(
                Line2D(
                    [], [], color="red", linestyle="--", linewidth=1.2,
                    label="Min training data",
                )
            )

        if max_line and max_val is not None:
            _vline(max_val, color="red", linestyle="--")
            line_handles.append(
                Line2D(
                    [], [], color="red", linestyle="--", linewidth=1.2,
                    label="Max training data",
                )
            )

    elif forcing_key not in std_dict and any([mean_line, sd_line, min_line, max_line]):
        print(
            f"Warning: '{forcing_key}' not found in std_dict; "
            "no mean/std/min/max lines drawn."
        )

    # --------- Axes labels & limits ---------
    if xlabel is not None:
        x_label = xlabel
    else:
        if forcing_key == "co2":
            x_label = r"CO$_2$ concentration (ppm)"
        elif forcing_key == "pre":
            x_label = "Precipitation percentage offset"
        elif forcing_key == "tmp":
            x_label = "Global mean temperature (°C)"
        else:
            x_label = forcing_var

    ax.set_xlabel(x_label, fontsize=11)

    if ylabel is not None:
        y_label = ylabel
    else:
        if units:
            y_label = f"{long_name} [{units}]"
        else:
            y_label = long_name
    ax.set_ylabel(y_label, fontsize=11)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.tick_params(labelsize=9)

    # --------- Legend (outside, with optional TRENDY + line entries) ---------
    handles, labels = ax.get_legend_handles_labels()

    for lh in line_handles:
        lbl = lh.get_label()
        if lbl and lbl not in labels:
            handles.append(lh)
            labels.append(lbl)

    if trendy_cross_legend:
        benchmark_handle = Line2D(
            [], [], color="black", marker="x", linestyle="None",
            markersize=7, markeredgewidth=1.5, label="TRENDY Mean (S0)",
        )
        if "TRENDY Mean (S0)" not in labels:
            handles.append(benchmark_handle)
            labels.append("TRENDY Mean (S0)")

    ax.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
        title="Legend",
        title_fontsize=10,
        fontsize=9,
    )

    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Combined plot per forcing
# -----------------------------------------------------------------------------
def make_combined_plot(
    df: pd.DataFrame,
    forcing: str,
    var_labels: Iterable[str],
    std_dict: dict,
    out_path: Path,
):
    """
    Multi-panel plot for one forcing, with up to 4 subplots (2x2), one per variable.

    Each subplot:
      - shows all regions (same as individual plots),
      - uses the same styling as plot_variable_response (lines with circle markers),
      - includes the same vertical training-stat lines as the single plots,
      - has its own y-label: <long_name> [units],
      - is labelled A–D in the top-left.

    The whole figure:
      - shares the x-axis,
      - has a single x-label at the bottom,
      - has one horizontal legend at the bottom for regions + training-stat lines.
    """
    forcing_key = forcing.lower()
    sub = df[df["Experiment"].str.lower() == forcing_key].copy()
    if sub.empty:
        print(f"[WARN] No data for forcing={forcing_key} in combined plot.")
        return

    # Only Model rows (no Benchmark lines or x's in the combined figure)
    model_rows = sub[sub["Type"] == "Model"].copy()
    if model_rows.empty:
        print(f"[WARN] No Model rows for forcing={forcing_key} in combined plot.")
        return

    # Resolve variables actually present
    resolved = []
    for token in var_labels:
        label = resolve_var_label(token, sub)
        if label is None:
            print(f"[WARN] Combined plot: could not resolve '{token}' for forcing={forcing}")
            continue
        if model_rows[model_rows["Variable"] == label].empty:
            continue
        resolved.append(label)

    if not resolved:
        print(f"[WARN] No valid variables to plot for forcing={forcing}.")
        return

    # Limit to max 4 variables (2x2)
    if len(resolved) > 4:
        print(f"[INFO] Combined plot: more than 4 variables requested for {forcing}; "
              f"using first 4: {resolved[:4]}")
        resolved = resolved[:4]

    n_panels = len(resolved)
    letters = "ABCD"

    # Decide which vertical lines to draw (match the single-plot logic)
    if forcing_key == "co2":
        mean_line = True
        sd_line = False
        min_line = True
        max_line = True
    elif forcing_key == "tmp":
        mean_line = True
        sd_line = False
        min_line = False
        max_line = False
    else:  # "pre"
        mean_line = False
        sd_line = False
        min_line = False
        max_line = False
        xlim = (0, -100)

    # Training stats for vertical lines
    stats = std_dict.get(forcing_key, {})
    mean = stats.get("mean", None)
    std = stats.get("std", None)
    min_val = stats.get("min", None)
    max_val = stats.get("max", None)

    # Convert tmp from K to °C for mean/min/max; std unchanged
    if forcing_key == "tmp":
        if mean is not None:
            mean = mean - 273.15
        if min_val is not None:
            min_val = min_val - 273.15
        if max_val is not None:
            max_val = max_val - 273.15

    def _draw_vertical_lines(ax):
        line_handles = []
        if forcing_key == "co2":
            # mean, min, max
            if mean_line and mean is not None:
                ax.axvline(mean, color="black", linestyle="--", linewidth=1.2)
                line_handles.append(
                    Line2D([], [], color="black", linestyle="--", linewidth=1.2,
                           label="Mean training data")
                )
            if min_line and min_val is not None:
                ax.axvline(min_val, color="red", linestyle="--", linewidth=1.2)
                line_handles.append(
                    Line2D([], [], color="red", linestyle="--", linewidth=1.2,
                           label="Min training data")
                )
            if max_line and max_val is not None:
                ax.axvline(max_val, color="red", linestyle="--", linewidth=1.2)
                line_handles.append(
                    Line2D([], [], color="red", linestyle="--", linewidth=1.2,
                           label="Max training data")
                )
        elif forcing_key == "tmp":
            if mean_line and mean is not None:
                ax.axvline(mean, color="black", linestyle="--", linewidth=1.2)
                line_handles.append(
                    Line2D([], [], color="black", linestyle="--", linewidth=1.2,
                           label="Mean training data")
                )
            if sd_line and (mean is not None) and (std is not None):
                ax.axvline(mean - std, color="grey", linestyle="--", linewidth=1.2)
                ax.axvline(mean + std, color="grey", linestyle="--", linewidth=1.2)
                line_handles.append(
                    Line2D([], [], color="grey", linestyle="--", linewidth=1.2,
                           label="Mean ±1σ")
                )
            if min_line and min_val is not None:
                ax.axvline(min_val, color="red", linestyle="--", linewidth=1.2)
                line_handles.append(
                    Line2D([], [], color="red", linestyle="--", linewidth=1.2,
                           label="Min training data")
                )
            if max_line and max_val is not None:
                ax.axvline(max_val, color="red", linestyle="--", linewidth=1.2)
                line_handles.append(
                    Line2D([], [], color="red", linestyle="--", linewidth=1.2,
                           label="Max training data")
                )
        return line_handles

    # Regions and colours (same across all subplots)
    regions = sorted(model_rows["Region"].dropna().unique())
    palette = [
        "#4c72b0",  # blue
        "#55a868",  # green
        "#c44e52",  # red
        "#8172b3",  # purple
        "#ccb974",  # ochre
        "#64b5cd",  # light blue
        "#8c8c8c",  # grey
    ]
    region_colors = {region: palette[i % len(palette)] for i, region in enumerate(regions)}

    # Create a 2x2 grid; hide unused axes if fewer than 4 panels
    fig, axes = plt.subplots(
        2,
        2,
        sharex=True,
        figsize=(9, 7),
        dpi=300,
    )
    axes_flat = axes.ravel()

    # For building a single legend at the bottom
    legend_handles = []
    legend_labels = []

    for i, label in enumerate(resolved):
        ax = axes_flat[i]
        ax.set_prop_cycle(None)  # we control colours ourselves

        vsub = model_rows[model_rows["Variable"] == label].copy()
        if vsub.empty:
            ax.set_visible(False)
            continue

        units_vals = vsub["Units"].dropna().unique()
        units = units_vals[0] if len(units_vals) > 0 else ""

        key = VARIABLE_TO_KEY.get(label, None)
        if key is None:
            long_name = label
            short_name = label
        else:
            long_name = output_attributes.get(key, label)
            short_name = key.upper()

        # Plot each region (same as in plot_variable_response)
        for region in regions:
            r_data = vsub[vsub["Region"] == region]
            if r_data.empty:
                continue
            r_data = r_data.sort_values("Model")
            color = region_colors[region]

            line, = ax.plot(
                r_data["Model"],
                r_data["Data"],
                marker="o",
                markersize=4.5,
                markerfacecolor="none",
                linewidth=1.8,
                linestyle="-",
                color=color,
                label=region,
            )

            # Collect region handles/labels from first time we see them
            if region not in legend_labels:
                legend_labels.append(region)
                legend_handles.append(line)

        # Vertical training-stat lines on this subplot
        line_handles_stats = _draw_vertical_lines(ax)
        for lh in line_handles_stats:
            lbl = lh.get_label()
            if lbl and lbl not in [h.get_label() for h in legend_handles]:
                legend_handles.append(lh)
                legend_labels.append(lbl)

        # Y label: variable name + units
        if units:
            ax.set_ylabel(f"Mean {short_name}\n[{units}]", fontsize=10)
        else:
            ax.set_ylabel(f"Mean {short_name}", fontsize=10)

        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=9)

        # Panel label A–D
        ax.text(
            0.02,
            0.95,
            letters[i],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
            ha="left",
        )

    # Hide any unused axes if fewer than 4 variables
    for j in range(len(resolved), 4):
        axes_flat[j].set_visible(False)

    # Shared x label
    if forcing_key == "co2":
        x_label = r"CO$_2$ concentration (ppm)"
    elif forcing_key == "pre":
        x_label = "Precipitation percentage offset"
    elif forcing_key == "tmp":
        x_label = "Global mean temperature (°C)"
    else:
        x_label = forcing_key
    axes_flat[2].set_xlabel(x_label, fontsize=11)
    axes_flat[3].set_xlabel(x_label, fontsize=11)
                
    # Optional x-limits for tmp to match your single plots
    if forcing_key == "tmp":
        for ax in axes_flat:
            if ax.get_visible():
                ax.set_xlim(4, 15)

    # Reverse x-axis for pre: 0 → -100
    if forcing_key == "pre":
        for ax in axes_flat:
            if ax.get_visible():
                ax.set_xlim(0, -100)

    # Layout with space at bottom for legend
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])

    # Single horizontal legend at the bottom
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            labels=[lbl.title() for lbl in legend_labels],
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            ncol=min(len(legend_labels), 4),
            frameon=False,
            title="Legend",
            title_fontsize=10,
            fontsize=9,
        )

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved combined multi-panel plot {out_path}")

# -----------------------------------------------------------------------------
# Build combined DataFrame
# -----------------------------------------------------------------------------

def build_combined_df() -> pd.DataFrame:
    df_tmp = pd.read_csv(CSV_TMP)
    df_co2 = pd.read_csv(CSV_CO2)
    df_pre = pd.read_csv(CSV_PRE)

    df_tmp["Experiment"] = "tmp"
    df_co2["Experiment"] = "co2"
    df_pre["Experiment"] = "pre"

    df = pd.concat([df_tmp, df_co2, df_pre], ignore_index=True)

    df = df.drop(
        columns=["Section", "Source", "AnalysisType", "ScalarType", "Weight"],
        errors="ignore",
    )

    df = df.rename(columns={"ScalarName": "Type"})

    # Keep only Benchmark / Model period means
    df = df[
        df["Type"].isin(
            [
                "Benchmark Period Mean (intersection)",
                "Model Period Mean (intersection)",
            ]
        )
    ]

    df.loc[df["Type"] == "Benchmark Period Mean (intersection)", "Type"] = "Benchmark"
    df.loc[df["Type"] == "Model Period Mean (intersection)", "Type"] = "Model"

    # Drop duplicate Benchmark rows per (Variable, Region)
    mask_bench = df["Type"] == "Benchmark"
    dupes_bench = df[mask_bench].duplicated(
        subset=["Variable", "Region"], keep="first"
    )
    df = df[~(mask_bench & dupes_bench)]

    df = df.sort_values(["Variable", "Region", "Type"])

    # --- CO2 experiment: parse percentage offsets -> absolute ppm ---
    co2_mask = df["Experiment"] == "co2"
    co2_str = df.loc[co2_mask, "Model"].astype(str)
    pct = co2_str.str.extract(r"([-+]?\d+)(?=pct$)", expand=False)

    valid_co2 = pct.notna()
    pct_int = pct[valid_co2].astype(int)

    # Assume 300 ppm baseline * (1 + pct/100)
    df.loc[co2_mask & valid_co2, "Model"] = 300.0 * (1.0 + pct_int / 100.0)

    # --- PRE experiment: signed percentage offsets (negative = drier) ---
    pre_mask = df["Experiment"] == "pre"
    pre_str = df.loc[pre_mask, "Model"].astype(str)
    pct_pre = pre_str.str.extract(r"([-+]?\d+)(?=pct$)", expand=False)

    valid_pre = pct_pre.notna()
    vals = pct_pre[valid_pre].astype(int)
    vals = vals.apply(lambda x: -abs(x) if x != 0 else 0)
    df.loc[pre_mask & valid_pre, "Model"] = vals

    # --- TMP experiment: convert "tmp_+/-X" to absolute global mean T (°C) ---
    tmp_mask = df["Experiment"] == "tmp"
    tmp_str = df.loc[tmp_mask, "Model"].astype(str)
    tmp_vals = tmp_str.str.extract(r"tmp_([-+]?\d+)$", expand=False)
    valid_tmp = tmp_vals.notna()

    if "tmp" in STD_DICT:
        tmp_stats = STD_DICT["tmp"]
        tmp_mean_K = float(tmp_stats["mean"])
        tmp_mean_C = tmp_mean_K - 273.15  # convert to °C

        offsets = tmp_vals[valid_tmp].astype(float)
        abs_tmp = tmp_mean_C + offsets  # absolute mean temperature in °C

        df.loc[tmp_mask & valid_tmp, "Model"] = abs_tmp.values
    else:
        df.loc[tmp_mask & valid_tmp, "Model"] = tmp_vals[valid_tmp].astype(int)
        print("Warning: 'tmp' not found in std_dict; using offsets only for tmp.")

    # --- Benchmarks: put at 0 on x-axis ---
    bench_mask = df["Type"] == "Benchmark"
    df.loc[bench_mask, "Model"] = 0.0

    # Ensure Model is numeric everywhere and drop any rows that still fail
    df["Model"] = pd.to_numeric(df["Model"], errors="coerce")
    df = df.dropna(subset=["Model"])

    return df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Make forcing-response plots for ILAMB counterfactuals."
    )
    ap.add_argument(
        "--combined-vars",
        type=str,
        default="",
        help=(
            "Comma-separated list of variables to include in combined plots "
            "(e.g. 'gpp,nbp,nee' or 'GrossPrimaryProduction,NetBiomeProductivity'). "
            "If empty, no combined plots are made."
        ),
    )
    ap.add_argument(
        "--no-trendy-cross-points",
        action="store_true",
        help="Disable plotting of benchmark 'x' markers.",
    )
    ap.add_argument(
        "--no-trendy-cross-legend",
        action="store_true",
        help="Disable TRENDY cross entry in the legend.",
    )
    return ap.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    df = build_combined_df()

    # Which ILAMB variable labels do we have?
    vars_in_df = sorted(df["Variable"].unique())

    # Process combined-vars tokens once
    if args.combined_vars.strip():
        combined_tokens = [t.strip() for t in args.combined_vars.split(",") if t.strip()]
    else:
        combined_tokens = []

    for forcing in ["co2", "tmp", "pre"]:
        out_dir = OUT_ROOT / forcing
        out_dir.mkdir(parents=True, exist_ok=True)

        # Per-variable plots
        for var_label in vars_in_df:
            sub = df[
                (df["Variable"] == var_label)
                & (df["Experiment"].str.lower() == forcing)
            ]
            if sub.empty:
                continue

            if forcing == "co2":
                mean_line = True
                sd_line = False
                min_line = True
                max_line = True
                xlim = None
            elif forcing == "tmp":
                mean_line = True
                sd_line = False
                min_line = False
                max_line = False
                
                xlim = (4, 15)
            else:  # "pre"
                mean_line = False
                sd_line = False
                min_line = False
                max_line = False
                xlim = None

            try:
                fig, ax = plot_variable_response(
                    df,
                    variable=var_label,
                    forcing_var=forcing,
                    std_dict=STD_DICT,
                    mean_line=mean_line,
                    sd_line=sd_line,
                    min_line=min_line,
                    max_line=max_line,
                    trendy_cross_legend=False,
                    trendy_cross_points=False,
                    xlim=xlim,
                )
            except ValueError as e:
                print(f"[WARN] Skipping {var_label} / {forcing}: {e}")
                continue

            fname = f"{var_label}_{forcing}.png"
            out_path = out_dir / fname
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"[INFO] Saved {out_path}")

        # Combined plot for this forcing if requested
        if combined_tokens:
            combined_out = out_dir / f"combined_{forcing}.png"
            make_combined_plot(
                df=df,
                forcing=forcing,
                var_labels=combined_tokens,
                std_dict=STD_DICT,
                out_path=combined_out,
            )


if __name__ == "__main__":
    main()