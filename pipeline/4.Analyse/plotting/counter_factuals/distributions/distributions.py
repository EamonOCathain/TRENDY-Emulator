#!/usr/bin/env python3
"""
sample_and_plot_distributions.py
--------------------------------

- Sample 200,000 values for each region and variable (tmp, pre)
  from a daily Zarr dataset.
- Save samples to .npz.
- Plot distributions per region as stacked horizontal panels.

TMP:
  - Original distribution: filled KDE line
  - Offset distributions: lines only, original ±1..±5 units (°C)

PRE:
  - Original distribution: filled KDE line
  - Offset distributions: lines only, scaled by [0.0 .. 1.0] in
    steps of -10% (1.0, 0.9, ..., 0.0)

Assumptions:
  - Zarr has coordinates 'lat' and 'lon'
  - Variables 'tmp' and 'pre' exist
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import xarray as xr
import zarr
import argparse

import matplotlib
matplotlib.use("Agg")  # for cluster use
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError

try:
    from scipy.stats import gaussian_kde
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ---------------------------------------------------------------------
# Region definitions
# ---------------------------------------------------------------------

REGIONS = [
    # name, long label, (lat_min, lat_max), (lon_min, lon_max)
    ("global",     "Global",     -90.0,  90.0, -180.0,  180.0),
    ("tropics",    "Tropics",    -15.0,  15.0, -180.0,  180.0),
    ("subtropics","Subtropics", -35.0,  35.0, -180.0,  180.0),  # combined NH+SH
    ("temperate",  "Temperate",  -60.0,  60.0, -180.0,  180.0),  # combined NH+SH
    ("boreal",     "Boreal",      60.0,  90.0, -180.0,  180.0),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sample_region_values(
    da,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    n_samples: int,
    rng: np.random.Generator,
):
    """
    Sample `n_samples` values from a region defined by lat/lon bounds.
    Assumes dimensions (..., lat, lon) or (time, lat, lon, ...).
    """
    lat = da["lat"]
    lon = da["lon"]

    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)

    # Compute combined mask before where(drop=True)
    combined_mask = (lat_mask & lon_mask).compute()

    da_region = da.where(combined_mask, drop=True)

    if da_region.size == 0:
        raise ValueError(
            f"No data found in region "
            f"lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]"
        )

    flat = da_region.values.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError("All values in region are NaN / non-finite.")

    if flat.size >= n_samples:
        idx = rng.choice(flat.size, size=n_samples, replace=False)
    else:
        idx = rng.choice(flat.size, size=n_samples, replace=True)

    return flat[idx]


def compute_density(
    data: np.ndarray,
    num_points: int = 512,
    x_pad: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a smooth density for plotting.
    Uses gaussian_kde if available; otherwise uses a normalized histogram.
    Falls back to histogram if KDE fails (e.g. degenerate data).
    """
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("Empty data passed to compute_density")

    d_min, d_max = np.percentile(data, [0.1, 99.9])
    if not np.isfinite(d_min) or not np.isfinite(d_max):
        d_min = np.nanmin(data)
        d_max = np.nanmax(data)

    if d_max == d_min:
        center = d_min
        span = 1.0
        x = np.linspace(center - span / 2, center + span / 2, num_points)
        y = np.exp(-0.5 * ((x - center) / (span / 6)) ** 2)
        y /= np.trapz(y, x)
        return x, y

    span = d_max - d_min
    d_min -= x_pad * span
    d_max += x_pad * span

    x = np.linspace(d_min, d_max, num_points)

    if HAVE_SCIPY and data.size > 10:
        try:
            kde = gaussian_kde(data)
            y = kde(x)
            return x, y
        except LinAlgError:
            pass

    hist, bin_edges = np.histogram(
        data,
        bins=80,
        range=(d_min, d_max),
        density=True,
    )
    x_hist = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    y_hist = hist.astype(float)
    return x_hist, y_hist


def setup_pub_style():
    """
    Global matplotlib rcParams for a clean, publication-ready look.
    """
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
        "figure.constrained_layout.use": True,
    })


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_tmp_distributions(samples_by_region: dict[str, np.ndarray], out_path: Path):
    """
    Plot TMP (in °C) distributions for each region (stacked 5×1):
      - original (filled)
      - offsets ±1..±5 °C (lines)
      - legend placed below all subplots
      - horizontal colour bar showing offset scale (-5 → +5 °C)
    """
    setup_pub_style()

    # Make room below for legend + colorbar
    fig, axes = plt.subplots(
        nrows=len(REGIONS),
        ncols=1,
        figsize=(4, 7),
        sharex=True,
        sharey=False,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1], "hspace": 0.15},
    )

    base_color = "#3b528b"
    offset_cmap = plt.cm.Blues
    offsets = np.array(list(range(-5, 0)) + list(range(1, 6)))  # [-5..-1, +1..+5]

    for ax, (key, label, *_rest) in zip(axes, REGIONS):
        data = samples_by_region[key]  # already in °C
        x0, y0 = compute_density(data)

        # Original filled distribution
        ax.fill_between(x0, y0, 0, color=base_color, alpha=0.30, linewidth=0)
        line_original, = ax.plot(
            x0, y0, color=base_color, linewidth=1.5, label="Original Distribution"
        )

        # Offset curves (±1..±5 °C)
        for j, off in enumerate(offsets):
            x_off = x0 + off
            # Normalise offset to [0,1] for color mapping
            norm_val = (off + 5) / 10  # -5→0, 0→0.5, +5→1
            c = offset_cmap(norm_val * 0.8 + 0.1)  # avoid too-pale edges
            ax.plot(x_off, y0, color=c, linewidth=1.0, alpha=0.9)

        ax.set_ylabel(label)
        ax.grid(True, which="both", axis="both", alpha=0.25)

    axes[-1].set_xlabel("Temperature (°C)")

    # -------------------------
    # Legend BELOW all subplots
    # -------------------------
    handles = [line_original]
    labels = ["Original Distribution"]

    # Add one dummy handle for the colour-bar label in the legend
    dummy_handle = Line2D([], [], color="none", label="Temperature Offsets (°C)")
    handles.append(dummy_handle)
    labels.append("Temperature Offsets (°C)")

    # Place legend below all subplots (centered)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
        handlelength=2.0,
    )

    # -------------------------
    # Colour bar for -5 → +5 °C
    # -------------------------
    # Create a ScalarMappable from the same colormap
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=-5, vmax=5)
    sm = plt.cm.ScalarMappable(cmap=offset_cmap, norm=norm)
    sm.set_array([])

    # Add colorbar UNDER the legend
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.04,
        pad=0.12,
        aspect=30,
    )
    cbar.set_label("Offset (°C)")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pre_distributions(samples_by_region: dict[str, np.ndarray], out_path: Path):
    """
    Plot PRE distributions for each region (stacked 5×1):
      - original (filled)
      - scaled distributions for factors 1.0, 0.9, ..., 0.0
    """
    setup_pub_style()

    fig, axes = plt.subplots(
        nrows=len(REGIONS),
        ncols=1,
        figsize=(5, 7),   # smaller figure
        sharex=True,
        sharey=False,
    )

    base_color = "#21918c"
    offset_cmap = plt.cm.Greens
    scales = np.linspace(1.0, 0.0, 11)  # 1.0, 0.9, ..., 0.0

    for ax, (key, label, *_rest) in zip(axes, REGIONS):
        data = samples_by_region[key]
        x0, y0 = compute_density(data)

        # Original filled
        ax.fill_between(x0, y0, 0, color=base_color, alpha=0.30, linewidth=0)
        ax.plot(x0, y0, color=base_color, linewidth=1.5, label="Original (100%)")

        # Scaled distributions
        for j, s in enumerate(scales[1:], start=1):  # skip 1.0 (already plotted)
            scaled = data * s
            xs, ys = compute_density(scaled)
            c = offset_cmap(0.2 + 0.6 * (j / (len(scales) - 1)))
            label_scale = None
            if s in (0.9, 0.5, 0.0):  # label a few for legend
                label_scale = f"{int(s*100)}% of original"
            ax.plot(xs, ys, color=c, linewidth=1.0, alpha=0.9, label=label_scale)

        ax.set_ylabel(label)
        ax.grid(True, which="both", axis="both", alpha=0.25)

    axes[-1].set_xlabel("Precipitation (units of pre)")

    # Build a clean legend with only labeled scales
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll and ll not in labels:
                handles.append(hh)
                labels.append(ll)

    if handles:
        axes[0].legend(
            handles,
            labels,
            loc="upper right",
            frameon=False,
            handlelength=2.0,
        )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample tmp and pre from a Zarr and plot regional distributions."
    )
    parser.add_argument(
        "--zarr",
        default="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/training_new/train/train_location_train_period/daily.zarr",
        help="Path to Zarr store (directory).",
    )
    parser.add_argument(
        "--out-dir",
        default="out/distribution_plots",
        help="Directory to write .npz samples and plots.",
    )
    parser.add_argument(
        "--samples-per-region",
        type=int,
        default=200_000,
        help="Number of samples per region per variable.",
    )
    parser.add_argument(
        "--overwrite-data",
        action="store_true",
        help="If set, overwrite existing sample .npz files. Otherwise reuse them."
    )
    args = parser.parse_args()

    zarr_path = Path(args.zarr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = xr.open_zarr(zarr_path)

    if "tmp" not in ds or "pre" not in ds:
        raise KeyError("Dataset must contain variables 'tmp' and 'pre'.")

    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise KeyError("Dataset must have 'lat' and 'lon' coordinates.")

    rng = np.random.default_rng(123)

    samples_tmp: dict[str, np.ndarray] = {}
    samples_pre: dict[str, np.ndarray] = {}

    # Determine whether NPZs exist
    expected_files = [
        out_dir / f"samples_tmp_{r[0]}.npz" for r in REGIONS
    ] + [
        out_dir / f"samples_pre_{r[0]}.npz" for r in REGIONS
    ]

    all_exist = all(f.exists() for f in expected_files)

    if all_exist and not args.overwrite_data:
        print("[INFO] All sample files found — reusing existing .npz files.")
        # Just load existing samples
        for key, label, *_ in REGIONS:
            tmp_data = np.load(out_dir / f"samples_tmp_{key}.npz")
            pre_data = np.load(out_dir / f"samples_pre_{key}.npz")

            samples_tmp[key] = tmp_data["values"]
            samples_pre[key] = pre_data["values"]
    else:
        print("[INFO] Sampling new data...")

        for key, label, lat_min, lat_max, lon_min, lon_max in REGIONS:
            print(f"[INFO] Sampling region {label} for tmp and pre")

            tmp_vals = sample_region_values(
                da=ds["tmp"],
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                n_samples=args.samples_per_region,
                rng=rng,
            )
            # Convert K -> °C for tmp
            tmp_vals = tmp_vals - 273.15

            pre_vals = sample_region_values(
                da=ds["pre"],
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                n_samples=args.samples_per_region,
                rng=rng,
            )

            samples_tmp[key] = tmp_vals
            samples_pre[key] = pre_vals

            # Save per-region samples
            np.savez_compressed(
                out_dir / f"samples_tmp_{key}.npz",
                values=tmp_vals,
                region=key,
                label=label,
            )
            np.savez_compressed(
                out_dir / f"samples_pre_{key}.npz",
                values=pre_vals,
                region=key,
                label=label,
            )

    # Plot publication-ready figures (smaller, tmp in °C, no titles)
    plot_tmp_distributions(samples_tmp, out_dir / "tmp_offsets_by_region.png")
    plot_pre_distributions(samples_pre, out_dir / "pre_offsets_by_region.png")

    print(f"[OK] Finished. Outputs written under {out_dir}")


if __name__ == "__main__":
    main()