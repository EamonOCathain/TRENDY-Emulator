#!/usr/bin/env python3
"""
plot_fFire_S3_region_timeseries.py
----------------------------------

Plot a regional mean timeseries of fFire for scenario S3 from:

  - TRENDY ensemble mean:
      .../data/preds_for_analysis/ensmean/S3/ENSMEAN_S3_fFire.nc
  - Stable-Emulator (with carry):
      .../data/preds_for_analysis/stable/carry/S3/fFire.nc

Region:
  lon: -120 to -100
  lat:  60 to  66

The script:
  - Computes the spatial mean over this lat/lon box for each timestep
  - Plots a single time series panel with:
        * TRENDY ensemble mean in purple
        * Stable-Emulator in orange
  - Uses a clean, publication-ready style, consistent with your LAI plots.

Output:
  PROJECT_ROOT/data/analysis/CSVs/plots/fFire_timeseries/
      fFire_S3_region_timeseries.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import sys
import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")

ENSMEAN_PATH = PROJECT_ROOT / "data/preds_for_analysis/ensmean/S3/ENSMEAN_S3_fFire.nc"
STABLE_PATH  = PROJECT_ROOT / "data/preds_for_analysis/stable/carry/S3/fFire.nc"

OUT_DIR = PROJECT_ROOT / "data/analysis/CSVs/plots/fFire_timeseries"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "fFire_S3_region_timeseries.png"

VAR_NAME = "fFire"

# Region (lon, lat)
LON_MIN, LON_MAX = -120.0, -100.0
LAT_MIN, LAT_MAX = 60.0, 66.0

# Colours (same as in your LAI script)
COLOR_ENS = "#410253"     # TRENDY ensemble (purple)
COLOR_STABLE = "#EF8F40"  # Stable-Emulator (orange)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def select_region_mean(
    da: xr.DataArray,
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
) -> xr.DataArray:
    """
    Select a lat/lon box and compute the spatial mean.

    Handles:
      - lat dimension named 'lat'
      - lon dimension named 'lon' in either [-180,180] or [0,360] format.
    """
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    dims = set(da.dims)
    if "lat" not in dims or "lon" not in dims:
        raise ValueError(f"DataArray lacks lat/lon dims: {da.dims}")

    # Latitude selection
    da_sel = da.sel(lat=slice(lat_min, lat_max))

    # Longitude selection: handle 0–360 vs -180–180
    lon = da["lon"]
    lon_min_data = float(lon.min())
    lon_max_data = float(lon.max())

    if lon_min_data >= 0.0 and lon_max_data > 180.0:
        # Dataset likely in 0–360; convert requested bounds
        lon_min_wrapped = (lon_min + 360.0) if lon_min < 0.0 else lon_min
        lon_max_wrapped = (lon_max + 360.0) if lon_max < 0.0 else lon_max
        da_sel = da_sel.sel(lon=slice(lon_min_wrapped, lon_max_wrapped))
    else:
        # Dataset likely already in -180–180
        da_sel = da_sel.sel(lon=slice(lon_min, lon_max))

    # Spatial mean over region
    da_mean = da_sel.mean(dim=("lat", "lon"), skipna=True)

    if "time" not in da_mean.dims:
        raise ValueError("Resulting DataArray has no 'time' dimension after averaging.")

    return da_mean


def load_region_timeseries(nc_path: Path, var: str) -> xr.DataArray:
    """Open NetCDF, extract var, compute regional spatial mean timeseries."""
    if not nc_path.is_file():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    # Use cftime to decode times
    ds = xr.open_dataset(nc_path, decode_times=True, use_cftime=True)
    if var not in ds.variables:
        ds.close()
        raise KeyError(f"Variable '{var}' not found in {nc_path.name}")

    da = ds[var]
    ts = select_region_mean(da, (LAT_MIN, LAT_MAX), (LON_MIN, LON_MAX))
    ds.close()
    return ts


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Plot regional mean fFire timeseries for S3 (ensemble vs emulator)."
    )
    ap.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Optional year range to trim the time axis (e.g. --years 1950 2000).",
    )
    args = ap.parse_args()

    # --- Load regional mean timeseries ---
    try:
        ts_ens = load_region_timeseries(ENSMEAN_PATH, VAR_NAME)
    except Exception as e:
        print(f"[ERROR] Failed to load ENSMEAN timeseries: {e}")
        return

    try:
        ts_stable = load_region_timeseries(STABLE_PATH, VAR_NAME)
    except Exception as e:
        print(f"[ERROR] Failed to load Stable-Emulator timeseries: {e}")
        return

    # Align on common time axis (just in case)
    common_time = np.intersect1d(ts_ens["time"].values, ts_stable["time"].values)
    if common_time.size == 0:
        print("[ERROR] No overlapping time steps between ENSMEAN and Stable-Emulator.")
        return

    ts_ens = ts_ens.sel(time=common_time)
    ts_stable = ts_stable.sel(time=common_time)

    # ------------------------------------------------------------------
    # Build numeric year axis: year + (month-1)/12 so monthly is visible
    # ------------------------------------------------------------------
    time_vals = ts_ens["time"].values  # cftime.DatetimeNoLeap objects
    years = np.array(
        [t.year + (t.month - 1) / 12.0 for t in time_vals],
        dtype=float,
    )

    # Optional trimming by year range
    if args.years:
        y0, y1 = args.years
        mask = (years >= y0) & (years <= y1)
        if not np.any(mask):
            print(f"[ERROR] No time steps in requested range {y0}–{y1}.")
            return
        years = years[mask]
        ts_ens = ts_ens.isel(time=mask)
        ts_stable = ts_stable.isel(time=mask)
        print(f"[INFO] Trimmed to years {y0}–{y1}")

    x = years
    x_label = "Year"

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        x,
        ts_ens.values,
        color=COLOR_ENS,
        linewidth=1.8,
        label="TRENDY Ensemble Mean",
    )
    ax.plot(
        x,
        ts_stable.values,
        color=COLOR_STABLE,
        linewidth=1.8,
        label="Emulator (Autoregressive)",
    )

    # Styling
    ax.set_title(
        "Regional Mean fFire (S3, -120° to -100°, 60° to 66°)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("fFire (kg m$^{-2}$ s$^{-1}$)", fontsize=11)

    ax.grid(True, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=400)
    plt.close(fig)
    print(f"[INFO] Saved fFire region timeseries to {OUT_FIG}")

if __name__ == "__main__":
    sys.path.append(str(PROJECT_ROOT))
    main()