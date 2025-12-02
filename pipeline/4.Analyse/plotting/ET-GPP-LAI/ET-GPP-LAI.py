#!/usr/bin/env python3
"""
lai_gpp_et_bin_relationship.py
------------------------------

- Open ENSMEAN, TL-Emulator, and observational LAI / GPP / ET fields.
- Check that all datasets share the same lat/lon grid; if not, raise
  an informative error.
- Infer the common overlapping time period across all datasets and
  trim them to this interval.
- Mask all fields to where tvt_mask ∈ {0,1,2}.
- Compute time-mean LAI, GPP, ET for each group.
- Bin LAI into 20 equal-width bins between 0 and the global maximum LAI.
- For each bin and each group, compute mean and standard deviation of
  GPP and ET.
- Plot:
    * Left panel: LAI bins vs GPP
    * Right panel: LAI bins vs ET
  with error bars (±1 SD) for each group:
    - Observations: dark brown
    - Ensemble mean: muted blue
    - TL-Emulator: muted red
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")

# Ensemble Mean
ens_dir = PROJECT_ROOT / "data/preds_for_analysis/ensmean/S3"
ens_lai = ens_dir / "ENSMEAN_S3_lai.nc"
ens_gpp = ens_dir / "ENSMEAN_S3_gpp.nc"
ens_et  = ens_dir / "ENSMEAN_S3_evapotrans.nc"

# Observations
obs_dir = PROJECT_ROOT / "data/preds_for_analysis/obs"
obs_lai = obs_dir / "lai/AVH15C1/lai_avh15c1_filled.nc"
obs_gpp = obs_dir / "gpp/FLUXCOM/gpp_remap.nc"
obs_et  = obs_dir / "evapotrans/GLEAMv3.3a/et_remap.nc"

# TL-Emulator
tl_dir = PROJECT_ROOT / "data/preds_for_analysis/TL/TL_1982_2018"
tl_lai = tl_dir / "lai.nc"
tl_gpp = tl_dir / "gpp.nc"
tl_et  = tl_dir / "evapotrans.nc"

# Train/val/test mask
TVT_MASK_PATH = PROJECT_ROOT / "data/masks/tvt_mask.nc"

# Output
OUT_DIR = PROJECT_ROOT / "data/analysis/CSVs/plots/lai_gpp_et_bins"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "lai_gpp_et_binned_relationships.png"

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans" 

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def open_single_var_ds(path: Path, preferred_names: List[str] | None = None) -> xr.DataArray:
    """
    Open a NetCDF file and return the main data variable as a DataArray.

    If preferred_names is given, return the first variable whose name matches.
    Otherwise, if there is only one data_var, return that.
    If multiple remain and no preferred name found, raise an error.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ds = xr.open_dataset(path)

    data_vars = list(ds.data_vars)

    if preferred_names is not None:
        for name in preferred_names:
            if name in ds.data_vars:
                return ds[name]

    if len(data_vars) == 1:
        return ds[data_vars[0]]

    raise ValueError(
        f"Could not unambiguously determine data variable in {path.name}. "
        f"Available data_vars: {data_vars}"
    )


def check_same_grid(datasets: Dict[str, xr.DataArray]) -> None:
    """
    Check that all DataArrays share the same lat/lon coordinates and shapes.

    Raises a ValueError with details if any mismatch is found.
    """
    names = list(datasets.keys())
    ref_name = names[0]
    ref = datasets[ref_name]

    required_coords = ["lat", "lon"]
    for coord in required_coords:
        if coord not in ref.coords:
            raise ValueError(f"Reference dataset '{ref_name}' lacks coord '{coord}'")

    ref_lat = ref["lat"]
    ref_lon = ref["lon"]
    ref_shape = (ref_lat.size, ref_lon.size)

    errors = []

    for name, da in datasets.items():
        if name == ref_name:
            continue

        for coord in required_coords:
            if coord not in da.coords:
                errors.append(f"Dataset '{name}' lacks coord '{coord}'")
                continue

        lat = da["lat"]
        lon = da["lon"]
        shape = (lat.size, lon.size)

        if shape != ref_shape:
            errors.append(
                f"Shape mismatch for '{name}': lat/lon shape {shape}, "
                f"expected {ref_shape}"
            )

        if not np.allclose(lat.values, ref_lat.values, equal_nan=True):
            errors.append(f"Latitude values differ between '{ref_name}' and '{name}'")

        if not np.allclose(lon.values, ref_lon.values, equal_nan=True):
            errors.append(f"Longitude values differ between '{ref_name}' and '{name}'")

    if errors:
        msg = "Grid mismatch detected among datasets:\n  - " + "\n  - ".join(errors)
        raise ValueError(msg)


def trim_to_common_time(
    data: Dict[str, Dict[str, xr.DataArray]]
) -> Dict[str, Dict[str, xr.DataArray]]:
    """
    Infer the common overlapping time interval across *all* DataArrays
    and trim each to that interval.

    Assumes each DataArray has a 'time' coordinate that is sortable
    (e.g. np.datetime64 or cftime.datetime).
    """
    starts = []
    ends = []
    for grp, vars_dict in data.items():
        for vname, da in vars_dict.items():
            if "time" not in da.dims:
                raise ValueError(f"Variable '{vname}' in group '{grp}' lacks 'time' dimension.")
            t = da["time"]
            if t.size == 0:
                raise ValueError(f"Variable '{vname}' in group '{grp}' has empty time axis.")
            starts.append(t.values[0])
            ends.append(t.values[-1])

    latest_start = max(starts)
    earliest_end = min(ends)

    if latest_start >= earliest_end:
        raise ValueError(
            "No overlapping time period found across all datasets:\n"
            f"  latest_start = {latest_start}\n"
            f"  earliest_end = {earliest_end}"
        )

    trimmed: Dict[str, Dict[str, xr.DataArray]] = {}
    for grp, vars_dict in data.items():
        trimmed[grp] = {}
        for vname, da in vars_dict.items():
            trimmed[grp][vname] = da.sel(time=slice(latest_start, earliest_end))

    print(f"[INFO] Trimmed all datasets to common time range "
          f"{latest_start} → {earliest_end}")
    return trimmed


def convert_gpp_obs_to_model_units(da_model: xr.DataArray, da_obs: xr.DataArray) -> xr.DataArray:
    """
    Convert obs GPP from g m-2 d-1 to the model units kg m-2 s-1.

    We *assume* obs is in g m-2 d-1, regardless of attrs.
    Model stays as-is; only obs is rescaled.
    """
    # g -> kg (× 1e-3)
    # d-1 -> s-1 (÷ 86400)
    fac = 1.0 / (1000.0 * 86400.0)

    da_conv = da_obs * fac
    # Set units to match model units, or a sensible default
    da_conv.attrs["units"] = da_model.attrs.get("units", "kg m-2 s-1")
    return da_conv


def load_and_check_all() -> Tuple[Dict[str, xr.DataArray], xr.DataArray]:
    """
    Load all LAI/GPP/ET datasets for ENSMEAN, TL, and OBS.
    Check that their grids match and that tvt_mask matches these grids.
    Trim all data variables to the common overlapping time interval.

    Returns
    -------
    data : dict
        Nested dict keyed by group ('ens', 'tl', 'obs') and variable ('lai','gpp','et').
    mask_da : xr.DataArray
        tvt_mask DataArray.
    """
    # Open data
    lai_ens = open_single_var_ds(ens_lai, preferred_names=["lai", "LAI"])
    gpp_ens = open_single_var_ds(ens_gpp, preferred_names=["gpp", "GPP"])
    et_ens  = open_single_var_ds(ens_et,  preferred_names=["evapotrans", "evapotranspiration", "et", "ET"])

    lai_tl = open_single_var_ds(tl_lai, preferred_names=["lai", "LAI"])
    gpp_tl = open_single_var_ds(tl_gpp, preferred_names=["gpp", "GPP"])
    et_tl  = open_single_var_ds(tl_et,  preferred_names=["evapotrans", "evapotranspiration", "et", "ET"])

    lai_obs = open_single_var_ds(obs_lai, preferred_names=["lai", "LAI"])
    gpp_obs = open_single_var_ds(obs_gpp, preferred_names=["gpp", "GPP"])
    et_obs  = open_single_var_ds(obs_et,  preferred_names=["et", "ET", "evapotrans", "evapotranspiration"])

    # Harmonise GPP units: convert obs to model units
    gpp_obs = convert_gpp_obs_to_model_units(gpp_ens, gpp_obs)

    data = {
        "ens": {"lai": lai_ens, "gpp": gpp_ens, "et": et_ens},
        "tl":  {"lai": lai_tl,  "gpp": gpp_tl,  "et": et_tl},
        "obs": {"lai": lai_obs, "gpp": gpp_obs, "et": et_obs},
    }

    # Trim all to common overlapping time range
    data = trim_to_common_time(data)

    # Check grids across all arrays (spatial)
    combined_for_check = {}
    for grp, vars_dict in data.items():
        for vname, da in vars_dict.items():
            combined_for_check[f"{grp}_{vname}"] = da

    check_same_grid(combined_for_check)

    # Load tvt_mask and check it matches spatial grid
    mask_ds = xr.open_dataset(TVT_MASK_PATH)
    mask_var_candidates = [
        v for v in mask_ds.data_vars
        if "tvt" in v.lower() or "mask" in v.lower()
    ]
    if len(mask_var_candidates) == 0:
        raise ValueError(
            f"No obvious tvt_mask variable found in {TVT_MASK_PATH.name}. "
            f"Data vars: {list(mask_ds.data_vars)}"
        )
    mask_da = mask_ds[mask_var_candidates[0]]

    # Check mask grid against reference LAI (ens)
    check_same_grid({"mask": mask_da, "ref_lai": data["ens"]["lai"]})

    return data, mask_da


def apply_tvt_mask(
    data: Dict[str, Dict[str, xr.DataArray]],
    mask_da: xr.DataArray
) -> Dict[str, Dict[str, xr.DataArray]]:
    """
    Mask all arrays to where tvt_mask ∈ {0,1,2}.
    """
    # You currently restrict to class 2 only:
    valid_mask = mask_da.isin([2])

    masked_data = {}
    for grp, vars_dict in data.items():
        masked_data[grp] = {}
        for vname, da in vars_dict.items():
            masked_data[grp][vname] = da.where(valid_mask)
    return masked_data


def time_mean(
    data: Dict[str, Dict[str, xr.DataArray]]
) -> Dict[str, Dict[str, xr.DataArray]]:
    """
    Take mean along 'time' for each array.

    Assumes there is a 'time' dimension; if not, raises an error.
    """
    mean_data = {}
    for grp, vars_dict in data.items():
        mean_data[grp] = {}
        for vname, da in vars_dict.items():
            if "time" not in da.dims:
                raise ValueError(f"Variable '{vname}' in group '{grp}' lacks 'time' dimension.")
            mean_data[grp][vname] = da.mean(dim="time", skipna=True)
    return mean_data


def bin_by_lai(
    lai: xr.DataArray,
    gpp: xr.DataArray,
    et: xr.DataArray,
    bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten LAI, GPP, ET to 1D, bin by LAI, and compute mean/std for GPP and ET.

    Returns
    -------
    bin_centres : (nbins,) array
    gpp_mean    : (nbins,) array
    gpp_std     : (nbins,) array
    et_mean     : (nbins,) array
    et_std      : (nbins,) array
    """
    lai_vals = lai.values.ravel()
    gpp_vals = gpp.values.ravel()
    et_vals  = et.values.ravel()

    mask = np.isfinite(lai_vals) & np.isfinite(gpp_vals) & np.isfinite(et_vals)
    lai_vals = lai_vals[mask]
    gpp_vals = gpp_vals[mask]
    et_vals  = et_vals[mask]

    bin_indices = np.digitize(lai_vals, bins) - 1
    nbins = len(bins) - 1

    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    gpp_mean = np.full(nbins, np.nan)
    gpp_std  = np.full(nbins, np.nan)
    et_mean  = np.full(nbins, np.nan)
    et_std   = np.full(nbins, np.nan)

    for i in range(nbins):
        idx = bin_indices == i
        if not np.any(idx):
            continue
        g = gpp_vals[idx]
        e = et_vals[idx]
        if g.size > 0:
            gpp_mean[i] = np.nanmean(g)
            gpp_std[i]  = np.nanstd(g)
        if e.size > 0:
            et_mean[i] = np.nanmean(e)
            et_std[i]  = np.nanstd(e)

    return bin_centres, gpp_mean, gpp_std, et_mean, et_std


def make_plot(
    bins: np.ndarray,
    stats: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    gpp_units: str,
    et_units: str,
):
    """
    Create figure with two subplots (GPP and ET) vs LAI bins.

    stats[group][quantity] where quantity in:
      'gpp_mean', 'gpp_std', 'et_mean', 'et_std', 'bin_centres'
    """
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
    })

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(7.0, 3.3),
        sharex=True,
    )

    ax_gpp, ax_et = axes
    
    ax_gpp.text(0.02, 0.98, "A", transform=ax_gpp.transAxes,
            va="top", ha="left", fontsize=12, fontweight="bold")
    ax_et.text(0.02, 0.98, "B", transform=ax_et.transAxes,
           va="top", ha="left", fontsize=12, fontweight="bold")

    COLOR_OBS = "#000000"  # dark brown
    COLOR_ENS = "#482878"  # viridis puple
    COLOR_TL  = "#6CCE59"  # muted green

    group_styles = {
        "obs": {"color": COLOR_OBS, "label": "Observations"},
        "ens": {"color": COLOR_ENS, "label": "TRENDY Ensemble Mean"},
        "tl":  {"color": COLOR_TL,  "label": "TL-Emulator"},
    }

    for grp, style in group_styles.items():
        bc  = stats[grp]["bin_centres"]
        gm  = stats[grp]["gpp_mean"]
        gs  = stats[grp]["gpp_std"]
        etm = stats[grp]["et_mean"]
        ets = stats[grp]["et_std"]

        ax_gpp.errorbar(
            bc,
            gm,
            yerr=gs,
            fmt="o-",
            color=style["color"],
            markersize=5,
            linewidth=1.5,
            capsize=0,
            label=style["label"],
        )

        ax_et.errorbar(
            bc,
            etm,
            yerr=ets,
            fmt="o-",
            color=style["color"],
            markersize=5,
            linewidth=1.5,
            capsize=0,
            label=style["label"],
        )

    ax_gpp.set_xlabel("Leaf Area Index")
    ax_et.set_xlabel("Leaf Area Index")

    if gpp_units:
        ax_gpp.set_ylabel(f"GPP ({gpp_units})")
    else:
        ax_gpp.set_ylabel("Mean GPP")

    if et_units:
        ax_et.set_ylabel(f"ET ({et_units})")
    else:
        ax_et.set_ylabel("Mean ET")

    for ax in axes:
        ax.grid(False)

    # Single legend below both subplots
    handles, labels = ax_gpp.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def assert_nonzero_and_print_means(
    data_mean: Dict[str, Dict[str, xr.DataArray]]
) -> None:
    """
    For each group, check that GPP is not identically zero and print its global mean.
    Also prints ET and LAI means for sanity.
    """
    for grp in ["ens", "tl", "obs"]:
        gpp_da = data_mean[grp]["gpp"]
        et_da  = data_mean[grp]["et"]
        lai_da = data_mean[grp]["lai"]

        gpp_vals = gpp_da.values
        et_vals  = et_da.values
        lai_vals = lai_da.values

        gpp_finite = np.isfinite(gpp_vals)
        et_finite  = np.isfinite(et_vals)
        lai_finite = np.isfinite(lai_vals)

        if not np.any(gpp_finite):
            raise ValueError(f"[CHECK] Group '{grp}' GPP has no finite values.")

        gpp_mean = float(np.nanmean(gpp_vals))
        et_mean  = float(np.nanmean(et_vals)) if np.any(et_finite) else np.nan
        lai_mean = float(np.nanmean(lai_vals)) if np.any(lai_finite) else np.nan

        print(f"[CHECK] Group '{grp}': mean GPP = {gpp_mean:.3e}, "
              f"mean ET = {et_mean:.3e}, mean LAI = {lai_mean:.3f}")

        # Assert not (almost) all zero
        if np.allclose(gpp_vals[gpp_finite], 0.0, atol=1e-12):
            raise ValueError(f"[CHECK] Group '{grp}' GPP appears to be (almost) all zeros.")
        
        print(f"[CHECK] Group '{grp}' GPP passed non-zero check.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("[INFO] Loading and checking datasets...")
    data_raw, tvt_mask = load_and_check_all()

    data_masked = apply_tvt_mask(data_raw, tvt_mask)

    data_mean = time_mean(data_masked)
    print("[INFO] Computing time means...")
    # Sanity check: print means and assert GPP not all-zero
    print("[INFO] Sanity check on time-mean data:")
    assert_nonzero_and_print_means(data_mean)

    # Build LAI bins: from 0 to global max LAI across all groups
    lai_max = 0.0
    for grp in ["ens", "tl", "obs"]:
        lai_max = max(lai_max, float(data_mean[grp]["lai"].max().values))

    nbins = 20
    bins = np.linspace(0.0, lai_max, nbins + 1)

    stats: Dict[str, Dict[str, np.ndarray]] = {}

    for grp in ["ens", "tl", "obs"]:
        lai = data_mean[grp]["lai"]
        gpp = data_mean[grp]["gpp"]
        et  = data_mean[grp]["et"]

        bin_centres, gpp_mean, gpp_std, et_mean, et_std = bin_by_lai(lai, gpp, et, bins)

        stats[grp] = {
            "bin_centres": bin_centres,
            "gpp_mean": gpp_mean,
            "gpp_std": gpp_std,
            "et_mean": et_mean,
            "et_std": et_std,
        }

    # Use ENS units for labels (all GPP harmonised to this)
    gpp_units = "kg m-2 s-1"
    et_units  = "kg m-2 s-1"

    make_plot(bins, stats, OUT_FIG, gpp_units=gpp_units, et_units=et_units)
    print(f"[INFO] Saved figure to {OUT_FIG}")


if __name__ == "__main__":
    main()