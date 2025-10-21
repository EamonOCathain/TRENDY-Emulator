#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xarray as xr
import numpy as np
import cftime

# ---------------- Project imports ---------------- #
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.analysis.visualisation import plot_three_maps, robust_limits
from src.dataset.variables import var_names
from src.utils.tools import slurm_shard

def _open_pair(labels_root: Path, preds_root: Path, res: str):
    assert res in {"monthly", "annual"}
    lab = xr.open_zarr(labels_root / f"{res}.zarr", consolidated=True, decode_times=False)
    pre = xr.open_zarr(preds_root  / f"{res}.zarr", consolidated=True, decode_times=False)
    print(pre)                     # quick summary
    print(pre.coords)              # all coords
    print(pre['lat'])              # lat coordinate values
    print(pre['lon'])              # lon coordinate values
    print(pre['lat'].values[:10])  # first 10 lat values as numpy array
    print(pre['lon'].values[:10])  # first 10 lon values as numpy array
    return lab, pre

def _time_indices(res: str, ntime: int):
    mid_default = 738 if res == "monthly" else 61
    i1 = 2
    im = min(max(0, mid_default), max(0, ntime - 1))
    il = max(0, ntime - 1)
    # guard against accidental OOB if ntime is tiny
    return sorted({i for i in (i1, im, il) if 0 <= i < ntime})

def _timestep_label(da: xr.DataArray, ti: int) -> str:
    if "time" not in da.coords:
        return f"timestep={ti}"
    time_coord = da["time"]
    units = time_coord.attrs.get("units")
    calendar = time_coord.attrs.get("calendar", "standard")
    try:
        val = time_coord.isel(time=ti).values.item()
        dt = cftime.num2date(val, units=units, calendar=calendar)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return f"timestep={ti}"

def _units_for(da: xr.DataArray) -> str:
    return da.attrs.get("units") or da.attrs.get("unit") or ""

def _normalize_1d_coord(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values)
    return vals

def normalize_lat_lon(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure lon in [-180, 180) (if needed), lat/lon strictly increasing, and drop duplicates.
    Works on a 2D (lat, lon) or 3D (time, lat, lon) DataArray.
    """
    out = da

    # Wrap longitudes if they look like 0..360
    try:
        lon = out["lon"].values
        if np.nanmax(lon) > 180:
            lon_wrapped = ((lon + 180) % 360) - 180
            out = out.assign_coords(lon=lon_wrapped)
    except Exception:
        pass

    # Sort and deduplicate lat
    if "lat" in out.coords:
        lat_vals = np.asarray(out["lat"].values)
        lat_order = np.argsort(lat_vals)
        lat_sorted = lat_vals[lat_order]
        lat_keep = np.concatenate(([True], np.diff(lat_sorted) > 0))
        out = out.isel(lat=lat_order[lat_keep])

    # Sort and deduplicate lon
    if "lon" in out.coords:
        lon_vals = np.asarray(out["lon"].values)
        lon_order = np.argsort(lon_vals)
        lon_sorted = lon_vals[lon_order]
        lon_keep = np.concatenate(([True], np.diff(lon_sorted) > 0))
        out = out.isel(lon=lon_order[lon_keep])

    return out

scenarios = ["S0", "S1", "S2", "S3"]
tasks = []
for res in ["monthly", "annual"]:
    for scenario in scenarios:
        vars_ = var_names["monthly_outputs"] if res == "monthly" else var_names["annual_outputs"]
        for v in vars_:
            tasks.append((res, scenario, v))

tasks_this_shard = slurm_shard(tasks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True, type=Path, help="Root containing per-scenario/<res>.zarr")
    ap.add_argument("--preds_dir",  required=True, type=Path, help="Root containing per-scenario/<res>.zarr")
    ap.add_argument("--out_dir",    required=True, type=Path)
    ap.add_argument("--data_cmap", default=None)
    ap.add_argument("--bias_cmap", default="RdBu_r")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--data_pct_low", type=float, default=1.0)
    ap.add_argument("--data_pct_high", type=float, default=99.0)
    ap.add_argument("--bias_pct", type=float, default=99.0)
    args = ap.parse_args()

    for res, scenario, var in tasks_this_shard:
        print(f"[INFO] Processing {res} {scenario} {var} …")

        labels_dir = args.labels_dir / scenario
        preds_dir  = args.preds_dir  / scenario / "zarr"
        out_root   = args.out_dir    / scenario / res / var
        out_root.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Opening {scenario}/{res} zarrs …")
        ds_labels, ds_preds = _open_pair(labels_dir, preds_dir, res)
        try:
            # must have time dim
            if "time" not in ds_labels.dims or "time" not in ds_preds.dims:
                print(f"[WARN] One of the datasets for {scenario}/{res} has no 'time' dim; skipping {var}.")
                continue

            # variable must exist in both
            if var not in ds_labels.data_vars or var not in ds_preds.data_vars:
                print(f"[SKIP] {scenario}/{res}: '{var}' not present in both datasets.")
                continue

            ntime = min(ds_labels.sizes["time"], ds_preds.sizes["time"])
            if ntime <= 0:
                print(f"[WARN] {scenario}/{res}: no overlapping time; skipping {var}.")
                continue

            # Select the variable and normalize coordinates BEFORE comparing/reindexing
            lab = normalize_lat_lon(ds_labels[var])
            pre = normalize_lat_lon(ds_preds[var])

            # must be gridded
            if not {"lat", "lon"}.issubset(lab.dims) or not {"lat", "lon"}.issubset(pre.dims):
                print(f"[SKIP] {scenario}/{res}:{var} is not lat/lon gridded.")
                continue

            # align grids if necessary (after normalization lon ranges should match)
            same_lat = np.array_equal(lab["lat"].values, pre["lat"].values)
            same_lon = np.array_equal(lab["lon"].values, pre["lon"].values)
            if not (same_lat and same_lon):
                try:
                    pre = pre.reindex_like(lab, method=None)
                except Exception as e:
                    print(f"[WARN] Could not reindex pred grid for {scenario}/{res}:{var} -> {e}; skipping.")
                    continue

            t_indices = _time_indices(res, ntime)
            print(f"[INFO] {scenario}/{res}:{var} plotting at steps={t_indices}")

            units = _units_for(lab)

            for ti in t_indices:
                l2d = lab.isel(time=ti)
                p2d = pre.isel(time=ti)
                b2d = l2d - p2d

                timestep_str = _timestep_label(lab, ti)

                # Robust limits
                data_vmin, data_vmax = robust_limits(
                    [l2d.values, p2d.values],
                    mode="range",
                    pct_low=args.data_pct_low,
                    pct_high=args.data_pct_high,
                    zero_floor_if_nonneg=True,
                )
                bmin, bmax = robust_limits(
                    b2d.values,
                    mode="symmetric",
                    pct=args.bias_pct,
                    default=1.0,
                )
                bias_halfwidth = max(abs(bmin), abs(bmax))

                out_path = out_root / f"{var}_{res}_t{ti:05d}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                titles = [
                    f"{var}: Labels · {timestep_str}",
                    f"{var}: Predictions · {timestep_str}",
                    f"{var}: Bias · {timestep_str}",
                ]
                cbar_labels = [
                    units or var,
                    units or var,
                    f"{units} (label - pred)" if units else "label - pred",
                ]

                plot_three_maps(
                    label2d=l2d,
                    pred2d=p2d,
                    bias2d=b2d,
                    titles=titles,
                    cbar_labels=cbar_labels,
                    out_path=out_path,
                    data_vmin=data_vmin,
                    data_vmax=data_vmax,
                    bias_vmax=bias_halfwidth,
                    data_cmap=args.data_cmap,
                    bias_cmap=args.bias_cmap,
                    overwrite=args.overwrite,
                )
        finally:
            ds_labels.close()
            ds_preds.close()

    print("[DONE] All figures written.")

if __name__ == "__main__":
    main()