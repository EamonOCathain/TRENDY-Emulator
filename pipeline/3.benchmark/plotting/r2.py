#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import math
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, List, Optional
dask.config.set(scheduler="threads")

# ---------------- Project root (for masks path only) ---------------- #
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))
from src.paths.paths import masks_dir
from src.dataset.variables import var_names  # only for expected var lists

# ---------------- Pretty names/units ---------------- #
ATTRS = {
    "nbp": {"units": "kg m-2 s-1", "long_name": "Net Biome Productivity"},
    "gpp": {"units": "kg m-2 s-1", "long_name": "Gross Primary Production"},
    "npp": {"units": "kg m-2 s-1", "long_name": "Net Primary Production"},
    "ra":  {"units": "kg m-2 s-1", "long_name": "Autotrophic Respiration"},
    "rh":  {"units": "kg m-2 s-1", "long_name": "Heterotrophic Respiration"},
    "fLuc":{"units": "kg m-2 s-1", "long_name": "Land-Use Change Emissions"},
    "fFire":{"units": "kg m-2 s-1", "long_name": "Fire Emissions"},
    "mrro":{"units": "kg m-2 s-1", "long_name": "Total Runoff"},
    "evapotrans":{"units": "kg m-2 s-1", "long_name": "Evapotranspiration"},
    "cLitter":{"units": "kg m-2", "long_name": "Carbon in Litter Pool"},
    "cSoil":{"units": "kg m-2", "long_name": "Carbon in Soil Pool"},
    "cVeg":{"units": "kg m-2", "long_name": "Carbon in Vegetation"},
    "cTotal_monthly":{"units": "kg m-2", "long_name": "Carbon in Ecosystem"},
    "mrso":{"units": "kg m-2", "long_name": "Total Soil Moisture Content"},
    "lai":{"units": "m2 m-2", "long_name": "Leaf Area Index"},
}

# ---------------- Utilities ---------------- #
def r2_from_xy(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot

def gather_points(
    preds: xr.DataArray,
    labs: xr.DataArray,
    *,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align, stack to 1D (lazy), randomly sample indices, fetch only those chunks,
    and drop NaNs on the small sample. Avoids boolean dask masks entirely.
    """
    # Align (just in case)
    preds, labs = xr.align(preds, labs, join="inner")

    # Stack lazily and keep as float32
    ps = preds.stack(z=preds.dims).astype("float32")
    ls = labs.stack(z=labs.dims).astype("float32")

    n = int(ps.sizes.get("z", 0))
    if n == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # We'll sample in batches to avoid big computes; with replacement is fine here
    rng = np.random.default_rng()
    target = min(max_points, n)
    batch_size = min(n, max(100_000, target * 2))  # 2x target is a good default
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    collected = 0
    max_attempts = 10  # safety

    attempts = 0
    while collected < target and attempts < max_attempts:
        k = min(batch_size, target - collected)
        idx = rng.integers(0, n, size=k)  # with replacement keeps it simple/light

        ps_s = ps.isel(z=idx)
        ls_s = ls.isel(z=idx)

        # Compute only this small sample
        y = ps_s.data.compute() if hasattr(ps_s.data, "compute") else ps_s.values
        x = ls_s.data.compute() if hasattr(ls_s.data, "compute") else ls_s.values
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        m = np.isfinite(x) & np.isfinite(y)
        if m.any():
            xs.append(x[m])
            ys.append(y[m])
            collected += int(m.sum())

        attempts += 1

    if not xs:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    if X.size > target:  # trim to exactly target points
        keep = rng.choice(X.size, size=target, replace=False)
        X = X[keep]
        Y = Y[keep]

    return X, Y

def select_timerange(da: xr.DataArray, t0: str | None, t1: str | None) -> xr.DataArray:
    if "time" in da.coords:
        if t0 and t1: return da.sel(time=slice(t0, t1))
        if t0:        return da.sel(time=slice(t0, None))
        if t1:        return da.sel(time=slice(None, t1))
    return da

def open_tvt_mask(mask_nc: Path) -> xr.DataArray:
    ds = xr.open_dataset(mask_nc)
    try:
        if "tvt_mask" in ds:
            da = ds["tvt_mask"]
        else:
            data_vars = list(ds.data_vars)
            if not data_vars:
                raise ValueError(f"No data variables found in {mask_nc}")
            da = ds[data_vars[0]]
        return da.load()
    finally:
        ds.close()

def resolve_pred_store(preds_root: Path, scenario: str, level: str, var: str) -> Optional[Path]:
    candidates = [
        preds_root / "zarr" / scenario / f"{level}.zarr" / var,
        preds_root / scenario / "zarr" / f"{level}.zarr" / var,
        preds_root / scenario / f"{level}.zarr" / var,
        preds_root / f"{level}.zarr" / var,
    ]
    for p in candidates:
        if (p / ".zarray").exists():
            return p
    return None

def labels_store(labels_root: Path, scenario: str, level: str, var: str) -> Path:
    return labels_root / scenario / f"{level}.zarr" / var

def list_available_scenarios(preds_dir: Path) -> List[str]:
    found: List[str] = []
    for s in ("S0", "S1", "S2", "S3"):
        base = preds_dir / "zarr" / s
        if (base / "monthly.zarr").exists() or (base / "annual.zarr").exists():
            found.append(s)
    return found

def _vars_in_store_dir(store_dir: Path) -> set[str]:
    if not store_dir.exists():
        return set()
    return {p.name for p in store_dir.iterdir() if p.is_dir() and (p / ".zarray").exists()}

def discover_available_vars(
    preds_dir: Path,
    scenarios: Iterable[str],
    monthly_expected: Iterable[str],
    annual_expected: Iterable[str],
) -> Tuple[List[str], List[str]]:
    monthly_vars_found: set[str] = set()
    annual_vars_found: set[str] = set()
    for scen in scenarios:
        mon_dir = preds_dir / "zarr" / scen / "monthly.zarr"
        ann_dir = preds_dir / "zarr" / scen / "annual.zarr"
        monthly_vars_found |= _vars_in_store_dir(mon_dir)
        annual_vars_found  |= _vars_in_store_dir(ann_dir)
        if monthly_vars_found or annual_vars_found:
            break
    monthly_vars = [v for v in monthly_expected if v in monthly_vars_found]
    annual_vars  = [v for v in annual_expected  if v in annual_vars_found]
    print(f"[INFO] Discovered monthly vars: {monthly_vars}")
    print(f"[INFO] Discovered annual  vars: {annual_vars}")
    return monthly_vars, annual_vars

def open_da_physical(
    preds_dir: Path, labels_dir: Path, scenario: str, var: str, is_annual: bool
) -> tuple[xr.DataArray, xr.DataArray]:
    level = "annual" if is_annual else "monthly"
    p_store = resolve_pred_store(preds_dir, scenario, level, var)
    if p_store is None:
        raise FileNotFoundError(f"preds zarr array not found for {scenario}/{level}/{var} under {preds_dir}")
    p_parent = p_store.parent
    ds_p = xr.open_zarr(p_parent, consolidated=(p_parent / ".zmetadata").exists())
    if var not in ds_p.data_vars:
        raise KeyError(f"{var} not found in preds dataset {p_parent} (have: {list(ds_p.data_vars)[:10]}...)")
    preds = ds_p[var].astype("float32")
    l_store = labels_store(labels_dir, scenario, level, var)
    if not (l_store / ".zarray").exists():
        raise FileNotFoundError(f"labels zarr array missing: {l_store} (expected <labels_dir>/<Sx>/{level}.zarr/{var})")
    l_parent = l_store.parent
    ds_l = xr.open_zarr(l_parent, consolidated=(l_parent / ".zmetadata").exists())
    if var not in ds_l.data_vars:
        raise KeyError(f"{var} not found in labels dataset {l_parent} (have: {list(ds_l.data_vars)[:10]}...)")
    labs = ds_l[var].astype("float32")
    return preds, labs

def get_subset_points_for_var(
    preds_dir: Path,
    labels_dir: Path,
    scenarios: List[str],
    var: str,
    *,
    is_annual: bool,
    tvt_mask_da: xr.DataArray | None,
    subset: str,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs_all: List[np.ndarray] = []
    ys_all: List[np.ndarray] = []
    EARLY_T0, EARLY_T1 = "1901-01-01", "1918-12-31"
    LATE_T0,  LATE_T1  = "2018-01-01", "2023-12-31"

    for scen in scenarios:
        try:
            preds, labs = open_da_physical(preds_dir, labels_dir, scen, var, is_annual=is_annual)
        except Exception as e:
            print(f"[WARN] Skipping scenario {scen} var {var}: {e}")
            continue

        preds_locs = labs_locs = None
        if subset in ("test_locs_full", "combined") and tvt_mask_da is not None:
            mask = tvt_mask_da
            if "time" in preds.coords:
                mask_b = mask.broadcast_like(preds, exclude=("time",))
                preds_locs = preds.where(mask_b == 2)
                labs_locs  = labs.where(mask_b == 2)
            else:
                preds_locs = preds.where(mask == 2)
                labs_locs  = labs.where(mask == 2)

        chunks: List[Tuple[xr.DataArray, xr.DataArray]] = []
        if subset == "test_locs_full":
            if preds_locs is not None and labs_locs is not None:
                chunks.append((preds_locs, labs_locs))
        elif subset == "global_early":
            chunks.append((select_timerange(preds, EARLY_T0, EARLY_T1),
                           select_timerange(labs,  EARLY_T0,  EARLY_T1)))
        elif subset == "global_late":
            chunks.append((select_timerange(preds, LATE_T0, LATE_T1),
                           select_timerange(labs,  LATE_T0,  LATE_T1)))
        elif subset == "combined":
            # Start with an all-False mask over the preds grid
            sel_mask = xr.zeros_like(preds, dtype=bool)

            # Test locations (value==2 in tvt_mask)
            if tvt_mask_da is not None:
                mloc = tvt_mask_da
                if "time" in preds.coords:
                    mloc = mloc.broadcast_like(preds, exclude=("time",))
                sel_mask = sel_mask | (mloc == 2)

            # Time windows: build a 1D time mask with cftime-safe logic, then broadcast
            if "time" in preds.coords:
                t = preds["time"]                      # 1D time coord (cftime-safe)
                # Select the indices (works with cftime calendars)
                t_early = select_timerange(t, EARLY_T0, EARLY_T1)
                t_late  = select_timerange(t, LATE_T0,  LATE_T1)
                # 1D boolean mask for time membership
                time_mask_1d = t.isin(t_early) | t.isin(t_late)
                # Auto-broadcast across spatial dims in the OR
                sel_mask = sel_mask | time_mask_1d

            # Apply the union mask once (no double counting)
            pr = preds.where(sel_mask)
            lb = labs.where(sel_mask)
            chunks.append((pr, lb))

        for (pr, lb) in chunks:
            x, y = gather_points(pr, lb, max_points=max_points)
            if x.size:
                xs_all.append(x); ys_all.append(y)

    if xs_all:
        return np.concatenate(xs_all), np.concatenate(ys_all)
    else:
        return np.array([]), np.array([])

def make_density_alpha(x: np.ndarray, y: np.ndarray, bins: int = 100, alpha_min: float = 0.05, alpha_max: float = 0.9) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=np.float32)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    if x_min == x_max: x_max = x_min + 1e-6
    if y_min == y_max: y_max = y_min + 1e-6
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    x_bin = np.clip(np.digitize(x, xedges) - 1, 0, H.shape[0] - 1)
    y_bin = np.clip(np.digitize(y, yedges) - 1, 0, H.shape[1] - 1)
    counts = H[x_bin, y_bin].astype(np.float32)
    cmin, cmax = counts.min(), counts.max()
    if cmax == cmin:
        return np.full_like(counts, (alpha_min + alpha_max) * 0.5)
    norm = (counts - cmin) / (cmax - cmin)
    return alpha_min + norm * (alpha_max - alpha_min)

def pretty_var_name(var: str) -> str:
    return ATTRS.get(var, {}).get("long_name", var)

def pretty_units(var: str, fallback: str = "") -> str:
    return ATTRS.get(var, {}).get("units", fallback)

def draw_panel(
    *,
    fig_title: str,
    out_path: Path,
    variables_monthly: List[str],
    variables_annual: List[str],
    preds_dir: Path,
    labels_dir: Path,
    scenarios: List[str],
    tvt_mask_da: xr.DataArray | None,
    subset: str,
    ncols: int = 3,
    max_points: int = 1_000_000
):
    all_vars = list(variables_monthly) + list(variables_annual)
    nvars = len(all_vars)
    if nvars == 0:
        print(f("[WARN] No variables available for plotting: {out_path.name}"))
        return

    nrows = math.ceil(nvars / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes = axes.ravel()

    for i, var in enumerate(all_vars):
        ax = axes[i]
        is_annual = var in variables_annual
        x, y = get_subset_points_for_var(
            preds_dir, labels_dir, scenarios, var,
            is_annual=is_annual, tvt_mask_da=tvt_mask_da, subset=subset,
            max_points=max_points,
        )

        meta = ATTRS.get(var, {})
        long_name = meta.get("long_name", var)
        units = meta.get("units", "")

        if x.size == 0:
            ax.set_title(f"{long_name} — no data", fontsize=10)
            ax.axis("off")
            continue

        alphas = make_density_alpha(x, y, bins=120, alpha_min=0.05, alpha_max=0.9)
        buckets = np.linspace(0.0, 1.0, 11)
        idx = np.digitize(alphas, buckets, right=True)
        for b in range(1, len(buckets)):
            m = idx == b
            if not np.any(m):
                continue
            ax.scatter(x[m], y[m], s=2, c="blue", alpha=float(buckets[b]), linewidths=0, edgecolors="none")

        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        if lo == hi: hi = lo + 1e-6
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)

        r2 = r2_from_xy(x, y)
        ax.set_title(f"{long_name}  (R²={r2:.3f})", fontsize=10)
        if units:
            ax.set_xlabel(f"Labels ({units})")
            ax.set_ylabel(f"Predictions ({units})")
        else:
            ax.set_xlabel("Labels"); ax.set_ylabel("Predictions")

    for j in range(nvars, len(axes)): axes[j].axis("off")
    fig.suptitle(fig_title, fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")

# ---------------- Main ---------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_name", required=True)
    ap.add_argument("--preds_dir", type=Path, required=True)
    ap.add_argument("--labels_dir", type=Path, required=True)
    ap.add_argument("--plot_dir", type=Path, required=True)
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--mask_nc", type=Path, default=masks_dir / "tvt_mask.nc",
                    help="Path to tvt_mask.nc (value==2 marks test locations).")
    ap.add_argument("--max_points", type=int, default=1_000_000,
                    help="Maximum number of points to sample for each variable.")
    ap.add_argument("--panel", choices=["combined", "test_locs_full", "global_early", "global_late"],
                    help="If provided, render only this panel (for SLURM array).")
    args = ap.parse_args()

    max_points = args.max_points

    scenarios = list_available_scenarios(args.preds_dir)
    if not scenarios:
        raise SystemExit(f"No scenarios (S0–S3) found under {args.preds_dir}")
    print(f"[INFO] Using scenarios: {', '.join(scenarios)}")

    monthly_expected = (
        list(var_names.get("monthly", []))
        or list(var_names.get("monthly_outputs", []))
        or list(var_names.get("monthly_fluxes", []))
    )
    annual_expected = (
        list(var_names.get("annual", []))
        or list(var_names.get("annual_outputs", []))
        or list(var_names.get("annual_states", []))
    )

    monthly_vars, annual_vars = discover_available_vars(
        preds_dir=args.preds_dir,
        scenarios=scenarios,
        monthly_expected=monthly_expected,
        annual_expected=annual_expected,
    )
    if not monthly_vars and not annual_vars:
        raise SystemExit("No monthly/annual variables found in prediction zarrs that match expected lists.")

    try:
        tvt_da = open_tvt_mask(args.mask_nc)
    except Exception as e:
        print(f"[WARN] Could not open tvt mask at {args.mask_nc}: {e}\nProceeding without test-location subsets.")
        tvt_da = None

    out_root = Path(args.plot_dir) / args.job_name / "global" / "r2_scatter_density"
    out_root.mkdir(parents=True, exist_ok=True)

    # Map each panel to its title and output filename
    PANEL_META = {
        "combined":      ("Combined Test Subsets (All Scenarios)", "combined_all_subsets_all_scenarios.png"),
        "test_locs_full":("Test Locations — 1901–2023 (All Scenarios)", "test_locations_full_all_scenarios.png"),
        "global_early":  ("Global — Early Test Period 1901–1918 (All Scenarios)", "global_early_1901_1918_all_scenarios.png"),
        "global_late":   ("Global — Late Test Period 2018–2023 (All Scenarios)", "global_late_2018_2023_all_scenarios.png"),
    }

    # If --panel provided: render only that one (for SLURM array)
    if args.panel:
        title, fname = PANEL_META[args.panel]
        print(f"[INFO] Rendering single panel '{args.panel}' ({title}) …")
        draw_panel(
            fig_title=title,
            out_path=out_root / fname,
            variables_monthly=monthly_vars,
            variables_annual=annual_vars,
            preds_dir=args.preds_dir,
            labels_dir=args.labels_dir,
            scenarios=scenarios,
            tvt_mask_da=tvt_da,
            subset=args.panel,
            ncols=args.ncols,
            max_points=max_points,
        )
        return

    for key in ["combined", "test_locs_full", "global_early", "global_late"]:
        title, fname = PANEL_META[key]
        print(f"[INFO] Starting panel '{key}' ({title}) …")
        draw_panel(
            fig_title=title,
            out_path=out_root / fname,
            variables_monthly=monthly_vars,
            variables_annual=annual_vars,
            preds_dir=args.preds_dir,
            labels_dir=args.labels_dir,
            scenarios=scenarios,
            tvt_mask_da=tvt_da,
            subset=key,
            ncols=args.ncols,
            max_points=max_points,
        )

if __name__ == "__main__":
    main()