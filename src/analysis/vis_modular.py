# plotkit.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import re
import sys
from matplotlib.ticker import FixedLocator, FixedFormatter  # add at top


project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.analysis.process_arrays import find_var_groups

# =========================
# Public types & registry
# =========================

# --- updated PlotSpec dataclass ---
@dataclass
class PlotSpec:
    """
    Rendering options independent of data.
    Supply only what you need; sensible defaults are used otherwise.
    """
    kind: Optional[str] = None        # "map" | "line" | "scatter" | "hist" (extensible)
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None

    # Color/limits
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    robust: bool = True
    robust_mode: str = "range"        # "range" or "symmetric"
    pct_low: float = 1.0
    pct_high: float = 99.0
    pct: float = 99.0                 # for symmetric

    # Scatter options
    s: float = 6.0
    alpha: float = 0.25
    add_1to1: bool = True
    equal_axes: bool = True
    show_r2: bool = True
    

    # Line options
    lw: float = 1.5

    # Colorbar
    add_colorbar: bool = True
    cbar_label: Optional[str] = None
    cbar_width_pct: float = 3.0       
    cbar_height_pct: float = 70.0    

    # Map configs
    use_cartopy: bool = True
    show_coastlines: bool = False        
    coastlines_lw: float = 0.1  
    show_borders: bool = False        

    # Saving configs
    dpi: int = 300
    bbox_inches: str = "tight"
    transparent: bool = False

    # Free-form extension point
    extras: Dict[str, Any] = field(default_factory=dict)
    
# Global registry: kind -> plotter function
# Each plotter: f(item, ax, spec) -> matplotlib Axes
PLOTTERS: Dict[str, Callable[[Any, plt.Axes, PlotSpec], plt.Axes]] = {}

def register_plotter(kind: str):
    """Decorator to register a plotting function under a 'kind'."""
    def _wrap(func: Callable[[Any, plt.Axes, PlotSpec], plt.Axes]):
        PLOTTERS[kind] = func
        return func
    return _wrap

# =========================
# Core entry points
# =========================

def plot_one(
    item: Any,
    *,
    ax: Optional[plt.Axes] = None,
    spec: Optional[PlotSpec] = None,
    out_path: Optional[str] = None,
) -> Optional[plt.Axes]:
    """
    Thin dispatcher: validate, choose a plotter, and draw.
      - 'item' can be a DataArray/np.ndarray or (x,y) tuple.
      - If spec.kind is None, a small inference is attempted.
      - If out_path is provided, saves the figure and returns None.
      - Otherwise returns the matplotlib Axes used.
    """
    spec = spec or PlotSpec()

    kind = spec.kind or _infer_kind(item)
    if kind not in PLOTTERS:
        raise ValueError(f"No plotter registered for kind='{kind}'")

    created = False
    fig = None

    # Create an Axes if needed. For maps + cartopy, create a GeoAxes centered on Greenwich.
    if ax is None:
        created = True
        if kind == "map" and spec.use_cartopy:
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
        else:
            fig, ax = plt.subplots(figsize=(5, 4))

    # Dispatch
    ax = PLOTTERS[kind](item, ax, spec)

    if created:
        plt.tight_layout()

    # Handle saving (only when we created the figure here)
    if out_path and created:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        (fig or ax.figure).savefig(
            out_path,
            dpi=spec.dpi,
            bbox_inches=spec.bbox_inches,
            transparent=spec.transparent,
        )
        plt.close(fig or ax.figure)
        print(f"[OK] Saved: {out_path}")
        return None

    return ax

def stack_plots(
    items: Sequence[Any],
    *,
    specs: Optional[Sequence[Optional[PlotSpec]]] = None,
    n_cols: int = 3,
    figsize_per_cell: Tuple[float, float] = (5.0, 4.0),
    suptitle: Optional[str] = None,
    sharex: bool | str = False,
    sharey: bool | str = False,
    out_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray] | None:
    """
    Build a grid of subplots and render each item via plot_one.
    Creates Cartopy GeoAxes for cells whose inferred kind is 'map' and spec.use_cartopy=True.
    """
    n = len(items)
    if n == 0:
        raise ValueError("No items to plot.")
    n_cols = max(1, int(n_cols))
    n_rows = (n + n_cols - 1) // n_cols
    fig_w = figsize_per_cell[0] * n_cols
    fig_h = figsize_per_cell[1] * n_rows

    # Prepare specs and infer kinds up front
    specs = list(specs) if specs is not None else []
    specs += [None] * (n - len(specs))
    kinds = [(sp.kind if sp and sp.kind else _infer_kind(it)) for it, sp in zip(items, specs)]
    use_cart = [(sp.use_cartopy if sp is not None else True) for sp in specs]

    # Manually create each Axes so maps get GeoAxes
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = np.empty((n_rows, n_cols), dtype=object)
    for idx in range(n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        if idx < n and kinds[idx] == "map" and use_cart[idx]:
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection=ccrs.PlateCarree(central_longitude=0))
        else:
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        axes[r, c] = ax

    # Render
    for ax, item, sp in zip(axes.ravel(), items, specs):
        plot_one(item, ax=ax, spec=sp)

    # Hide unused cells
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sp0 = next((s for s in specs if s is not None), PlotSpec())
        fig.savefig(out_path, dpi=sp0.dpi, bbox_inches=sp0.bbox_inches, transparent=sp0.transparent)
        plt.close(fig)
        print(f"[OK] Saved: {out_path}")
        return None
    return fig, axes

def stack_maps(
    items: Sequence[xr.DataArray],
    *,
    specs: Optional[Sequence[Optional[PlotSpec]]] = None,
    figsize_per_cell: Tuple[float, float] = (6.0, 3.5),
    ncols: int = 1,
    suptitle: Optional[str] = None,
    out_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray] | None:
    """
    Stack multiple map plots in a grid layout.

    Parameters
    ----------
    items : sequence of DataArray
        Each item must be 2D with ('lat','lon').
    specs : sequence of PlotSpec or None
        Per-plot specs. If fewer than items, missing ones are filled with defaults.
    figsize_per_cell : (float, float)
        Width, height per subplot.
    ncols : int
        Number of columns in the grid.
    suptitle : str, optional
        Title for the whole figure.
    out_path : str or Path, optional
        If given, save to this path and return None.

    Returns
    -------
    (fig, axes) if out_path is None, else None.
    """
    n = len(items)
    if n == 0:
        raise ValueError("No items to plot.")

    # prepare specs
    specs = list(specs) if specs is not None else []
    specs += [None] * (n - len(specs))

    # layout
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))

    # figure size
    fig_w = figsize_per_cell[0] * ncols
    fig_h = figsize_per_cell[1] * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": ccrs.PlateCarree()} if any(
            (sp or PlotSpec()).use_cartopy for sp in specs
        ) else None,
        squeeze=False,
    )

    # flatten axes
    axes_flat = axes.ravel()

    for idx, (item, sp) in enumerate(zip(items, specs)):
        sp = sp or PlotSpec(kind="map")
        ax = axes_flat[idx]
        # if projection mismatches, re-add properly
        if sp.use_cartopy and not hasattr(ax, "projection"):
            ax.remove()
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection=ccrs.PlateCarree())
            axes_flat[idx] = ax
        plot_one(item, ax=ax, spec=sp)

    # remove unused axes
    for ax in axes_flat[n:]:
        ax.remove()

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sp0 = next((s for s in specs if s is not None), PlotSpec())
        fig.savefig(out_path, dpi=sp0.dpi, bbox_inches=sp0.bbox_inches, transparent=sp0.transparent)
        plt.close(fig)
        print(f"[OK] Saved: {out_path}")
        return None

    return fig, axes

# ------------------ Stacks which take in DataSets (of pred-label(-bias)) ------------------

# Take in a ds and plot them as prediction label pair maps
def stack_map_pairs(
    ds: xr.Dataset,
    *,
    suptitle: Optional[str] = None,
    out_path: Optional[Path] = None,
    use_cartopy: bool = True,
    show_coastlines: bool = False,
    robust: bool = True,
    robust_mode: str = "range",          # "range" or "symmetric"
    pct_low: float = 1.0,
    pct_high: float = 99.0,
    cmap_label: Optional[str] = None,    # e.g., "cividis"
    cmap_pred: Optional[str] = None,     # e.g., "cividis"
    cmap_bias: Optional[str] = "RdBu_r", # symmetric default for bias
    ncols: int = 3,                      # used only if no pairs are found
) -> Tuple[plt.Figure, np.ndarray] | None:
    """
    Plot a grid of maps from `ds`.

    If (Label_*, Predicted_*[, Bias_*]) groups exist:
      - Uses 2 columns (Label | Predicted) if no Bias in any group.
      - Uses 3 columns (Label | Predicted | Bias) if any group includes Bias.
    Otherwise:
      - Plots every 2D ('lat','lon') variable in `ds` in a grid with `ncols` columns.

    If only one panel results, use a single column.
    """
    pairs = find_var_groups(ds)

    # Fallback: no pairs → plot all 2D lat/lon vars with given ncols
    if not pairs:
        items: List[xr.DataArray] = []
        specs: List[PlotSpec] = []
        for name, da in ds.data_vars.items():
            if isinstance(da, xr.DataArray) and da.ndim == 2 and {"lat", "lon"}.issubset(da.dims):
                items.append(da)
                specs.append(PlotSpec(
                    kind="map",
                    title=name,
                    use_cartopy=use_cartopy,
                    show_coastlines=show_coastlines,
                    robust=robust,
                    robust_mode=robust_mode,
                    pct_low=pct_low,
                    pct_high=pct_high,
                    cmap=cmap_label,  # reuse label cmap for generic maps
                ))
        if not items:
            raise ValueError("No 2D ('lat','lon') variables found to plot.")

        # If only one array, force one column
        fallback_cols = 1 if len(items) == 1 else max(1, int(ncols))
        return stack_maps(
            items,
            specs=specs,
            ncols=fallback_cols,
            suptitle=suptitle,
            out_path=out_path,
        )

    # Paired mode
    has_bias = any(len(grp) == 3 for grp in pairs)
    paired_default_cols = 3 if has_bias else 2

    items: List[xr.DataArray] = []
    specs: List[PlotSpec] = []

    def _suffix(name: str) -> str:
        m = re.match(r"^(Label|Predicted|Bias)_(.+)$", name)
        return m.group(2) if m else name

    for grp in pairs:
        label_name, pred_name = grp[0], grp[1]
        da_label = ds[label_name]
        da_pred  = ds[pred_name]

        for da in (da_label, da_pred):
            if not isinstance(da, xr.DataArray) or da.ndim != 2 or not {"lat","lon"}.issubset(da.dims):
                raise ValueError(f"Expected 2D ('lat','lon') DataArray, got {da.name} with dims {da.dims}")

        suf = _suffix(label_name)

        items.append(da_label)
        specs.append(PlotSpec(
            kind="map",
            title=f"Label • {suf}",
            use_cartopy=use_cartopy,
            show_coastlines=show_coastlines,
            robust=robust,
            robust_mode=robust_mode,
            pct_low=pct_low,
            pct_high=pct_high,
            cmap=cmap_label,
        ))

        items.append(da_pred)
        specs.append(PlotSpec(
            kind="map",
            title=f"Predicted • {suf}",
            use_cartopy=use_cartopy,
            show_coastlines=show_coastlines,
            robust=robust,
            robust_mode=robust_mode,
            pct_low=pct_low,
            pct_high=pct_high,
            cmap=cmap_pred,
        ))

        if len(grp) == 3:
            bias_name = grp[2]
            da_bias = ds[bias_name]
            if not isinstance(da_bias, xr.DataArray) or da_bias.ndim != 2 or not {"lat","lon"}.issubset(da_bias.dims):
                raise ValueError(f"Expected 2D ('lat','lon') DataArray, got {da_bias.name} with dims {da_bias.dims}")
            items.append(da_bias)
            specs.append(PlotSpec(
                kind="map",
                title=f"Bias • {suf}",
                use_cartopy=use_cartopy,
                show_coastlines=show_coastlines,
                robust=robust,
                robust_mode="symmetric" if robust_mode == "range" else robust_mode,
                pct_low=pct_low,
                pct_high=pct_high,
                cmap=cmap_bias,
            ))

    # If only one panel overall, force one column; else use paired default
    paired_cols = 1 if len(items) == 1 else paired_default_cols

    return stack_maps(
        items,
        specs=specs,
        ncols=paired_cols,
        suptitle=suptitle,
        out_path=out_path,
    )
    
def stack_global_r2_scatter(
    ds: xr.Dataset,
    *,
    ncols: int = 2,
    suptitle: str | None = None,
    out_path: str | Path | None = None,
):
    """
    For each (Label_*, Predicted_*) pair in `ds`, make a scatter subplot of
    Observed (Label) vs Predicted. Points are flattened across all dims and
    NaNs are ignored. R² is shown in the title via _plot_scatter.

    If only one pair, force a single column.
    """
    pairs = find_var_groups(ds)
    if not pairs:
        raise ValueError("No (Label_*, Predicted_*) pairs found in dataset.")

    items = []
    specs = []
    for grp in pairs:
        label_name, pred_name = grp[0], grp[1]
        y = ds[label_name]
        yhat = ds[pred_name]
        y_aligned, yhat_aligned = xr.align(y, yhat, join="inner")

        items.append((y_aligned, yhat_aligned))
        suffix = label_name.split("_", 1)[1] if "_" in label_name else label_name
        specs.append(PlotSpec(
            kind="scatter",
            title=suffix,
            xlabel="Observed",
            ylabel="Predicted",
            add_1to1=True,
            equal_axes=True,
            show_r2=True,
            s=6.0,
            alpha=0.25,
        ))

    # collapse to one column if only one panel
    use_cols = 1 if len(items) == 1 else ncols

    return stack_plots(
        items,
        specs=specs,
        n_cols=use_cols,
        figsize_per_cell=(5.0, 4.0),
        suptitle=suptitle,
        out_path=out_path,
    )
    
def plot_timeseries_pairs_grid(
    ds: xr.Dataset,
    *,
    ncols: int = 2,
    suptitle: str | None = None,
    out_path: str | Path | None = None,
):
    """
    For each (Label_*, Predicted_*) pair in `ds`, make one subplot that overlays
    the two 1D time series (Label in blue, Predicted in orange). Plots are laid
    out in a grid with `ncols` columns (default 2). If only one pair, use one column.
    """
    pairs = find_var_groups(ds)
    if not pairs:
        raise ValueError("No (Label_*, Predicted_*) pairs found in dataset.")

    items = []
    specs = []
    for grp in pairs:
        label_name, pred_name = grp[0], grp[1]
        y = ds[label_name]
        yhat = ds[pred_name]
        y_aligned, yhat_aligned = xr.align(y, yhat, join="inner")

        items.append([y_aligned, yhat_aligned])

        suffix = label_name.split("_", 1)[1] if "_" in label_name else label_name
        specs.append(PlotSpec(
            kind="line_multi",
            title=suffix,
            extras={
                "series_labels": ["Label", "Predicted"],
                "line_cmap": "tab10",
                "legend_title": None,
                "legend_ncol": 1,
                "legend_fontsize": 8,
                "width_factor": 1.3,
            },
            lw=1.6,
        ))

    # collapse to one column if only one panel
    use_cols = 1 if len(items) == 1 else ncols

    return stack_plots(
        items,
        specs=specs,
        n_cols=use_cols,
        figsize_per_cell=(5.0, 3.2),
        suptitle=suptitle,
        out_path=out_path,
    )
    
def scatter_grid_from_pairs(
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    ncols: int = 3,
    suptitle: Optional[str] = None,
    out_path: Optional[str | Path] = None,
    subsample: Optional[int] = 200_000,   # light memory guard
    density_alpha: bool = True,
) -> Tuple[plt.Figure, np.ndarray] | None:
    """
    Render a grid of per-variable scatter plots from {name: (y_obs, y_pred)}.
    """
    items: List[Tuple[np.ndarray, np.ndarray]] = []
    specs: List[PlotSpec] = []
    for name, (y, yhat) in pairs.items():
        items.append((y, yhat))
        specs.append(PlotSpec(
            kind="scatter",
            title=name,
            xlabel="Observed",
            ylabel="Predicted",
            add_1to1=True,
            equal_axes=True,
            show_r2=True,
            s=6.0,
            alpha=0.25,  # fallback when density_alpha=False
            extras={
                "density_alpha": density_alpha,
                "bins": 200,
                "alpha_min": 0.05,
                "alpha_max": 0.9,
                "subsample": subsample,
            },
        ))
    use_cols = 1 if len(items) == 1 else ncols
    return stack_plots(
        items,
        specs=specs,
        n_cols=use_cols,
        figsize_per_cell=(5.0, 4.0),
        suptitle=suptitle,
        out_path=out_path,
    )
    
# =========================
# Kind inference & helpers
# =========================

def _infer_kind(item: Any) -> str:
    """
    Minimal 'kind' inference:
      - tuple/list of length 2 -> 'scatter'
      - DataArray with 'time' dim and 1D -> 'line'
      - DataArray with ('lat','lon') dims (2D) -> 'map'
      - fallback -> 'hist'
    You can override by passing PlotSpec(kind=...).
    """
    # Scatter: (x, y)
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return "scatter"

    # xarray heuristics
    if isinstance(item, xr.DataArray):
        dims = set(item.dims)
        if "time" in dims and item.ndim == 1:
            return "line"
        if {"lat", "lon"}.issubset(dims) and item.ndim == 2:
            return "map"

    # Default
    return "hist"


def _as_numpy(da: Union[xr.DataArray, np.ndarray]) -> np.ndarray:
    """Safely convert an xarray or numpy array to a numpy ndarray."""
    if isinstance(da, xr.DataArray):
        return da.values
    return np.asarray(da)


def _robust_limits_one(
    arr: Union[xr.DataArray, np.ndarray],
    *,
    mode: str = "range",            # "range" or "symmetric"
    pct_low: float = 1.0,
    pct_high: float = 99.0,
    pct: float = 99.0,
    zero_floor_if_nonneg: bool = True,
    default: float = 1.0
) -> Tuple[Optional[float], Optional[float]]:
    """
    Robust plotting limits for a single array (NaNs ignored).
    Returns (vmin, vmax) or (-default,+default) if no finite data for symmetric mode.
    """
    x = _as_numpy(arr).ravel()
    m = np.isfinite(x)
    if not m.any():
        return (-default, default) if mode == "symmetric" else (None, None)

    x = x[m]
    if mode == "range":
        data_min = float(np.nanmin(x))
        data_max = float(np.nanmax(x))
        vmin = float(np.nanpercentile(x, pct_low))
        vmax = float(np.nanpercentile(x, pct_high))
        vmin = max(vmin, data_min)
        vmax = min(vmax, data_max)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return data_min, data_max
        if zero_floor_if_nonneg and data_min >= 0:
            vmin = 0.0
        return vmin, vmax
    elif mode == "symmetric":
        a = float(np.nanpercentile(np.abs(x), pct))
        if not np.isfinite(a) or a == 0:
            a = default
        return -a, a
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def _years_from_time_coord(time_coord, time_da: Optional[xr.DataArray] = None) -> Optional[np.ndarray]:
    """
    Return an array of integer years inferred from a time coordinate, or None if not possible.
    """
    if time_coord is None:
        return None

    # 1) numpy datetime64
    if np.issubdtype(np.asarray(time_coord).dtype, np.datetime64):
        # cast to year, then convert to int year = 1970 + ordinal
        yrs = time_coord.astype('datetime64[Y]').astype(int) + 1970
        return yrs.astype(int)

    # 2) cftime objects (DatetimeNoLeap, etc.)
    try:
        # cftime has .year attribute
        first = time_coord[0]
        if hasattr(first, "year"):
            return np.array([int(t.year) for t in time_coord], dtype=int)
    except Exception:
        pass

    # 3) Numeric "days since YYYY-MM-DD..." units
    if time_da is not None and "units" in time_da.attrs:
        m = re.match(r"^\s*days\s+since\s+(\d{4})-(\d{2})-(\d{2})", str(time_da.attrs["units"]), re.I)
        if m:
            base_year = int(m.group(1))
            days = np.asarray(time_coord, dtype=float)
            # assume 365-day (noleap) if not otherwise specified
            yrs = base_year + np.floor(days / 365.0).astype(int)
            return yrs.astype(int)

    return None


def _decade_ticks_from_years(years: np.ndarray) -> tuple[list[int], list[str]]:
    """
    Given an array of integer years (len N), return (tick_positions, tick_labels) such that:
    - ticks at indices where the year is the first in its decade sequence from the first year
    - labels are 4-digit years
    """
    if years is None or len(years) == 0:
        return [], []

    y0 = int(years[0])
    # Put ticks every 10 years aligned to y0 (e.g., 1901, 1911, ...)
    positions, labels = [], []
    for i, y in enumerate(years):
        if (y - y0) % 10 == 0:
            positions.append(i)
            labels.append(f"{y}")
    return positions, labels


def _r2_score(x: np.ndarray, y: np.ndarray) -> float:
    """Compute R² with numerical guards; returns NaN if degenerate."""
    m = np.isfinite(x) & np.isfinite(y)
    if not m.any():
        return np.nan
    x = x[m]; y = y[m]
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


# =========================
# Built-in plotters
# =========================

@register_plotter("scatter")
def _plot_scatter(item: Tuple[Union[xr.DataArray, np.ndarray], Union[xr.DataArray, np.ndarray]],
                  ax: plt.Axes,
                  spec: PlotSpec) -> plt.Axes:
    """
    Scatter of two matching arrays (x vs y).
    'item' must be a length-2 tuple/list. NaNs are ignored.

    New (optional) extras:
      - density_alpha: bool = False
      - bins: int = 200               # 2D hist bins
      - alpha_min: float = 0.05       # alpha at lowest nonzero density
      - alpha_max: float = 0.9        # alpha at highest density
      - subsample: Optional[int]      # random subsample cap (for memory)
    """
    if not (isinstance(item, (tuple, list)) and len(item) == 2):
        raise ValueError("scatter plot expects a (x, y) tuple/list.")

    x = _as_numpy(item[0]).ravel()
    y = _as_numpy(item[1]).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]

    # Optional very-light subsample (keeps memory modest if huge)
    subsample = spec.extras.get("subsample", None)
    if subsample is not None and isinstance(subsample, int) and subsample > 0 and x.size > subsample:
        idx = np.random.default_rng(42).choice(x.size, size=subsample, replace=False)
        x = x[idx]; y = y[idx]

    use_density = bool(spec.extras.get("density_alpha", False))
    if not use_density:
        ax.scatter(x, y, s=spec.s, alpha=spec.alpha, color="C0")
    else:
        # 2D histogram → per-point alpha based on bin count (high density → higher alpha)
        bins = int(spec.extras.get("bins", 200))
        alpha_min = float(spec.extras.get("alpha_min", 0.05))
        alpha_max = float(spec.extras.get("alpha_max", 0.9))

        # Compute bin edges and counts
        H, xedges, yedges = np.histogram2d(x, y, bins=bins)
        # Find each point's bin (clip to valid range)
        xbin = np.clip(np.digitize(x, xedges) - 1, 0, H.shape[0] - 1)
        ybin = np.clip(np.digitize(y, yedges) - 1, 0, H.shape[1] - 1)
        counts = H[xbin, ybin]

        if counts.max() > 0:
            a = counts / counts.max()
        else:
            a = np.zeros_like(counts)

        # Map density to alpha: low density → closer to alpha_min (more transparent),
        # high density → closer to alpha_max (less transparent)
        alphas = alpha_min + (alpha_max - alpha_min) * a
        # Draw as a single PathCollection with per-point alpha by passing RGBA
        # (using blue channel only)
        colors = np.zeros((x.size, 4), dtype=float)
        colors[:, 2] = 1.0  # blue
        colors[:, 3] = alphas
        ax.scatter(x, y, s=spec.s, c=colors)

    if x.size and y.size:
        mn = float(min(x.min(), y.min()))
        mx = float(max(x.max(), y.max()))
        if spec.add_1to1:
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.2)
        if spec.equal_axes:
            ax.set_xlim(mn, mx)
            ax.set_ylim(mn, mx)

    ax.set_xlabel(spec.xlabel or "Observed")
    ax.set_ylabel(spec.ylabel or "Predicted")

    title = spec.title or "Scatter"
    if spec.show_r2 and x.size > 1:
        r2 = _r2_score(x, y)
        title = f"{title}\nR² = {r2:.3f}"
    ax.set_title(title)
    return ax


@register_plotter("line")
def _plot_line(item: Union[xr.DataArray, np.ndarray], ax: plt.Axes, spec: PlotSpec) -> plt.Axes:
    """
    1D line plot. If xarray with 'time', infer years and show decade ticks.
    """
    arr = item
    years = None
    if isinstance(item, xr.DataArray):
        # infer years from item's time coord (if any)
        if "time" in item.coords:
            years = _years_from_time_coord(item["time"].values, item["time"])
        label = item.name or ""
        arr = item.values
    else:
        label = ""

    # x for plotting: we always plot against index positions, then format ticks as years (if available)
    n = np.asarray(arr).size
    x_ix = np.arange(n)
    ax.plot(x_ix, np.asarray(arr), linewidth=spec.lw)

    # decade ticks if we have years
    if years is not None and len(years) == n:
        ticks, ticklabs = _decade_ticks_from_years(years)
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabs)
        ax.set_xlabel(spec.xlabel or "Year")
    else:
        ax.set_xlabel(spec.xlabel or "Index")

    ax.set_ylabel(spec.ylabel or (label or "Value"))
    if spec.title:
        ax.set_title(spec.title)
    return ax


# --- updated map plotter ---
@register_plotter("map")
def _plot_map(item: xr.DataArray, ax: plt.Axes, spec: PlotSpec) -> plt.Axes:
    """
    2D lat/lon map using pcolormesh.
    
    - Expects a 2D DataArray with dims ('lat','lon').
    - Accepts longitudes in 0–360; renders as −180…180.
    - Uses robust limits when vmin/vmax are not provided.
    - If a Cartopy GeoAxes is passed and spec.use_cartopy=True,
      draws coastlines and borders.
    """
    # ---- validate
    if not isinstance(item, xr.DataArray) or item.ndim != 2 or not {"lat", "lon"}.issubset(item.dims):
        raise ValueError("map plot expects a 2D xarray.DataArray with dims ('lat','lon').")

    # ---- wrap longitudes: 0–360 -> −180…180 (sorted)
    lon = item["lon"].values
    lat = item["lat"].values
    if np.nanmin(lon) >= 0 and np.nanmax(lon) <= 360:
        lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(lon_wrapped)
        lon_plot = lon_wrapped[order]
        data_plot = item.values[:, order]
    else:
        lon_plot = lon
        data_plot = item.values

    # ---- limits & colormap
    vmin, vmax = spec.vmin, spec.vmax
    if spec.robust and (vmin is None or vmax is None):
        vmin, vmax = _robust_limits_one(
            data_plot,
            mode=spec.robust_mode,
            pct_low=spec.pct_low,
            pct_high=spec.pct_high,
            pct=spec.pct,
        )
    cmap = spec.cmap or ("RdBu_r" if spec.robust_mode == "symmetric" else "cividis")

    # ---- cartopy detection
    is_geoaxes = spec.use_cartopy and hasattr(ax, "projection") and hasattr(ax, "add_feature")

    # ---- cartopy outlines
    if is_geoaxes:
        if spec.show_coastlines:
            ax.coastlines(resolution="110m", linewidth=spec.coastlines_lw)
        if spec.show_borders:
            ax.add_feature(cfeature.BORDERS, linewidth=0.2)

    # ---- draw data
    if is_geoaxes:
        mappable = ax.pcolormesh(
            lon_plot, lat, data_plot,
            transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
        )
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    else:
        mappable = ax.pcolormesh(
            lon_plot, lat, data_plot,
            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
        )

    # ---- colorbar / labels / title
    if spec.add_colorbar:
        cbar = plt.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
        if spec.cbar_label:
            cbar.set_label(spec.cbar_label)

    ax.set_xlabel(spec.xlabel or "Longitude")
    ax.set_ylabel(spec.ylabel or "Latitude")
    if spec.title:
        ax.set_title(spec.title)
    return ax

@register_plotter("hist")
def _plot_hist(item: Union[xr.DataArray, np.ndarray], ax: plt.Axes, spec: PlotSpec) -> plt.Axes:
    """
    Fallback histogram of finite values.
    """
    x = _as_numpy(item).ravel()
    x = x[np.isfinite(x)]
    ax.hist(x, bins=spec.extras.get("bins", 40), alpha=0.8)

    ax.set_xlabel(spec.xlabel or "Value")
    ax.set_ylabel(spec.ylabel or "Count")
    if spec.title:
        ax.set_title(spec.title)
    else:
        ax.set_title("Histogram")
    return ax


@register_plotter("line_multi")
def _plot_line_multi(
    items: Sequence[Union[xr.DataArray, np.ndarray]],
    ax: plt.Axes,
    spec: PlotSpec
) -> plt.Axes:
    if not isinstance(items, (list, tuple)) or len(items) == 0:
        raise ValueError("line_multi expects a non-empty list/tuple of 1D arrays.")

    ax.cla()

    arrays, names, years = [], [], None
    for arr in items:
        if isinstance(arr, xr.DataArray):
            arrays.append(arr.values)
            names.append(arr.name or "")
            if years is None and "time" in arr.coords:
                years = _years_from_time_coord(arr["time"].values, arr["time"])
        else:
            arrays.append(np.asarray(arr))
            names.append("")

    n = arrays[0].size
    x_ix = np.arange(n)

    labels = spec.extras.get("series_labels") or [nm if nm else f"Series {i+1}" for i, nm in enumerate(names)]

    cmap_name = spec.extras.get("line_cmap", "tab10")
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(len(arrays))]

    for y, lab, col in zip(arrays, labels, colors):
        ax.plot(x_ix, y, lw=spec.lw, label=lab, color=col)

    # lock ticks/labels so they render once
    if years is not None and len(years) == n:
        ticks, ticklabs = _decade_ticks_from_years(years)
        if ticks:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter(ticklabs))
        ax.set_xlabel(spec.xlabel or "Year")
    else:
        ax.set_xlabel(spec.xlabel or "Index")

    ax.set_ylabel(spec.ylabel or "Value")
    if spec.title:
        ax.set_title(spec.title)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
        title=spec.extras.get("legend_title", None),
        ncol=spec.extras.get("legend_ncol", 1),
        fontsize=spec.extras.get("legend_fontsize", 8),
    )

    fig = ax.figure
    try:
        fig.subplots_adjust(right=0.78)
        w, h = fig.get_size_inches()
        width_factor = spec.extras.get("width_factor", 1.5)
        fig.set_size_inches(w * width_factor, h, forward=True)
    except Exception:
        pass

    return ax