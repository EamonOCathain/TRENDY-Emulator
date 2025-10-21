import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import sys
import json

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.paths.paths import (masks_dir, masking_dir)

current_dir = masking_dir / "tiling"

plot_dir = current_dir / "val_plots"
plot_dir.mkdir(parents=True, exist_ok=True)


# ================= HARD-CODED SETTINGS ==================
mask_nc   = masks_dir /  "forcing_nan_mask.nc"
varname   = "finite_mask"             
tile_lat  = 30                   
tile_lon  = 30                    
out_png   = plot_dir / "tiles_over_mask.png"
# ========================================================

def load_binary_mask(nc_path: Path, varname: str) -> xr.DataArray:
    with xr.open_dataset(nc_path, decode_times=False) as ds:
        da = ds[varname]
        land = (da.values != 0).astype(bool)
        return xr.DataArray(
            land, dims=("lat", "lon"),
            coords={"lat": ds["lat"], "lon": ds["lon"]},
            name="land_mask"
        )

def centers_to_edges(arr):
    if arr.size == 1:
        step = 1.0
    else:
        step = np.median(np.diff(arr))
    edges = np.concatenate(([arr[0] - 0.5*step], arr[:-1] + 0.5*step, [arr[-1] + 0.5*step]))
    return edges

def plan_tiles(nlat, nlon, tile_lat, tile_lon):
    tiles = []
    for i0 in range(0, nlat, tile_lat):
        i1 = min(i0 + tile_lat, nlat)
        for j0 in range(0, nlon, tile_lon):
            j1 = min(j0 + tile_lon, nlon)
            tiles.append((slice(i0, i1), slice(j0, j1)))
    return tiles

def compute_ocean_only_indices(land_da: xr.DataArray, tile_lat: int, tile_lon: int):
    """Return metadata including indices of tiles that are entirely ocean (all False)."""
    land = land_da.values.astype(bool)
    nlat, nlon = land.shape
    tiles = plan_tiles(nlat, nlon, tile_lat, tile_lon)

    ocean_only_indices = []
    for idx, (s_lat, s_lon) in enumerate(tiles):
        block = land[s_lat, s_lon]
        if block.sum() == 0:
            ocean_only_indices.append(idx)

    meta = {
        "tile_shape": [tile_lat, tile_lon],
        "nlat": int(nlat),
        "nlon": int(nlon),
        "num_tiles": int(len(tiles)),
        "ocean_only_indices": ocean_only_indices,
        "valid_indices": [i for i in range(len(tiles)) if i not in set(ocean_only_indices)],
    }
    return meta

def save_ocean_only_indices(meta: dict, out_json: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[WRITE] Ocean-only index list -> {out_json}")

def plot_tiles_over_mask(land_da, tile_lat, tile_lon, out_png):
    land = land_da.values
    nlat, nlon = land.shape
    lat = np.asarray(land_da["lat"].values)
    lon = np.asarray(land_da["lon"].values)

    lat_edges = centers_to_edges(lat)
    lon_edges = centers_to_edges(lon)

    tiles = plan_tiles(nlat, nlon, tile_lat, tile_lon)

    ocean_only = []
    for s_lat, s_lon in tiles:
        block = land[s_lat, s_lon]
        if block.sum() == 0:
            ocean_only.append((s_lat, s_lon))

    print(f"Tiles total: {len(tiles)} | fully-ocean: {len(ocean_only)}")

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.pcolormesh(lon_edges, lat_edges, land.astype(float), shading="auto")

    # draw grid lines
    for i in range(0, nlat + 1, tile_lat):
        if i < lat_edges.size:
            ax.plot([lon_edges[0], lon_edges[-1]], [lat_edges[i], lat_edges[i]],
                    lw=0.7, color="k", alpha=0.35)
    for j in range(0, nlon + 1, tile_lon):
        if j < lon_edges.size:
            ax.plot([lon_edges[j], lon_edges[j]], [lat_edges[0], lat_edges[-1]],
                    lw=0.7, color="k", alpha=0.35)

    # shade fully-ocean tiles
    for s_lat, s_lon in ocean_only:
        i0, i1 = s_lat.start, s_lat.stop
        j0, j1 = s_lon.start, s_lon.stop
        x0, x1 = lon_edges[j0], lon_edges[j1]
        y0, y1 = lat_edges[i0], lat_edges[i1]
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                         facecolor=(0.1, 0.4, 0.8, 0.25), edgecolor="none")
        ax.add_patch(rect)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Tiling for Inference - Shaded Tiles are 100% Ocean")
    ax.set_xlim(lon_edges[0], lon_edges[-1])
    ax.set_ylim(lat_edges[0], lat_edges[-1])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def main():
    land_da = load_binary_mask(mask_nc, varname)
    plot_tiles_over_mask(land_da, tile_lat, tile_lon, out_png)

    # export indices of fully-ocean tiles for inference skipping
    meta = compute_ocean_only_indices(land_da, tile_lat, tile_lon)
    out_json = masks_dir / f"tiles_ocean_only.json"
    save_ocean_only_indices(meta, out_json)
    print(f"Tiles total: {meta['num_tiles']} | fully-ocean: {len(meta['ocean_only_indices'])}")

if __name__ == "__main__":
    main()