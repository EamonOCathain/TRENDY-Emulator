import xarray as xr
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from pathlib import Path
import os

OVERWRITE = True

# Some Paths
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.paths.paths import (
    scripts_dir, 
    preprocessing_dir,
    raw_data_dir,
    historical_dir,
    raw_outputs_dir,
    model_outputs_dir,
    masking_dir,
    preprocessed_dir,
    masks_dir
)

from src.utils.tools import slurm_shard, finite_mask
from src.utils.visualisation import first_timestep

# ============ functions ============
def create_split_mask_longitudinal_bands(land_mask_path: str | Path, band_centers: List[int],
                                         percent_train:int=0.7, percent_val:int=0.1, percent_test:int=0.2) -> xr.DataArray:
        """
        Create a split mask using vertical longitudinal bands centered on given longitudes,
        ensuring the output mask aligns exactly with the land_mask grid.
            
        0 -> Train
        1 -> Val
        2 -> Test
        """

        land_ds = xr.open_dataset(land_mask_path, decode_times=False)
        land_mask = land_ds["finite_mask"]

        assert abs(percent_train + percent_val + percent_test - 1.0) < 1e-6, "Percentages must sum to 1.0"

        num_lat, num_lon = land_mask.shape
        lons = land_mask.lon.values  # use exact lon coords from file

        total_land_cells = land_mask.sum().item()
        target_test = int(total_land_cells * percent_test)
        target_val = int(total_land_cells * percent_val)

        # Initialize mask array (numpy) to -1 (unassigned)
        mask_arr = np.full((num_lat, num_lon), fill_value=-1, dtype=int)

        # Helper: find closest lon index for each band_center
        def closest_lon_idx(lon_value):
            diffs = np.abs(lons - lon_value)
            return int(np.argmin(diffs))

        test_indices = set()
        band_radius = 0

        # Expand test indices longitudinally until target count or max bands
        while len(test_indices) < target_test and band_radius < num_lon:
            for lon_center in band_centers:
                center_idx = closest_lon_idx(lon_center)
                for offset in [-band_radius, band_radius]:
                    col = (center_idx + offset) % num_lon  # wrap around
                    for i in range(num_lat):
                        if land_mask[i, col]:
                            test_indices.add((i, col))
            band_radius += 1

        for i, j in test_indices:
            mask_arr[i, j] = 2  # test

        val_indices = set()
        # Expand val indices similarly after test bands
        while len(val_indices) < target_val and band_radius < num_lon:
            for lon_center in band_centers:
                center_idx = closest_lon_idx(lon_center)
                for offset in [-band_radius, band_radius]:
                    col = (center_idx + offset) % num_lon
                    for i in range(num_lat):
                        if mask_arr[i, col] == -1 and land_mask[i, col]:
                            val_indices.add((i, col))
            band_radius += 1

        for i, j in val_indices:
            mask_arr[i, j] = 1  # val

        # Assign remaining unassigned land cells as train
        for i in range(num_lat):
            for j in range(num_lon):
                if mask_arr[i, j] == -1 and land_mask[i, j]:
                    mask_arr[i, j] = 0

        # Convert mask array back to xarray DataArray with coords
        mask_da = xr.DataArray(mask_arr, coords=land_mask.coords, dims=land_mask.dims, name="split_mask")

        # Print summary stats
        land_total = total_land_cells
        train_pct = ((mask_da == 0) & (land_mask == 1)).sum().item() / land_total * 100
        val_pct = ((mask_da == 1) & (land_mask == 1)).sum().item() / land_total * 100
        test_pct = ((mask_da == 2) & (land_mask == 1)).sum().item() / land_total * 100

        print(f"Train Land %: {train_pct:.1f}%")
        print(f"Val Land %:   {val_pct:.1f}%")
        print(f"Test Land %:  {test_pct:.1f}%")

        land_ds.close()

        return mask_da
    
def write_split_mask_to_netcdf(mask:xr.DataArray | xr.Dataset, output_path:str | Path, var_name:str ="split_mask") -> None:
        """
        Write a mask matrix to a NetCDF file, using lat and lon coords from the mask itself
        if available (i.e., mask is an xarray DataArray or Dataset).
        """
        if output_path.exists():
            output_path.unlink()
        
        if hasattr(mask, "coords") and "lat" in mask.coords and "lon" in mask.coords:
            lat = mask.coords["lat"].values
            lon = mask.coords["lon"].values
            da = mask.rename(var_name) if hasattr(mask, "name") else mask
            da.name = var_name
        else:
            raise ValueError("Mask must be an xarray DataArray with 'lat' and 'lon' coordinates")

        ds = da.to_dataset()
        ds.to_netcdf(path=output_path)
        print(f"NetCDF saved to: {output_path}")

def plot_split_mask(file_path, output_path, var_name="tvt_mask", title="Train/Val/Test Split", overwrite = True):
    """
    Plot train/val/test mask using lat/lon coordinates from a file or DataArray.
    """
    
    file_path = Path(file_path)

    if overwrite and output_path.exists():
        output_path.unlink()
    
    # Load mask from file
    ds = xr.open_dataset(file_path)
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in {file_path}. Available: {list(ds.data_vars)}")
    mask = ds[var_name]

    # Validate coords
    if not ("lat" in mask.coords and "lon" in mask.coords):
        raise ValueError("Mask must have 'lat' and 'lon' coordinates.")

    lat = mask.coords["lat"].values
    lon = mask.coords["lon"].values
    Lon, Lat = np.meshgrid(lon, lat)

    # Percentages
    valid = mask >= 0
    total = np.sum(valid)
    percent_train = np.sum(mask == 0) / total * 100
    percent_val = np.sum(mask == 1) / total * 100
    percent_test = np.sum(mask == 2) / total * 100
    title += f"\nTrain: {percent_train:.1f}%, Val: {percent_val:.1f}%, Test: {percent_test:.1f}%"

    # Plot
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_title(title)

    cmap = ListedColormap(["white", "#1f77b4", "#ff7f0e", "#2ca02c"])
    mesh = ax.pcolormesh(Lon, Lat, mask, cmap=cmap, vmin=-1, vmax=2, alpha=0.6, shading="auto")

    xticks = np.arange(-180, 181, 60)
    yticks = np.arange(-90, 91, 30)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels([f"{x}°" for x in xticks])
    ax.set_yticklabels([f"{y}°" for y in yticks])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
    cbar.set_label("Region")
    cbar.set_ticks([-1, 0, 1, 2])
    cbar.set_ticklabels(["Unused", "Train", "Val", "Test"])

    output_path = Path(output_path)
    plt.savefig(str(output_path), dpi=200)
    print("Plot saved to:", os.path.abspath(output_path))
    plt.close()
        
# ============ main ============
# this does the same manually but centered on europe
band_centers = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

# Make the Mask
land_mask_path = masks_dir / "training_nan_mask.nc"
split_mask = create_split_mask_longitudinal_bands(
    band_centers=band_centers,
    percent_train=0.8,
    percent_val=0.075,
    percent_test=0.125,
    land_mask_path=land_mask_path
)

# Write to NetCDF
out_path = masks_dir / "tvt_mask.nc"
out_path.parent.mkdir(parents=True, exist_ok=True)
if out_path.exists() and not OVERWRITE:
    print("skipping")
else:
    write_split_mask_to_netcdf(split_mask, out_path, var_name = 'tvt_mask')

# Plot the mask
current_dir = masking_dir / "tvt_mask"
plot_dir = masking_dir / "tvt_mask/val_plots"
plot_dir.mkdir(parents=True, exist_ok=True)
plot_split_mask(out_path, plot_dir / "tvt_mask.png")