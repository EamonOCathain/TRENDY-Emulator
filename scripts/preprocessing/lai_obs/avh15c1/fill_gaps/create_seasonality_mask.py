#!/usr/bin/env python3
import xarray as xr
import numpy as np
from pathlib import Path
from numcodecs import Blosc

# ------------------------------ Paths ------------------------------
masks_dir = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks")
masks_dir.mkdir(parents=True, exist_ok=True)

src_nc   = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/avh15c1_lai.nc")
dst_zarr = masks_dir / "lai_avh15c1_seasonality.zarr"
var_out  = "lai_avh15c1"

print(f"[INFO] Opening {src_nc}")
# Use reasonably large chunks for faster compute; we’ll rechunk before writing
ds = xr.open_dataset(
    src_nc,
    use_cftime=True,
    chunks={"time": 120, "lat": 64, "lon": 64},
)

# ----------------------- Compute climatology -----------------------
# Mask fill values
da = ds["avh15c1_lai"].where(ds["avh15c1_lai"] != -999)

# Pixels with any nonzero finite signal over time -> keep; else NaN in output
valid_any  = ((da.notnull()) & (da != 0)).any(dim="time")
bad_pixels = ~valid_any

# 12-month climatology (NaNs ignored)
clim = da.groupby("time.month").mean(dim="time", skipna=True)
clim = clim.where(~bad_pixels)

# Rename 'month' -> time (1..12)
clim = clim.rename({"month": "time"}).assign_coords(time=np.arange(1, 13))
clim.attrs["description"] = "Monthly climatology (1..12) for lai_avh15c1; NaN where pixel has no signal."

# Pack into a dataset with desired variable name and final Zarr chunks
out = clim.to_dataset(name=var_out).chunk({"time": 12, "lat": 1, "lon": 1})

# Optional: light compression
encoding = {
    var_out: {
        "compressor": Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE),
    }
}

# ----------------------------- Write ------------------------------
if dst_zarr.exists():
    print(f"[INFO] Removing existing {dst_zarr} to write fresh store")
    import shutil
    shutil.rmtree(dst_zarr)

print(f"[INFO] Writing Zarr to {dst_zarr} with chunks (12,1,1)")
out.to_zarr(
    str(dst_zarr),
    mode="w",
    encoding=encoding,
    consolidated=True,
)

print("[OK] Done.")