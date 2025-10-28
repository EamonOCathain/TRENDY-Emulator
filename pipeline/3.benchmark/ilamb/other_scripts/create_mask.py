from netCDF4 import Dataset
import numpy as np

src = "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/tvt_mask.nc"
with Dataset(src, "r") as ds:
    lat = ds.variables["lat"][:].astype(np.float64)   # [-89.75 .. 89.75], size 360
    lon = ds.variables["lon"][:].astype(np.float64)   # likely 0..359.5 or -180..179.5
    tvt = ds.variables["tvt_mask"][:].astype(np.int64)  # (lat,lon)

# Ensure lon is on [-180,180)
if np.nanmax(lon) > 180:
    lon_shift = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon_shift)
    lon = lon_shift[order]
    tvt = tvt[:, order]
else:
    order = np.arange(lon.size)

# Build CF-style bounds (half-grid)
dlat = float(np.median(np.diff(lat)))
dlon = float(np.median(np.diff(lon)))
lat_bounds = np.column_stack((lat - 0.5*dlat, lat + 0.5*dlat)).astype(np.float64)
lon_bounds = np.column_stack((lon - 0.5*dlon, lon + 0.5*dlon)).astype(np.float64)
# keep bounds within [-180,180]
lon_bounds[:, 0] = np.maximum(lon_bounds[:, 0], -180.0)
lon_bounds[:, 1] = np.minimum(lon_bounds[:, 1],  180.0)

# Map tvt values -> region ids
# ids: 0=train, 1=val, 2=test  (labels order defines the id)
mapping = {0: 0, 1: 1, 2: 2}
labels  = np.asarray(["train", "val", "test"])

MISS = -999  # sentinel for “no region”
ids = np.full(tvt.shape, MISS, dtype=np.int32)
for tvt_val, reg_id in mapping.items():
    ids[tvt == tvt_val] = reg_id

# ---- Write NetCDF --- 
out = "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/ilamb_tvt/ilamb_tvt.nc"
dset = Dataset(out, "w")
try:
    dset.createDimension("lat", lat.size)
    dset.createDimension("lon", lon.size)
    dset.createDimension("nb",  2)
    dset.createDimension("n",   labels.size)

    X  = dset.createVariable("lat",        "f8", ("lat",))
    XB = dset.createVariable("lat_bounds", "f8", ("lat","nb"))
    Y  = dset.createVariable("lon",        "f8", ("lon",))
    YB = dset.createVariable("lon_bounds", "f8", ("lon","nb"))
    # the key bit: declare fill_value=MISS
    I  = dset.createVariable("ids",        "i4", ("lat","lon"), fill_value=MISS)
    L  = dset.createVariable("labels",     str,  ("n",))

    # coordinates
    X[:] = lat; X.units = "degrees_north"
    XB[:] = lat_bounds
    Y[:] = lon; Y.units = "degrees_east"
    YB[:] = lon_bounds

    # ids + attributes ILAMB expects
    I[:] = ids  # background will be MISS on disk
    I.labels = "labels"
    I.missing_value = np.int32(MISS)  # optional, but helpful

    L[:] = labels
finally:
    dset.close()

print(f"Wrote {out}")