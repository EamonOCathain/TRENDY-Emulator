import xarray as xr
path = "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/base_model_new_loss/no_carry/zarr/S3/annual.zarr"
ds = xr.open_zarr(path, consolidated=True, decode_times=False, chunks="auto")

print(ds)