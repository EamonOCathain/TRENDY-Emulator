from pathlib import Path
import os
import xarray as xr

def rename_variable_inplace(nc_path: str | Path, old_name: str, new_name: str):
    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    tmp = nc_path.with_suffix(nc_path.suffix + ".tmp")
    bak = nc_path.with_suffix(nc_path.suffix + ".bak")

    try:
        with xr.open_dataset(nc_path) as ds:
            if old_name not in ds.data_vars:
                raise KeyError(f"{old_name} not found. Vars: {list(ds.data_vars)}")
            ds_renamed = ds.rename({old_name: new_name})
            ds_renamed.to_netcdf(tmp, engine="netcdf4")

        # optional backup then atomic replace
        if bak.exists():
            bak.unlink()
        os.replace(nc_path, bak)
        os.replace(tmp, nc_path)
        bak.unlink(missing_ok=True)
        print(f"[OK] {nc_path.name}: {old_name} -> {new_name}")
    except Exception:
        # clean up temp on failure
        try: tmp.unlink()
        except Exception: pass
        raise

base = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/model_outputs")
for scen in ["S0","S1","S2","S3"]:
    p = base / f"ENSMEAN_{scen}_cTotal_annual.nc"
    rename_variable_inplace(p, "cTotal_monthly", "cTotal_annual")