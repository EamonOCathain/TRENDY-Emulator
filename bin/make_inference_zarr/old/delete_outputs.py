from pathlib import Path
import zarr
import xarray as xr
import sys

# --- configure here ---
# Some Paths
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

root_dir = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference") 

from src.dataset.variables import var_names

targets = set(var_names["outputs"])

for store in root_dir.rglob("*.zarr"):
    try:
        ds = xr.open_zarr(store, consolidated=True)
        vars_in_store = list(ds.data_vars)
        ds.close()
    except Exception as e:
        print(f"[WARN] Could not open {store}: {e}", flush=True)
        continue

    to_delete = [v for v in vars_in_store if v in targets]
    if not to_delete:
        continue

    print(f"[INFO] {store}: deleting {to_delete}", flush=True)
    root = zarr.open_group(str(store), mode="a")
    for v in to_delete:
        try:
            del root[v]
            print(f"  - removed {v}", flush=True)
        except Exception as e:
            print(f"  - failed to remove {v}: {e}", flush=True)