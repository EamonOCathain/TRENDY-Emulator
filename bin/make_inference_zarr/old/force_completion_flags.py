from pathlib import Path
import zarr
import xarray as xr
import sys

def bootstrap_completion_flags(dst_store, vars_expected):
    """
    Retroactively mark variables complete (per-scenario + global) if arrays exist.
    No data is rewritten. Safe if you trust the existing data.
    """
    dst_store = Path(dst_store)
    root = zarr.open_group(str(dst_store), mode="a")

    # infer num scenarios from coords
    with xr.open_zarr(dst_store, consolidated=True, decode_times=False) as ds:
        S = int(ds.sizes["scenario"])

    for var in vars_expected:
        if var not in root:
            print(f"[SKIP] {var}: not present in {dst_store.name}")
            continue

        # nuke any progress masks from prior runs
        for s in range(S):
            key = f"__written_loc__{var}:scen{s}"
            if key in root:
                del root[key]

        # set per-scenario + global complete flags
        for s in range(S):
            root.attrs[f"complete:{var}:scen{s}"] = True
        root.attrs[f"complete:{var}"] = True
        print(f"[OK] marked complete: {var} (all scenarios)")

    # keep metadata snappy
    try:
        zarr.consolidate_metadata(str(dst_store))
    except Exception:
        pass
    

# Some Paths
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))
    
from src.dataset.variables import var_names

zarr_dir = project_root / "data/zarrs/inference"
scenarios = ['S0','S1','S2','S3']
for scen in scenarios:
    store = zarr_dir / "{scen}/daily.zarr"
bootstrap_completion_flags(store, var_names["daily_forcing"])