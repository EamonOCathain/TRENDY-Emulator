#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import zarr
import xarray as xr

# ---------------- Config ----------------
OVERWRITE_SKELETON = True     # recreate coords-only zarrs
OVERWRITE_VARS     = True     # delete & recreate arrays for variables

LAT_CHUNK = 5
LON_CHUNK = 5
TIME_CHUNK_BY_RES = {"daily": 44895, "monthly": 1476, "annual": 123}

# ---------------- Paths -----------------
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.paths.paths import zarr_dir
from src.utils.zarr_tools import make_zarr_skeleton
from src.dataset.variables import var_names  # expects var_names['daily'|'monthly'|'annual']

out_dir = zarr_dir / "inference_new"
out_dir.mkdir(parents=True, exist_ok=True)

SCENARIOS = ("S0", "S1", "S2", "S3")
TIME_RESES = ("annual", "monthly", "daily")

# ---------------- Helpers ----------------
def ensure_var_in_store_timechunks(
    store: Path | str,
    var_name: str,
    time_len: int,
    lat_len: int,
    lon_len: int,
    *,
    time_chunk: int,
    lat_chunk: int,
    lon_chunk: int,
    dtype: str = "float32",
    overwrite: bool = False,
):
    """
    Ensure a (time, lat, lon) array exists with given chunks & NaN fill.
    If overwrite=True and var exists, delete and recreate.
    """
    store = Path(store)
    root = zarr.open_group(str(store), mode="a")

    if var_name in root:
        if overwrite:
            del root[var_name]
        else:
            # make sure xarray can read dims
            if "_ARRAY_DIMENSIONS" not in root[var_name].attrs:
                root[var_name].attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]
            return

    from numcodecs import Blosc
    comp = Blosc(cname="zstd", clevel=8, shuffle=Blosc.BITSHUFFLE)

    arr = root.create_dataset(
        name=var_name,
        shape=(int(time_len), int(lat_len), int(lon_len)),
        chunks=(int(time_chunk), int(lat_chunk), int(lon_chunk)),
        dtype=dtype,
        fill_value=np.nan,
        compressor=comp,
        overwrite=False,
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]


# ---------------- Main ----------------
def main():
    for scen in SCENARIOS:
        for tres in TIME_RESES:
            store = out_dir / f"{scen}/{tres}.zarr"
            store.parent.mkdir(parents=True, exist_ok=True)

            # 1) Skeleton (coords only)
            if (not store.exists()) or OVERWRITE_SKELETON:
                make_zarr_skeleton(
                    out_path=store,
                    time_res=tres,
                    start="1901-01-01",
                    end="2023-12-31",
                    overwrite=True,
                )
                print(f"[OK] skeleton: {store}")
            else:
                print(f"[SKIP] skeleton exists: {store}")

            # 2) Pre-create variables (no data write)
            with xr.open_zarr(store, consolidated=True, decode_times=False) as ds:
                T = int(ds.sizes["time"])
                Y = int(ds.sizes["lat"])
                X = int(ds.sizes["lon"])

            tchunk = TIME_CHUNK_BY_RES[tres]
            vars_this_res = var_names[tres]  # e.g., daily_forcing + outputs as you define it

            for v in vars_this_res:
                ensure_var_in_store_timechunks(
                    store=store,
                    var_name=v,
                    time_len=T,
                    lat_len=Y,
                    lon_len=X,
                    time_chunk=tchunk,
                    lat_chunk=LAT_CHUNK,
                    lon_chunk=LON_CHUNK,
                    overwrite=OVERWRITE_VARS,
                )
                print(f"[OK] ensured var '{v}' in {store.name} (chunks t={tchunk}, y={LAT_CHUNK}, x={LON_CHUNK})")

            # Consolidate once per store
            try:
                zarr.consolidate_metadata(str(store))
            except Exception:
                pass

    print("Done: skeletons + empty variables are ready for the writing phase.")


if __name__ == "__main__":
    main()