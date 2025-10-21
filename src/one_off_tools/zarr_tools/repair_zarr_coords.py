# repair_coords_now.py
from pathlib import Path
import zarr
import numpy as np

def _time_axis_noleap(time_res: str, length: int):
    import cftime
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    if time_res == "daily":
        return np.asarray([(cftime.DatetimeNoLeap(1901,1,1) + cftime.timedelta(days=i) - ref).days
                           for i in range(length)], dtype="int32")
    elif time_res == "monthly":
        vals = []
        y, m = 1901, 1
        for _ in range(length):
            vals.append((cftime.DatetimeNoLeap(y, m, 1) - ref).days)
            m += 1
            if m > 12: m, y = 1, y + 1
        return np.asarray(vals, dtype="int32")
    elif time_res == "annual":
        return np.asarray([(cftime.DatetimeNoLeap(1901 + i, 1, 1) - ref).days
                           for i in range(length)], dtype="int32")
    else:
        raise ValueError(time_res)

def _fix_group(store_path: Path, time_res: str):
    g = zarr.open_group(store=zarr.DirectoryStore(str(store_path)), mode="a")

    # infer T
    if "time" in g:
        T = int(g["time"].shape[0])
    else:
        T = next((int(a.shape[0]) for _, a in g.arrays() if a.ndim == 3), None)
        if T is None:
            raise RuntimeError(f"Cannot infer time length in {store_path}")

    # target coords
    time_arr = _time_axis_noleap(time_res, T)
    lat = np.arange(-89.75, 90.0, 0.5, dtype="float32")   # 360
    lon = np.arange(0.0, 360.0, 0.5, dtype="float32")     # 720

    # time (ensure no fill_value)
    t = g.require_dataset("time", shape=(T,), chunks=(T,), dtype="i4", compressor=None, fill_value=None)
    if getattr(t, "fill_value", None) is not None: t.fill_value = None
    t[:] = time_arr
    t.attrs["units"] = "days since 1901-01-01 00:00:00"
    t.attrs["calendar"] = "noleap"
    t.attrs["_ARRAY_DIMENSIONS"] = ["time"]

    # lat/lon (ensure no fill_value)
    for name, vals in (("lat", lat), ("lon", lon)):
        d = g.require_dataset(name, shape=(len(vals),), chunks=(len(vals),), dtype="f4",
                              compressor=None, fill_value=None)
        if getattr(d, "fill_value", None) is not None: d.fill_value = None
        d[:] = vals.astype("float32")
        d.attrs["_ARRAY_DIMENSIONS"] = [name]

    # annotate var dims (optional)
    for v, arr in g.arrays():
        if v not in ("time", "lat", "lon"):
            arr.attrs.setdefault("_ARRAY_DIMENSIONS", ["time", "lat", "lon"])

    # refresh consolidated metadata
    zarr.consolidate_metadata(zarr.DirectoryStore(str(store_path)))
    print(f"[OK] Repaired: {store_path}")

def repair_all(run_root: Path):
    z = run_root / "zarr"
    _fix_group(z / "daily.zarr",   "daily")
    _fix_group(z / "monthly.zarr", "monthly")
    _fix_group(z / "annual.zarr",  "annual")

if __name__ == "__main__":
    # EDIT THIS to point at the run you want to fix:
    run_root = Path("")
    repair_all(run_root)