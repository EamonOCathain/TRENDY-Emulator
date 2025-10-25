#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import xarray as xr

target_path = "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference/S3/monthly.zarr"
src_path    = "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/lai_avh15c1_filled_30x30.zarr"

var_name    = "lai_avh15c1"  # name to write in target zarr

def _first_var_name(ds: xr.Dataset) -> str:
    if not ds.data_vars:
        raise RuntimeError("Dataset has no data variables.")
    return list(ds.data_vars)[0]

def _maybe_transpose(da: xr.DataArray) -> xr.DataArray:
    if tuple(da.dims) != ("time", "lat", "lon"):
        return da.transpose("time", "lat", "lon")
    return da

def _year_month_keys(time_coord: xr.DataArray) -> xr.DataArray:
    """
    Return an int32 YearMonth key like 198201 for each timestamp.
    Works with numpy datetime64 or cftime (noleap).
    """
    try:
        ym = (time_coord.dt.year * 100 + time_coord.dt.month).astype("int32")
        return ym
    except Exception:
        # Fallback for exotic objects: pull attributes directly
        vals = time_coord.values
        years, months = [], []
        for t in vals:
            y = getattr(t, "year", None)
            m = getattr(t, "month", None)
            if y is None or m is None:
                # last resort: parse strings
                ts = str(t)
                y = int(ts[:4])
                m = int(ts[5:7])
            years.append(int(y))
            months.append(int(m))
        years = np.asarray(years, dtype=np.int32)
        months = np.asarray(months, dtype=np.int32)
        return xr.DataArray(years * 100 + months, dims=["time"], name="ym").astype("int32")

def main():
    print(f"[INFO] Opening target monthly.zarr: {target_path}")
    target = xr.open_zarr(target_path, consolidated=True, use_cftime=True)

    if not target.data_vars:
        raise SystemExit("Target Zarr has no data variables to copy encoding/chunking from.")
    ref_var = list(target.data_vars)[0]
    ref_da  = target[ref_var]

    # Extract chunking/encoding to reuse
    enc = dict(ref_da.encoding) if ref_da.encoding is not None else {}
    chunks = enc.get("chunksizes", enc.get("chunks", None))
    if chunks is None and getattr(ref_da.data, "chunks", None):
        # dask-style chunks -> a tuple (t, y, x)
        chunks = tuple(c[0] for c in ref_da.data.chunks)
    # Build a mapping for xarray .chunk()
    chunk_map = None
    if chunks and isinstance(chunks, (tuple, list)) and len(chunks) == 3:
        chunk_map = {"time": chunks[0], "lat": chunks[1], "lon": chunks[2]}

    print(f"[INFO] Opening source zarr: {src_path}")
    src = xr.open_zarr(src_path, consolidated=True, use_cftime=True)

    src_var = var_name if var_name in src.data_vars else _first_var_name(src)
    if src_var != var_name:
        print(f"[WARN] Variable '{var_name}' not found in source; using '{src_var}' instead.")
    src_da = _maybe_transpose(src[src_var]).astype("float32")

    # Safety: dimension sizes must match for lat/lon
    for dim in ("lat", "lon"):
        if src_da.sizes[dim] != target.sizes[dim]:
            raise SystemExit(
                f"Dimension mismatch for {dim}: source={src_da.sizes[dim]} target={target.sizes[dim]}"
            )
        # Use target coord objects for exact equality of coords
        src_da = src_da.assign_coords({dim: target[dim]})

    # ---- Align by Year-Month on the existing 'time' dimension (no swap_dims) ----
    print("[INFO] Aligning by Year-Month on 'time' …")

    # Compute integer YM keys like 198201
    src_ym = _year_month_keys(src_da["time"]).values      # 1D numpy array
    tgt_ym = _year_month_keys(target["time"]).values      # 1D numpy array

    # Temporarily replace the *labels* of the time coordinate with YM keys
    src_with_ym = src_da.assign_coords(time=("time", src_ym))

    # Reindex along 'time' using target YM keys (introduces NaNs where src is missing)
    aligned = src_with_ym.reindex(time=tgt_ym)

    # Restore the target's actual time coordinate (preserves cftime/calendar)
    aligned = aligned.assign_coords(time=target["time"])

    # Chunk like target
    if chunk_map is not None:
        aligned = aligned.chunk(chunk_map)

    # Compose encoding (preserve compressor/filters/chunks)
    var_encoding = {}
    if "compressor" in enc:
        var_encoding["compressor"] = enc["compressor"]
    if "filters" in enc:
        var_encoding["filters"] = enc["filters"]
    if chunks:
        var_encoding["chunks"] = chunks
    var_encoding["dtype"] = "float32"

    # Write to the target Zarr
    out_ds = aligned.to_dataset(name=var_name)
    print(f"[WRITE] Writing '{var_name}' into target Zarr …")
    out_ds.to_zarr(
        target_path,
        mode="a",
        consolidated=False,   # writing doesn't need consolidated=True
        encoding={var_name: var_encoding},
    )

    # Now consolidate metadata for fast future reads
    import zarr
    zarr.consolidate_metadata(zarr.DirectoryStore(target_path))

    print("[DONE] Insert complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", repr(e))
        sys.exit(1)