#!/usr/bin/env python3
from pathlib import Path
import itertools as it
import zarr
import numpy as np

# Some Paths

def _iter_chunk_slices(shape, chunks):
    """
    Yield (chunk_index_tuple, slice_tuple) for all chunks in an array.
    """
    # normalize chunks to a tuple of ints per dim
    ch = tuple(int(c) for c in chunks)
    # number of chunks along each dim
    n_per_dim = tuple((s + c - 1) // c for s, c in zip(shape, ch))
    for idx in it.product(*[range(n) for n in n_per_dim]):
        slices = []
        for d, (i, c, s) in enumerate(zip(idx, ch, shape)):
            start = i * c
            stop  = min(start + c, s)
            slices.append(slice(start, stop))
        yield idx, tuple(slices)

def check_zarr_store(store_path, vars_include=None, verbose=True):
    """
    Scan a .zarr directory and attempt to read/decompress every chunk
    for each array. Returns a dict: {var_name: [(chunk_idx, error_str), ...]}.
    
    - vars_include: optional set/list of variable names to restrict checking.
    - If empty result -> everything read/decompressed cleanly.
    """
    store_path = Path(store_path)
    root = zarr.open_group(str(store_path), mode="r")
    bad = {}

    # List array-like members only (skip groups)
    array_names = [k for k, v in root.arrays()]  # zarr >= 2.16 has .arrays()
    if vars_include:
        array_names = [v for v in array_names if v in set(vars_include)]

    if verbose:
        print(f"[INFO] Checking {store_path} … arrays={len(array_names)}")

    for name in array_names:
        arr = root[name]
        shape, chunks = arr.shape, arr.chunks
        errs = []
        if verbose:
            print(f"  - {name}: shape={shape}, chunks={chunks}")

        for idx, slices in _iter_chunk_slices(shape, chunks):
            try:
                # This triggers read+decompress of that chunk region
                _ = arr[slices]
            except Exception as e:
                errs.append((idx, repr(e)))
                if verbose:
                    print(f"    [CORRUPT] {name} chunk {idx}: {e}")

        if errs:
            bad[name] = errs

    if not bad and verbose:
        print("[OK] All chunks decompressed successfully.")
    else:
        if verbose:
            total_bad = sum(len(v) for v in bad.values())
            print(f"[WARN] Found {total_bad} corrupt chunk(s) in {len(bad)} array(s).")

    return bad

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Verify Zarr chunks are readable (detect partial/corrupt writes).")
    p.add_argument("zarr_dir", help="Path to a .zarr directory")
    p.add_argument("--vars", nargs="*", help="Optional list of variable names to check")
    args = p.parse_args()

    bad = check_zarr_store(args.zarr_dir, vars_include=args.vars)
    if bad:
        # Non-zero exit so this can be used in CI or SLURM post-step checks.
        import sys
        sys.exit(2)