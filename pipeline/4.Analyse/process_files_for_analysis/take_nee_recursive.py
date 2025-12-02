#!/usr/bin/env python3
"""
make_nee_from_gpp_rh_ra.py
--------------------------
Recursively walk a root directory and, for each subdirectory that contains
NetCDF files, try to locate exactly one GPP file, one Rh file, and one Ra file.

Matching rule (per directory):
  - gpp file: filename endswith 'gpp.nc'  (case-insensitive)
  - rh file : filename endswith 'rh.nc'   (case-insensitive)
  - ra file : filename endswith 'ra.nc'   (case-insensitive)

If:
  - more than one match for any of {gpp, rh, ra}, OR
  - any of {gpp, rh, ra} is missing

then log an error and skip that directory.

Otherwise:
  - open the three files with xarray (decode_times=False)
  - pick the data variable whose name contains 'gpp', 'rh', 'ra' respectively
    (case-insensitive). If none matches but there is exactly one data_var,
    that one is used. If ambiguous, log and skip.
  - compute NEE = GPP - Rh - Ra
  - create a new file alongside the originals, named by replacing 'gpp'
    (case-insensitive) in the GPP filename with 'nee', e.g.
      ensmean_s3_gpp.nc -> ensmean_s3_nee.nc
  - copy the coords/time and global attrs from the GPP file
  - set NEE's attrs:
      long_name     = "Net Ecosystem Exchange"
      standard_name = "net_ecosystem_exchange"
      units         = "kg m-2 s-1"
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import xarray as xr


TARGET_KEYS = ["gpp", "rh", "ra"]


def find_triplet_in_dir(d: Path) -> Optional[Dict[str, Path]]:
    """
    Return dict {key: Path} for key in [gpp, rh, ra] or None if invalid.

    Matching rule: filename endswith '<key>.nc' (case-insensitive).
    """
    nc_files = [p for p in d.iterdir() if p.is_file() and p.suffix == ".nc"]
    if not nc_files:
        return None

    result: Dict[str, Path] = {}
    for key in TARGET_KEYS:
        suffix = f"{key}.nc"
        matches = [f for f in nc_files if f.name.lower().endswith(suffix)]
        if len(matches) == 0:
            logging.warning(f"[SKIP] {d}: no file ending with '{suffix}'")
            return None
        if len(matches) > 1:
            logging.error(
                f"[SKIP] {d}: multiple files ending with '{suffix}': "
                + ", ".join(m.name for m in matches)
            )
            return None
        result[key] = matches[0]

    return result


def select_data_var(ds: xr.Dataset, key: str, file_path: Path) -> Optional[str]:
    """
    Select a variable name from ds.data_vars whose name contains key
    (case-insensitive). If none matches but there is exactly one data_var,
    return that one. If still ambiguous, log and return None.
    """
    candidates = [name for name in ds.data_vars if key in name.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logging.error(
            f"[SKIP] {file_path}: multiple data variables contain '{key}': "
            + ", ".join(candidates)
        )
        return None

    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]

    logging.error(
        f"[SKIP] {file_path}: could not identify a unique '{key}' variable "
        f"(data_vars={list(ds.data_vars)})"
    )
    return None


def make_output_name(gpp_file: Path) -> Path:
    """
    Replace 'gpp' (case-insensitive) in filename with 'nee'.

    Example: 'ensmean_s3_gpp.nc' -> 'ensmean_s3_nee.nc'.

    If 'gpp' is not found, append '_nee' before extension.
    """
    name = gpp_file.name
    lower = name.lower()
    idx = lower.find("gpp")
    if idx == -1:
        return gpp_file.with_name(gpp_file.stem + "_nee" + gpp_file.suffix)

    new_name = name[:idx] + "nee" + name[idx + 3 :]
    return gpp_file.with_name(new_name)


def process_dir(d: Path, overwrite: bool = False, dry_run: bool = False):
    triplet = find_triplet_in_dir(d)
    if triplet is None:
        return

    gpp_path = triplet["gpp"]
    rh_path = triplet["rh"]
    ra_path = triplet["ra"]

    out_path = make_output_name(gpp_path)

    if out_path.exists() and not overwrite:
        logging.info(f"[SKIP] {d}: output exists and overwrite=False: {out_path.name}")
        return

    logging.info(
        f"[DIR] {d}\n"
        f"      GPP: {gpp_path.name}\n"
        f"      Rh : {rh_path.name}\n"
        f"      Ra : {ra_path.name}\n"
        f"      -> NEE: {out_path.name}"
    )

    if dry_run:
        logging.info("      [dry-run] Skipping computation/write")
        return

    ds_gpp = xr.open_dataset(gpp_path, decode_times=False)
    ds_rh = xr.open_dataset(rh_path, decode_times=False)
    ds_ra = xr.open_dataset(ra_path, decode_times=False)

    try:
        gpp_var = select_data_var(ds_gpp, "gpp", gpp_path)
        rh_var = select_data_var(ds_rh, "rh", rh_path)
        ra_var = select_data_var(ds_ra, "ra", ra_path)

        if not (gpp_var and rh_var and ra_var):
            return

        gpp = ds_gpp[gpp_var]
        rh = ds_rh[rh_var]
        ra = ds_ra[ra_var]

        nee = gpp - rh - ra
        nee.name = "nee"

        nee.attrs.update(
            {
                "long_name": "Net Ecosystem Exchange",
                "standard_name": "net_ecosystem_exchange",
                "units": "kg m-2 s-1",
            }
        )
        if "_FillValue" in gpp.attrs:
            nee.attrs["_FillValue"] = gpp.attrs["_FillValue"]

        new_ds = xr.Dataset(
            data_vars={nee.name: nee},
            coords=ds_gpp.coords,
            attrs=ds_gpp.attrs,
        )

        logging.info(f"      Writing: {out_path}")
        new_ds.to_netcdf(out_path)
    finally:
        ds_gpp.close()
        ds_rh.close()
        ds_ra.close()


def main():
    ap = argparse.ArgumentParser(
        description="Compute NEE = GPP - Rh - Ra from per-var NetCDFs."
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Root directory to walk recursively.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_nee.nc files if present.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done but do not write any files.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")

    logging.info(f"Walking root: {root}")

    # Process each directory that actually has .nc files
    for d in sorted({p.parent for p in root.rglob("*.nc")}):
        process_dir(d, overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()