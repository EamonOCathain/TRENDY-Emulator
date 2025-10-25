#!/usr/bin/env python3
"""
fix_time_axis.py
----------------
Utility to reset the time axis of an existing Zarr dataset.

Edits only the `time` coordinate:
  • sets units="days since 1901-01-01 00:00"
  • sets calendar="noleap"
  • generates a new sequence (daily/monthly/annual) between START and END
  • anchored to first-of-period by default (month start, year start, etc.)

Usage: just edit the CONFIG section below.
"""

from pathlib import Path
import numpy as np
import zarr
import cftime
import shutil

# ====================== CONFIG ======================

# Path to the Zarr dataset to fix
ZARR_PATH = Path(
    "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/modis_lai/modis_lai_monthly_filled.zarr"
)

# Backup (auto-created before modifying)
BACKUP_PATH = ZARR_PATH.with_suffix(".zarr.bak")

# Desired time range and resolution
START_YEAR = 2000
END_YEAR   = 2021
FREQ       = "monthly"   # one of: "daily", "monthly", "annual"

# Anchor choice: "start" = 1st of month/year, "end" = last day of period
ANCHOR     = "start"
MAKE_BACKUP = False

# =====================================================

_NOLEAP_DPM = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=int)

def _noleap_days_since_1901(y, m, d):
    """Return integer days since 1901-01-01 (noleap)."""
    return (y - 1901) * 365 + _NOLEAP_DPM[:m-1].sum() + (d - 1)

def generate_dates(start_year, end_year, freq, anchor):
    """Return list of cftime.DatetimeNoLeap values."""
    out = []
    for y in range(start_year, end_year + 1):
        if freq == "annual":
            m, d = (1, 1) if anchor == "start" else (12, 31)
            out.append(cftime.DatetimeNoLeap(y, m, d))
        elif freq == "monthly":
            for m in range(1, 13):
                d = 1 if anchor == "start" else int(_NOLEAP_DPM[m-1])
                out.append(cftime.DatetimeNoLeap(y, m, d))
        elif freq == "daily":
            for m in range(1, 13):
                for d in range(1, _NOLEAP_DPM[m-1] + 1):
                    out.append(cftime.DatetimeNoLeap(y, m, d))
        else:
            raise ValueError("freq must be daily/monthly/annual")
    return out

def main():
    # Backup first
    if MAKE_BACKUP:
        if not BACKUP_PATH.exists():
            print(f"[INFO] Creating backup → {BACKUP_PATH}")
            shutil.copytree(ZARR_PATH, BACKUP_PATH)
        else:
            print(f"[WARN] Backup already exists at {BACKUP_PATH}")

    print(f"[INFO] Opening Zarr: {ZARR_PATH}")
    root = zarr.open_group(str(ZARR_PATH), mode="a")

    if "time" not in root:
        raise KeyError("Zarr group has no 'time' coordinate array")

    n_time = root["time"].shape[0]
    print(f"[INFO] Existing time length = {n_time}")

    # Generate target date list
    dates = generate_dates(START_YEAR, END_YEAR, FREQ, ANCHOR)
    if len(dates) != n_time:
        print(f"[WARN] Length mismatch: generated {len(dates)} != existing {n_time}")
        raise SystemExit(1)

    # Convert to integer days since 1901-01-01
    ref = cftime.DatetimeNoLeap(1901, 1, 1)
    new_vals = np.array([(d - ref).days for d in dates], dtype=np.int64)

    print(f"[INFO] Writing new time axis ({FREQ}, {ANCHOR})")
    root["time"][:] = new_vals
    root["time"].attrs.update({
        "units": "days since 1901-01-01 00:00",
        "calendar": "noleap",
        "description": f"Reset from {START_YEAR}-{END_YEAR} ({FREQ}, anchored={ANCHOR})"
    })

    print(f"[INFO] Reconsolidating metadata...")
    zarr.consolidate_metadata(str(ZARR_PATH))
    print("[OK] Time axis fixed successfully.")

if __name__ == "__main__":
    main()