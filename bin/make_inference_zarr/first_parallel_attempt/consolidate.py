#!/usr/bin/env python3
"""
Consolidate all inference Zarr stores *only* if all variables are marked complete.

- Validates S0..S3 × (annual, monthly, daily).
- For each store, every var in var_names[time_res] must have root.attrs["complete:<var>"] = True.
- If ANY store fails the check, nothing is consolidated and the script exits with code 1.

Run on a login/head node (it does only metadata ops).
"""

from __future__ import annotations
from pathlib import Path
import sys
import zarr

# --- Project imports ---
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.paths.paths import zarr_dir
from src.dataset.variables import var_names

SCENARIOS = ("S0", "S1", "S2", "S3")
TIME_RESES = ("annual", "monthly", "daily")

def missing_completions(store: Path, vars_for_res) -> list[str]:
    """
    Return list of variables that are NOT marked complete in this store.
    Completeness is checked via group attrs: 'complete:<var>' == True.
    """
    root = zarr.open_group(str(store), mode="r")
    missing = []
    for v in vars_for_res:
        if not bool(root.attrs.get(f"complete:{v}", False)):
            missing.append(v)
    return missing


def main():
    # First pass: verify all stores are complete
    problems = []
    stores = []

    for scen in SCENARIOS:
        for tres in TIME_RESES:
            store = zarr_dir / f"inference/{scen}/{tres}.zarr"
            stores.append((store, scen, tres))

    # Check existence & completeness
    for store, scen, tres in stores:
        if not store.exists():
            problems.append((store, scen, tres, "STORE_MISSING", []))
            continue

        vars_for_res = list(var_names[tres])  # expects keys 'annual'|'monthly'|'daily'
        missing = missing_completions(store, vars_for_res)
        if missing:
            problems.append((store, scen, tres, "INCOMPLETE_VARS", missing))

    if problems:
        print("[ERROR] One or more stores are not ready for consolidation:\n", file=sys.stderr)
        for store, scen, tres, kind, details in problems:
            if kind == "STORE_MISSING":
                print(f"  - {store}  (scenario={scen}, time_res={tres}) -> MISSING", file=sys.stderr)
            elif kind == "INCOMPLETE_VARS":
                preview = ", ".join(details[:8])
                more = f" (+{len(details)-8} more)" if len(details) > 8 else ""
                print(f"  - {store}  (scenario={scen}, time_res={tres}) "
                      f"-> incomplete vars: {preview}{more}", file=sys.stderr)
        sys.exit(1)

    # Second pass: consolidate all
    for store, scen, tres in stores:
        print(f"[OK] Consolidating {store} (scenario={scen}, time_res={tres}) ...")
        zarr.consolidate_metadata(str(store))
        print(f"[DONE] Consolidated: {store}")

    print("\nAll stores consolidated successfully.")


if __name__ == "__main__":
    main()