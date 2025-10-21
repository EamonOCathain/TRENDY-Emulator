#!/usr/bin/env python3
from pathlib import Path
import sys, time, shutil
import zarr
import xarray as xr

ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/training_new")

def try_open_consolidated(p: Path) -> tuple[bool, str]:
    """Return (ok, msg) for consolidated=True open."""
    try:
        ds = xr.open_zarr(p, consolidated=True, decode_times=False)
        # touch something small so we actually parse metadata:
        _ = ds.sizes
        ds.close()
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def try_open_unconsolidated(p: Path) -> tuple[bool, str]:
    """Return (ok, msg) for consolidated=False open (sanity check)."""
    try:
        ds = xr.open_zarr(p, consolidated=False, decode_times=False)
        _ = ds.sizes
        ds.close()
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def backup_zmetadata(p: Path) -> Path | None:
    zmeta = p / ".zmetadata"
    if not zmeta.exists():
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = p / f".zmetadata.bak.{ts}"
    shutil.copy2(zmeta, bak)
    return bak

def main():
    zarrs = sorted(ROOT.rglob("*.zarr"))
    if not zarrs:
        print(f"[ERR] No *.zarr under {ROOT}", file=sys.stderr)
        sys.exit(1)

    bad, fixed, still_bad = [], [], []

    print(f"[INFO] scanning {len(zarrs)} stores under {ROOT}")
    for p in zarrs:
        ok, msg = try_open_consolidated(p)
        if ok:
            print(f"[OK] consolidated open: {p}")
            continue

        print(f"[WARN] consolidated open failed: {p} -> {msg}")
        # If even unconsolidated fails, log and continue (store may be corrupt)
        ok_unc, msg_unc = try_open_unconsolidated(p)
        if not ok_unc:
            print(f"[FAIL] unconsolidated open also failed: {p} -> {msg_unc}")
            bad.append((p, msg, msg_unc))
            continue

        # Backup and reconsolidate
        bak = backup_zmetadata(p)
        if bak:
            print(f"[INFO] backed up {p/'.zmetadata'} -> {bak.name}")
        try:
            zarr.consolidate_metadata(str(p))
            print(f"[INFO] reconsolidated: {p}")
        except Exception as e:
            print(f"[FAIL] consolidate_metadata failed: {p} -> {e}")
            bad.append((p, msg, f"consolidate_metadata: {e}"))
            continue

        # Verify consolidated open now works
        ok2, msg2 = try_open_consolidated(p)
        if ok2:
            print(f"[FIXED] consolidated open now ok: {p}")
            fixed.append(p)
        else:
            print(f"[FAIL] still failing consolidated open: {p} -> {msg2}")
            still_bad.append((p, msg2))

    # Summary
    print("\n=== SUMMARY ===")
    total = len(zarrs)
    good = total - len(fixed) - len(bad) - len(still_bad)
    print(f"Total: {total}")
    print(f"Good initially: {good}")
    print(f"Fixed by reconsolidation: {len(fixed)}")
    print(f"Unconsolidated also failed (likely corrupt): {len(bad)}")
    print(f"Still failing consolidated after reconsolidation: {len(still_bad)}")

    if bad:
        print("\n[CORRUPT or unreadable stores]")
        for p, m1, m2 in bad:
            print(f" - {p}\n     consolidated error: {m1}\n     unconsolidated error: {m2}")
    if still_bad:
        print("\n[Reconsolidated but still failing consolidated open]")
        for p, m in still_bad:
            print(f" - {p} -> {m}")

if __name__ == "__main__":
    main()