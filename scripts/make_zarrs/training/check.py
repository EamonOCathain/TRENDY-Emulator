# --- Basic Zarr checks + NaN coverage for target window ---
import zarr
import numpy as np

p = "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/training_new/test/test_location_whole_period/monthly.zarr"

g = zarr.open_group(p, mode="r")
if "avh15c1_lai" not in g:
    raise SystemExit("avh15c1_lai not found in store")

a = g["avh15c1_lai"]
print("shape:", a.shape)          # (T, 4, L)
print("chunks:", a.chunks)

# quick overall NaN rate for S3 (note: reads all time)
print("nan% S3 (overall):", float(np.isnan(a[:, 3, :]).mean()))

# --- Check time axis ---
t = g["time"][:].astype(int)
print("time range:", t[0], "→", t[-1])

def to_ym(d: int) -> str:
    base = np.datetime64("1901-01-01")
    return str(base + np.timedelta64(int(d), "D"))

print("first date:", to_ym(t[0]))
print("last date:",  to_ym(t[-1]))

# --- Expected overlap range (1981-01-01 .. 2019-12-31) ---
start = (np.datetime64("1981-01-01") - np.datetime64("1901-01-01")).astype("timedelta64[D]").astype(int)
end   = (np.datetime64("2019-12-31") - np.datetime64("1901-01-01")).astype("timedelta64[D]").astype(int)

mask = (t >= start) & (t <= end)
idx = np.flatnonzero(mask)

if idx.size == 0:
    print("No indices in the 1981–2019 window; nothing to check.")
else:
    # Advanced indexing must use oindex with integer positions
    sel = a.oindex[idx, 3, :]   # (time_in_window, locations)

    frac_nan_time  = np.isnan(sel).all(axis=1).mean()
    frac_nan_total = np.isnan(sel).mean()
    months_with_data = int((~np.isnan(sel).all(axis=1)).sum())

    print("Window dates:", to_ym(t[idx[0]]), "→", to_ym(t[idx[-1]]), f"({idx.size} months)")
    print("NaN fraction over 1981–2019 window:")
    print(f"  • All-NaN months fraction: {frac_nan_time:.3f}")
    print(f"  • Overall cell NaN fraction: {frac_nan_total:.3f}")
    print(f"  • Months with any data: {months_with_data} / {idx.size} (~{months_with_data/idx.size:.1%})")