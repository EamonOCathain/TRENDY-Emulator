# --- Fully-NaN + NaN summary check for 1982–2018 window ---
import zarr
import numpy as np

p = "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/training_new/train/train_location_train_period/monthly.zarr"

g = zarr.open_group(p, mode="r")
a = g["lai_avh15c1"]             # (time, scenario, location)
t = g["time"][:].astype(int)     # days since 1901-01-01 (noleap)
L = a.shape[2]                   # number of locations

def to_days(datestr: str) -> int:
    base = np.datetime64("1901-01-01")
    return int((np.datetime64(datestr) - base) / np.timedelta64(1, "D"))

start = to_days("1982-01-01")
end   = to_days("2018-12-31")

mask = (t >= start) & (t <= end)
idx = np.flatnonzero(mask)
if idx.size == 0:
    raise SystemExit("No indices in 1982–2018; check time axis.")

# Slice S3 and window
sel = a.oindex[idx, 3, :]   # (T_window, L)
T = sel.shape[0]

# --- NaN stats ---
nan_mask = np.isnan(sel)

# (1) Overall NaN fraction across all cells
frac_nan_total = nan_mask.mean()

# (2) Months that are entirely NaN across all locations
months_all_nan = nan_mask.all(axis=1)
num_all_nan_months = int(months_all_nan.sum())
pct_all_nan_months = 100.0 * num_all_nan_months / T

# (3) Locations that are entirely NaN across all months in window
locs_all_nan = nan_mask.all(axis=0)
num_all_nan_locs = int(locs_all_nan.sum())
pct_all_nan_locs = 100.0 * num_all_nan_locs / L

# --- Report ---
print(f"\n[1982–2018 window] shape = {T} months × {L} locations")
print(f"  • Overall NaN fraction: {frac_nan_total:.3%}")
print(f"  • Fully-NaN months: {num_all_nan_months} / {T} ({pct_all_nan_months:.2f}%)")
print(f"  • Fully-NaN locations: {num_all_nan_locs} / {L} ({pct_all_nan_locs:.2f}%)")

# (Optional) show a few example location indices
show_k = 10
if num_all_nan_locs > 0:
    idxs = np.flatnonzero(locs_all_nan)[:show_k]
    print("Examples (fully NaN location indices):", idxs.tolist())