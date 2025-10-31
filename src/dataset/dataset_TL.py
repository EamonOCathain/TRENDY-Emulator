# src/dataset/dataset_TL.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Subset, random_split
import sys

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.dataset.variables import var_names, luh2_deltas
from src.training.varschema import VarSchema


class CustomDatasetTL(Dataset):
    """
    Transfer-learning dataset:
      - Identical to the standard CustomDataset where possible.
      - Applies replace_map ONLY to label groups (monthly_fluxes, monthly_states, annual_states).
      - Uses scenario index 3 exclusively (zero-based), but validates that the Zarr has ≥4 scenarios.
      - Supports inclusive TL time slicing [tl_start, tl_end] on 365-day 'days since 1901-01-01'.

    One SAMPLE = (one zarr group, scenario=3, one contiguous location block):
      inputs:   [C_in, 365*(Y-1),   L]
      labels_m: [C_m,  12*(Y-1),    L]
      labels_a: [C_a,  (Y-1),       L]

    Shifting:
      - Monthly states: Jan(t) ← Dec(t-1); first-year Jan is dummy (dropped with first year).
      - Annual states : t-1 with zero on first year; first year dropped.
    """

    def __init__(
        self,
        data_dir: str,
        std_dict: Dict[str, Dict[str, float]],
        tensor_type: str,                  # "train" | "val" | "test"
        block_locs: int = 70,              # locations per block (floor division)
        exclude_vars: Sequence[str] | None = None,
        delta_luh: bool = False,
        tl_activated: bool = False,
        tl_start: Optional[int] = None,
        tl_end: Optional[int] = None,
        replace_map: Optional[Dict[str, str]] = None,
    ):
        self.std_dict = std_dict
        self.tensor_type = tensor_type
        self.block_locs = int(block_locs)
        self.base_path = Path(data_dir) / tensor_type
        self.delta_luh = bool(delta_luh)
        self.exclude_set = set(exclude_vars or [])
        self.unfiltered_var_names = var_names

        # We only ever read scenario index 3; we do not enumerate scenarios.
        self.required_scenario_index = 3
        self.n_scenarios = 1  # planning enumerates only blocks, not scenarios

        # TL config
        self.transfer_learn = bool(tl_activated)
        self.tl_start = int(tl_start) if tl_start is not None else None
        self.tl_end   = int(tl_end)   if tl_end   is not None else None
        self.replace_map = replace_map or {}

        if self.transfer_learn:
            if self.tl_start is None or self.tl_end is None:
                raise ValueError("tl_activated=True requires tl_start and tl_end.")
            if self.tl_start > self.tl_end:
                raise ValueError(f"Invalid TL window: {self.tl_start}>{self.tl_end}")

        # Paths
        self._get_paths()

        # Open stores
        opts = dict(consolidated=True, decode_times=False, chunks={})
        self.ds_daily   = [xr.open_zarr(p, **opts) for p in self.daily_paths]
        self.ds_monthly = [xr.open_zarr(p, **opts) for p in self.monthly_paths]
        self.ds_annual  = [xr.open_zarr(p, **opts) for p in self.annual_paths]
        self._all = self.ds_daily + self.ds_monthly + self.ds_annual

        # Validate scenarios present (but we will only use index 3)
        for ds in (self.ds_daily + self.ds_monthly + self.ds_annual):
            if "scenario" not in ds.dims:
                raise AssertionError("All Zarr datasets must have a 'scenario' dimension.")
            if int(ds.sizes["scenario"]) <= self.required_scenario_index:
                raise AssertionError(
                    f"Expected at least {self.required_scenario_index+1} scenarios, "
                    f"got {int(ds.sizes['scenario'])}"
                )

        # Calendar (define before schema)
        self._month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int32)
        self._day_to_month  = np.repeat(np.arange(12, dtype=np.int64), self._month_lengths)

        # TL slicing (if enabled) — slice all opened datasets and drop unusable stores
        if self.transfer_learn:
            self._apply_transfer_learning_window(self.tl_start, self.tl_end)

        # Variable filtering + schema (strict)
        self._filter_var_names()

        # Plan samples (floor division)
        self._plan_samples()

    # ----------------------------- Paths -----------------------------
    def _get_paths(self) -> None:
        if self.tensor_type == "train":
            stem = "train_location_train_period"
            self.daily_paths   = [self.base_path / f"{stem}/daily.zarr"]
            self.monthly_paths = [self.base_path / f"{stem}/monthly.zarr"]
            self.annual_paths  = [self.base_path / f"{stem}/annual.zarr"]
        elif self.tensor_type == "val":
            stems = [
                "train_location_val_period_early",
                "train_location_val_period_late",
                "val_location_whole_period",
            ]
            self.daily_paths   = [self.base_path / f"{s}/daily.zarr"   for s in stems]
            self.monthly_paths = [self.base_path / f"{s}/monthly.zarr" for s in stems]
            self.annual_paths  = [self.base_path / f"{s}/annual.zarr"  for s in stems]
        elif self.tensor_type == "test":
            stems = [
                "test_location_whole_period",
                "train_location_test_period_early",
                "train_location_test_period_late",
            ]
            self.daily_paths   = [self.base_path / f"{s}/daily.zarr"   for s in stems]
            self.monthly_paths = [self.base_path / f"{s}/monthly.zarr" for s in stems]
            self.annual_paths  = [self.base_path / f"{s}/annual.zarr"  for s in stems]
        else:
            raise ValueError(f"Unknown tensor_type: {self.tensor_type!r}")

    # -------------------------- TL slicing --------------------------
    def _slice_days_since_1901(self, ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
        if "time" not in ds:
            return ds
        time_vals = ds["time"].values  # numeric "days since 1901-01-01", noleap
        start_day = (int(start_year) - 1901) * 365
        end_day_excl = (int(end_year) - 1901 + 1) * 365
        mask = (time_vals >= start_day) & (time_vals < end_day_excl)
        idx = np.where(mask)[0]
        if idx.size == 0:
            return ds.isel(time=slice(0, 0))
        return ds.isel(time=idx)

    def _apply_transfer_learning_window(self, start_year: int, end_year: int) -> None:
        self.ds_daily   = [self._slice_days_since_1901(ds, start_year, end_year) for ds in self.ds_daily]
        self.ds_monthly = [self._slice_days_since_1901(ds, start_year, end_year) for ds in self.ds_monthly]
        self.ds_annual  = [self._slice_days_since_1901(ds, start_year, end_year) for ds in self.ds_annual]

        keep = []
        for i, ds_m in enumerate(self.ds_monthly):
            months = int(ds_m.sizes.get("time", 0))
            years = months // 12 if months % 12 == 0 else 0
            if years >= 2:  # need ≥2 years because we drop the first
                keep.append(i)
        if not keep:
            raise RuntimeError(f"[TL] Window {start_year}-{end_year} left no usable stores in split='{self.tensor_type}'.")

        self.ds_daily   = [self.ds_daily[i]   for i in keep]
        self.ds_monthly = [self.ds_monthly[i] for i in keep]
        self.ds_annual  = [self.ds_annual[i]  for i in keep]
        self._all = self.ds_daily + self.ds_monthly + self.ds_annual

    # --------------- Var filtering (strict) + schema ----------------
    def _present_in_any_zarr(self, name: str) -> bool:
        return any(name in ds.data_vars for ds in self._all)

    def _require_stats(self, name: str) -> None:
        stats = self.std_dict.get(name)
        if not stats:
            raise AssertionError(f"Missing standardisation stats for '{name}'.")
        std = stats.get("std", None)
        if std is None or std <= 0:
            raise AssertionError(f"Non-positive std for '{name}'.")

    def _filter_var_names(self) -> None:
        base = {k: list(v) for k, v in self.unfiltered_var_names.items()}

        # Optional LUH deltas into annual_forcing
        if self.delta_luh:
            for v in luh2_deltas:
                if v not in base["annual_forcing"]:
                    base["annual_forcing"].append(v)

        label_groups = {"monthly_fluxes", "monthly_states", "annual_states"}

        filtered: Dict[str, List[str]] = {}
        for group, var_list in base.items():
            keep: List[str] = []
            for logical in var_list:
                # Apply replace_map ONLY to label groups
                actual = self.replace_map.get(logical, logical) if group in label_groups else logical

                if logical in self.exclude_set or actual in self.exclude_set:
                    continue

                # Stats required
                self._require_stats(actual)

                # Presence required; LUH deltas may be optional if requested but missing
                if not self._present_in_any_zarr(actual):
                    if self.delta_luh and group == "annual_forcing" and logical in luh2_deltas:
                        continue
                    raise AssertionError(
                        f"Zarr datasets are missing variable{(' after replacement' if actual != logical else '')}: {actual}"
                    )

                if actual not in keep:
                    keep.append(actual)
            filtered[group] = keep

        # Persist sorted lists
        self.daily_forcing   = sorted(filtered["daily_forcing"])
        self.monthly_forcing = sorted(filtered["monthly_forcing"])
        self.monthly_states  = sorted(filtered["monthly_states"])
        self.annual_forcing  = sorted(filtered["annual_forcing"])
        self.annual_states   = sorted(filtered["annual_states"])
        self.monthly_fluxes  = sorted(filtered["monthly_fluxes"])

        # Schema & orders
        self.schema = VarSchema(
            daily_forcing   = list(self.daily_forcing),
            monthly_forcing = list(self.monthly_forcing),
            monthly_states  = list(self.monthly_states),
            annual_forcing  = list(self.annual_forcing),
            annual_states   = list(self.annual_states),
            monthly_fluxes  = list(self.monthly_fluxes),
            month_lengths   = self._month_lengths.tolist(),
        )
        self.input_order = (
            self.daily_forcing + self.monthly_forcing + self.monthly_states + self.annual_forcing + self.annual_states
        )
        self.output_order = (
            self.monthly_fluxes + self.monthly_states + self.annual_states
        )

    # ----------------------- Planning (floor) -----------------------
    def _plan_samples(self) -> None:
        """
        meta[i] = (dataset_idx, scenario=3, loc0, loc1)
        Floor partitioning: drop tail < block_locs.
        """
        self.meta: List[Tuple[int, int, int, int]] = []
        self.tail_dropped: List[int] = []

        for k in range(len(self.ds_daily)):
            # location sizes must match
            Ld = int(self.ds_daily[k].sizes["location"])
            Lm = int(self.ds_monthly[k].sizes["location"])
            La = int(self.ds_annual[k].sizes["location"])
            if not (Ld == Lm == La):
                raise AssertionError(f"location size mismatch at ds[{k}]: daily={Ld}, monthly={Lm}, annual={La}")
            L = Ld

            # time alignment
            Td = int(self.ds_daily[k].sizes["time"])
            Tm = int(self.ds_monthly[k].sizes["time"])
            Ta = int(self.ds_annual[k].sizes["time"])
            if not (Td % 365 == 0 and Tm % 12 == 0 and Ta == Td // 365 == Tm // 12):
                raise AssertionError(f"time alignment mismatch at ds[{k}]: Td={Td}, Tm={Tm}, Ta={Ta}")
            if Ta < 2:
                raise AssertionError(f"Need at least 2 years (to drop first year) at ds[{k}], got {Ta}")

            n_blocks = L // self.block_locs  # floor
            remainder = L % self.block_locs
            self.tail_dropped.append(remainder)

            for b in range(n_blocks):
                loc0 = b * self.block_locs
                loc1 = loc0 + self.block_locs
                self.meta.append((k, self.required_scenario_index, loc0, loc1))

    def __len__(self) -> int:
        return len(self.meta)

    # --------------------- Standardisation (strict) ---------------------
    def _standardise(self, arr: np.ndarray, name: str) -> np.ndarray:
        stats = self.std_dict.get(name)
        if stats is None:
            raise AssertionError(f"Missing standardisation stats for '{name}'.")
        std = stats.get("std", None)
        mean = stats.get("mean", None)
        if std is None or std <= 0 or mean is None:
            raise AssertionError(f"Invalid stats for '{name}': mean={mean}, std={std}")
        return ((arr - mean) / std).astype(np.float32, copy=False)

    def _standardise_dataset(self, ds: xr.Dataset, var_list: List[str]) -> xr.Dataset:
        if len(var_list) == 0:
            raise RuntimeError("Variable list empty in dataloader standardisation")
        out = {}
        for v in var_list:
            arr = ds[v].transpose("time", "location", ...).values
            out[v] = xr.DataArray(
                self._standardise(arr, v),
                dims=("time", "location"),
                coords={"time": ds["time"].values, "location": ds["location"].values},
            )
        return xr.Dataset(out, coords={"time": ds["time"].values, "location": ds["location"].values})

    # ------------------ Monthly/Annual → Daily (shared) ------------------
    def _expand_monthly_to_daily(self, ds_m: xr.Dataset) -> xr.Dataset:
        months = int(ds_m.sizes["time"])
        if months % 12 != 0:
            raise AssertionError(f"Monthly time length {months} not divisible by 12.")
        years = months // 12
        L = int(ds_m.sizes["location"])
        days = np.arange(years * 365)
        loc = ds_m["location"].values

        out = {}
        for name, da in ds_m.data_vars.items():
            arr = da.transpose("time", "location").values.reshape(years, 12, L)
            arr_d = arr[:, self._day_to_month, :].reshape(years * 365, L)
            out[name] = xr.DataArray(arr_d, dims=("time", "location"), coords={"time": days, "location": loc})
        return xr.Dataset(out, coords={"time": days, "location": loc})

    def _expand_annual_to_daily(self, ds_a: xr.Dataset) -> xr.Dataset:
        years = int(ds_a.sizes["time"])
        L = int(ds_a.sizes["location"])
        days = np.arange(years * 365)
        loc = ds_a["location"].values

        out = {}
        for name, da in ds_a.data_vars.items():
            arr = da.transpose("time", "location").values  # [Y, L]
            arr_d = np.repeat(arr[:, None, :], 365, axis=1).reshape(years * 365, L)
            out[name] = xr.DataArray(arr_d, dims=("time", "location"), coords={"time": days, "location": loc})
        return xr.Dataset(out, coords={"time": days, "location": loc})

    # ---------------------- Shifting (standard) ----------------------
    def _shift_monthly_states_across_years(self, ds: xr.Dataset) -> xr.Dataset:
        """t-1 month shift with January(t) = December(t-1). First year's Jan is dummy (dropped later)."""
        out = {}
        T = int(ds.sizes["time"])
        if T % 12 != 0:
            raise AssertionError(f"Monthly time length {T} not divisible by 12.")
        Y = T // 12
        L = int(ds.sizes["location"])

        time_vals = ds["time"].values
        loc_vals  = ds["location"].values

        for v, da in ds.data_vars.items():
            arr = da.transpose("time", "location").values.reshape(Y, 12, L)  # [Y,12,L]
            shifted = np.empty_like(arr, dtype=arr.dtype)
            shifted[:, 1:, :] = arr[:, 0:11, :]   # Feb..Dec(t) ← Jan..Nov(t)
            shifted[1:, 0, :] = arr[:-1, 11, :]   # Jan(t)     ← Dec(t-1)
            shifted[0, 0, :] = 0.0                # first-year Jan (dropped later)
            out[v] = xr.DataArray(
                shifted.reshape(T, L), dims=("time", "location"),
                coords={"time": time_vals, "location": loc_vals}
            )
        return xr.Dataset(out, coords={"time": time_vals, "location": loc_vals})

    def _shift_annual_states_tminus1(self, ds: xr.Dataset) -> xr.Dataset:
        out = {}
        Y = int(ds.sizes["time"])
        L = int(ds.sizes["location"])
        time_vals = ds["time"].values
        loc_vals  = ds["location"].values
        for v, da in ds.data_vars.items():
            arr = da.transpose("time", "location").values  # [Y,L]
            shf = np.zeros_like(arr)
            shf[1:, :] = arr[:-1, :]
            out[v] = xr.DataArray(shf, dims=("time", "location"),
                                  coords={"time": time_vals, "location": loc_vals})
        return xr.Dataset(out, coords={"time": time_vals, "location": loc_vals})

    # ------------------------------- Item -------------------------------
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds_idx, scenario, loc0, loc1 = self.meta[i]  # scenario is fixed to index 3 in meta

        # slice scenario=3
        Dd = self.ds_daily[ds_idx].isel(location=slice(loc0, loc1), scenario=scenario)
        Dm = self.ds_monthly[ds_idx].isel(location=slice(loc0, loc1), scenario=scenario)
        Da = self.ds_annual[ds_idx].isel(location=slice(loc0, loc1), scenario=scenario)

        # split groups
        inputs_daily        = Dd[self.daily_forcing]
        monthly_forc_ds     = Dm[self.monthly_forcing]
        monthly_states_ds   = Dm[self.monthly_states]
        annual_forc_ds      = Da[self.annual_forcing]
        annual_states_ds    = Da[self.annual_states]

        # shifts
        monthly_states_t1 = self._shift_monthly_states_across_years(monthly_states_ds)
        annual_states_t1  = self._shift_annual_states_tminus1(annual_states_ds)

        # standardise
        daily_std   = self._standardise_dataset(inputs_daily,        self.daily_forcing)
        monthly_std = self._standardise_dataset(
            xr.merge([monthly_forc_ds, monthly_states_t1]),
            self.monthly_forcing + self.monthly_states
        )
        annual_std  = self._standardise_dataset(
            xr.merge([annual_forc_ds,  annual_states_t1]),
            self.annual_forcing + self.annual_states
        )

        # expand to daily
        monthly_daily = self._expand_monthly_to_daily(monthly_std)
        annual_daily  = self._expand_annual_to_daily(annual_std)

        # assemble inputs [C_in, 365*Y, L]
        arr_daily   = np.stack([daily_std[v]    .transpose("time", "location").values
                                for v in self.daily_forcing])
        arr_monthly = np.stack([monthly_daily[v].transpose("time", "location").values
                                for v in (self.monthly_forcing + self.monthly_states)])
        arr_annual  = np.stack([annual_daily[v] .transpose("time", "location").values
                                for v in (self.annual_forcing  + self.annual_states)])
        inputs = np.concatenate([arr_daily, arr_monthly, arr_annual], axis=0)

        # labels (no shift)
        out_m_std = self._standardise_dataset(
            Dm[self.monthly_fluxes + self.monthly_states],
            self.monthly_fluxes + self.monthly_states
        )
        out_a_std = self._standardise_dataset(
            Da[self.annual_states],
            self.annual_states
        )
        labels_m = np.stack([out_m_std[v].transpose("time", "location").values
                             for v in (self.monthly_fluxes + self.monthly_states)])
        labels_a = np.stack([out_a_std[v].transpose("time", "location").values
                             for v in self.annual_states])

        # drop first year everywhere
        inputs    = inputs[:,   365:, :]   # [C_in, 365*(Y-1), L]
        labels_m  = labels_m[:,  12:, :]   # [C_m,  12*(Y-1),  L]
        labels_a  = labels_a[:,   1:, :]   # [C_a,  (Y-1),     L]

        # sanity
        Y = int(Da.sizes["time"])
        L = int(Dd.sizes["location"])
        assert inputs.shape[1]   == 365*(Y-1) and inputs.shape[2]   == L
        assert labels_m.shape[1] ==  12*(Y-1) and labels_m.shape[2] == L
        assert labels_a.shape[1] ==      (Y-1) and labels_a.shape[2] == L

        return (
            torch.from_numpy(inputs.astype(np.float32,   copy=False)),
            torch.from_numpy(labels_m.astype(np.float32, copy=False)),
            torch.from_numpy(labels_a.astype(np.float32, copy=False)),
        )
        
# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_subset(dataset: Dataset, frac: float = 0.01, seed: int = 42) -> Subset:
    """
    Take a random subset of a dataset (useful for smoke tests).

    Args:
        dataset: Dataset to subset.
        frac:    Fraction of dataset to keep (0 < frac ≤ 1).
        seed:    RNG seed for reproducibility.

    Returns:
        Subset object pointing to the selected samples.
    """
    n_total = len(dataset)
    n_subset = max(1, int(n_total * frac))
    subset, _ = random_split(
        dataset,
        [n_subset, n_total - n_subset],
        generator=torch.Generator().manual_seed(seed),
    )
    return subset


def base(ds: Dataset | Subset) -> Dataset:
    """
    Return the underlying Dataset if `ds` is a Subset; otherwise return `ds`.
    Handy when accessing custom attributes on the original dataset.
    """
    return ds.dataset if isinstance(ds, Subset) else ds