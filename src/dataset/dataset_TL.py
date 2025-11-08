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
    Transfer-learning dataset with parity to the standard dataset:

      • Same strict variable filtering policy:
          - respect exclude list
          - silently drop vars with invalid/missing std stats
          - require presence in the Zarrs (soft-skip LUH deltas if delta_luh=True)
      • Same public surface:
          - self.var_names (dict of filtered groups)
          - self.schema (VarSchema with month_lengths)
          - self.input_order / self.output_order
      • Same t-1 shifting for states + drop first year after shifting.

    TL-specific features retained:
      • Hard-codes scenario index = 3 (zero-based); validates scenario size == 4.
      • TL windowing by [tl_start, tl_end] years on days-since-1901 (noleap).
      • Label-only replacement via replace_map on {monthly_fluxes, monthly_states, annual_states}.
      • Planning builds blocks only for scenario 3.

    Each SAMPLE:
      inputs:   [C_in, 365*(Y-1),   L]
      labels_m: [C_m,  12*(Y-1),    L]
      labels_a: [C_a,  (Y-1),       L]
    """

    def __init__(
        self,
        data_dir: str,
        std_dict: Dict[str, Dict[str, float]],
        tensor_type: str,                  # "train" | "val" | "test"
        block_locs: int = 70,
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

        # TL config
        self.transfer_learn = bool(tl_activated)
        self.tl_start = int(tl_start) if tl_start is not None else None
        self.tl_end   = int(tl_end)   if tl_end   is not None else None
        self.replace_map = replace_map or {}

        if self.transfer_learn:
            if self.tl_start is None or self.tl_end is None:
                raise ValueError("tl_activated=True requires tl_start and tl_end (years).")
            if self.tl_start > self.tl_end:
                raise ValueError(f"Invalid TL window: {self.tl_start}>{self.tl_end}")

        # Always use scenario index 3 (zero-based); require exactly 4 scenarios.
        self.required_scenario_index = 3
        self.n_scenarios = 4  # invariant of the stores; we only read index 3

        # ---- resolve split paths + tags (match standard) ----
        self._get_paths()

        # ---- open stores ----
        opts = dict(consolidated=True, decode_times=False, chunks={})
        self.ds_daily   = [xr.open_zarr(p, **opts) for p in self.daily_paths]
        self.ds_monthly = [xr.open_zarr(p, **opts) for p in self.monthly_paths]
        self.ds_annual  = [xr.open_zarr(p, **opts) for p in self.annual_paths]
        self._all = self.ds_daily + self.ds_monthly + self.ds_annual

        # ---- calendar helpers (before schema) ----
        self._month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int32)
        self._day_to_month  = np.repeat(np.arange(12, dtype=np.int64), self._month_lengths)

        # ---- TL windowing (slice all datasets; drop stores with <2 years) ----
        if self.transfer_learn:
            self._apply_transfer_learning_window(self.tl_start, self.tl_end)

        # ---- variable filtering + schema (STRICT; parity with standard) ----
        self._filter_var_names()

        # ---- planning: blocks over scenario index 3 only ----
        self._plan_samples()

        # ---- scenario cardinality check (same guarantees as standard) ----
        for ds in (self.ds_daily + self.ds_monthly + self.ds_annual):
            if "scenario" not in ds.dims:
                raise AssertionError("All Zarr datasets must have a 'scenario' dimension.")
            if int(ds.sizes["scenario"]) != self.n_scenarios:
                raise AssertionError(
                    f"Expected scenario size {self.n_scenarios}, got {int(ds.sizes['scenario'])}"
                )

    # ---------------------------------------------------------------------
    # Paths + tags (mirror standard)
    # ---------------------------------------------------------------------
    def _get_paths(self) -> None:
        if self.tensor_type == "train":
            stem = "train_location_train_period"
            self.daily_paths   = [self.base_path / f"{stem}/daily.zarr"]
            self.monthly_paths = [self.base_path / f"{stem}/monthly.zarr"]
            self.annual_paths  = [self.base_path / f"{stem}/annual.zarr"]
            self.dataset_tags  = ["full"]

        elif self.tensor_type == "val":
            stems = [
                "train_location_val_period_early",
                "train_location_val_period_late",
                "val_location_whole_period",
            ]
            self.daily_paths   = [self.base_path / f"{s}/daily.zarr"   for s in stems]
            self.monthly_paths = [self.base_path / f"{s}/monthly.zarr" for s in stems]
            self.annual_paths  = [self.base_path / f"{s}/annual.zarr"  for s in stems]
            self.dataset_tags  = ["early", "late", "full"]

        elif self.tensor_type == "test":
            stems = [
                "test_location_whole_period",
                "train_location_test_period_early",
                "train_location_test_period_late",
            ]
            self.daily_paths   = [self.base_path / f"{s}/daily.zarr"   for s in stems]
            self.monthly_paths = [self.base_path / f"{s}/monthly.zarr" for s in stems]
            self.annual_paths  = [self.base_path / f"{s}/annual.zarr"  for s in stems]
            self.dataset_tags  = ["full", "early", "late"]
        else:
            raise ValueError(f"Unknown tensor_type: {self.tensor_type!r}")

    # ---------------------------------------------------------------------
    # TL slicing by days-since-1901 (noleap)
    # ---------------------------------------------------------------------
    def _slice_days_since_1901(self, ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
        if "time" not in ds:
            return ds
        time_vals = ds["time"].values  # numeric days since 1901-01-01 (noleap)
        start_day = (int(start_year) - 1901) * 365
        end_day_excl = (int(end_year) - 1901 + 1) * 365  # exclusive
        mask = (time_vals >= start_day) & (time_vals < end_day_excl)
        idx = np.where(mask)[0]
        if idx.size == 0:
            return ds.isel(time=slice(0, 0))
        return ds.isel(time=idx)

    def _apply_transfer_learning_window(self, start_year: int, end_year: int) -> None:
        self.ds_daily   = [self._slice_days_since_1901(ds, start_year, end_year) for ds in self.ds_daily]
        self.ds_monthly = [self._slice_days_since_1901(ds, start_year, end_year) for ds in self.ds_monthly]
        self.ds_annual  = [self._slice_days_since_1901(ds, start_year, end_year) for ds in self.ds_annual]

        # Keep only stores with >=2 years after slicing (to support dropping first year)
        keep = []
        for i, ds_m in enumerate(self.ds_monthly):
            months = int(ds_m.sizes.get("time", 0))
            years = months // 12 if months % 12 == 0 else 0
            if years >= 2:
                keep.append(i)
        if not keep:
            raise RuntimeError(f"[TL] Window {start_year}-{end_year} leaves no usable stores in split='{self.tensor_type}'.")

        self.ds_daily   = [self.ds_daily[i]   for i in keep]
        self.ds_monthly = [self.ds_monthly[i] for i in keep]
        self.ds_annual  = [self.ds_annual[i]  for i in keep]
        self._all = self.ds_daily + self.ds_monthly + self.ds_annual

    # ---------------------------------------------------------------------
    # Var filtering (STRICT; parity with standard) + schema
    # ---------------------------------------------------------------------
    def _present_in_any_zarr(self, name: str) -> bool:
        return any(name in ds.data_vars for ds in self._all)

    def _has_valid_stats(self, name: str) -> bool:
        st = self.std_dict.get(name)
        if not st:
            return False
        try:
            mu = float(st.get("mean", np.nan))
            sd = float(st.get("std", np.nan))
            return np.isfinite(mu) and np.isfinite(sd) and sd > 0.0
        except Exception:
            return False

    def _filter_var_names(self) -> None:
        """
        Build filtered var lists by:
        - respecting exclude list,
        - requiring valid std stats (std > 0, finite) and silently dropping if not,
        - requiring presence in any Zarr (softly skip LUH deltas if requested but not present),
        then freeze stable input/output orders and construct VarSchema.

        Additionally (TL only):
        - apply replace_map *only* to label groups (monthly_fluxes, monthly_states, annual_states).
        """
        # safe copy of groups
        base = {k: list(v) for k, v in self.unfiltered_var_names.items()}

        # optional LUH deltas into annual_forcing
        if self.delta_luh:
            for v in luh2_deltas:
                if v not in base["annual_forcing"]:
                    base["annual_forcing"].append(v)

        label_groups = {"monthly_fluxes", "monthly_states", "annual_states"}

        filtered: Dict[str, List[str]] = {}
        for group, var_list in base.items():
            keep: List[str] = []
            for logical in var_list:
                # TL-only: label replacement
                actual = self.replace_map.get(logical, logical) if group in label_groups else logical

                # exclusions
                if logical in self.exclude_set or actual in self.exclude_set:
                    continue

                # stats (strict, silent drop on invalid/missing)
                if not self._has_valid_stats(actual):
                    continue

                # presence (strict; soft for LUH deltas when delta_luh=True)
                if not self._present_in_any_zarr(actual):
                    if self.delta_luh and group == "annual_forcing" and logical in luh2_deltas:
                        continue
                    raise AssertionError(
                        f"Zarr datasets are missing variable"
                        f"{' after replacement' if actual != logical else ''}: {actual}"
                    )

                if actual not in keep:
                    keep.append(actual)

            filtered[group] = keep

        # --- expose filtered names just like the standard dataset ---
        self.var_names = filtered

        # persist sorted lists for stable channel layout
        self.daily_forcing   = sorted(filtered["daily_forcing"])
        self.monthly_forcing = sorted(filtered["monthly_forcing"])
        self.monthly_states  = sorted(filtered["monthly_states"])
        self.annual_forcing  = sorted(filtered["annual_forcing"])
        self.annual_states   = sorted(filtered["annual_states"])
        self.monthly_fluxes  = sorted(filtered["monthly_fluxes"])

        # schema (unchanged)
        self.schema = VarSchema(
            daily_forcing   = list(self.daily_forcing),
            monthly_forcing = list(self.monthly_forcing),
            monthly_states  = list(self.monthly_states),
            annual_forcing  = list(self.annual_forcing),
            annual_states   = list(self.annual_states),
            monthly_fluxes  = list(self.monthly_fluxes),
            month_lengths   = self._month_lengths.tolist(),
        )

        # handy orders (unchanged)
        self.input_order = (
            self.daily_forcing + self.monthly_forcing + self.monthly_states + self.annual_forcing + self.annual_states
        )
        self.output_order = (
            self.monthly_fluxes + self.monthly_states + self.annual_states
        )

        if not self.input_order:
            raise RuntimeError("Empty input_order after filtering.")
        if not self.output_order:
            raise RuntimeError("Empty output_order after filtering.")

    # ---------------------------------------------------------------------
    # Planning (blocks over scenario=3 only; floor division)
    # ---------------------------------------------------------------------
    def _plan_samples(self) -> None:
        """
        meta[i] = (dataset_idx, scenario=3, loc0, loc1)
        Floor partitioning: drop any tail block with < block_locs locations.
        """
        self.meta: List[Tuple[int, int, int, int]] = []
        self.tail_dropped: List[int] = []

        for k in range(len(self.ds_daily)):
            # location sizes must match across resolutions
            Ld = int(self.ds_daily[k].sizes["location"])
            Lm = int(self.ds_monthly[k].sizes["location"])
            La = int(self.ds_annual[k].sizes["location"])
            if not (Ld == Lm == La):
                raise AssertionError(f"location size mismatch at ds[{k}]: daily={Ld}, monthly={Lm}, annual={La}")
            L = Ld

            # years must align (noleap)
            Td = int(self.ds_daily[k].sizes["time"])
            Tm = int(self.ds_monthly[k].sizes["time"])
            Ta = int(self.ds_annual[k].sizes["time"])
            if not (Td % 365 == 0 and Tm % 12 == 0 and Ta == Td // 365 == Tm // 12):
                raise AssertionError(f"time alignment mismatch at ds[{k}]: Td={Td}, Tm={Tm}, Ta={Ta}")
            if Ta < 2:
                raise AssertionError(f"Need at least 2 years (to drop first year) at ds[{k}], got {Ta}")

            # blocks (floor division)
            n_blocks = L // self.block_locs
            remainder = L % self.block_locs
            self.tail_dropped.append(remainder)

            # scenario is fixed to 3
            s = self.required_scenario_index
            for b in range(n_blocks):
                loc0 = b * self.block_locs
                loc1 = loc0 + self.block_locs
                self.meta.append((k, s, loc0, loc1))

    def __len__(self) -> int:
        return len(self.meta)

    # ---------------------------------------------------------------------
    # Standardisation (strict)
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Monthly/Annual → Daily expansion (shared)
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # State shifting (shared)
    # ---------------------------------------------------------------------
    def _shift_monthly_states_across_years(self, ds: xr.Dataset) -> xr.Dataset:
        """t-1 month shift with January(t) ← December(t-1). First year's Jan is dummy (dropped later)."""
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
            shifted[0, 0, :] = 0.0                # first-year Jan (dummy; dropped later)
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

    # ---------------------------------------------------------------------
    # Item
    # ---------------------------------------------------------------------
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        ds_idx, scenario, loc0, loc1 = self.meta[i]  # scenario fixed to 3
        period_tag = self.dataset_tags[ds_idx]

        # slice
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
            period_tag,
        )


# ---------------------------------------------------------------------------
# Utilities (parity with standard)
# ---------------------------------------------------------------------------

def get_subset(dataset: Dataset, frac: float = 0.01, seed: int = 42) -> Subset:
    n_total = len(dataset)
    n_subset = max(1, int(n_total * frac))
    subset, _ = random_split(
        dataset,
        [n_subset, n_total - n_subset],
        generator=torch.Generator().manual_seed(seed),
    )
    return subset


def base(ds: Dataset | Subset) -> Dataset:
    return ds.dataset if isinstance(ds, Subset) else ds