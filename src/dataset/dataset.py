# src/dataset/dataset_unified_block.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Subset, random_split
import sys

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.dataset.variables import var_names, luh2_deltas
from src.training.varschema import VarSchema

class CustomDataset(Dataset):
    """
    Block-partitioned dataset (floor division) with uniform t-1 shifting.

    Each SAMPLE = (one zarr group, one scenario ∈ {0..3}, one contiguous location block):
      inputs:   [C_in, 365*(Y-1),   L]
      labels_m: [C_m,  12*(Y-1),    L]
      labels_a: [C_a,  (Y-1),       L]

    where Y is the number of years available in the store (must be consistent across daily/monthly/annual).
    This dataset prepares monthly/annual states with a global t-1 shift (fill with 0 at the first step),
    then drops the first year so all timesteps have valid context. Works for both carry and non-carry models.
    """

    def __init__(
        self,
        data_dir: str,
        std_dict: Dict[str, Dict[str, float]],
        tensor_type: str,                # "train" | "val" | "test"
        block_locs: int = 70,
        exclude_vars: Sequence[str] | None = None,
        delta_luh: bool = False,
    ):
        self.std_dict = std_dict
        self.tensor_type = tensor_type
        self.block_locs = int(block_locs)
        self.base_path = Path(data_dir) / tensor_type
        self.delta_luh = bool(delta_luh)
        self.exclude_set = set(exclude_vars or [])
        self.unfiltered_var_names = var_names
        self.n_scenarios = 4  # required

        # ---- resolve split paths ----
        self._get_paths()

        # ---- open stores ----
        opts = dict(consolidated=True, decode_times=False, chunks={})
        self.ds_daily   = [xr.open_zarr(p, **opts) for p in self.daily_paths]
        self.ds_monthly = [xr.open_zarr(p, **opts) for p in self.monthly_paths]
        self.ds_annual  = [xr.open_zarr(p, **opts) for p in self.annual_paths]
        self._all = self.ds_daily + self.ds_monthly + self.ds_annual
        
        
        # ---- calendar (must be defined before schema) ----
        self._month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int32)
        self._day_to_month  = np.repeat(np.arange(12, dtype=np.int64), self._month_lengths)

        # ---- variable filtering + schema (strict) ----
        self._filter_var_names()  # sets self.schema & per-group lists

        # ---- build block meta with floor division ----
        self._plan_samples()
        
        # sanity: all stores must have scenario=4
        for ds in (self.ds_daily + self.ds_monthly + self.ds_annual):
            if "scenario" not in ds.dims:
                raise AssertionError("All Zarr datasets must have a 'scenario' dimension.")
            if int(ds.sizes["scenario"]) != self.n_scenarios:
                raise AssertionError(f"Expected scenario size {self.n_scenarios}, "
                                     f"got {int(ds.sizes['scenario'])}")

    # ---------------------------------------------------------------------
    # Paths
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
    # Var filtering (strict) + schema
    # ---------------------------------------------------------------------
    def _present_in_any_zarr(self, name: str) -> bool:
        return any(name in ds.data_vars for ds in self._all)

    def _require_stats(self, name: str) -> None:
        stats = self.std_dict.get(name)
        if not stats:
            raise AssertionError(f"Missing standardisation stats for '{name}'.")
        std = stats.get("std", None)
        if std is None or std <= 0:
            raise AssertionError(f"Non-positive std for '{name}'.")
        # (no NaN/finite probing by request)

    def _filter_var_names(self) -> None:
        """
        Build filtered var lists by:
        - respecting exclude list,
        - requiring valid std stats (std > 0, finite),
        - requiring presence in any Zarr (but softly skip LUH deltas if absent when delta_luh=True),
        then freeze stable input/output orders and construct VarSchema.
        """
        def _has_valid_stats(name: str) -> bool:
            st = self.std_dict.get(name)
            if not st:
                return False
            try:
                mu = float(st.get("mean", np.nan))
                sd = float(st.get("std", np.nan))
                return np.isfinite(mu) and np.isfinite(sd) and sd > 0.0
            except Exception:
                return False

        def _present_in_any_zarr(name: str) -> bool:
            return any(name in ds.data_vars for ds in self._all)

        # safe copy of groups
        base = {k: list(v) for k, v in self.unfiltered_var_names.items()}

        # optionally append LUH deltas to annual_forcing
        if self.delta_luh:
            for v in luh2_deltas:
                if v not in base["annual_forcing"]:
                    base["annual_forcing"].append(v)

        filtered: Dict[str, List[str]] = {}
        for group, var_list in base.items():
            keep: List[str] = []
            for v in var_list:
                if v in self.exclude_set:
                    continue

                # 1) require valid stats; silently drop if std<=0 or missing
                if not _has_valid_stats(v):
                    continue

                # 2) require presence; softly skip LUH deltas if requested but not present
                if not _present_in_any_zarr(v):
                    if self.delta_luh and v in luh2_deltas:
                        continue  # optional channel; skip quietly
                    # hard error for everything else
                    raise AssertionError(f"Zarr datasets are missing variable: {v}")

                if v not in keep:
                    keep.append(v)

            filtered[group] = keep

        # expose filtered names for downstream snapshotting/debug
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
    # Planning with block partitioning and floor division to exclude last block 
    # ---------------------------------------------------------------------
    def _plan_samples(self) -> None:
        """
        meta[i] = (dataset_idx, scenario, loc0, loc1)
        Floor-partitioning: drop any tail block with < block_locs locations.
        """
        self.meta: List[Tuple[int, int, int, int]] = []
        self.tail_dropped: List[int] = []  # optional: track how many locations were dropped per store

        for k in range(len(self.ds_daily)):
            # location sizes must match across resolutions
            Ld = int(self.ds_daily[k].sizes["location"])
            Lm = int(self.ds_monthly[k].sizes["location"])
            La = int(self.ds_annual[k].sizes["location"])
            if not (Ld == Lm == La):
                raise AssertionError(f"location size mismatch at ds[{k}]: daily={Ld}, monthly={Lm}, annual={La}")
            L = Ld

            # years must align (noleap: daily=365*Y, monthly=12*Y, annual=Y)
            Td = int(self.ds_daily[k].sizes["time"])
            Tm = int(self.ds_monthly[k].sizes["time"])
            Ta = int(self.ds_annual[k].sizes["time"])
            if not (Td % 365 == 0 and Tm % 12 == 0 and Ta == Td // 365 == Tm // 12):
                raise AssertionError(f"time alignment mismatch at ds[{k}]: Td={Td}, Tm={Tm}, Ta={Ta}")
            if Ta < 2:
                raise AssertionError(f"Need at least 2 years (to drop first year) at ds[{k}], got {Ta}")

            # --- floor blocks (drop remainder) ---
            n_blocks = L // self.block_locs
            remainder = L % self.block_locs
            self.tail_dropped.append(remainder)

            for s in range(self.n_scenarios):
                for b in range(n_blocks):
                    loc0 = b * self.block_locs
                    loc1 = loc0 + self.block_locs  # exact block size
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
    
    def _shift_monthly_states_across_years(self, ds: xr.Dataset) -> xr.Dataset:
        """t-1 month shift with January(t) = December(t-1). First year's Jan is a dummy (dropped later)."""
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

            # within-year shifts: Feb..Dec(t) ← Jan..Nov(t)
            shifted[:, 1:, :] = arr[:, 0:11, :]

            # across-year carry for January: Jan(t) ← Dec(t-1)
            shifted[1:, 0, :] = arr[:-1, 11, :]

            # first year's January (no previous year): any sentinel; it will be dropped
            shifted[0, 0, :] = 0.0

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
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds_idx, scenario, loc0, loc1 = self.meta[i]
        period_tag = self.dataset_tags[ds_idx] 
        
        # slice
        Dd = self.ds_daily[ds_idx].isel(location=slice(loc0, loc1), scenario=scenario)
        Dm = self.ds_monthly[ds_idx].isel(location=slice(loc0, loc1), scenario=scenario)
        Da = self.ds_annual[ds_idx].isel(location=slice(loc0, loc1), scenario=scenario)

        # --- split into groups
        inputs_daily        = Dd[self.daily_forcing]
        monthly_forc_ds     = Dm[self.monthly_forcing]
        monthly_states_ds   = Dm[self.monthly_states]
        annual_forc_ds      = Da[self.annual_forcing]
        annual_states_ds    = Da[self.annual_states]

        # --- t-1 shift for states (global, uniform logic)
        monthly_states_t1 = self._shift_monthly_states_across_years(monthly_states_ds)
        annual_states_t1 = self._shift_annual_states_tminus1(annual_states_ds)

        # --- standardise (strict)
        daily_std   = self._standardise_dataset(inputs_daily,        self.daily_forcing)
        monthly_std = self._standardise_dataset(
            xr.merge([monthly_forc_ds, monthly_states_t1]),
            self.monthly_forcing + self.monthly_states
        )
        annual_std  = self._standardise_dataset(
            xr.merge([annual_forc_ds,  annual_states_t1]),
            self.annual_forcing + self.annual_states
        )

        # --- expand month/annual to daily
        monthly_daily = self._expand_monthly_to_daily(monthly_std)
        annual_daily  = self._expand_annual_to_daily(annual_std)

        # --- assemble input channels in explicit order
        arr_daily   = np.stack([daily_std[v]    .transpose("time", "location").values
                                for v in self.daily_forcing])
        arr_monthly = np.stack([monthly_daily[v].transpose("time", "location").values
                                for v in (self.monthly_forcing + self.monthly_states)])
        arr_annual  = np.stack([annual_daily[v] .transpose("time", "location").values
                                for v in (self.annual_forcing  + self.annual_states)])

        inputs = np.concatenate([arr_daily, arr_monthly, arr_annual], axis=0)  # [C_in, 365*Y, L]

        # --- labels (no shift; drop first year for alignment)
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

        # --- drop first year everywhere (carry & non-carry)
        inputs    = inputs[:,    365:, :]   # [C_in, 365*(Y-1), L]
        labels_m  = labels_m[:,   12:, :]   # [C_m,  12*(Y-1),  L]
        labels_a  = labels_a[:,    1:, :]   # [C_a,  (Y-1),     L]

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