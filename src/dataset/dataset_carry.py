# src/dataset/dataset_carry.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

import sys
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.dataset.variables import var_names, luh2_deltas
from src.training.varschema import VarSchema


class CarryBlockDataset(Dataset):
    """
    Carry-optimised dataset.

    One SAMPLE = (all years, a block of locations, one scenario),
    pulled as year-sized slices (leveraging Zarr time chunking):

      inputs:   [nin, 365*Y,  L]
      labels_m: [nm,  12*Y,   L]
      labels_a: [na,   Y,     L]

    Where:
      - nin = channels in (daily forcings + monthly forcings & states(t-1) + annual forcings & states(t-1))
      - Y   = number of years in the sample’s period
      - L   = number of locations in the block
    """

    def __init__(
        self,
        data_dir: str,
        std_dict: Dict,
        tensor_type: str,
        block_locs: int = 512,
        delta_luh: bool = False,
        exclude_vars: Sequence[str] | None = None,
    ):
        self.std_dict    = std_dict
        self.tensor_type = tensor_type
        self.block_locs  = int(block_locs)
        self.base_path   = Path(data_dir) / tensor_type
        self.unfiltered_var_names = var_names  # same source as other datasets
        self.delta_luh  = bool(delta_luh)
        self.exclude_set = set(exclude_vars or [])

        # ------------------------- Resolve split paths -------------------------
        if tensor_type == "train":
            self.daily_paths   = [self.base_path / "train_location_train_period/daily.zarr"]
            self.monthly_paths = [self.base_path / "train_location_train_period/monthly.zarr"]
            self.annual_paths  = [self.base_path / "train_location_train_period/annual.zarr"]

        elif tensor_type == "val":
            self.daily_paths = [
                self.base_path / "train_location_val_period_early/daily.zarr",
                self.base_path / "train_location_val_period_late/daily.zarr",
                self.base_path / "val_location_whole_period/daily.zarr",
            ]
            self.monthly_paths = [
                self.base_path / "train_location_val_period_early/monthly.zarr",
                self.base_path / "train_location_val_period_late/monthly.zarr",
                self.base_path / "val_location_whole_period/monthly.zarr",
            ]
            self.annual_paths = [
                self.base_path / "train_location_val_period_early/annual.zarr",
                self.base_path / "train_location_val_period_late/annual.zarr",
                self.base_path / "val_location_whole_period/annual.zarr",
            ]

        elif tensor_type == "test":
            self.daily_paths = [
                self.base_path / "test_location_whole_period/daily.zarr",
                self.base_path / "train_location_test_period_early/daily.zarr",
                self.base_path / "train_location_test_period_late/daily.zarr",
            ]
            self.monthly_paths = [
                self.base_path / "test_location_whole_period/monthly.zarr",
                self.base_path / "train_location_test_period_early/monthly.zarr",
                self.base_path / "train_location_test_period_late/monthly.zarr",
            ]
            self.annual_paths = [
                self.base_path / "test_location_whole_period/annual.zarr",
                self.base_path / "train_location_test_period_early/annual.zarr",
                self.base_path / "train_location_test_period_late/annual.zarr",
            ]
        else:
            raise ValueError(f"Unknown tensor_type: {tensor_type!r}")

        # ---------------------------- Open Zarrs ------------------------------
        opts = dict(consolidated=True, decode_times=False, chunks={})
        self.daily   = [xr.open_zarr(p, **opts) for p in self.daily_paths]
        self.monthly = [xr.open_zarr(p, **opts) for p in self.monthly_paths]
        self.annual  = [xr.open_zarr(p, **opts) for p in self.annual_paths]
        self._all    = self.daily + self.monthly + self.annual

        # -------------------------- Plan sample grid --------------------------
        self._plan_samples()  # builds self.meta = [(ds_idx, scenario, loc0, loc1), ...]

        # ---------------------- Build filtered variable lists -----------------
        self._filter_var_names()  # sets self.* lists and self.schema

        # Optional: expose canonical orders (if referenced downstream)
        self.input_order    = self.schema.input_order()
        self.output_order_m = self.schema.out_monthly_names()
        self.output_order_a = self.schema.out_annual_names()

    # -----------------------------------------------------------------------
    # Filtering / schema (harmonised with other datasets)
    # -----------------------------------------------------------------------
    def _present_in_any_zarr(self, name: str) -> bool:
        return any(name in ds.data_vars for ds in self._all)

    def _has_valid_stats(self, name: str) -> bool:
        stats = self.std_dict.get(name)
        try:
            return (
                stats is not None
                and np.isfinite(float(stats.get("mean", np.nan)))
                and np.isfinite(float(stats.get("std",  np.nan)))
                and float(stats["std"]) > 0.0
            )
        except Exception:
            return False

    def _add_unique(self, dst: List[str], name: str):
        if name not in dst:
            dst.append(name)

    def _filter_var_names(self) -> None:
        """
        Build filtered var lists by:
        - (optionally) extending annual_forcing with LUH deltas,
        - requiring valid std stats,
        - requiring presence in the Zarr stores (LUH deltas skipped quietly if missing),
        - applying exclude list.
        Then build stable input/output orders and a VarSchema snapshot.
        """
        # Safe copy so we don't mutate global map
        base_vars = {k: list(v) for k, v in self.unfiltered_var_names.items()}

        # If requested, append LUH deltas into annual_forcing (dedup here)
        if self.delta_luh:
            extra = [v for v in luh2_deltas if v not in base_vars["annual_forcing"]]
            base_vars["annual_forcing"].extend(extra)

        filtered: Dict[str, List[str]] = {}
        for group, var_list in base_vars.items():
            keep: List[str] = []
            for v in var_list:
                actual = v

                # respect excludes
                if actual in self.exclude_set:
                    continue

                # require std stats
                if not self._has_valid_stats(actual):
                    continue

                # require presence in Zarr; for LUH deltas, skip quietly if absent
                if not self._present_in_any_zarr(actual):
                    if self.delta_luh and group == "annual_forcing" and actual in luh2_deltas:
                        continue
                    raise AssertionError(f"Zarr datasets are missing variable: {actual}")

                self._add_unique(keep, actual)

            filtered[group] = keep

        # Persist lists for __getitem__ (sorted for stable channel layout)
        self.daily_forc     = sorted(filtered["daily_forcing"])
        self.monthly_forc   = sorted(filtered["monthly_forcing"])
        self.monthly_state  = sorted(filtered["monthly_states"])
        self.annual_forc    = sorted(filtered["annual_forcing"])
        self.annual_state   = sorted(filtered["annual_states"])
        self.monthly_fluxes = sorted(filtered["monthly_fluxes"])

        # Var schema (matches other datasets)
        self.schema = VarSchema(
            daily_forcing   = list(self.daily_forc),
            monthly_forcing = list(self.monthly_forc),
            monthly_states  = list(self.monthly_state),
            annual_forcing  = list(self.annual_forc),
            annual_states   = list(self.annual_state),
            monthly_fluxes  = list(self.monthly_fluxes),
            month_lengths   = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        )

    # -----------------------------------------------------------------------
    # Planning & length
    # -----------------------------------------------------------------------
    def _plan_samples(self) -> None:
        """
        Build a flat index mapping for samples:
          meta[i] = (dataset_idx, scenario, start_loc, end_loc)
        """
        self.meta: List[Tuple[int, int, int, int]] = []
        for ds_idx, ds_d in enumerate(self.daily):
            L = int(ds_d.sizes["location"])
            S = int(ds_d.sizes.get("scenario", 1))
            n_blocks = (L + self.block_locs - 1) // self.block_locs  # ceil_div

            for s in range(S):
                for b in range(n_blocks):
                    loc0 = b * self.block_locs
                    loc1 = min(L, loc0 + self.block_locs)
                    if loc0 < loc1:
                        self.meta.append((ds_idx, s, loc0, loc1))

    def __len__(self) -> int:
        return len(self.meta)

    # -----------------------------------------------------------------------
    # Stats helpers (keep original grouped/flat compatibility)
    # -----------------------------------------------------------------------
    def _std(self, arr: np.ndarray, name: str, group: str | None = None) -> np.ndarray:
        """
        Standardize with stats that may be stored either flat (std_dict[name]) or
        grouped (std_dict[group][name]). If std <= 0 or not finite, return zeros.
        """
        stats = None
        if group is not None and isinstance(self.std_dict.get(group), dict):
            stats = self.std_dict[group].get(name)
        if stats is None:
            stats = self.std_dict.get(name)

        if stats is None:
            # Fallback: zeros (robust in long jobs)
            return np.zeros_like(arr, dtype=np.float32)

        mean = stats.get("mean", 0.0)
        std  = stats.get("std",  None)
        if std is None or not np.isfinite(std) or std <= 0:
            return np.zeros_like(arr, dtype=np.float32)

        return ((arr - mean) / std).astype(np.float32, copy=False)

    # -----------------------------------------------------------------------
    # Item creation (unchanged carry + block logic)
    # -----------------------------------------------------------------------
    def __getitem__(self, idx: int):
        """
        Return (inputs, labels_monthly, labels_annual) for a single sample.

        Reads one scenario and a contiguous location block across *all* years,
        builds standardized inputs/labels, and expands monthly/annual inputs
        to daily resolution for the model input.
        """
        ds_idx, scenario, loc0, loc1 = self.meta[idx]

        # Zarr slices for this (dataset, scenario, location block)
        Dd = self.daily[ds_idx].isel(location=slice(loc0, loc1), scenario=scenario)
        Dm = self.monthly[ds_idx].isel(location=slice(loc0, loc1), scenario=scenario)
        Da = self.annual[ds_idx].isel(location=slice(loc0, loc1), scenario=scenario)

        # Dimensions
        Y = int(Da.sizes["time"])       # years in this sample
        L = int(Dd.sizes["location"])   # locations in this block

        # Convenience bindings
        daily_forc    = self.daily_forc
        monthly_forc  = self.monthly_forc
        monthly_state = self.monthly_state
        annual_forc   = self.annual_forc
        annual_state  = self.annual_state

        # Accumulators across years
        in_daily_chunks:   List[np.ndarray] = []
        in_monthly_chunks: List[np.ndarray] = []
        in_annual_chunks:  List[np.ndarray] = []
        out_m_chunks:      List[np.ndarray] = []
        out_a_chunks:      List[np.ndarray] = []

        # Month meta
        mlens = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int32)
        day2m = np.repeat(np.arange(12, dtype=np.int64), mlens)

        for y in range(Y):
            # Year slices
            d_y = Dd.isel(time=slice(y * 365, (y + 1) * 365))
            m_y = Dm.isel(time=slice(y * 12,  (y + 1) * 12))
            a_y = Da.isel(time=slice(y,       y + 1))

            # --- Monthly inputs (forcings + states t-1) ---
            m_states_t1_np = {}
            for v in monthly_state:
                arr = m_y[v].transpose("time", "location").values  # [12, L]
                shf = np.roll(arr, 1, axis=0)
                shf[0, :] = 0.0
                m_states_t1_np[v] = shf

            m_std = {}
            for v in monthly_forc:
                m_std[v] = self._std(m_y[v].transpose("time", "location").values, v, "monthly")
            for v in monthly_state:
                m_std[v] = self._std(m_states_t1_np[v], v, "monthly")

            # --- Annual inputs (forcings + states t-1) ---
            a_states_t1_np = {}
            for v in annual_state:
                a_states_t1_np[v] = np.zeros_like(a_y[v].transpose("time", "location").values)  # [1, L]

            a_std = {}
            for v in annual_forc:
                a_std[v] = self._std(a_y[v].transpose("time", "location").values, v, "annual")
            for v in annual_state:
                a_std[v] = self._std(a_states_t1_np[v], v, "annual")

            # --- Daily inputs (forcings only) ---
            d_std = {v: self._std(d_y[v].transpose("time", "location").values, v, "daily")
                     for v in daily_forc}

            # --- Expand month/annual to daily ---
            Lloc = int(d_y.sizes["location"])
            m_daily_stacked = np.stack(
                [m_std[v].reshape(12, Lloc)[day2m, :] for v in (monthly_forc + monthly_state)],
                axis=0,
            )  # [Cm, 365, L]
            a_daily_stacked = np.stack(
                [np.repeat(a_std[v], 365, axis=0) for v in (annual_forc + annual_state)],
                axis=0,
            )  # [Ca, 365, L]
            d_daily_stacked = np.stack([d_std[v] for v in daily_forc], axis=0)  # [Cd, 365, L]

            in_daily_chunks.append(d_daily_stacked)
            in_monthly_chunks.append(m_daily_stacked)
            in_annual_chunks.append(a_daily_stacked)

            # --- Labels (no shifts) ---
            m_lab_std = {v: self._std(m_y[v].transpose("time", "location").values, v, "monthly")
                         for v in (self.monthly_fluxes + monthly_state)}
            a_lab_std = {v: self._std(a_y[v].transpose("time", "location").values, v, "annual")
                         for v in annual_state}

            out_m = np.stack([m_lab_std[v] for v in (self.monthly_fluxes + monthly_state)], axis=0)  # [nm, 12, L]
            out_a = np.stack([a_lab_std[v] for v in annual_state], axis=0)                           # [na,  1, L]

            out_m_chunks.append(out_m)
            out_a_chunks.append(out_a)

        # --- Concatenate years along time ---
        in_daily   = np.concatenate(in_daily_chunks,   axis=1)  # [Cd, 365*Y, L]
        in_monthly = np.concatenate(in_monthly_chunks, axis=1)  # [Cm, 365*Y, L]
        in_annual  = np.concatenate(in_annual_chunks,  axis=1)  # [Ca, 365*Y, L]
        inputs = np.concatenate([in_daily, in_monthly, in_annual], axis=0)  # [nin, 365*Y, L]

        out_m_all = np.concatenate(out_m_chunks, axis=1)  # [nm, 12*Y, L]
        out_a_all = np.concatenate(out_a_chunks, axis=1)  # [na,  1*Y, L]

        # Sanity checks
        assert inputs.ndim == 3 and inputs.shape[1] == 365 * Y and inputs.shape[2] == L
        assert out_m_all.ndim == 3 and out_m_all.shape[1] == 12 * Y and out_m_all.shape[2] == L
        assert out_a_all.ndim == 3 and out_a_all.shape[1] == 1 * Y  and out_a_all.shape[2] == L

        return (
            torch.from_numpy(inputs.astype(np.float32,   copy=False)),
            torch.from_numpy(out_m_all.astype(np.float32, copy=False)),
            torch.from_numpy(out_a_all.astype(np.float32, copy=False)),
        )