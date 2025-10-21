# src/dataset/dataset_carry.py
from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

# Project root on path (unchanged)
import sys
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.dataset.variables import var_names
from src.training.varschema import VarSchema


# ---------------------------------------------------------------------------
# Dataset (carry-optimised)
# ---------------------------------------------------------------------------

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
    ):
        self.std_dict    = std_dict
        self.tensor_type = tensor_type
        self.block_locs  = int(block_locs)
        self.base_path   = Path(data_dir) / tensor_type
        self.var_names_all = var_names  # same global variable map used elsewhere

        # ------------------------- Resolve split paths -------------------------
        if tensor_type == "train":
            self.daily_paths =   [self.base_path / "train_location_train_period/daily.zarr"]
            self.monthly_paths = [self.base_path / "train_location_train_period/monthly.zarr"]
            self.annual_paths =  [self.base_path / "train_location_train_period/annual.zarr"]

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
        # Open group handles (no chunk reads yet); decode_times disabled for speed.
        opts = dict(consolidated=True, decode_times=False, chunks={})
        self.daily   = [xr.open_zarr(p, **opts) for p in self.daily_paths]
        self.monthly = [xr.open_zarr(p, **opts) for p in self.monthly_paths]
        self.annual  = [xr.open_zarr(p, **opts) for p in self.annual_paths]

        # -------------------------- Plan sample grid --------------------------
        self._plan_samples()  # builds self.meta = [(ds_idx, scenario, loc0, loc1), ...]

        # ------------------ Build & save filtered variable lists --------------
        df   = self._filter_by_stats(sorted(self.var_names_all["daily_forcing"]),   "daily")
        mf   = self._filter_by_stats(sorted(self.var_names_all["monthly_forcing"]), "monthly")
        ms   = self._filter_by_stats(sorted(self.var_names_all["monthly_states"]),  "monthly")
        af   = self._filter_by_stats(sorted(self.var_names_all["annual_forcing"]),  "annual")
        a_s  = self._filter_by_stats(sorted(self.var_names_all["annual_states"]),   "annual")
        mfx  = self._filter_by_stats(sorted(self.var_names_all["monthly_fluxes"]),  "monthly")

        # Persist lists for __getitem__
        self.daily_forc     = df
        self.monthly_forc   = mf
        self.monthly_state  = ms
        self.annual_forc    = af
        self.annual_state   = a_s
        self.monthly_fluxes = mfx

        # Var schema (useful elsewhere; preserves channel orders)
        self.schema = VarSchema(
            daily_forcing   = list(self.daily_forc),
            monthly_forcing = list(self.monthly_forc),
            monthly_states  = list(self.monthly_state),
            annual_forcing  = list(self.annual_forc),
            annual_states   = list(self.annual_state),
            monthly_fluxes  = list(self.monthly_fluxes),
            month_lengths   = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        )

        # Optional: expose canonical orders (if referenced downstream)
        self.input_order    = self.schema.input_order()
        self.output_order_m = self.schema.out_monthly_names()
        self.output_order_a = self.schema.out_annual_names()

    # -----------------------------------------------------------------------
    # Planning & length
    # -----------------------------------------------------------------------

    def _plan_samples(self) -> None:
        """
        Build a flat index mapping for samples:
          meta[i] = (dataset_idx, scenario, start_loc, end_loc)

        We tile each Zarr dataset into blocks of locations (≈ block_locs),
        and replicate for each available scenario.
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
    # Stats helpers
    # -----------------------------------------------------------------------

    def _has_stats(self, name: str, group: str | None = None) -> bool:
        """
        True if stats for 'name' exist either under std_dict[group][name] or std_dict[name].
        """
        if group is not None and isinstance(self.std_dict.get(group), dict):
            if name in self.std_dict[group]:
                return True
        return name in self.std_dict

    def _filter_by_stats(self, names: List[str], group: str | None) -> List[str]:
        """
        Keep only variables that have stats; return missing list silently (no error).
        """
        keep = [v for v in names if self._has_stats(v, group)]
        # (You can log 'missing' here if desired)
        # missing = sorted(set(names) - set(keep))
        return keep

    def _std(self, arr: np.ndarray, name: str, group: str | None = None) -> np.ndarray:
        """
        Standardize with stats that may be stored either flat (std_dict[name]) or
        grouped (std_dict[group][name]).
        If std <= 0 or not finite, return zeros (avoid NaNs).
        """
        stats = None
        if group is not None and isinstance(self.std_dict.get(group), dict):
            stats = self.std_dict[group].get(name)
        if stats is None:
            stats = self.std_dict.get(name)

        if stats is None:
            available = list(self.std_dict.keys())
            raise KeyError(
                f"No stats for '{name}' (group={group!r}). "
                f"Available top-level keys: {available[:10]}{'...' if len(available) > 10 else ''}"
            )

        mean = stats.get("mean", 0.0)
        std  = stats.get("std",  None)
        if std is None:
            raise KeyError(f"Stats for '{name}' missing 'std' (group={group!r}): {stats}")

        if not np.isfinite(std) or std <= 0:
            return np.zeros_like(arr, dtype=np.float32)

        return ((arr - mean) / std).astype(np.float32, copy=False)

    # -----------------------------------------------------------------------
    # Item creation
    # -----------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (inputs, labels_monthly, labels_annual) for a single sample.

        Reads one scenario and a contiguous location block across *all* years
        from the chosen dataset index, builds standardized inputs/labels, and
        expands monthly/annual inputs to daily resolution for the model input.
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

        # Accumulators across years (we concatenate at the end)
        in_daily_chunks:   List[np.ndarray] = []
        in_monthly_chunks: List[np.ndarray] = []
        in_annual_chunks:  List[np.ndarray] = []
        out_m_chunks:      List[np.ndarray] = []
        out_a_chunks:      List[np.ndarray] = []

        # Month meta
        mlens = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int32)
        day2m = np.repeat(np.arange(12, dtype=np.int64), mlens)

        for y in range(Y):
            # ------------------ Slice this year's data ------------------
            d_y = Dd.isel(time=slice(y * 365, (y + 1) * 365))   # daily year window
            m_y = Dm.isel(time=slice(y * 12,  (y + 1) * 12))    # monthly year window
            a_y = Da.isel(time=slice(y,       y + 1))           # annual single step

            # ---------------- Monthly inputs (forcings + states t-1) ----------------
            # Shift monthly states by 1 month (t-1) within the year; Jan gets zeros.
            m_states_t1_np = {}
            for v in monthly_state:
                arr = m_y[v].transpose("time", "location").values  # [12, L]
                shf = np.roll(arr, 1, axis=0)
                shf[0, :] = 0.0
                m_states_t1_np[v] = shf

            # Standardize monthly forcings & shifted states separately
            m_std = {}
            for v in monthly_forc:
                m_std[v] = self._std(m_y[v].transpose("time", "location").values, v, "monthly")
            for v in monthly_state:
                m_std[v] = self._std(m_states_t1_np[v], v, "monthly")

            # ---------------- Annual inputs (forcings + states t-1) -----------------
            # Shift annual states by 1 year (t-1) INSIDE this 1-year window => zeros.
            a_states_t1_np = {}
            for v in annual_state:
                a_states_t1_np[v] = np.zeros_like(a_y[v].transpose("time", "location").values)  # [1, L]

            a_std = {}
            for v in annual_forc:
                a_std[v] = self._std(a_y[v].transpose("time", "location").values, v, "annual")
            for v in annual_state:
                a_std[v] = self._std(a_states_t1_np[v], v, "annual")

            # ---------------- Daily inputs (forcings only) --------------------------
            d_std = {
                v: self._std(d_y[v].transpose("time", "location").values, v, "daily")
                for v in daily_forc
            }

            # ---------------- Expand month/annual to daily --------------------------
            Lloc = int(d_y.sizes["location"])

            # Monthly: [12, L] → index to [365, L] by repeating per month length
            m_daily_stacked = np.stack(
                [m_std[v].reshape(12, Lloc)[day2m, :] for v in (monthly_forc + monthly_state)],
                axis=0,
            )  # [Cm, 365, L]

            # Annual: [1, L] → repeat 365 days → [365, L]
            a_daily_stacked = np.stack(
                [np.repeat(a_std[v], 365, axis=0) for v in (annual_forc + annual_state)],
                axis=0,
            )  # [Ca, 365, L]

            # Daily forcings already daily: [365, L]
            d_daily_stacked = np.stack([d_std[v] for v in daily_forc], axis=0)  # [Cd, 365, L]

            # Append inputs (per-year)
            in_daily_chunks.append(d_daily_stacked)
            in_monthly_chunks.append(m_daily_stacked)
            in_annual_chunks.append(a_daily_stacked)

            # --------------------------- Labels -------------------------------------
            # Monthly labels: monthly_fluxes + monthly_states (no shift)
            m_lab_std = {
                v: self._std(m_y[v].transpose("time", "location").values, v, "monthly")
                for v in (self.monthly_fluxes + monthly_state)
            }
            # Annual labels: annual_states (no shift)
            a_lab_std = {
                v: self._std(a_y[v].transpose("time", "location").values, v, "annual")
                for v in annual_state
            }

            out_m = np.stack([m_lab_std[v] for v in (self.monthly_fluxes + monthly_state)], axis=0)  # [nm, 12, L]
            out_a = np.stack([a_lab_std[v] for v in annual_state], axis=0)                           # [na,  1, L]

            out_m_chunks.append(out_m)
            out_a_chunks.append(out_a)

        # --------------------- Concatenate years along time -------------------------
        in_daily   = np.concatenate(in_daily_chunks,   axis=1)  # [Cd, 365*Y, L]
        in_monthly = np.concatenate(in_monthly_chunks, axis=1)  # [Cm, 365*Y, L]
        in_annual  = np.concatenate(in_annual_chunks,  axis=1)  # [Ca, 365*Y, L]
        inputs = np.concatenate([in_daily, in_monthly, in_annual], axis=0)  # [nin, 365*Y, L]

        out_m_all = np.concatenate(out_m_chunks, axis=1)  # [nm, 12*Y, L]
        out_a_all = np.concatenate(out_a_chunks, axis=1)  # [na,  1*Y, L]

        # ------------------------------ Sanity checks ------------------------------
        assert inputs.ndim == 3 and inputs.shape[1] == 365 * Y and inputs.shape[2] == L
        assert out_m_all.ndim == 3 and out_m_all.shape[1] == 12 * Y and out_m_all.shape[2] == L
        assert out_a_all.ndim == 3 and out_a_all.shape[1] == 1 * Y  and out_a_all.shape[2] == L

        # -------------------------- Convert to torch -------------------------------
        return (
            torch.from_numpy(inputs.astype(np.float32,   copy=False)),
            torch.from_numpy(out_m_all.astype(np.float32, copy=False)),
            torch.from_numpy(out_a_all.astype(np.float32, copy=False)),
        )