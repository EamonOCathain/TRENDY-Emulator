# src/dataset/dataset_carry.py
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
import sys
# set project root
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.training.varschema import VarSchema

from src.dataset.variables import var_names  

class CarryBlockDataset(Dataset):
    """
    Carry-optimised dataset:
      - returns one SAMPLE = (all years, many locations, one scenario)
      - builds tensors by reading year-sized slices (leveraging time=1y Zarr chunks)
      - output shapes match your trainer carry path:
          inputs:   [nin, 365*Y,  L]
          labels_m: [nm,  12*Y,   L]
          labels_a: [na,   Y,     L]
    """
    def __init__(self, data_dir: str, std_dict: Dict, tensor_type: str,
                 block_locs: int = 512):
        self.std_dict = std_dict
        self.tensor_type = tensor_type
        self.block_locs = int(block_locs)
        self.base_path = Path(data_dir) / tensor_type
        self.var_names_all = var_names  # same source as before

        # Resolve split paths (same structure as your current CustomDataset)
        if tensor_type == "train":
            self.daily_paths   = [self.base_path / "train_location_train_period/daily.zarr"]
            self.monthly_paths = [self.base_path / "train_location_train_period/monthly.zarr"]
            self.annual_paths  = [self.base_path / "train_location_train_period/annual.zarr"]
        elif tensor_type == "val":
            self.daily_paths   = [
                self.base_path / "train_location_val_period_early/daily.zarr",
                self.base_path / "train_location_val_period_late/daily.zarr",
                self.base_path / "val_location_whole_period/daily.zarr",
            ]
            self.monthly_paths = [
                self.base_path / "train_location_val_period_early/monthly.zarr",
                self.base_path / "train_location_val_period_late/monthly.zarr",
                self.base_path / "val_location_whole_period/monthly.zarr",
            ]
            self.annual_paths  = [
                self.base_path / "train_location_val_period_early/annual.zarr",
                self.base_path / "train_location_val_period_late/annual.zarr",
                self.base_path / "val_location_whole_period/annual.zarr",
            ]
        elif tensor_type == "test":
            self.daily_paths   = [
                self.base_path / "test_location_whole_period/daily.zarr",
                self.base_path / "train_location_test_period_early/daily.zarr",
                self.base_path / "train_location_test_period_late/daily.zarr",
            ]
            self.monthly_paths = [
                self.base_path / "test_location_whole_period/monthly.zarr",
                self.base_path / "train_location_test_period_early/monthly.zarr",
                self.base_path / "train_location_test_period_late/monthly.zarr",
            ]
            self.annual_paths  = [
                self.base_path / "test_location_whole_period/annual.zarr",
                self.base_path / "train_location_test_period_early/annual.zarr",
                self.base_path / "train_location_test_period_late/annual.zarr",
            ]
        else:
            raise ValueError(f"Unknown tensor_type: {tensor_type}")

        # Open Zarr groups once (consolidated) — no chunks read yet
        opts = dict(consolidated=True, decode_times=False, chunks={})
        self.daily   = [xr.open_zarr(p, **opts) for p in self.daily_paths]
        self.monthly = [xr.open_zarr(p, **opts) for p in self.monthly_paths]
        self.annual  = [xr.open_zarr(p, **opts) for p in self.annual_paths]

        # Basic dims and planning
        self._plan_samples()
        
        # ----- build & SAVE filtered var lists -----
        df  = self._filter_by_stats(sorted(self.var_names_all['daily_forcing']),   "daily")
        mf  = self._filter_by_stats(sorted(self.var_names_all['monthly_forcing']), "monthly")
        ms  = self._filter_by_stats(sorted(self.var_names_all['monthly_states']),  "monthly")
        af  = self._filter_by_stats(sorted(self.var_names_all['annual_forcing']),  "annual")
        as_ = self._filter_by_stats(sorted(self.var_names_all['annual_states']),   "annual")
        mflux = self._filter_by_stats(sorted(self.var_names_all['monthly_fluxes']), "monthly")

        # save per-group lists for reuse in __getitem__
        self.daily_forc     = df
        self.monthly_forc   = mf
        self.monthly_state  = ms
        self.annual_forc    = af
        self.annual_state   = as_
        self.monthly_fluxes = mflux
        
        self.schema = VarSchema(
            daily_forcing   = list(self.daily_forc),
            monthly_forcing = list(self.monthly_forc),
            monthly_states  = list(self.monthly_state),
            annual_forcing  = list(self.annual_forc),
            annual_states   = list(self.annual_state),
            monthly_fluxes  = list(self.monthly_fluxes),
            month_lengths   = [31,28,31,30,31,30,31,31,30,31,30,31],
        )

        # If you still reference these elsewhere in the class:
        self.input_order    = self.schema.input_order()
        self.output_order_m = self.schema.out_monthly_names()
        self.output_order_a = self.schema.out_annual_names()

    def _plan_samples(self):
        """Plan (ds_idx, scenario, location_block) grid -> flat indexing."""
        self.meta = []  # list of tuples (ds_idx, scenario, start_loc, end_loc)
        for i, ds_d in enumerate(self.daily):
            L = int(ds_d.sizes["location"])
            S = int(ds_d.sizes.get("scenario", 1))
            n_blocks = (L + self.block_locs - 1) // self.block_locs
            for s in range(S):
                for b in range(n_blocks):
                    start = b * self.block_locs
                    end   = min(L, start + self.block_locs)
                    if start < end:
                        self.meta.append((i, s, start, end))

    def __len__(self):
        return len(self.meta)

    # ---------- helpers ----------
    def _std(self, arr: np.ndarray, name: str, group: str | None = None) -> np.ndarray:
        """
        Standardize with stats that may be stored either flat (std_dict[name]) or
        grouped (std_dict[group][name]). If std == 0 or invalid, return zeros.
        """
        stats = None

        # Try grouped first if present
        if group is not None and isinstance(self.std_dict.get(group), dict):
            stats = self.std_dict[group].get(name)

        # Fallback to flat structure
        if stats is None:
            stats = self.std_dict.get(name)

        if stats is None:
            available = list(self.std_dict.keys())
            raise KeyError(f"std_dict has no stats for '{name}' (group={group!r}). "
                        f"Available top-level keys: {available[:10]}{'...' if len(available)>10 else ''}")

        mean = stats.get("mean", 0.0)
        std  = stats.get("std",  None)

        if std is None:
            raise KeyError(f"stats for '{name}' missing 'std' (group={group!r}): {stats}")

        # Guard zero/invalid variance: return zeros (since (x-mean)/0 is undefined)
        if not np.isfinite(std) or std <= 0:
            return np.zeros_like(arr, dtype=np.float32)

        return ((arr - mean) / std).astype(np.float32, copy=False)
    
    def _has_stats(self, name: str, group: str | None = None) -> bool:
        if group is not None and isinstance(self.std_dict.get(group), dict):
            if name in self.std_dict[group]:
                return True
        return name in self.std_dict  # flat fallback

    def _filter_by_stats(self, names: list[str], group: str | None) -> list[str]:
        keep = [v for v in names if self._has_stats(v, group)]
        missing = sorted(set(names) - set(keep))
        return keep

    # ---------- main ----------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds_idx, scenario, loc0, loc1 = self.meta[idx]
        Dd = self.daily[ds_idx].isel(location=slice(loc0, loc1), scenario=scenario)
        Dm = self.monthly[ds_idx].isel(location=slice(loc0, loc1), scenario=scenario)
        Da = self.annual[ds_idx].isel(location=slice(loc0, loc1), scenario=scenario)

        # Dimensions
        Y = int(Da.sizes["time"])          # years
        L = int(Dd.sizes["location"])      # locations in this block

        # use the filtered lists saved in __init__
        daily_forc    = self.daily_forc
        monthly_forc  = self.monthly_forc
        monthly_state = self.monthly_state
        annual_forc   = self.annual_forc
        annual_state  = self.annual_state

        # Accumulators
        in_daily_chunks   = []
        in_monthly_chunks = []
        in_annual_chunks  = []
        out_m_chunks      = []
        out_a_chunks      = []

        for y in range(Y):
            # slice each resolution for this year
            d_y = Dd.isel(time=slice(y*365, (y+1)*365))
            m_y = Dm.isel(time=slice(y*12,  (y+1)*12))
            a_y = Da.isel(time=slice(y,     y+1))

            # ---------- MONTHLY: build inputs without xr.merge / xr.shift ----------
            # monthly states shifted by 1 month (t-1), zeros for January
            m_states_t1_np = {}
            for v in monthly_state:
                arr = m_y[v].transpose("time", "location").values  # [12, L]
                shf = np.roll(arr, 1, axis=0)
                shf[0, :] = 0.0
                m_states_t1_np[v] = shf

            # standardise monthly forcings (native) and shifted states separately
            m_std = {}
            # monthly forcings
            for v in monthly_forc:
                arr = m_y[v].transpose("time", "location").values  # [12, L]
                m_std[v] = self._std(arr, v, "monthly")

            # shifted monthly states
            for v in monthly_state:
                m_std[v] = self._std(m_states_t1_np[v], v, "monthly")

            # ---------- ANNUAL: build inputs without xr.merge / xr.shift ----------
            # annual states shifted by 1 year (t-1) within this 1-year window => zeros
            a_states_t1_np = {}
            for v in annual_state:
                # this slice is [1, L]; t-1 within this window is zero
                a_states_t1_np[v] = np.zeros_like(a_y[v].transpose("time", "location").values)

            a_std = {}
            # annual forcings
            for v in annual_forc:
                arr = a_y[v].transpose("time", "location").values  # [1, L]
                a_std[v] = self._std(arr, v, "annual")

            # shifted annual states
            for v in annual_state:
                a_std[v] = self._std(a_states_t1_np[v], v, "annual")  # [1, L]

            # ---------- DAILY forcing ----------
            d_std = {
                v: self._std(d_y[v].transpose("time", "location").values, v, "daily")
                for v in daily_forc
            }

            # ---------- Expand monthly/annual to daily ----------
            mlens = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=np.int32)
            day2m = np.repeat(np.arange(12, dtype=np.int64), mlens)
            Lloc  = int(d_y.sizes["location"])

            m_daily_stacked = np.stack(
                [m_std[v].reshape(12, Lloc)[day2m, :] for v in (monthly_forc + monthly_state)],
                axis=0
            )  # [Cm,365,L]
            a_daily_stacked = np.stack(
                [np.repeat(a_std[v], 365, axis=0) for v in (annual_forc + annual_state)],
                axis=0
            )  # [Ca,365,L]
            d_daily_stacked = np.stack([d_std[v] for v in daily_forc], axis=0)  # [Cd,365,L]

            in_daily_chunks.append(d_daily_stacked)
            in_monthly_chunks.append(m_daily_stacked)
            in_annual_chunks.append(a_daily_stacked)

            # ---------- labels (no merge) ----------
            m_lab_std = {
                v: self._std(m_y[v].transpose("time","location").values, v, "monthly")
                for v in (self.monthly_fluxes + monthly_state)
            }
            a_lab_std = {
                v: self._std(a_y[v].transpose("time","location").values, v, "annual")
                for v in annual_state
            }

            out_m = np.stack([m_lab_std[v] for v in (self.monthly_fluxes + monthly_state)], axis=0)  # [nm,12,L]
            out_a = np.stack([a_lab_std[v] for v in annual_state], axis=0)                           # [na,1,L]
            
            out_m_chunks.append(out_m)
            out_a_chunks.append(out_a)

        # Concatenate years along time
        in_daily   = np.concatenate(in_daily_chunks,   axis=1)   # [Cd, 365*Y, L]
        in_monthly = np.concatenate(in_monthly_chunks, axis=1)   # [Cm, 365*Y, L]
        in_annual  = np.concatenate(in_annual_chunks,  axis=1)   # [Ca, 365*Y, L]

        inputs = np.concatenate([in_daily, in_monthly, in_annual], axis=0)  # [nin, 365*Y, L]

        out_m_all = np.concatenate(out_m_chunks, axis=1)  # [nm, 12*Y, L]
        out_a_all = np.concatenate(out_a_chunks, axis=1)  # [na, 1*Y,  L]
        
        # inputs: [nin, 365*Y, L]
        assert inputs.ndim == 3 and inputs.shape[1] == 365 * Y and inputs.shape[2] == L
        # labels: [nm, 12*Y, L], [na, 1*Y, L]
        assert out_m_all.ndim == 3 and out_m_all.shape[1] == 12 * Y and out_m_all.shape[2] == L
        assert out_a_all.ndim == 3 and out_a_all.shape[1] == 1 * Y  and out_a_all.shape[2] == L

        return (torch.from_numpy(inputs.astype(np.float32, copy=False)),
                torch.from_numpy(out_m_all.astype(np.float32, copy=False)),
                torch.from_numpy(out_a_all.astype(np.float32, copy=False)))