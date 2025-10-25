from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CustomDataset(Dataset):
    """
    Dataset for loading and processing climate data from Zarr stores.

    Produces tuples:
      (inputs[C_in, 365*Y-365, L], monthly_labels[C_m, 12*Y-12, L], annual_labels[C_a, Y-1, L])

    Notes on alignment:
      - Inputs include monthly/annual states shifted by 1 step (t-1 context),
        so we drop the first year from inputs and labels to remove the cold-start.
      - Monthly/annual inputs are expanded to daily (by repetition) before concatenation.
    """

    def __init__(
            self,
            data_dir: str,
            std_dict: Dict,
            tensor_type: str,        # "train" | "val" | "test"
            chunk_size: int = 70,    # locations per sample
            exclude_vars: Sequence[str] | None = None, 
            delta_luh: bool = False  
        ):
            self.std_dict = std_dict
            self.tensor_type = tensor_type
            self.chunk_size = chunk_size
            self.n_scenarios = 4
            self.base_path = Path(data_dir) / tensor_type
            self.unfiltered_var_names = var_names
            self.delta_luh = delta_luh  
            self.exclude_vars = set(exclude_vars or [])

            # Discover file layout and open datasets
            self._get_paths()
            self._open_datasets()

            # Select/validate variable names and build I/O orders
            self._filter_var_names()

            # Precompute sample plan for __len__/__getitem__
            self._plan_samples() 

    # -----------------------------------------------------------------------
    # Paths / opening
    # -----------------------------------------------------------------------

    def _get_paths(self) -> None:
        """Fill lists of daily/monthly/annual Zarr paths based on tensor_type."""
        if self.tensor_type == "train":
            self.daily_paths   = [self.base_path / "train_location_train_period/daily.zarr"]
            self.monthly_paths = [self.base_path / "train_location_train_period/monthly.zarr"]
            self.annual_paths  = [self.base_path / "train_location_train_period/annual.zarr"]

        elif self.tensor_type == "val":
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

        elif self.tensor_type == "test":
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
            raise ValueError(f"Unknown tensor_type: {self.tensor_type!r}")

    def _open_datasets(self) -> None:
        """Open all Zarr datasets and store in lists."""
        opts = dict(consolidated=True, decode_times=False, chunks={})
        self.daily_datasets   = [xr.open_zarr(p, **opts) for p in self.daily_paths]
        self.monthly_datasets = [xr.open_zarr(p, **opts) for p in self.monthly_paths]
        self.annual_datasets  = [xr.open_zarr(p, **opts) for p in self.annual_paths]
        self.all_datasets     = self.daily_datasets + self.monthly_datasets + self.annual_datasets

    # -----------------------------------------------------------------------
    # Variable filtering / ordering
    # -----------------------------------------------------------------------

    def _filter_var_names(self) -> None:
        """
        Build filtered var lists by:
        - applying optional renames (e.g., "lai" -> "lai_avh15c1"),
        - requiring valid std stats,
        - requiring presence in the Zarr stores,
        - applying exclude list.
        Then build stable input/output orders and a VarSchema snapshot.
        """
        def present_in_any_zarr(name: str) -> bool:
            return any(name in ds.data_vars for ds in self.all_datasets)

        def add_unique(dst: List[str], name: str):
            if name not in dst:
                dst.append(name)
                
        # safe copy so we don't mutate the global var_names
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
                if actual in self.exclude_vars:
                    continue

                # require std stats
                stats = self.std_dict.get(actual)
                if not stats or float(stats.get("std", 0.0)) <= 0.0:
                    continue

                # require presence in Zarr; for LUH deltas, skip quietly if absent
                if not present_in_any_zarr(actual):
                    if self.delta_luh and actual in luh2_deltas:
                        # silently skip optional LUH deltas if not present in the data
                        continue
                    raise AssertionError(f"Zarr datasets are missing variable: {actual}")

                add_unique(keep, actual)

            filtered[group] = keep

        # Freeze final filtered lists (sorted for stable channel layout)
        self.var_names = filtered

        self.input_order = (
            sorted(self.var_names["daily_forcing"]) +
            sorted(self.var_names["monthly_forcing"]) +
            sorted(self.var_names["monthly_states"]) +
            sorted(self.var_names["annual_forcing"]) +
            sorted(self.var_names["annual_states"])
        )
        self.output_order = (
            sorted(self.var_names["monthly_fluxes"]) +
            sorted(self.var_names["monthly_states"]) +
            sorted(self.var_names["annual_states"])
        )

        # Guardrails
        if not self.input_order:
            raise RuntimeError("Empty input_order after filtering.")
        if not self.output_order:
            raise RuntimeError("Empty output_order after filtering.")

        # Optional indices
        self.input_index  = {v: i for i, v in enumerate(self.input_order)}
        self.output_index = {v: i for i, v in enumerate(self.output_order)}

        # EXPOSE a canonical schema so main can consume it (and checkpoints record it)
        self.schema = VarSchema(
            daily_forcing   = sorted(self.var_names["daily_forcing"]),
            monthly_forcing = sorted(self.var_names["monthly_forcing"]),
            monthly_states  = sorted(self.var_names["monthly_states"]),
            annual_forcing  = sorted(self.var_names["annual_forcing"]),
            annual_states   = sorted(self.var_names["annual_states"]),
            monthly_fluxes  = sorted(self.var_names["monthly_fluxes"]),
        )

    # -----------------------------------------------------------------------
    # Sampling plan
    # -----------------------------------------------------------------------

    def _plan_samples(self) -> None:
        """
        Compute per-dataset sample counts, cumulative thresholds, total samples,
        and number of location-chunks per dataset.

        Sets:
          - n_location_chunks_list: List[int]
          - sample_counts:          List[int]
          - idx_thresholds:         np.ndarray[len = len(datasets)+1]
          - n_samples:              int
        """
        # sanity: matching location sizes across resolutions for each dataset index
        for i in range(len(self.daily_datasets)):
            Ld = int(self.daily_datasets[i].sizes["location"])
            Lm = int(self.monthly_datasets[i].sizes["location"])
            La = int(self.annual_datasets[i].sizes["location"])
            if not (Ld == Lm == La):
                raise ValueError(
                    f"location size mismatch at ds[{i}]: daily={Ld}, monthly={Lm}, annual={La}"
                )

        # number of location-chunks per dataset (floor division)
        self.n_location_chunks_list = [
            int(ds.sizes["location"]) // self.chunk_size for ds in self.daily_datasets
        ]

        # samples per dataset = (#loc-chunks) * (#scenarios)
        self.sample_counts = [
            n_chunks * int(self.n_scenarios) for n_chunks in self.n_location_chunks_list
        ]

        # cumulative thresholds to map global idx -> dataset index
        self.idx_thresholds = np.cumsum([0] + self.sample_counts)

        # total across all datasets
        self.n_samples = int(sum(self.sample_counts))

    # -----------------------------------------------------------------------
    # Basic dataset methods
    # -----------------------------------------------------------------------

    def __len__(self) -> int:
        """Return total number of samples across all datasets."""
        return self.n_samples

    # -----------------------------------------------------------------------
    # Standardization helpers
    # -----------------------------------------------------------------------

    def _standardise(self, arr: np.ndarray, var_name: str, std_dict: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        Standardize array by (x - mean) / std using precomputed stats from std_dict.
        """
        stats = std_dict.get(var_name)
        if not stats:
            raise ValueError(f"No stats for '{var_name}'")
        std = stats["std"]
        if std <= 0:
            raise ValueError(f"Non-positive std for '{var_name}'")
        return (arr - stats["mean"]) / std

    def _standardise_dataset(self, ds: xr.Dataset, var_list: List[str]) -> xr.Dataset:
        """
        Return a new xr.Dataset where each variable in var_list is standardized.
        Preserves (time, location) dims and original coords.

        Iterates in var_list order so later stacking preserves INPUT/OUTPUT order.
        """
        if len(var_list) == 0:
            raise RuntimeError("Variable list empty in dataloader standardisation")

        out = {}
        for var in var_list:  # preserves caller’s order
            arr = ds[var].transpose("time", "location", ...).values
            arr_std = self._standardise(arr, var, self.std_dict)
            out[var] = xr.DataArray(
                arr_std,
                dims=("time", "location"),
                coords={"time": ds[var]["time"].values, "location": ds[var]["location"].values},
            )
        return xr.Dataset(out, coords={"time": ds["time"].values, "location": ds["location"].values})

    # -----------------------------------------------------------------------
    # Chunk extraction / time expansion
    # -----------------------------------------------------------------------

    def _extract_chunk_from_ds(
        self,
        dataset_idx: int,
        local_idx: int,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Extract a single chunk (location window, scenario slice) from a dataset.

        Mapping:
          local_idx ∈ [0, n_chunks*n_scenarios)
          scenario = local_idx // n_chunks
          chunk    = local_idx %  n_chunks
        """
        n_location_chunks = self.n_location_chunks_list[dataset_idx]

        scenario = local_idx // n_location_chunks
        chunk = local_idx % n_location_chunks

        start_loc = chunk * self.chunk_size
        end_loc = start_loc + self.chunk_size

        return ds.isel(location=slice(start_loc, end_loc), scenario=scenario)

    def _expand_monthly_to_daily(self, chunk: xr.Dataset) -> xr.Dataset:
        """Repeat monthly values to daily using month lengths (31-28-31-...)."""
        month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int32)
        day_to_month = np.repeat(np.arange(12, dtype=np.int64), month_lengths)

        months = int(chunk.sizes["time"])
        if months % 12 != 0:
            raise ValueError(f"Monthly time length {months} is not divisible by 12.")
        years = months // 12

        locations = int(chunk.sizes["location"])
        days = np.arange(years * 365)
        loc = chunk["location"].values

        out_vars = {}
        for name, da in chunk.data_vars.items():
            # [months, L] → [Y, 12, L] → index to [Y, 365, L] → flatten days
            arr = da.transpose("time", "location", ...).values
            arr_reshaped = arr.reshape(years, 12, locations)
            arr_indexed = arr_reshaped[:, day_to_month, :]
            arr_daily = arr_indexed.reshape(years * 365, locations)
            out_vars[name] = xr.DataArray(arr_daily, dims=("time", "location"),
                                          coords={"time": days, "location": loc})

        return xr.Dataset(out_vars, coords={"time": days, "location": loc})

    def _expand_annual_to_daily(self, chunk: xr.Dataset) -> xr.Dataset:
        """Repeat annual values to daily (each year repeated for 365 days)."""
        years = int(chunk.sizes["time"])
        locations = int(chunk.sizes["location"])

        days = np.arange(years * 365)
        loc = chunk["location"].values

        out_vars = {}
        for name, da in chunk.data_vars.items():
            arr = da.transpose("time", "location", ...).values  # [Y, L]
            arr_rep = np.repeat(arr[:, None, :], 365, axis=1)   # [Y, 365, L]
            arr_daily = arr_rep.reshape(years * 365, locations) # [365*Y, L]
            out_vars[name] = xr.DataArray(arr_daily, dims=("time", "location"),
                                          coords={"time": days, "location": loc})
        return xr.Dataset(out_vars, coords={"time": days, "location": loc})

    # -----------------------------------------------------------------------
    # Input / output tensor builders
    # -----------------------------------------------------------------------

    def _create_input_tensor(
        self,
        chunk_daily: xr.Dataset,
        chunk_monthly: xr.Dataset,
        chunk_annual: xr.Dataset,
    ) -> torch.Tensor:
        """
        Build input tensor with:
          - daily forcings (t),
          - monthly forcings (t) + monthly states shifted by 1 month (t-1),
          - annual  forcings (t) + annual  states shifted by 1 year  (t-1),
        expanded to daily and concatenated along channel,
        then drop the first year (remove shift cold-start).
        """
        # Variables (orders are sorted in _filter_var_names)
        daily_forcing_vars   = sorted(self.var_names["daily_forcing"])
        monthly_forcing_vars = sorted(self.var_names["monthly_forcing"])
        monthly_state_vars   = sorted(self.var_names["monthly_states"])
        annual_forcing_vars  = sorted(self.var_names["annual_forcing"])
        annual_state_vars    = sorted(self.var_names["annual_states"])

        # Slice datasets into forcings/states
        inputs_daily        = chunk_daily[daily_forcing_vars]
        inputs_monthly_forc = chunk_monthly[monthly_forcing_vars]
        inputs_annual_forc  = chunk_annual[annual_forcing_vars]
        monthly_states_ds   = chunk_monthly[monthly_state_vars]
        annual_states_ds    = chunk_annual[annual_state_vars]

        # Shift states by 1 step to provide t-1 context
        monthly_states_shifted = monthly_states_ds.shift(time=1, fill_value=0)
        annual_states_shifted  = annual_states_ds.shift(time=1, fill_value=0)

        # Merge forcings (t) + shifted states (t-1)
        monthly_merged = xr.merge([inputs_monthly_forc, monthly_states_shifted])
        annual_merged  = xr.merge([inputs_annual_forc,  annual_states_shifted])

        # Standardize
        daily_std   = self._standardise_dataset(inputs_daily,        daily_forcing_vars)
        monthly_std = self._standardise_dataset(monthly_merged,      monthly_forcing_vars + monthly_state_vars)
        annual_std  = self._standardise_dataset(annual_merged,       annual_forcing_vars  + annual_state_vars)

        # Expand to daily
        monthly_daily = self._expand_monthly_to_daily(monthly_std)
        annual_daily  = self._expand_annual_to_daily(annual_std)

        # To [C, T, L] in explicit list order
        arr_daily   = np.stack([daily_std[v]    .transpose("time", "location", ...).values
                                for v in daily_forcing_vars])
        arr_monthly = np.stack([monthly_daily[v].transpose("time", "location", ...).values
                                for v in (monthly_forcing_vars + monthly_state_vars)])
        arr_annual  = np.stack([annual_daily[v] .transpose("time", "location", ...).values
                                for v in (annual_forcing_vars  + annual_state_vars)])

        # Concatenate channels
        input_tensor = np.concatenate([arr_daily, arr_monthly, arr_annual], axis=0)

        # Drop first year to remove cold-start from shifts
        input_tensor = input_tensor[:, 365:, :]

        return torch.from_numpy(input_tensor.astype(np.float32, copy=False))

    def _create_output_tensor(
        self,
        chunk_monthly: xr.Dataset,
        chunk_annual: xr.Dataset,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert monthly & annual datasets into standardized label tensors.

        No repetition to daily is needed since the loss is computed on
        monthly/annual aggregates. We drop the first year to align with inputs.
        """
        monthly_vars = sorted(self.var_names["monthly_fluxes"]) + sorted(self.var_names["monthly_states"])
        annual_vars  = sorted(self.var_names["annual_states"])

        outputs_monthly = chunk_monthly[monthly_vars]
        outputs_annual  = chunk_annual[annual_vars]

        monthly_std = self._standardise_dataset(outputs_monthly, monthly_vars)
        annual_std  = self._standardise_dataset(outputs_annual,  annual_vars)

        monthly_arr = np.stack([monthly_std[v].transpose("time", "location", ...).values
                                for v in monthly_vars])
        annual_arr  = np.stack([annual_std[v] .transpose("time", "location", ...).values
                                for v in annual_vars])

        # Drop first year to align with inputs
        monthly_arr = monthly_arr[:, 12:, :]
        annual_arr  = annual_arr[:,  1:, :]

        monthly_tensor = torch.from_numpy(monthly_arr.astype(np.float32, copy=False))
        annual_tensor  = torch.from_numpy(annual_arr.astype(np.float32,  copy=False))
        return monthly_tensor, annual_tensor

    # -----------------------------------------------------------------------
    # __getitem__
    # -----------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (inputs, labels_monthly, labels_annual) for a given sample index.

        Maps global idx across multiple zarr stores to:
          - dataset index (which store),
          - local index   (scenario, location-chunk within the store),
        then extracts chunks and builds tensors.
        """
        if idx >= self.n_samples:
            raise IndexError("Index out of range")

        # Which dataset (zarr store) does this sample fall into?
        dataset_idx = np.searchsorted(self.idx_thresholds, idx, side="right") - 1
        # Local index within that dataset
        local_idx = idx - self.idx_thresholds[dataset_idx]

        # Datasets at this index
        ds_daily   = self.daily_datasets[dataset_idx]
        ds_monthly = self.monthly_datasets[dataset_idx]
        ds_annual  = self.annual_datasets[dataset_idx]

        # Extract chunks (same location/scenario across daily/monthly/annual)
        chunk_daily   = self._extract_chunk_from_ds(dataset_idx, local_idx, ds_daily)
        chunk_monthly = self._extract_chunk_from_ds(dataset_idx, local_idx, ds_monthly)
        chunk_annual  = self._extract_chunk_from_ds(dataset_idx, local_idx, ds_annual)

        # Build tensors
        input_tensor = self._create_input_tensor(chunk_daily, chunk_monthly, chunk_annual)
        label_tensor_monthly, label_tensor_annual = self._create_output_tensor(chunk_monthly, chunk_annual)

        return input_tensor, label_tensor_monthly, label_tensor_annual


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