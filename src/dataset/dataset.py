import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import torch
from torch.utils.data import Dataset, Subset, random_split
from src.dataset.variables import var_names

# ------------------------------ Dataset Class -----------------------------------
class CustomDataset(Dataset):
    """Dataset class for loading and processing climate data from Zarr files"""
    def __init__(self, data_dir: str, std_dict: Dict, tensor_type: str, chunk_size=70):
        self.std_dict = std_dict
        self.tensor_type = tensor_type
        self.chunk_size = chunk_size
        self.n_scenarios = 4
        self.base_path = Path(data_dir) / tensor_type
        self.unfiltered_var_names = var_names
        
        # Get Paths
        self._get_paths()

        # Load datasets
        self._open_datasets()
        
        # Store names all input vars
        self._filter_var_names()
        
        # Calculate sample distribution
        self._plan_samples()
        
    def _get_paths(self) -> None:        
        # Train Paths
        if self.tensor_type == "train":
            self.daily_paths = [self.base_path/ "train_location_train_period/daily.zarr"]
            self.monthly_paths = [self.base_path / "train_location_train_period/monthly.zarr"]
            self.annual_paths = [self.base_path / "train_location_train_period/annual.zarr"]
        
        # Validation Paths
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
        
        # Test Paths
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
    
    def _open_datasets(self) -> None:
        """Open all Zarr datasets and store in lists"""
        opts = dict(consolidated=True, decode_times=False, chunks={})
        self.daily_datasets   = [xr.open_zarr(p, **opts) for p in self.daily_paths]
        self.monthly_datasets = [xr.open_zarr(p, **opts) for p in self.monthly_paths]
        self.annual_datasets  = [xr.open_zarr(p, **opts) for p in self.annual_paths]
        self.all_datasets = self.daily_datasets + self.monthly_datasets + self.annual_datasets
         
    def _filter_var_names(self) -> None:
        """
        Filter out variables that have no stats, non-positive std, or are missing from the Zarr datasets.
        Then build sorted per-section input/output orders for consistency with training.
        """
        filtered = {}
        for group, var_list in self.unfiltered_var_names.items():
            keep = []
            for v in var_list:
                stats = self.std_dict.get(v)
                if not stats: 
                    continue
                if stats.get("std", 0) <= 0:
                    continue
                if not any(v in ds.data_vars for ds in self.all_datasets):
                    raise AssertionError(f"Zarr datasets are missing variable: {v}")
                keep.append(v)
            filtered[group] = keep
        
        # Build sorted lists for reproducible I/O order
        self.var_names = filtered
        self.input_order = (
            sorted(self.var_names['daily_forcing']) +
            sorted(self.var_names['monthly_forcing']) +
            sorted(self.var_names['monthly_states']) +
            sorted(self.var_names['annual_forcing']) +
            sorted(self.var_names['annual_states'])
        )
        self.output_order = (
            sorted(self.var_names['monthly_fluxes']) +
            sorted(self.var_names['monthly_states']) +
            sorted(self.var_names['annual_states'])
        )

        # Optional: store index maps for debugging
        self.input_index  = {v:i for i,v in enumerate(self.input_order)}
        self.output_index = {v:i for i,v in enumerate(self.output_order)}

        # Guardrails
        if len(self.input_order) == 0:
            raise RuntimeError("Empty input_order after filtering.")
        if len(self.output_order) == 0:
            raise RuntimeError("Empty output_order after filtering.")
    
    def _plan_samples(self) -> None:
        """
        Compute per-dataset sample counts, cumulative thresholds, total samples,
        and number of location-chunks per dataset.
        Sets:
        - sample_counts: List[int]
        - idx_thresholds: np.ndarray  (len = len(datasets)+1)
        - n_samples: int
        - n_location_chunks_list: List[int]
        """
        # sanity: matching locations across resolutions per dataset index
        for i in range(len(self.daily_datasets)):
            Ld = int(self.daily_datasets[i].sizes["location"])
            Lm = int(self.monthly_datasets[i].sizes["location"])
            La = int(self.annual_datasets[i].sizes["location"])
            if not (Ld == Lm == La):
                raise ValueError(f"location size mismatch at ds[{i}]: daily={Ld}, monthly={Lm}, annual={La}")
            
        # number of location-chunks per dataset (floor division)
        self.n_location_chunks_list = [
            int(ds.sizes["location"]) // self.chunk_size for ds in self.daily_datasets
        ]

        # samples per dataset = (#loc-chunks) * (#scenarios)
        self.sample_counts = [
            n_chunks * int(self.n_scenarios) for n_chunks in self.n_location_chunks_list
        ]

        # cumulative thresholds to map global idx -> dataset idx
        self.idx_thresholds = np.cumsum([0] + self.sample_counts)

        # total samples across all datasets
        self.n_samples = int(sum(self.sample_counts))
        
    def __len__(self) -> int:
        "Just return the number of samples in the dataset"
        return self.n_samples
    
    def _standardise(self, arr: np.ndarray, var_name: str, std_dict: Dict[str, Dict[str, float]]) -> np.ndarray:
        """Apply standardization to data using precomputed mean and std"""
        stats = std_dict.get(var_name)
        if not stats:
            raise ValueError(f"No stats for '{var_name}'")
        std = stats["std"]
        if std <= 0:
            raise ValueError(f"Non-positive std for '{var_name}'")
        return (arr - stats["mean"]) / std
    
    def _standardise_dataset(self, ds: xr.Dataset, var_list: List[str]) -> xr.Dataset:
        """
        Return a new xr.Dataset where each variable in var_list (or all ds vars) is standardised
        using self.std_dict. Keeps (time, location) dims and original coords.

        Iterates in the order of var_list so stacking later preserves INPUT_ORDER / OUTPUT_ORDER.
        """
        if len(var_list) == 0:
            raise RuntimeError("Variable list empty in dataloader standardisation")
        out = {}
        for var in var_list:  # preserves caller’s order
            arr = ds[var].transpose("time", "location", ...).values
            arr_std = self._standardise(arr, var, self.std_dict)
            out[var] = xr.DataArray(
                arr_std, dims=("time", "location"),
                coords={"time": ds[var]["time"].values, "location": ds[var]["location"].values},
            )
        return xr.Dataset(out, coords={"time": ds["time"].values, "location": ds["location"].values})
        
    def _extract_chunk_from_ds(
            self,
            dataset_idx: int,
            local_idx: int,
            ds: xr.Dataset,
            ) -> xr.Dataset:
        """Extract a single chunk from the dataset by figuring out the exact indexes and slicing"""
        
        # Get number of location chunks in the dataset index
        n_location_chunks = self.n_location_chunks_list[dataset_idx]

        # Define the indexes of the chunk
        # One chunk = 70 locations * 1 scenario * time
        scenario = local_idx // n_location_chunks # Local_idx has length  n_chunks * n_scenarios -> so by integer dividision by the n_chunks gives the scenario you're in  
        chunk = local_idx % n_location_chunks # Division by n_scenarios gives the chunk you're in
        start_loc = chunk * self.chunk_size
        end_loc = start_loc + self.chunk_size
        
        chunk = ds.isel(location=slice(start_loc, end_loc), scenario=scenario)
        
        return chunk
    
    def _expand_monthly_to_daily(self, chunk: xr.Dataset) -> xr.Dataset:
        """Expands a monthly tensor to daily values by repeating each month by the appropriate amount of days."""
        # Create masks
        month_lengths = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=np.int32)
        day_to_month = np.repeat(np.arange(12, dtype=np.int64), month_lengths) 
        
        # Store number of months and years
        months = int(chunk.sizes["time"])
        
        if months % 12 != 0:
            raise ValueError(f"Monthly time length {months} is not divisible by 12.")
        
        years = months // 12 
        
        # n_locations
        locations = int(chunk.sizes["location"])
        
        # Build the coordinates of the output
        days = np.arange(years * 365)
        loc = chunk['location'].values
        
        out_vars = {}
        for name, da in chunk.data_vars.items():
            array = da.transpose("time", "location", ...).values # [months, location]
            array_reshaped = array.reshape(years, 12, locations) # [year, 12 months, locations]
            array_indexed_monthly = array_reshaped[:, day_to_month, :] #[years, 365, locations]
            array_daily = array_indexed_monthly.reshape(years * 365, locations) # [days, locations]
            # store the result for each variable in a new array
            out_vars[name] = xr.DataArray(array_daily, dims=("time", "location"),
                                      coords={"time": days, "location": loc})
        
        # return the entire array
        return xr.Dataset(out_vars, coords={"time": days, "location": loc})
        
        
    def _expand_annual_to_daily(self, chunk: xr.Dataset) -> xr.Dataset:
        """Expands an annual tensor to daily values by repeating each year by the appropriate amount of days."""
        years = int(chunk.sizes["time"])
        locations = int(chunk.sizes["location"])

        days = np.arange(years * 365)
        loc = chunk["location"].values

        out_vars = {}
        for name, da in chunk.data_vars.items():
            array = da.transpose("time", "location", ...).values   # shape [Ya, L]
            # repeat each year across 365 days -> [Ya, 365, L]
            array_repeated = np.repeat(array[:, None, :], 365, axis=1)
            # flatten years -> [365*Ya, L]
            array_daily = array_repeated.reshape(years * 365, locations)
            out_vars[name] = xr.DataArray(array_daily, dims=("time", "location"),
                                        coords={"time": days, "location": loc})

        return xr.Dataset(out_vars, coords={"time": days, "location": loc})
        
    def _create_input_tensor(
        self, 
        chunk_daily: xr.Dataset, 
        chunk_monthly: xr.Dataset, 
        chunk_annual: xr.Dataset,
    ) -> torch.Tensor:
        """
        Build a single input tensor with:
        - daily forcings (unshifted, kept daily),
        - monthly forcings (unshifted) + monthly states shifted by 1 month,
        - annual  forcings (unshifted) + annual  states shifted by 1 year,
        then expand monthly/annual blocks to daily, concatenate along channel,
        and drop the first year to remove the shift cold-start.
        """

        # --- pick variable lists (sorted inside _filter_var_names) ---
        daily_forcing_vars   = sorted(self.var_names['daily_forcing'])
        monthly_forcing_vars = sorted(self.var_names['monthly_forcing'])
        monthly_state_vars   = sorted(self.var_names['monthly_states'])
        annual_forcing_vars  = sorted(self.var_names['annual_forcing'])
        annual_state_vars    = sorted(self.var_names['annual_states'])

        # --- slice datasets into forcings vs states (monthly/annual) ---
        inputs_daily        = chunk_daily[daily_forcing_vars]
        inputs_monthly_forc = chunk_monthly[monthly_forcing_vars]     # UNshifted
        inputs_annual_forc  = chunk_annual[annual_forcing_vars]       # UNshifted
        monthly_states_ds   = chunk_monthly[monthly_state_vars]       # ONLY states
        annual_states_ds    = chunk_annual[annual_state_vars]         # ONLY states

        # --- shift states back one step (introduce t-1 context) ---
        monthly_states_shifted = monthly_states_ds.shift(time=1, fill_value=0)
        annual_states_shifted  = annual_states_ds.shift(time=1, fill_value=0)

        # --- merge forcings (t) + shifted states (t-1) ---
        monthly_merged = xr.merge([inputs_monthly_forc, monthly_states_shifted])
        annual_merged  = xr.merge([inputs_annual_forc,  annual_states_shifted])

        # --- standardize (per var using std_dict, preserves order) ---
        daily_std   = self._standardise_dataset(inputs_daily,        daily_forcing_vars)
        monthly_std = self._standardise_dataset(monthly_merged,      monthly_forcing_vars + monthly_state_vars)
        annual_std  = self._standardise_dataset(annual_merged,       annual_forcing_vars  + annual_state_vars)

        # --- expand monthly/annual to daily (repeat to 365) ---
        monthly_daily = self._expand_monthly_to_daily(monthly_std)
        annual_daily  = self._expand_annual_to_daily(annual_std)

        # --- to [C, T, L] arrays (explicitly by list order) ---
        arr_daily   = np.stack([daily_std[v]    .transpose("time", "location", ...).values 
                                for v in daily_forcing_vars])
        arr_monthly = np.stack([monthly_daily[v].transpose("time", "location", ...).values 
                                for v in (monthly_forcing_vars + monthly_state_vars)])
        arr_annual  = np.stack([annual_daily[v] .transpose("time", "location", ...).values 
                                for v in (annual_forcing_vars  + annual_state_vars)])

        # --- concat channels [Cin, T, L] ---
        input_tensor = np.concatenate([arr_daily, arr_monthly, arr_annual], axis=0)

        # --- drop first year to remove cold-start introduced by shift(fill_value=0) ---
        input_tensor = input_tensor[:, 365:, :]

        # --- to torch tensor ---
        return torch.from_numpy(input_tensor.astype(np.float32, copy=False))
    
    def _create_output_tensor(
        self, 
        chunk_monthly: xr.Dataset, 
        chunk_annual: xr.Dataset,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the monthly and annual datasets into output tensors.
        - No need to repeat to daily as loss is on monthly/annual.
        - No need to shift states. 
        - Returns float32 torch.Tensors in the OUTPUT_ORDER section order.
        """

        # --- Build lists of vars (sorted per section) ---
        monthly_vars = sorted(self.var_names['monthly_fluxes']) + sorted(self.var_names['monthly_states'])
        annual_vars  = sorted(self.var_names['annual_states'])

        # --- separate by variable type ---
        outputs_monthly = chunk_monthly[monthly_vars]  # xr.Dataset
        outputs_annual  = chunk_annual[annual_vars]    # xr.Dataset

        # --- standardise (per var using std_dict, preserves order) ---
        monthly_std = self._standardise_dataset(outputs_monthly, monthly_vars)
        annual_std  = self._standardise_dataset(outputs_annual,  annual_vars)

        # --- convert and stack to NumPy [C, T, L] ---
        monthly_arr = np.stack([monthly_std[v].transpose("time", "location", ...).values 
                                for v in monthly_vars])
        annual_arr  = np.stack([annual_std[v] .transpose("time", "location", ...).values 
                                for v in annual_vars])

        # --- drop the first year to align with inputs (shift cold-start removed) ---
        monthly_arr = monthly_arr[:, 12:, :]
        annual_arr  = annual_arr[:,  1:, :]

        # --- to torch tensors ---
        monthly_tensor = torch.from_numpy(monthly_arr.astype(np.float32, copy=False))
        annual_tensor  = torch.from_numpy(annual_arr.astype(np.float32,  copy=False))

        return monthly_tensor, annual_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Main get item function"""
        if idx >= self.n_samples:
            raise IndexError("Index out of range")
        # Define which zarr we're in based on the thresholds (there can be multiple zarrs for a val / eval ds)
        dataset_idx = np.searchsorted(self.idx_thresholds, idx, side='right') - 1
        # Define index within the dataset
        local_idx = idx - self.idx_thresholds[dataset_idx]

        # Get datasets
        ds_daily = self.daily_datasets[dataset_idx]
        ds_monthly = self.monthly_datasets[dataset_idx]
        ds_annual = self.annual_datasets[dataset_idx]

        # Extract data chunks
        chunk_daily = self._extract_chunk_from_ds(dataset_idx, local_idx, ds_daily)
        chunk_monthly = self._extract_chunk_from_ds(dataset_idx, local_idx, ds_monthly)
        chunk_annual = self._extract_chunk_from_ds(dataset_idx, local_idx, ds_annual)
        
        input_tensor = self._create_input_tensor(chunk_daily, chunk_monthly, chunk_annual)
        label_tensor_monthly, label_tensor_annual = self._create_output_tensor(chunk_monthly, chunk_annual)

        return input_tensor, label_tensor_monthly, label_tensor_annual
    
def get_subset(dataset, frac: float = 0.01, seed: int = 42):
    """
    Take a random subset of a dataset.
    
    Args:
        dataset: Dataset to subset.
        frac (float): Fraction of dataset to keep.
        seed (int): RNG seed for reproducibility.
    
    Returns:
        torch.utils.data.Subset: Subset of the dataset.
    """
    n_total = len(dataset)
    n_subset = max(1, int(n_total * frac))
    subset, _ = random_split(
        dataset,
        [n_subset, n_total - n_subset],
        generator=torch.Generator().manual_seed(seed)
    )
    return subset


def base(ds):
    """
    Return the underlying dataset if input is a Subset, otherwise return as-is.
    
    Args:
        ds: Dataset or Subset.
    
    Returns:
        Dataset: The underlying dataset.
    """
    return ds.dataset if isinstance(ds, Subset) else ds