# ------------------------------ Data Loading Helpers -----------------------------------
import os
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset, random_split
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict
import sys
import torch.distributed as dist

project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.dataset.dataset import CustomDataset
from src.paths.paths import training_zarr_dir, training_zarr_rechunked_dir
from src.dataset.dataset_carry import CarryBlockDataset
from src.training.carry import parse_carry_years_flag 


def custom_collate(batch):
    """
    Custom collate function for DataLoader.
    
    Takes a batch of (inputs, labels_monthly, labels_annual) tuples and stacks them
    into batched tensors.
    
    Args:
        batch (list of tuples): Each item is (inputs, monthly_labels, annual_labels).
    
    Returns:
        tuple: (batched_inputs, batched_monthly, batched_annual), each stacked into a tensor.
    """
    inputs, labels_monthly, labels_annual = zip(*batch)
    batched_inputs = torch.stack(inputs)
    batched_monthly = torch.stack(labels_monthly)
    batched_annual = torch.stack(labels_annual)
    return batched_inputs, batched_monthly, batched_annual

def get_train_val_test(std_dict: dict, block_locs: int = 70, carry_years_flag: str = "0") -> dict:
    """
    If carry is enabled (progressive or any static carry > 0),
    use the carry-optimised dataset; otherwise use the base CustomDataset.

    Now supports multi-value carry flags (e.g. "1 2 3 6 9").
    """
    carry_mode, carry_vals = parse_carry_years_flag(carry_years_flag)

    # Normalise carry values into a list of floats
    if isinstance(carry_vals, (list, tuple)):
        vals = [float(v) for v in carry_vals]
    else:
        vals = [float(carry_vals)]

    any_positive = any(v > 0.0 for v in vals)
    use_carry_ds = (carry_mode == "progressive") or any_positive

    data_dir = training_zarr_dir
    rechunked_data_dir = data_dir  # TEMPORARY UNTIL FINDING NANS IN RECHUNKED ZARR
    if not data_dir:
        raise RuntimeError("ZARR path not set")

    # ---- DDP rank check ----
    is_main_rank = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    if use_carry_ds:
        ds_train = CarryBlockDataset(
            data_dir=training_zarr_rechunked_dir, std_dict=std_dict,
            tensor_type="train", block_locs=block_locs
        )
        ds_val = CarryBlockDataset(
            data_dir=training_zarr_rechunked_dir, std_dict=std_dict,
            tensor_type="val", block_locs=block_locs
        )
        ds_test = CarryBlockDataset(
            data_dir=training_zarr_rechunked_dir, std_dict=std_dict,
            tensor_type="test", block_locs=block_locs
        )
        if is_main_rank:
            print(f"[INFO] Dataloader in carry mode Using data directory: {training_zarr_rechunked_dir}")
    else:
        ds_train = CustomDataset(data_dir=data_dir, std_dict=std_dict, tensor_type="train")
        ds_val   = CustomDataset(data_dir=data_dir, std_dict=std_dict, tensor_type="val")
        ds_test  = CustomDataset(data_dir=data_dir, std_dict=std_dict, tensor_type="test")
        if is_main_rank:
            print(f"[INFO] Dataloader in non-carry mode using data directory: {data_dir}")

    return {"train": ds_train, "val": ds_val, "test": ds_test}

def get_data(
    train_ds,
    valid_ds,
    test_ds,
    bs: int = 1,
    collate_fn=custom_collate,
    ddp: bool = False,
    num_workers: int = 1,
):
    """
    Wrap datasets into DataLoaders for training, validation, and testing.

    Args:
        train_ds: Dataset for training.
        valid_ds: Dataset for validation.
        test_ds: Dataset for testing.
        bs (int): Batch size.
        collate_fn (callable): Function to collate samples into batches.
        ddp (bool): If True, use DistributedSampler for multi-GPU training.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    common = dict(
        batch_size=bs,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0 and not ddp),
    )
    if num_workers > 0:
        # Keep prefetch small to control memory use
        common["prefetch_factor"] = 1
    if torch.cuda.is_available():
        # Pin directly to GPU if available
        common["pin_memory_device"] = "cuda"

    if ddp:
        # Use distributed samplers to shard datasets across GPUs
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(valid_ds, shuffle=False, drop_last=False)
        test_sampler  = DistributedSampler(test_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = val_sampler = test_sampler = None

    train_loader = DataLoader(train_ds, shuffle=(train_sampler is None), sampler=train_sampler, **common)
    val_loader   = DataLoader(valid_ds, shuffle=False, sampler=val_sampler, **common)
    test_loader  = DataLoader(test_ds, shuffle=False, sampler=test_sampler, **common)

    return train_loader, val_loader, test_loader