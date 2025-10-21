# ------------------------------ Data Loading Helpers -----------------------------------
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# Project root (so local imports resolve when running scripts directly)
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

# Project imports
from src.dataset.dataset import CustomDataset
from src.dataset.dataset_carry import CarryBlockDataset
from src.paths.paths import training_zarr_dir, training_zarr_rechunked_dir

def custom_collate(batch):
    """
    Collate function for (inputs, labels_monthly, labels_annual) samples.

    Stacks a list of tuples into three batched tensors:
      inputs          → [B, ...]
      labels_monthly  → [B, ...]
      labels_annual   → [B, ...]
    """
    inputs, labels_monthly, labels_annual = zip(*batch)
    batched_inputs = torch.stack(inputs)
    batched_monthly = torch.stack(labels_monthly)
    batched_annual = torch.stack(labels_annual)
    return batched_inputs, batched_monthly, batched_annual


def get_train_val_test(
    std_dict: dict,
    block_locs: int = 70,
    carry_years: float = 0,
) -> dict:
    """
    Return train/val/test datasets depending on *single* carry setting.

    If carry_years > 0, use the carry-optimized dataset; otherwise use the base CustomDataset.

    Args:
        std_dict:     Normalization/standardization dictionary used by datasets.
        block_locs:   Number of locations per block (only used by CarryBlockDataset).
        carry_years:  Single value (e.g., 0, 1, 2.5 or "0", "1"). No progressive/multi runs.

    Returns:
        dict with keys {"train", "val", "test"} mapping to Dataset objects.
    """
    # Robustly parse to float (supports int/float/str inputs)
    try:
        carry_val = float(str(carry_years).strip())
    except Exception:
        carry_val = 0.0

    use_carry_ds = (carry_val > 0.0)

    # Base locations for data
    data_dir = training_zarr_dir
    rechunked_dir = training_zarr_rechunked_dir  # used for carry mode
    if not data_dir:
        raise RuntimeError("ZARR path not set")

    # Print only from main rank (avoid duplicated messages under DDP)
    is_main_rank = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    if use_carry_ds:
        ds_train = CarryBlockDataset(
            data_dir=rechunked_dir, std_dict=std_dict, tensor_type="train", block_locs=block_locs
        )
        ds_val = CarryBlockDataset(
            data_dir=rechunked_dir, std_dict=std_dict, tensor_type="val", block_locs=block_locs
        )
        ds_test = CarryBlockDataset(
            data_dir=rechunked_dir, std_dict=std_dict, tensor_type="test", block_locs=block_locs
        )
        if is_main_rank:
            print(f"[INFO] Dataloader in carry mode (carry_years={carry_val}) using: {rechunked_dir}")
    else:
        ds_train = CustomDataset(data_dir=data_dir, std_dict=std_dict, tensor_type="train")
        ds_val   = CustomDataset(data_dir=data_dir, std_dict=std_dict, tensor_type="val")
        ds_test  = CustomDataset(data_dir=data_dir, std_dict=std_dict, tensor_type="test")
        if is_main_rank:
            print(f"[INFO] Dataloader in non-carry mode (carry_years={carry_val}) using: {data_dir}")

    return {"train": ds_train, "val": ds_val, "test": ds_test}

def get_data(
    train_ds,
    valid_ds,
    test_ds,
    bs: int = 1,
    collate_fn: Callable = custom_collate,
    ddp: bool = False,
    num_workers: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap datasets into DataLoaders for training, validation, and testing.

    Args:
        train_ds:     Training dataset.
        valid_ds:     Validation dataset.
        test_ds:      Test dataset.
        bs:           Batch size.
        collate_fn:   Collate function to pack samples into batches.
        ddp:          If True, use DistributedSampler for multi-GPU training.
        num_workers:  Number of DataLoader workers.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Common DataLoader kwargs
    common = dict(
        batch_size=bs,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0 and not ddp),
    )

    # Keep prefetch small to control host memory use (when workers > 0).
    if num_workers > 0:
        common["prefetch_factor"] = 1

    # If CUDA is available, pin pages directly to the GPU
    if torch.cuda.is_available():
        common["pin_memory_device"] = "cuda"

    # DDP-aware sampling/shuffling
    if ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True,  drop_last=False)
        val_sampler   = DistributedSampler(valid_ds, shuffle=False, drop_last=False)
        test_sampler  = DistributedSampler(test_ds,  shuffle=False, drop_last=False)
    else:
        train_sampler = val_sampler = test_sampler = None

    # For non-DDP, enable shuffle only for the training loader.
    train_loader = DataLoader(train_ds, shuffle=(train_sampler is None), sampler=train_sampler, **common)
    val_loader   = DataLoader(valid_ds, shuffle=False, sampler=val_sampler, **common)
    test_loader  = DataLoader(test_ds, shuffle=False, sampler=test_sampler, **common)

    return train_loader, val_loader, test_loader