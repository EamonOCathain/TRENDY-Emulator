# ------------------------------ Data Loading Helpers -----------------------------------
from __future__ import annotations

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
from src.dataset.dataset_TL import CustomDatasetTL
from src.paths.paths import training_zarr_dir


def custom_collate(batch):
    """
    Supports samples of:
      (inputs, labels_m, labels_a, period_tag:str)
    Returns:
      inputs[B,...], labels_m[B,...], labels_a[B,...], extra: dict(full=bool, tags=List[str]|None)
    """
    first = batch[0]
    if len(first) == 4:
        inputs, labels_m, labels_a, tags = zip(*batch)      # tags: tuple[str]
        period_tags = list(tags)
        # consider the batch "full" if ANY sample tag equals 'full'
        is_full = any((t == "full") or (t == "val_full") or (t == True) for t in period_tags)
        extra = {"full": bool(is_full), "tags": period_tags}
    else:
        inputs, labels_m, labels_a = zip(*batch)
        extra = {"full": False, "tags": None}

    return (
        torch.stack(inputs),
        torch.stack(labels_m),
        torch.stack(labels_a),
        extra,
    )

def get_train_val_test(
    std_dict: dict,
    block_locs: int = 70,                
    tl_activated: bool = False,
    exclude_vars: Optional[list] = None,
    tl_start: Optional[int] = None,
    tl_end: Optional[int] = None,
    replace_map: Optional[Dict[str, str]] = None,
    delta_luh: bool = False,
) -> dict:
    """
    Return train/val/test datasets. Always uses CustomDataset for both carry and non-carry.
    Transfer Learning path still uses CustomDatasetTL.
    """
    exclude_vars_set = set(exclude_vars or [])
    data_dir = training_zarr_dir
    if not data_dir:
        raise RuntimeError("ZARR path not set")

    is_main_rank = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    if tl_activated:
        ds_train = CustomDatasetTL(
            data_dir=data_dir, std_dict=std_dict, tensor_type="train",
            exclude_vars=exclude_vars_set, tl_activated=True, block_locs = block_locs,
            tl_start=tl_start, tl_end=tl_end, replace_map=replace_map, delta_luh=delta_luh,
        )
        ds_val = CustomDatasetTL(
            data_dir=data_dir, std_dict=std_dict, tensor_type="val",
            exclude_vars=exclude_vars_set, tl_activated=True, block_locs = block_locs,
            tl_start=tl_start, tl_end=tl_end, replace_map=replace_map, delta_luh=delta_luh
        )
        ds_test = CustomDatasetTL(
            data_dir=data_dir, std_dict=std_dict, tensor_type="test",
            exclude_vars=exclude_vars_set, tl_activated=True, block_locs = block_locs,
            tl_start=tl_start, tl_end=tl_end, replace_map=replace_map, delta_luh=delta_luh
        )
        if is_main_rank:
            print(f"[INFO] Dataloader in transfer learning mode using years {tl_start}-{tl_end} from: {data_dir}")
    else:
        ds_train = CustomDataset(
            data_dir=data_dir, std_dict=std_dict, tensor_type="train", block_locs = block_locs,
            exclude_vars=exclude_vars_set, delta_luh=delta_luh
        )
        ds_val = CustomDataset(
            data_dir=data_dir, std_dict=std_dict, tensor_type="val", block_locs = block_locs,
            exclude_vars=exclude_vars_set, delta_luh=delta_luh
        )
        ds_test = CustomDataset(
            data_dir=data_dir, std_dict=std_dict, tensor_type="test", block_locs = block_locs,
            exclude_vars=exclude_vars_set, delta_luh=delta_luh
        )
        if is_main_rank:
            print(f"[INFO] Dataloader using CustomDataset (carry handled by model) from: {data_dir}")

    return {"train": ds_train, "val": ds_val, "test": ds_test}


def get_data(
    train_ds,
    valid_ds,
    test_ds,
    batch_size: int = 1,
    collate_fn: Callable = custom_collate,
    ddp: bool = False,
    num_workers: int = 1,
    prefetch_factor: int = 1,
    val_prefetch_factor: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap datasets into DataLoaders for training, validation, and testing.

    Args:
        train_ds, valid_ds, test_ds: datasets
        batch_size: batch size for all loaders
        collate_fn: collate function
        ddp: use DistributedDataParallel samplers
        num_workers: number of workers per loader
        prefetch_factor: train/test prefetch factor (if workers > 0)
        val_prefetch_factor: validation prefetch factor (defaults to prefetch_factor)
    """
    if val_prefetch_factor is None:
        val_prefetch_factor = prefetch_factor

    common_base = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0 and not ddp),
    )
    if torch.cuda.is_available():
        common_base["pin_memory_device"] = "cuda"

    # Per-loader copies so val can use a different prefetch
    common_train = dict(common_base)
    common_val   = dict(common_base)
    common_test  = dict(common_base)

    if num_workers > 0:
        common_train["prefetch_factor"] = int(prefetch_factor)
        common_test["prefetch_factor"]  = int(val_prefetch_factor)
        common_val["prefetch_factor"]   = int(val_prefetch_factor)

    if ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True,  drop_last=False)
        val_sampler   = DistributedSampler(valid_ds, shuffle=False, drop_last=False)
        test_sampler  = DistributedSampler(test_ds,  shuffle=False, drop_last=False)
    else:
        train_sampler = val_sampler = test_sampler = None

    train_loader = DataLoader(train_ds, shuffle=(train_sampler is None), sampler=train_sampler, **common_train)
    val_loader   = DataLoader(valid_ds, shuffle=False, sampler=val_sampler, **common_val)
    test_loader  = DataLoader(test_ds, shuffle=False, sampler=test_sampler, **common_test)

    return train_loader, val_loader, test_loader