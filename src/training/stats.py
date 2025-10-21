from __future__ import annotations
from typing import Iterable, Optional, Dict, Tuple, List, Set
from pathlib import Path
import random
import json
import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. Batch & window statistics
# ---------------------------------------------------------------------------

def count_batches_and_windows(dl: Iterable, split_name: str = "dataset") -> Tuple[int, int]:
    """
    Count how many *outer batches* and (location × year) *windows* exist in a DataLoader.

    Used for reporting dataset geometry and computing steps per epoch.

    Args:
        dl: DataLoader-like iterable returning (inputs, monthly_labels, annual_labels).
        split_name: Human-readable name for error reporting (e.g., 'train').

    Returns:
        tuple[int, int]: (num_batches, total_windows)
    """
    num_batches = len(dl)
    if num_batches == 0:
        raise ValueError(f"{split_name} is empty (no batches). Cannot proceed with training.")

    # Peek at first batch (no persistent consumption) to infer geometry
    first_inputs, _, first_annual = next(iter(dl))

    # Shapes: inputs [nin, 365 * n_years, n_locations]; annual [na, n_years, n_locations]
    s_in  = first_inputs.squeeze(0)
    s_ann = first_annual.squeeze(0)

    n_locations = s_in.shape[2]
    n_years     = s_ann.shape[1]

    # One less year used for supervision (since we use previous year’s state)
    effective_years = n_years - 1
    windows_per_batch = n_locations * effective_years

    return num_batches, num_batches * windows_per_batch


def get_split_stats(
    train_dl: Iterable,
    valid_dl: Iterable,
    test_dl: Iterable,
    accum_steps: Optional[int] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Compute per-split batch/window counts and steps-per-epoch (considering grad accumulation).

    Args:
        train_dl, valid_dl, test_dl: Dataloaders for each split.
        accum_steps: Gradient accumulation steps (optional).

    Returns:
        dict[str, dict[str, int]]: Summary stats for {train, val, test}.
    """
    stats: Dict[str, Dict[str, int]] = {}

    # --- Train split ---
    t_batches, t_windows = count_batches_and_windows(train_dl, "train")
    eff_accum = 1 if (accum_steps is None or accum_steps <= 1) else int(accum_steps)
    steps_per_epoch = (t_windows // eff_accum) + (1 if (t_windows % eff_accum) else 0)
    stats["train"] = {
        "batches": t_batches,
        "windows": t_windows,
        "steps_per_epoch": steps_per_epoch,
        "eff_accum": eff_accum,
    }

    # --- Validation split ---
    v_batches, v_windows = count_batches_and_windows(valid_dl, "validation")
    stats["val"] = {"batches": v_batches, "windows": v_windows}

    # --- Test split ---
    te_batches, te_windows = count_batches_and_windows(test_dl, "test")
    stats["test"] = {"batches": te_batches, "windows": te_windows}

    return stats


# ---------------------------------------------------------------------------
# 2. Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set all relevant random seeds for reproducibility across Python, NumPy, and Torch.
    Also enforces deterministic CuDNN kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 3. Standardisation dictionary loader & filter
# ---------------------------------------------------------------------------

def load_and_filter_standardisation(
    standardisation_path: Path,
    all_vars: List[str],
    daily_vars: List[str],
    monthly_vars: List[str],
    annual_vars: List[str],
    monthly_states: List[str],
    annual_states: List[str],
    exclude_vars: Set[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[str]]]:
    """
    Load and filter the standardisation dictionary from JSON, pruning invalid or excluded variables.

    The returned dicts ensure all entries have finite, positive std values.

    Args:
        standardisation_path: Path to `standardisation_dict.json`.
        all_vars: Complete variable list.
        daily_vars, monthly_vars, annual_vars: Time-resolution variable lists.
        monthly_states, annual_states: State variable lists.
        exclude_vars: Extra variables to exclude manually.

    Returns:
        tuple:
            - std_dict: Filtered normalisation dictionary {var: {"mean": µ, "std": σ}}
            - pruned_vars: Dict with pruned variable lists keyed by category.
    """

    # --- Helper: validate that a stats record is usable ---
    def is_valid_stat(obj: Dict[str, float]) -> bool:
        try:
            return (
                np.isfinite(float(obj["mean"])) and
                np.isfinite(float(obj["std"])) and
                float(obj["std"]) > 0
            )
        except Exception:
            return False

    # --- Load raw stats JSON ---
    with open(standardisation_path, "r") as f:
        raw_stats = json.load(f)

    # Keep only finite & positive entries
    std_dict = {
        k: {"mean": float(v["mean"]), "std": float(v["std"])}
        for k, v in raw_stats.items()
        if is_valid_stat(v)
    }

    # --- Build exclusion set ---
    exclude = set(exclude_vars or set()) | (set(raw_stats) - set(std_dict))

    # --- Helper: prune invalid vars from a list ---
    def prune(vars_list: List[str]) -> List[str]:
        return [v for v in vars_list if v not in exclude]

    # --- Apply pruning to all variable groups ---
    pruned_vars = {
        "all_vars": prune(all_vars),
        "daily_vars": prune(daily_vars),
        "monthly_vars": prune(monthly_vars),
        "annual_vars": prune(annual_vars),
        "monthly_states": prune(monthly_states),
        "annual_states": prune(annual_states),
    }

    # --- Remove excluded keys from stats ---
    std_dict = {k: v for k, v in std_dict.items() if k not in exclude}

    return std_dict, pruned_vars