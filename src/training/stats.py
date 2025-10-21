from typing import Iterable, Optional, Dict, Tuple
import random
import numpy as np
import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set

# ------------------------------ Stats & scheduler helpers -----------------------------------
def count_batches_and_windows(dl: Iterable, split_name: str = "dataset") -> Tuple[int, int]:
    """
    Infer how many *outer batches* and (location x year) *windows* we have.

    Returns:
        (num_batches, total_windows)
    """
    num_batches = len(dl)
    if num_batches == 0:
        raise ValueError(f"{split_name} is empty (no batches). Cannot proceed with training.")

    # Peek at first batch to compute windowing geometry
    first_inputs, _, first_annual = next(iter(dl))
    s_in  = first_inputs.squeeze(0)   # [nin, 365 * n_years, n_locations]
    s_ann = first_annual.squeeze(0)   # [na, 1 * n_years,    n_locations]

    n_locations = s_in.shape[2]
    n_years     = s_ann.shape[1]
    effective_years = n_years - 1     # skip first (per training/val logic)

    windows_per_batch = n_locations * effective_years
    return num_batches, num_batches * windows_per_batch

def get_split_stats(
    train_dl: Iterable,
    valid_dl: Iterable,
    test_dl: Iterable,
    accum_steps: Optional[int] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Compute batch/window counts and steps-per-epoch (considering grad accumulation).
    """
    stats: Dict[str, Dict[str, int]] = {}

    # Train
    t_batches, t_windows = count_batches_and_windows(train_dl, "train")
    eff_accum = 1 if (accum_steps is None or accum_steps <= 1) else int(accum_steps)
    steps_per_epoch = (t_windows // eff_accum) + (1 if (t_windows % eff_accum) else 0)
    stats["train"] = {
        "batches": t_batches,
        "windows": t_windows,
        "steps_per_epoch": steps_per_epoch,
        "eff_accum": eff_accum,
    }

    # Val
    v_batches, v_windows = count_batches_and_windows(valid_dl, "validation")
    stats["val"] = {"batches": v_batches, "windows": v_windows}

    # Test
    te_batches, te_windows = count_batches_and_windows(test_dl, "test")
    stats["test"] = {"batches": te_batches, "windows": te_windows}

    return stats

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    Load and filter the standardisation dictionary, pruning invalid or excluded variables.

    Args:
        standardisation_path (Path): Path to `standardisation_dict.json`.
        all_vars (list[str]): List of all variable names.
        daily_vars (list[str]): Daily variable names.
        monthly_vars (list[str]): Monthly variable names.
        annual_vars (list[str]): Annual variable names.
        monthly_states (list[str]): Monthly state variable names.
        annual_states (list[str]): Annual state variable names.
        exclude_vars (set[str], optional): Extra variables to exclude.

    Returns:
        tuple:
            - std_dict (dict): Filtered normalisation dict with valid mean/std.
            - pruned_vars (dict[str, list[str]]): Dict containing pruned variable lists.
    """

    def is_valid_stat(obj) -> bool:
        try:
            return (
                np.isfinite(float(obj["mean"]))
                and np.isfinite(float(obj["std"]))
                and float(obj["std"]) > 0
            )
        except Exception:
            return False

    # --- Load stats ---
    with open(standardisation_path, "r") as f:
        raw_stats = json.load(f)

    std_dict = {
        k: {"mean": float(v["mean"]), "std": float(v["std"])}
        for k, v in raw_stats.items()
        if is_valid_stat(v)
    }

    # --- Build exclusion set ---
    exclude = set(exclude_vars or set()) | (set(raw_stats) - set(std_dict))

    # --- Pruning helper ---
    prune = lambda vars_list: [v for v in vars_list if v not in exclude]

    # --- Apply pruning ---
    pruned_vars = {
        "all_vars": prune(all_vars),
        "daily_vars": prune(daily_vars),
        "monthly_vars": prune(monthly_vars),
        "annual_vars": prune(annual_vars),
        "monthly_states": prune(monthly_states),
        "annual_states": prune(annual_states),
    }

    # --- Remove excluded keys from std_dict ---
    std_dict = {k: v for k, v in std_dict.items() if k not in exclude}

    return std_dict, pruned_vars
    
