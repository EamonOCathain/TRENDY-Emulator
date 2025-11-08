from __future__ import annotations
from typing import Iterable, Optional, Dict, Tuple, Set, Any
from pathlib import Path
import random
import json
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Internal helper: safely unpack a batch that may include a 4th "full"/meta item
# ---------------------------------------------------------------------------

def _extract_inputs_and_annual(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (inputs, annual_labels) from a DataLoader batch.

    Supports:
      • tuple/list of length 3: (x, m, a)
      • tuple/list of length ≥4: (x, m, a, meta/flag, ...)
      • dict-like: expects keys 'inputs' and 'annual' (or fallback 'labels_a')

    Raises:
      RuntimeError if required tensors cannot be found.
    """
    # tuple/list
    if isinstance(batch, (tuple, list)):
        if len(batch) < 3:
            raise RuntimeError("Batch must have at least 3 items: (inputs, monthly, annual)")
        x = batch[0]
        a = batch[2]
        return x, a

    # dict-like
    if isinstance(batch, dict):
        # Preferred keys
        x = batch.get("inputs", None)
        a = batch.get("annual", batch.get("labels_a", None))
        if x is not None and a is not None:
            return x, a

    # Fallback: try positional unpack if it behaves like (x, m, a, ...)
    try:
        x, _m, a = batch[0], batch[1], batch[2]
        return x, a
    except Exception:
        pass

    raise RuntimeError("Unrecognised batch structure; expected (x,m,a[,...]) or dict-like with 'inputs'/'annual'.")


# ---------------------------------------------------------------------------
# 1. Batch & window statistics
# ---------------------------------------------------------------------------

def count_batches_and_windows(dl: Iterable, split_name: str = "dataset") -> Tuple[int, int]:
    """
    Count how many *outer batches* and (location × year) *windows* exist in a DataLoader.

    Assumes shape conventions:
      inputs   : [batch=1, nin, 365*Y, L]  or [nin, 365*Y, L]
      annual   : [batch=1, na,      Y, L]  or [na,      Y, L]

    Args:
        dl: DataLoader-like iterable returning (inputs, monthly_labels, annual_labels, [meta/full]).
        split_name: Used for friendly error messages.

    Returns:
        (num_batches, total_windows)
    """
    # Number of outer batches
    try:
        num_batches = len(dl)
    except Exception:
        # If the DataLoader has no __len__, fall back to single pass (rare in your setup)
        num_batches = sum(1 for _ in dl)

    if num_batches == 0:
        raise ValueError(f"{split_name} is empty (no batches). Cannot proceed with training.")

    # Peek the first batch to infer geometry (does not consume the loader globally)
    first_batch = next(iter(dl))
    first_inputs, first_annual = _extract_inputs_and_annual(first_batch)

    # Squeeze leading batch dim if present
    s_in  = first_inputs.squeeze(0)
    s_ann = first_annual.squeeze(0)

    # Expect inputs [nin, 365*Y, L] and annual [na, Y, L]
    if s_in.ndim != 3 or s_ann.ndim != 3:
        raise ValueError(
            f"{split_name}: unexpected tensor ranks — inputs.ndim={s_in.ndim}, annual.ndim={s_ann.ndim} "
            "(expected 3 after squeeze)."
        )

    nin, Ttot, n_locations = int(s_in.shape[0]), int(s_in.shape[1]), int(s_in.shape[2])
    na, n_years, n_locations_b = int(s_ann.shape[0]), int(s_ann.shape[1]), int(s_ann.shape[2])

    if n_locations_b != n_locations:
        raise ValueError(
            f"{split_name}: location dimension mismatch between inputs (L={n_locations}) "
            f"and annual labels (L={n_locations_b})."
        )

    # Robust year inference from annual head (preferred)
    Y = n_years
    # (Optionally sanity-check: Ttot should be 365 * Y for noleap calendars)
    # If needed, you could relax this but it matches your dataset convention.
    if Y <= 0:
        raise ValueError(f"{split_name}: inferred Y={Y} from annual labels; must be > 0.")

    # Supervision uses previous-year state → effective target years = Y-1
    effective_years = max(0, Y - 1)
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

    Returns:
        dict with keys 'train', 'val', 'test' and sub-keys:
          - batches
          - windows
          - steps_per_epoch (train only)
          - eff_accum (train only)
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

def load_standardisation_only(
    standardisation_path: Path,
) -> Tuple[Dict[str, Dict[str, float]], Set[str]]:
    """
    Load standardisation stats and drop invalid entries.

    Returns:
        std_dict: {var: {"mean": float, "std": float}} with finite mean, std>0
        invalid_vars: set of variable names dropped due to invalid stats
    """

    def is_valid_stat(obj) -> bool:
        try:
            mu = float(obj["mean"])
            sd = float(obj["std"])
            return np.isfinite(mu) and np.isfinite(sd) and sd > 0.0
        except Exception:
            return False

    with open(standardisation_path, "r") as f:
        raw_stats = json.load(f)

    std_dict: Dict[str, Dict[str, float]] = {}
    invalid_vars: Set[str] = set()

    for k, v in raw_stats.items():
        if is_valid_stat(v):
            std_dict[k] = {"mean": float(v["mean"]), "std": float(v["std"])}
        else:
            invalid_vars.add(k)

    return std_dict, invalid_vars