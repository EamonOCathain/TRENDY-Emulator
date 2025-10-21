import logging
from typing import Dict, Optional, Tuple
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# ---------------------------------------------------------------------------
# Utility: resolve scheduler warm-restart period (T_0)
# ---------------------------------------------------------------------------
def _resolve_sched_t0(sched_t0_arg: str, steps_per_epoch: int) -> int:
    """
    Interpret `sched_t0_arg` to determine the warm-restart period (T_0).

    Accepts:
      - "epoch" → use one full epoch (T_0 = steps_per_epoch)
      - string or numeric values like "5" or "5.0" → use as integer
      - anything else → fallback to steps_per_epoch

    Args:
        sched_t0_arg: User-supplied argument for T_0, may be "epoch" or number-like.
        steps_per_epoch: Number of optimizer steps per training epoch.

    Returns:
        int: Warm-restart period in steps.
    """
    if isinstance(sched_t0_arg, str):
        s = sched_t0_arg.strip().lower()

        # "epoch" keyword → one cycle per epoch
        if s == "epoch":
            return max(1, int(steps_per_epoch))

        # purely numeric strings → interpret as integer
        if s.isdigit():
            return max(1, int(s))

        # tolerate float or accidental extra symbols (e.g. "+5", "5.0")
        try:
            t0 = int(float(s))
            return max(1, t0)
        except Exception:
            pass

    # Final fallback if argument type/format invalid
    return max(1, int(steps_per_epoch))


# ---------------------------------------------------------------------------
# CosineAnnealingWarmRestarts factory
# ---------------------------------------------------------------------------
def build_cosine_wr_scheduler(
    args,
    opt: torch.optim.Optimizer,
    train_stats: Dict[str, int],
    log: Optional[logging.Logger] = None,
) -> Tuple[Optional[CosineAnnealingWarmRestarts], Optional[Dict[str, int]]]:
    """
    Construct a CosineAnnealingWarmRestarts scheduler, or return None if disabled.

    Expected CLI args:
      - args.scheduler     : must be "cosine_wr" to activate
      - args.sched_t0      : "epoch" or integer/float string
      - args.sched_tmult   : multiplicative cycle length factor (>=1)
      - args.eta_min       : minimum LR at cycle trough

    Args:
        args: Namespace or object with scheduler arguments.
        opt: Optimizer to attach the scheduler to.
        train_stats: Dict containing 'steps_per_epoch' and optionally 'eff_accum'.
        log: Optional logger to record scheduler configuration.

    Returns:
        (scheduler, info_dict):
            - scheduler: instance of CosineAnnealingWarmRestarts or None
            - info_dict: dictionary with scheduler parameters for record-keeping
    """
    if args.scheduler != "cosine_wr":
        return None, None

    # Resolve cycle length (T_0)
    steps_per_epoch = int(train_stats["steps_per_epoch"])
    T0 = _resolve_sched_t0(args.sched_t0, steps_per_epoch)

    # Create scheduler
    scheduler = CosineAnnealingWarmRestarts(
        opt,
        T_0=T0,
        T_mult=max(1, int(args.sched_tmult)),
        eta_min=float(args.eta_min),
    )

    # Log summary
    if log:
        mode = "epoch-aligned" if str(args.sched_t0).strip().lower() == "epoch" else "manual"
        log.info(
            "Cosine WR %s: T_0=%d, steps_per_epoch=%d, T_mult=%d",
            mode, T0, steps_per_epoch, int(args.sched_tmult)
        )

    # Return metadata for reproducibility / debugging
    info = {
        "mode": ("epoch" if str(args.sched_t0).strip().lower() == "epoch" else "manual"),
        "T0": T0,
        "steps_per_epoch": steps_per_epoch,
        "t_mult": int(args.sched_tmult),
        "eff_accum": train_stats.get("eff_accum"),
    }
    return scheduler, info