import logging
from typing import Dict, Optional, Tuple
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def _resolve_sched_t0(sched_t0_arg: str, steps_per_epoch: int) -> int:
    """
    Accepts "epoch" or a string integer. Falls back to steps_per_epoch if invalid.
    """
    if isinstance(sched_t0_arg, str):
        s = sched_t0_arg.strip().lower()
        if s == "epoch":
            return max(1, int(steps_per_epoch))
        # integer-like?
        if s.isdigit():
            t0 = int(s)
            return max(1, t0)
        # allow leading '+' or spaces, or accidental floats that are ints
        try:
            t0 = int(float(s))
            return max(1, t0)
        except Exception:
            pass

    # Final fallback
    return max(1, int(steps_per_epoch))

def build_cosine_wr_scheduler(
    args,
    opt: torch.optim.Optimizer,
    train_stats: Dict[str, int],
    log: Optional[logging.Logger] = None,
) -> Tuple[Optional[CosineAnnealingWarmRestarts], Optional[Dict[str, int]]]:
    """
    CosineAnnealingWarmRestarts configured with:
      - T_0: "epoch" => equals steps_per_epoch, or an integer (steps)
      - T_mult: args.sched_tmult
    """
    if args.scheduler != "cosine_wr":
        return None, None

    steps_per_epoch = int(train_stats["steps_per_epoch"])
    T0 = _resolve_sched_t0(args.sched_t0, steps_per_epoch)

    scheduler = CosineAnnealingWarmRestarts(
        opt,
        T_0=T0,
        T_mult=max(1, int(args.sched_tmult)),
        eta_min=float(args.eta_min),
    )

    if log:
        mode = "epoch-aligned" if str(args.sched_t0).strip().lower() == "epoch" else "manual"
        log.info(
            "Cosine WR %s: T_0=%d, steps_per_epoch=%d, T_mult=%d",
            mode, T0, steps_per_epoch, int(args.sched_tmult)
        )

    info = {
        "mode": ("epoch" if str(args.sched_t0).strip().lower() == "epoch" else "manual"),
        "T0": T0,
        "steps_per_epoch": steps_per_epoch,
        "t_mult": int(args.sched_tmult),
        "eff_accum": train_stats.get("eff_accum"),
    }
    return scheduler, info