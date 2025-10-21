from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# ---------------------------------------------------------------------------
# Single-file checkpoint (minimal payload)
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    opt: Optimizer,
    epoch_1based: int,                # store 1-based epoch index
    best_val: float,
    history: Any,
    extra_cfg: Optional[Dict[str, Any]] = None,
    scheduler: Optional[_LRScheduler] = None,
) -> None:
    """
    Write a single checkpoint file with a compact, consumer-friendly payload.

    Fields (unchanged):
      - epoch (int, 1-based)
      - model_state (state_dict)
      - opt_state (optimizer state)
      - sched_state (scheduler state or None)
      - best_val (float)
      - history.train_loss / history.val_loss (lists)
      - config (dict, as provided via extra_cfg)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "epoch": int(epoch_1based),  # 1-based in the file
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "sched_state": (scheduler.state_dict() if scheduler is not None else None),
        "best_val": float(best_val),
        # keep a minimal history here to stay compatible with any consumers
        "history": {
            "train_loss": list(getattr(history, "train_loss", [])),
            "val_loss":   list(getattr(history, "val_loss",   [])),
        },
        "config": (extra_cfg or {}),
    }

    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Rolling checkpoint pruning
# ---------------------------------------------------------------------------

def _prune_checkpoints(ckpt_dir: Path, keep: int = 3) -> None:
    """
    Keep only the most recent `keep` files matching 'epoch*.pt'.
    Note: 'best.pt' is not touched (pattern excludes it).
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(
        (p for p in ckpt_dir.glob("epoch*.pt") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in ckpts[keep:]:
        try:
            p.unlink()
        except Exception:
            # best-effort cleanup
            pass


# ---------------------------------------------------------------------------
# Trainer callback: rich checkpoint + rolling + weights dumps
# ---------------------------------------------------------------------------

def save_cb(
    epoch: int,                # 0-based loop index coming from the trainer
    best: bool,
    val: float,
    history: Any,
    *,
    args: Any,
    run_dir: Path,
    model: torch.nn.Module,
    opt: Optimizer,
    scheduler: Optional[_LRScheduler],
    input_dim: int,
    output_dim: int,
    input_order=None,
    output_order=None,
    var_names_snapshot=None,
    schema_sig: str | None = None,
    schema_dims: dict | None = None,
) -> None:
    """
    Checkpoint callback used during training.

    Saves:
      - checkpoints/best.pt when `best=True`
      - checkpoints/epoch{N}.pt every `args.ckpt_every_epochs`
      - saves/best_weights.pt (copy of weights when best)
      - saves/latest_weights.pt (always)

    Payload fields (unchanged):
      epoch (1-based), best_val, model_state, opt_state, sched_state,
      input_dim, output_dim, history (full dict if available), early_state,
      config = {"args": jsonable_args, "extra_cfg": extra_cfg}
      Optional: schema snapshot under key "schema".
    """
    from pathlib import Path as _P  # local alias for JSON conversion

    # JSON-safe CLI args (Path → str)
    args_jsonable = {k: (str(v) if isinstance(v, _P) else v) for k, v in vars(args).items()}

    # Extra config snapshot (structure preserved)
    extra_cfg = {
        "args": args_jsonable,
        "model_kwargs": {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "d": 128, "h": 1024, "g": 256, "num_layers": 4, "nhead": 8, "dropout": 0.1, "max_len": 31,
        },
        "io_orders": {
            "input":  list(input_order or []),
            "output": list(output_order or []),
        },
        "var_names_snapshot": (var_names_snapshot or {}),
    }

    ckpt_dir = run_dir / "checkpoints"
    saves_dir = run_dir / "saves"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    saves_dir.mkdir(parents=True, exist_ok=True)

    # Build payload (store 1-based epoch)
    epoch_1based = epoch + 1
    payload = {
        "epoch": epoch_1based,
        "best_val": float(val),
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "sched_state": (scheduler.state_dict() if scheduler is not None else None),
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "history": (history.to_dict() if hasattr(history, "to_dict") else {
            "train_loss":       list(getattr(history, "train_loss", [])),
            "val_loss":         list(getattr(history, "val_loss", [])),
            "batch_loss":       list(getattr(history, "batch_loss", [])),
            "batch_step":       list(getattr(history, "batch_step", [])),
            "epoch_edges":      list(getattr(history, "epoch_edges", [0])),
            "val_loss_batches": list(getattr(history, "val_loss_batches", [])),
            "val_loss_steps":   list(getattr(history, "val_loss_steps", [])),
            "lr_values":        list(getattr(history, "lr_values", [])),
            "lr_steps":         list(getattr(history, "lr_steps", [])),
            "samples_seen":     int(getattr(history, "samples_seen", 0)),
        }),
        "early_state": getattr(history, "_early_state", None),
        "config": {"args": args_jsonable, "extra_cfg": extra_cfg},
    }

    # Optional schema snapshot (attach to payload)
    if (schema_sig is not None) or (schema_dims is not None):
        _in_dim  = int((schema_dims or {}).get("input_dim",  input_dim))
        _out_dim = int((schema_dims or {}).get("output_dim", output_dim))
        payload["schema"] = {
            "signature": schema_sig,
            "input_dim": _in_dim,
            "output_dim": _out_dim,
            # helpful context
            "input_order":  list(input_order or []),
            "output_order": list(output_order or []),
            "varnames_snapshot": dict(var_names_snapshot or {}),
        }

    # Save "best" payload and a copy of weights
    if best:
        torch.save(payload, ckpt_dir / "best.pt")
        torch.save(model.state_dict(), saves_dir / "best_weights.pt")

    # Rolling checkpoints every N epochs (and prune)
    keep_last  = int(getattr(args, "keep_last", 3) or 0)
    ckpt_every = int(getattr(args, "ckpt_every_epochs", 0) or 0)
    if ckpt_every > 0 and (epoch_1based % ckpt_every == 0):
        torch.save(payload, ckpt_dir / f"epoch{epoch_1based}.pt")
        _prune_checkpoints(ckpt_dir, keep=keep_last)

    # Always dump latest weights for convenience
    torch.save(model.state_dict(), saves_dir / "latest_weights.pt")


# ---------------------------------------------------------------------------
# Optional: extract a model state_dict from various checkpoint shapes
# ---------------------------------------------------------------------------

def extract_state_dict_for_foundation(ckpt: dict | object) -> dict:
    """
    Accepts either:
      - a raw state_dict (mapping param_name → Tensor), or
      - a training checkpoint containing one of:
          {"model_state" | "state_dict" | "model_state_dict" | "model"} → dict

    Returns a cleaned state_dict with common prefixes ('module.' / 'model.') stripped.
    """
    # Accept a raw state_dict (all values are Tensors)
    if isinstance(ckpt, dict) and ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd = ckpt
    # Or a training checkpoint containing a nested model state
    elif isinstance(ckpt, dict):
        for key in ("model_state", "state_dict", "model_state_dict", "model"):
            blob = ckpt.get(key)
            if isinstance(blob, dict):
                sd = blob
                break
        else:
            raise RuntimeError("Checkpoint has no model state dict (looked for model_state/state_dict/...)")
    else:
        raise RuntimeError("Unrecognized checkpoint format.")

    # Strip common DDP / wrapper prefixes
    def strip_prefix(d: dict, prefix: str) -> dict:
        if not any(k.startswith(prefix) for k in d.keys()):
            return d
        return {k[len(prefix):]: v for k, v in d.items()}

    sd = strip_prefix(sd, "module.")
    sd = strip_prefix(sd, "model.")
    return sd