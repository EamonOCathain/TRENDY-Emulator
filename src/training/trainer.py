from __future__ import annotations

# ---------------------------------------------------------------------------
# Library imports
# ---------------------------------------------------------------------------
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project imports / path setup
# ---------------------------------------------------------------------------
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.training.distributed import ddp_mean_scalar
from src.training.history import History

# --- Carry rollout helpers (from carry branch) ------------------------------
from src.training.carry import rollout_outer_batch

# --- Mass-balance logging helpers -------------------------------------------
def _accum_bd_sums(acc: Dict[str, float], bd: Optional[dict], mult: float) -> None:
    """
    Accumulate per-call weighted contributions from loss_fn.last_breakdown["weighted"].
    `mult` should be the number of windows represented by this loss call.
    """
    if not bd or "weighted" not in bd:
        return
    for k, v in bd["weighted"].items():
        if v is None:
            continue
        acc[k] = acc.get(k, 0.0) + float(v) * float(mult)

def _normalize_bd_sums(acc: Dict[str, float], denom: float) -> Dict[str, float]:
    """Convert accumulated sums to averages per window."""
    if denom <= 0:
        return {k: float("nan") for k in acc.keys()}
    return {k: (v / denom) for k, v in acc.items()}

# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop when the monitored metric fails to improve by at least `min_delta`
    for `patience` epochs (after an optional warmup). Assumes 'min' mode.
    """
    def __init__(self, patience: int, min_delta: float = 0.0, warmup_epochs: int = 0):
        assert patience > 0
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup_epochs = int(warmup_epochs)
        self.best = float("inf")
        self.bad_epochs = 0
        self.should_stop = False
        self.best_epoch = -1

    def step(self, value: float, epoch_idx: int) -> None:
        if epoch_idx < self.warmup_epochs:
            return
        if value < self.best - self.min_delta:
            self.best = value
            self.best_epoch = epoch_idx
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True

    def step_on_check(self, value: float, epoch_idx: int) -> None:
        if epoch_idx < self.warmup_epochs:
            return
        if value < self.best - self.min_delta:
            self.best = value
            self.best_epoch = epoch_idx
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def unwrap(model):
    """Return the underlying model when wrapped in DDP; otherwise return as-is."""
    return model.module if isinstance(model, DDP) else model



# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def fit(
    epochs: int,
    model: torch.nn.Module,
    loss_func,
    opt,
    train_dl,
    valid_dl,
    log=None,
    save_cb=None,
    accum_steps: Optional[int] = None,
    grad_clip: Optional[float] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    val_plan: Optional[dict] = None,
    train_mb_size: Optional[int] = 1470,
    eval_mb_size: Optional[int] = 1470,
    ddp: bool = False,
    early_stop_patience: Optional[int] = None,
    early_stop_min_delta: float = 0.0,
    early_stop_warmup_epochs: int = 0,
    start_dt: Optional[datetime] = None,
    run_dir: Optional[Path] = None,
    args: Optional[object] = None,
    start_epoch: int = 0,
    best_val_init: float = float("inf"),
    history_seed: Optional[dict] = None,
    samples_seen_seed: int = 0,
    rollout_cfg: Optional[dict] = None,
    validate_only: bool = False,
) -> Tuple["History", float, bool]:
    """
    Train with micro-batching + grad accumulation, optional DDP, mid-epoch validation,
    carry rollout (when enabled via rollout_cfg), and checkpointing.

    Returns:
      (history, best_val, stopped_early_flag)
    """
    history = History(model)
    stopped_early_flag = False
    
    train_mb  = 1470 if (train_mb_size is None or train_mb_size <= 0) else int(train_mb_size)
    eval_mb   = 1470 if (eval_mb_size  is None or eval_mb_size  <= 0) else int(eval_mb_size)

    # --- Rehydrate history (resume) ---
    if history_seed:
        history.train_loss       = list(history_seed.get("train_loss", []))
        history.val_loss         = list(history_seed.get("val_loss", []))
        history.batch_loss       = list(history_seed.get("batch_loss", []))
        history.batch_step       = list(history_seed.get("batch_step", []))
        history.epoch_edges      = list(history_seed.get("epoch_edges", [0]))
        history.val_loss_batches = list(history_seed.get("val_loss_batches", []))
        history.val_loss_steps   = list(history_seed.get("val_loss_steps", []))
        history.lr_values        = list(history_seed.get("lr_values", []))
        history.lr_steps         = list(history_seed.get("lr_steps", []))

    history.samples_seen = int(samples_seen_seed)

    best_val = float(best_val_init)

    # --- Effective knobs ---
    eff_accum = 1 if (accum_steps is None or accum_steps <= 1) else int(accum_steps)
    eff_clip  = grad_clip if (grad_clip is not None and grad_clip > 0) else None

    # Helper flags
    is_main_fit = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
    model_device = next(model.parameters()).device

    # --- Carry flag (controls training/eval path) ---
    def _carry_on(cfg: Optional[dict]) -> bool:
        try:
            return float((cfg or {}).get("carry_horizon", 0.0) or 0.0) > 0.0
        except Exception:
            return False

    carry_enabled = _carry_on(rollout_cfg)

    if log and is_main_fit:
        sig = (rollout_cfg or {}).get("schema_sig")
        if sig:
            log.info("[schema] signature=%s", sig)

    # ---------- Validation-only short-circuit ----------
    if validate_only:
        # Use deterministic subset if present in plan
        batch_indices = None
        if val_plan is not None and "fixed_val_batch_ids" in val_plan:
            batch_indices = list(val_plan["fixed_val_batch_ids"])

        avg_val_loss, val_cnt, val_mb_avgs = validate(
            model, loss_func, valid_dl,
            device=model_device,
            batch_indices=batch_indices,
            rollout_cfg=rollout_cfg,
            eval_mb_size=(getattr(args, "eval_mb_size", None) or eval_mb),
        )
        history.update(train_loss=float("nan"), val_loss=avg_val_loss)
        if log and is_main_fit:
            log.info("Validation-only: val_loss=%.6f (windows=%d)", avg_val_loss, val_cnt)
            if getattr(args, "use_mass_balances", False) and val_mb_avgs:
                log.info("Val-only MB (weighted avg per-window): %s",
                         " | ".join(f"{k}={v:.6f}" for k, v in sorted(val_mb_avgs.items())))
        return history, avg_val_loss, False

    # --- Early stopping state (rehydrate if provided) ---
    early = None
    if early_stop_patience is not None and early_stop_patience > 0:
        early = EarlyStopping(
            patience=early_stop_patience,
            min_delta=early_stop_min_delta,
            warmup_epochs=early_stop_warmup_epochs,
        )
    if early is not None and history_seed is not None:
        es = history_seed.get("_early_state") or history_seed.get("early_state")
        if es and es.get("enabled", False):
            early.best       = float(es.get("best", early.best))
            early.bad_epochs = int(es.get("bad_epochs", 0))
            early.best_epoch = int(es.get("best_epoch", -1))

    # --- Nothing to do if already finished ---
    if start_epoch >= epochs:
        if log and is_main_fit:
            log.info("Resume requested at epoch %d (>= total epochs %d). Nothing to do.", start_epoch, epochs)
        return history, best_val, False

    global_batch_idx = len(history.batch_step) if history.batch_step else 0
    global_opt_step  = len(history.lr_steps)   if history.lr_steps   else 0

    # -----------------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):

        # DDP per-epoch reshuffle
        if ddp:
            if hasattr(train_dl.sampler, "set_epoch"):
                train_dl.sampler.set_epoch(epoch)
            if hasattr(valid_dl.sampler, "set_epoch"):
                valid_dl.sampler.set_epoch(epoch)

        if log and is_main_fit:
            log.info("Starting Training - [Epoch %d/%d] accum_steps=%s, train_mb_size=%s (carry=%s)",
                     epoch + 1, epochs, eff_accum, train_mb, carry_enabled)

        every_n = max(1, len(train_dl) // 20)

        model.train()
        train_losses: List[float] = []
        opt.zero_grad(set_to_none=True)

        # --- Mass-balance epoch accumulators (train) ------------------------
        epoch_train_mb_sums: Dict[str, float] = {}
        epoch_train_windows: int = 0

        # ============================== outer-batch loop ==============================
        for batch_idx, (batch_inputs, batch_monthly, batch_annual) in enumerate(train_dl):
            # inputs:   [nin, 365*n_years, n_locs]
            # labels_m: [nm,  12*n_years,  n_locs]
            # labels_a: [na,    n_years,   n_locs]
            inputs   = batch_inputs.squeeze(0).float().to(model_device)
            labels_m = batch_monthly.squeeze(0).float().to(model_device)
            labels_a = batch_annual.squeeze(0).float().to(model_device)

            n_years = int(labels_a.shape[1])
            n_locs  = int(inputs.shape[2])

            # Determine samples_seen according to carry semantics
            if carry_enabled:
                from math import ceil
                H = float((rollout_cfg or {}).get("carry_horizon", 0.0) or 0.0)
                D = int(ceil(H))
                history.samples_seen += max(0, n_years - D) * n_locs
            else:
                history.samples_seen += max(0, n_years - 1) * n_locs

            # -------------------------- carry rollout path --------------------------
            if carry_enabled:
                rt = rollout_outer_batch(
                    model=model,
                    loss_func=loss_func,
                    opt=opt,
                    scheduler=scheduler,
                    inputs=inputs,
                    labels_m=labels_m,
                    labels_a=labels_a,
                    mb_size=train_mb,
                    eff_accum=eff_accum,
                    eff_clip=eff_clip,
                    history=history,
                    global_opt_step=global_opt_step,
                    device=model_device,
                    rollout_cfg=rollout_cfg,
                    training=True,
                )
                # Returns (sum_loss, n_windows, global_opt_step, optional_mb_sums)
                batch_sum_loss, n_windows, global_opt_step = float(rt[0]), int(rt[1]), int(rt[2])
                avg_batch_loss = (batch_sum_loss / max(1, n_windows)) if n_windows > 0 else float("inf")

                # If you track mass-balance breakdowns:
                if len(rt) >= 4 and isinstance(rt[3], dict) and n_windows > 0:
                    for k, v in rt[3].items():
                        epoch_train_mb_sums[k] = epoch_train_mb_sums.get(k, 0.0) + float(v)
                    epoch_train_windows += int(n_windows)

                train_losses.append(float(avg_batch_loss))
                history.add_batch(float(avg_batch_loss), global_batch_idx)
                global_batch_idx += 1

            # ----------------------- teacher-forced path -----------------------
            else:
                # Stack all (year, location) windows -> per-window mini-batch
                x_list, yM_list, yA_list = [], [], []

                for y in range(1, n_years):
                    # slices for current supervision window y
                    x_slice  = inputs[:,   y*365:(y+1)*365, :].permute(2, 1, 0)  # [nl,365,nin]
                    yM_slice = labels_m[:, y*12:(y+1)*12,  :].permute(2, 1, 0)  # [nl,12,nm]
                    yA_slice = labels_a[:, y:(y+1),        :].permute(2, 1, 0)  # [nl,1, na]

                    x_list.append(x_slice);      yM_list.append(yM_slice);  yA_list.append(yA_slice)

                Xw   = torch.cat(x_list, dim=0)  # [Nw,365,nin]
                YMw  = torch.cat(yM_list, dim=0) # [Nw,12,nm]
                YAw  = torch.cat(yA_list, dim=0) # [Nw,1, na]
                Nw   = int(Xw.shape[0])

                if getattr(args, "shuffle_windows", False) and Nw > 1:
                    perm = torch.randperm(Nw, device=Xw.device)
                    Xw, YMw, YAw = Xw[perm], YMw[perm], YAw[perm]

                micro_size = min(train_mb, Nw)
                num_micro  = (Nw + micro_size - 1) // micro_size
                microbatches_done = 0
                running_sum_loss  = 0.0

                for mb_idx in range(num_micro):
                    s = mb_idx * micro_size
                    e = min((mb_idx + 1) * micro_size, Nw)

                    xb   = Xw[s:e]
                    yb_m = YMw[s:e]
                    yb_a = YAw[s:e]

                    # Guard minibatch inputs
                    if not torch.isfinite(xb).all():
                        if log:
                            log.warning("[train/teacher] non-finite INPUT minibatch %s", tuple(xb.shape))
                        del xb, yb_m, yb_a
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    preds = model(xb)  # [B,365,nm+na]

                    # Build extra_daily for water-balance (physical Pre)
                    extra_daily = None
                    try:
                        in_order = (rollout_cfg or {}).get("input_order", [])
                        if "pre" in in_order:
                            pre_idx = in_order.index("pre")
                            pre_norm = xb[:, :, pre_idx]
                            st = ((rollout_cfg or {}).get("std_stats_in", {}) or {}).get("pre", {})
                            mu = float(st.get("mean", 0.0))
                            sd = float(st.get("std", 1.0))
                            pre_phys = pre_norm * sd + mu
                            extra_daily = {"pre": pre_phys}
                    except Exception:
                        extra_daily = None

                    if not torch.isfinite(preds).all():
                        if log:
                            log.warning("[train/teacher] non-finite PRED minibatch %s", tuple(preds.shape))
                        del preds, xb, yb_m, yb_a
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    loss = loss_func(preds, yb_m, yb_a, extra_daily=extra_daily)

                    # Accumulate MB weighted contributions
                    bd = getattr(loss_func, "last_breakdown", None)
                    _accum_bd_sums(epoch_train_mb_sums, bd, mult=(e - s))
                    epoch_train_windows += (e - s)

                    running_sum_loss += float(loss.detach().cpu()) * (e - s)

                    (loss / eff_accum).backward()
                    microbatches_done += 1

                    if microbatches_done % eff_accum == 0:
                        if eff_clip is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=eff_clip)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        if scheduler is not None:
                            scheduler.step()
                        lr_now = opt.param_groups[0]["lr"]
                        history.lr_values.append(float(lr_now))
                        history.lr_steps.append(global_opt_step)
                        global_opt_step += 1

                # Flush remainder if needed
                if microbatches_done % eff_accum != 0:
                    if eff_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=eff_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    lr_now = opt.param_groups[0]["lr"]
                    history.lr_values.append(float(lr_now))
                    history.lr_steps.append(global_opt_step)
                    global_opt_step += 1

                avg_batch_loss = running_sum_loss / max(1, Nw)
                train_losses.append(avg_batch_loss)
                history.add_batch(avg_batch_loss, global_batch_idx)
                global_batch_idx += 1

            # --------------------------- in-epoch validation ---------------------------
            if val_plan is not None:
                veb = val_plan["validate_every_batches"]
                if (veb < len(train_dl)) and ((batch_idx + 1) % veb == 0):
                    total_val_batches = val_plan["val_total_batches"]

                    if "fixed_val_batch_ids" in val_plan:
                        batch_ids = list(val_plan["fixed_val_batch_ids"])
                    else:
                        use_k = max(1, int(round(val_plan["val_batches_to_use"])))
                        batch_ids = _sample_validation_batch_indices(total_val_batches, use_k)

                    # DDP: broadcast chosen val batches to all ranks
                    if ddp and dist.is_available() and dist.is_initialized():
                        is_main_rank = (dist.get_rank() == 0)
                        dev = model_device
                        if is_main_rank:
                            ids  = torch.tensor(batch_ids, dtype=torch.int64, device=dev)
                            size = torch.tensor([ids.numel()], dtype=torch.int64, device=dev)
                        else:
                            size = torch.zeros(1, dtype=torch.int64, device=dev)
                        dist.broadcast(size, src=0)
                        if not is_main_rank:
                            ids = torch.empty(int(size.item()), dtype=torch.int64, device=dev)
                        dist.broadcast(ids, src=0)
                        batch_ids = ids.tolist()

                    interim_avg, interim_cnt, _ = validate(
                        model, loss_func, valid_dl, device=model_device, batch_indices=batch_ids,
                        rollout_cfg=rollout_cfg,
                        eval_mb_size=(getattr(args, "eval_mb_size", None) or eval_mb),
                    )
                    
                    # Mid-epoch early stopping check
                    if early is not None:
                        early.step_on_check(interim_avg, epoch)
                        stop_now = torch.tensor(
                            [1 if (is_main_fit and early.should_stop) else 0],
                            device=model_device,
                            dtype=torch.int32,
                        )
                        if ddp and dist.is_available() and dist.is_initialized():
                            dist.broadcast(stop_now, src=0)

                        if int(stop_now.item()) == 1:
                            if log and is_main_fit:
                                log.info(
                                    "[Early stop mid-epoch] epoch %d (best @ epoch %d, best_val=%.6f)",
                                    epoch + 1, early.best_epoch + 1, early.best
                                )
                            stopped_early_flag = True
                            break  # break outer-batch loop for this epoch

                    if log and is_main_fit:
                        log.info("[Validation @ batch %d] avg_loss=%.6f on %d selected val batches (windows=%d)",
                                 batch_idx + 1, interim_avg, len(batch_ids), interim_cnt)

                    history.add_val_batch(interim_avg, global_batch_idx)

                    # Snapshot early state for resume
                    if early is not None:
                        history._early_state = {
                            "enabled": True,
                            "patience": early.patience,
                            "min_delta": early.min_delta,
                            "warmup_epochs": early.warmup_epochs,
                            "best": early.best,
                            "bad_epochs": early.bad_epochs,
                            "best_epoch": early.best_epoch,
                        }
                    else:
                        history._early_state = {"enabled": False}

                    # Save best-on-interim if desired
                    if interim_avg < best_val and save_cb:
                        best_val = interim_avg
                        save_cb(epoch, best=True, val=best_val, history=history)

            # ------------------------------ progress log ------------------------------
            if log and is_main_fit and (
                ((batch_idx + 1) % every_n == 0) or ((batch_idx + 1) == len(train_dl))
            ):
                total_outer_batches = max(1, epochs * len(train_dl))
                overall_done = epoch * len(train_dl) + (batch_idx + 1)
                percent_done = 100.0 * overall_done / total_outer_batches
                log.info(
                    "Progress %5.1f%% — epoch %d/%d, batch %d/%d, avg_batch_loss=%.6f",
                    percent_done, epoch + 1, epochs, batch_idx + 1, len(train_dl), train_losses[-1]
                )

        # ============================ end outer-batch loop ============================
        history.close_epoch()

        # --- HARD CLEANUP before epoch-end validation (from carry branch) ---
        for _name in (
            "Xw","YMw","YAw","xb","yb_m","yb_a","preds",
            "x_list","yM_list","yA_list",
            "inputs","labels_m","labels_a",
        ):
            if _name in locals():
                try: del locals()[_name]
                except Exception: pass
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Break out if we stopped mid-epoch
        if stopped_early_flag:
            break

        # Epoch-level metrics (+ DDP reduce)
        avg_train_loss = float(np.mean(train_losses)) if len(train_losses) else float("nan")
        avg_train_loss = ddp_mean_scalar(avg_train_loss, model_device)

        epoch_end_batch_indices = val_plan["fixed_val_batch_ids"] if (val_plan is not None) else None
        avg_val_loss, val_cnt, val_mb_avgs = validate(
            model, loss_func, valid_dl, device=model_device,
            batch_indices=epoch_end_batch_indices,
            rollout_cfg=rollout_cfg,
            eval_mb_size=(getattr(args, "eval_mb_size", None) or eval_mb),
        )
        
        if val_cnt == 0:
            if log and is_main_fit:
                log.warning("Validation produced 0 windows — check batch_indices/filters.")
            avg_val_loss = float("inf")

        # Train MB avgs for this epoch (weighted contributions per window)
        train_mb_avgs = _normalize_bd_sums(epoch_train_mb_sums, max(1, epoch_train_windows))

        if log and is_main_fit:
            log.info("Epoch average train loss=%.6f, val loss=%.6f (val_cnt=%d)",
                     avg_train_loss, avg_val_loss, val_cnt)

            # --- Mass-balance epoch logs (only when enabled) -------------------
            if getattr(args, "use_mass_balances", False):
                if train_mb_avgs:
                    log.info("Epoch MB train (weighted avg per-window): %s",
                             " | ".join(f"{k}={v:.6f}" for k, v in sorted(train_mb_avgs.items())))
                if val_mb_avgs:
                    log.info("Epoch MB val   (weighted avg per-window): %s",
                             " | ".join(f"{k}={v:.6f}" for k, v in sorted(val_mb_avgs.items())))

        history.update(train_loss=avg_train_loss, val_loss=avg_val_loss)

        # Persist MB series for plotting (if history supports it)
        if getattr(args, "use_mass_balances", False) and hasattr(history, "add_mass_balance_epoch"):
            history.add_mass_balance_epoch(train_mb_avgs, val_mb_avgs)

        # Update early stopping on epoch end
        if early is not None:
            early.step_on_check(avg_val_loss, epoch)

        # Best-effort per-epoch plots
        if log and is_main_fit and (run_dir is not None and args is not None and start_dt is not None):
            try:
                elapsed_seconds = (datetime.now() - start_dt).total_seconds()
                history.save_epoch_plots_overwrite(run_dir, args, start_dt, elapsed_seconds)
            except Exception as e:
                log.warning("Per-epoch plotting failed: %s", e)

        # Snapshot early state before saving
        if early is not None:
            history._early_state = {
                "enabled": True,
                "patience": early.patience,
                "min_delta": early.min_delta,
                "warmup_epochs": early.warmup_epochs,
                "best": early.best,
                "bad_epochs": early.bad_epochs,
                "best_epoch": early.best_epoch,
            }
        else:
            history._early_state = {"enabled": False}

        # Checkpointing (best + rolling)
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            if save_cb:
                save_cb(epoch, best=True, val=best_val, history=history)
        if save_cb:
            save_cb(epoch, best=False, val=avg_val_loss, history=history)

        # Final early-stopping decision for this epoch
        if early is not None:
            stop_flag = torch.tensor(
                [1 if (is_main_fit and early.should_stop) else 0],
                device=model_device,
                dtype=torch.int32,
            )
            if ddp and dist.is_available() and dist.is_initialized():
                dist.broadcast(stop_flag, src=0)

            if int(stop_flag.item()) == 1:
                if log and is_main_fit:
                    log.info(
                        "Early stopping triggered at epoch %d (best @ epoch %d, best_val=%.6f)",
                        epoch + 1, early.best_epoch + 1, early.best
                    )
                stopped_early_flag = True
                break

    # DDP: make samples_seen global
    if ddp and dist.is_available() and dist.is_initialized():
        t = torch.tensor([history.samples_seen], device=next(model.parameters()).device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        history.samples_seen = int(t.item())

    return history, best_val, stopped_early_flag


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------

def plan_validation(
    train_stats: Dict[str, int],
    valid_dl,
    validation_frequency: float,
    validation_size: float,
) -> Dict[str, int]:
    """
    Plan how often and how many batches to use for validation, and (optionally)
    precompute a fixed subset of validation batch indices to reuse every time.
    """
    # frequency -> every N train batches
    train_batches = int(train_stats["batches"])
    validate_every_batches = max(1, int(round(validation_frequency * train_batches)))

    # how many val batches exist / to use each probe
    val_total_batches = len(valid_dl)
    val_batches_to_use = int(round(validation_size * val_total_batches))
    val_batches_to_use = min(val_total_batches, max(1, val_batches_to_use))

    # fixed subset of val batch indices (same across the run)
    rng = random.Random(42)  # constant seed (deterministic subset)
    indices = list(range(val_total_batches))
    rng.shuffle(indices)
    fixed_ids = sorted(indices[:val_batches_to_use])

    return {
        "validate_every_batches": validate_every_batches,
        "val_batches_to_use": val_batches_to_use,
        "train_batches": train_batches,
        "val_total_batches": val_total_batches,
        "fixed_val_batch_ids": fixed_ids,
    }

def _sample_validation_batch_indices(total_batches: int, k: int) -> List[int]:
    """Randomly select k validation batch indices from total_batches (sorted)."""
    k = min(max(0, int(k)), int(total_batches))
    if k == 0:
        return []
    if k >= total_batches:
        return list(range(total_batches))
    return random.sample(range(total_batches), k)

def validate(
    model: torch.nn.Module,
    loss_func,
    valid_dl,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    batch_indices: Optional[Iterable[int]] = None,
    rollout_cfg: Optional[dict] = None,
    eval_mb_size: Optional[int] = 1470,
) -> Tuple[float, int, Dict[str, float]]:
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    def _carry_on(cfg: Optional[dict]) -> bool:
        try:
            return float((cfg or {}).get("carry_horizon", 0.0) or 0.0) > 0.0
        except Exception:
            return False

    carry_enabled = _carry_on(rollout_cfg)

    use_index_subset = batch_indices is not None
    index_set = set(batch_indices) if use_index_subset else None
    seen_selected = 0
    need_selected = len(index_set) if use_index_subset else None

    total_loss = 0.0
    total_cnt  = 0
    mb_sums: Dict[str, float] = {}

    with torch.inference_mode():
        for b_idx, (batch_inputs, batch_monthly, batch_annual) in enumerate(valid_dl):
            if use_index_subset and (b_idx not in index_set):
                continue
            if (not use_index_subset) and (max_batches is not None) and (b_idx >= max_batches):
                break

            inputs   = batch_inputs.squeeze(0).float().to(device, non_blocking=True)
            labels_m = batch_monthly.squeeze(0).float().to(device, non_blocking=True)
            labels_a = batch_annual.squeeze(0).float().to(device, non_blocking=True)

            if carry_enabled:
                result = rollout_outer_batch(
                    model=model,
                    loss_func=loss_func,
                    inputs=inputs,
                    labels_m=labels_m,
                    labels_a=labels_a,
                    device=device,
                    rollout_cfg=rollout_cfg,
                    mb_size=(int(eval_mb_size) if eval_mb_size else 1470),
                    training=False,
                )
                batch_sum_loss, n_windows = float(result[0]), int(result[1])
                total_loss += batch_sum_loss
                total_cnt  += n_windows
                if len(result) >= 4 and isinstance(result[3], dict):
                    for k, v in result[3].items():
                        mb_sums[k] = mb_sums.get(k, 0.0) + float(v)

            else:
                # Teacher-forced: iterate per (year, location) window
                n_locations = int(inputs.shape[2])
                n_years     = int(labels_a.shape[1])

                for loc in range(n_locations):
                    for year_idx in range(1, n_years):
                        x_win   = inputs[:, year_idx*365:(year_idx+1)*365, loc]   # [nin,365]
                        yM_win  = labels_m[:, year_idx*12:(year_idx+1)*12, loc]   # [nm,12]
                        yA_win  = labels_a[:, year_idx:(year_idx+1),        loc]  # [na,1]

                        xb   = x_win.T.unsqueeze(0)  # [1,365,nin]
                        yb_m = yM_win.T.unsqueeze(0) # [1,12,nm]
                        yb_a = yA_win.T.unsqueeze(0) # [1,1,na]

                        # Optional: extra_daily (physical Pre) if you use it
                        extra_daily = None
                        try:
                            in_order = (rollout_cfg or {}).get("input_order", [])
                            if "pre" in in_order:
                                pre_idx = in_order.index("pre")
                                pre_norm = xb[:, :, pre_idx]
                                st = ((rollout_cfg or {}).get("std_stats_in", {}) or {}).get("pre", {})
                                mu = float(st.get("mean", 0.0))
                                sd = float(st.get("std", 1.0))
                                pre_phys = pre_norm * sd + mu
                                extra_daily = {"pre": pre_phys}
                        except Exception:
                            extra_daily = None

                        preds_daily = model(xb)
                        val_loss = loss_func(preds_daily, yb_m, yb_a, extra_daily=extra_daily)
                        total_loss += float(val_loss.item())
                        total_cnt  += 1

                        # If your loss exposes a breakdown:
                        bd = getattr(loss_func, "last_breakdown", None)
                        if isinstance(bd, dict) and ("weighted" in bd):
                            for k, v in bd["weighted"].items():
                                if v is None:
                                    continue
                                mb_sums[k] = mb_sums.get(k, 0.0) + float(v)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if use_index_subset:
                seen_selected += 1
                if seen_selected >= need_selected:
                    break

    # DDP reduce if needed
    if dist.is_available() and dist.is_initialized():
        dev = device
        t_sum = torch.tensor([total_loss], device=dev, dtype=torch.float64)
        t_cnt = torch.tensor([total_cnt],  device=dev, dtype=torch.float64)
        dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_cnt, op=dist.ReduceOp.SUM)

        if mb_sums:
            keys = sorted(mb_sums.keys())
            buf  = torch.tensor([mb_sums[k] for k in keys], device=dev, dtype=torch.float64)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            mb_sums = {k: float(buf[i].item()) for i, k in enumerate(keys)}

        total_loss = float(t_sum.item())
        total_cnt  = int(t_cnt.item())

    avg = (total_loss / total_cnt) if total_cnt > 0 else float("inf")

    # Convert MB sums to per-window averages
    mb_avgs = {k: (v / max(1, total_cnt)) for k, v in mb_sums.items()}
    return avg, total_cnt, mb_avgs