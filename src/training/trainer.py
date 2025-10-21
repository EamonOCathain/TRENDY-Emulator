from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable, Optional, Tuple, Dict, Iterable, List, Any
import time
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import sys
import logging
from datetime import datetime 
from copy import deepcopy
import json
import math

# set project root
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

from src.training.distributed import ddp_mean_scalar
from src.training.history import History
from src.training.carry import rollout_train_outer_batch, rollout_eval_outer_batch, _tsum
class EarlyStopping:
    """
    Stop when the monitored metric fails to improve by at least `min_delta`
    for `patience` epochs. Assumes 'min' mode (lower is better).
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
        # Ignore during warmup
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

def fit(
    epochs: int,
    model: torch.nn.Module,
    loss_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    opt: Optimizer,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    log: Optional[logging.Logger] = None,
    save_cb: Optional[Callable[[int, bool, float, "History"], None]] = None,
    accum_steps: Optional[int] = None,
    grad_clip: Optional[float] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    val_plan: Optional[dict] = None,
    mb_size: Optional[int] = None,
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
    stage_carry: Optional[float] = None,
) -> Tuple["History", float, bool]:
    """
    Train loop with micro-batching + grad accumulation, optional DDP sync,
    periodic in-epoch validation, and epoch-level checkpointing.

    Returns: (history, best_val, stopped_early_flag)
    """
    history = History(model)
    stopped_early_flag = False

    # ---- Rehydrate history on resume (if provided) ----
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
    
    try:
        history.record_carry_stage(stage_carry)
    except Exception:
        pass

    best_val = float(best_val_init)

    eff_accum = 1 if (accum_steps is None or accum_steps <= 1) else int(accum_steps)
    eff_clip  = grad_clip if (grad_clip is not None and grad_clip > 0) else None
    mb_size   = 512 if (mb_size is None or mb_size <= 0) else int(mb_size)

    is_main_fit = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
    model_device = next(model.parameters()).device

    # Early stopping helper
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

    # Nothing to do if already finished
    if start_epoch >= epochs:
        if log and is_main_fit:
            log.info("Resume requested at epoch %d, which >= total epochs %d. Nothing to do.",
                     start_epoch, epochs)
        return history, best_val, False

    global_batch_idx = len(history.batch_step) if history.batch_step else 0
    global_opt_step  = len(history.lr_steps)   if history.lr_steps   else 0

    # Enable carry?
    carry_on = False
    if rollout_cfg is not None:
        try:
            carry_on = float(rollout_cfg.get("carry_horizon", 0.0)) > 0.0
        except Exception:
            carry_on = False
    
    if log and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
        sig = (rollout_cfg or {}).get("schema_sig")
        if sig:
            log.info(f"[schema] signature={sig}")
    
    # --- Downshift default microbatch when carry is on ---
    '''if carry_on:
        mb_size = min(mb_size, 64)'''

    # --- Override if we already autotuned in a previous stage ---
    if carry_on and isinstance(rollout_cfg, dict) and rollout_cfg.get("autotuned_mb_size"):
        mb_size = int(rollout_cfg["autotuned_mb_size"])
        if ddp and dist.is_available() and dist.is_initialized():
            t = torch.tensor([mb_size], device=model_device, dtype=torch.int64)
            dist.broadcast(t, src=0)
            mb_size = int(t.item())

    for epoch in range(start_epoch, epochs):

        # DDP per-epoch reshuffle
        if ddp:
            if hasattr(train_dl.sampler, "set_epoch"):
                train_dl.sampler.set_epoch(epoch)
            if hasattr(valid_dl.sampler, "set_epoch"):
                valid_dl.sampler.set_epoch(epoch)

        if log and is_main_fit:
            log.info("Starting Training - [Epoch %d/%d] accum_steps=%s, mb_size=%s",
                     epoch + 1, epochs, eff_accum, mb_size)

        every_n = max(1, len(train_dl) // 20)

        model.train()
        train_losses = []
        opt.zero_grad(set_to_none=True)

        # ---------------- outer-batch loop ----------------
        for batch_idx, (batch_inputs, batch_monthly, batch_annual) in enumerate(train_dl):
            # inputs:   [nin, 365*n_years, n_locs]
            # labels_m: [nm,  12*n_years,  n_locs]
            # labels_a: [na,    n_years,   n_locs]
            inputs   = batch_inputs.squeeze(0).float().to(model_device)
            labels_m = batch_monthly.squeeze(0).float().to(model_device)
            labels_a = batch_annual.squeeze(0).float().to(model_device)

            n_years = int(labels_a.shape[1])
            n_locs  = int(inputs.shape[2])

            carry_on = False
            # Determine horizon and update samples_seen (tail-only ownership if H>0)
            try:
                H = float((rollout_cfg or {}).get("carry_horizon", 0.0) or 0.0)
            except Exception:
                H = 0.0
            carry_on = H > 0.0

            if carry_on:
                D = int(math.ceil(H))
                history.samples_seen += max(0, n_years - D) * n_locs
            else:
                history.samples_seen += max(0, n_years - 1) * n_locs

            if not carry_on:
                # ---------- existing teacher-forced path (unchanged) ----------
                # Stack all (year, location) windows -> per-window mini-batch
                x_list, yM_list, yA_list = [], [], []
                yM_prev_last_list, yA_prev_list = [], []

                for y in range(1, n_years):
                    # slices for current supervision window y
                    x_slice  = inputs[:,   y*365:(y+1)*365, :].permute(2, 1, 0)  # [nl,365,nin]
                    yM_slice = labels_m[:, y*12:(y+1)*12,  :].permute(2, 1, 0)   # [nl,12,nm]
                    yA_slice = labels_a[:, y:(y+1),        :].permute(2, 1, 0)   # [nl,1, na]

                    # previous-year anchors (Dec of y-1, and annual of y-1)
                    dec_prev = labels_m[:, y*12 - 1, :].permute(1, 0).unsqueeze(1)   # [nl,1,nm]
                    ann_prev = labels_a[:, y - 1,    :].permute(1, 0).unsqueeze(1)   # [nl,1,na]

                    x_list.append(x_slice);      yM_list.append(yM_slice);  yA_list.append(yA_slice)
                    yM_prev_last_list.append(dec_prev);  yA_prev_list.append(ann_prev)

                Xw   = torch.cat(x_list,            dim=0)  # [Nw,365,nin]
                YMw  = torch.cat(yM_list,           dim=0)  # [Nw,12,nm]
                YAw  = torch.cat(yA_list,           dim=0)  # [Nw,1, na]
                YMwP = torch.cat(yM_prev_last_list, dim=0)  # [Nw,1, nm]  (Dec_{y-1})
                YAwP = torch.cat(yA_prev_list,      dim=0)  # [Nw,1, na]  (Annual_{y-1})
                Nw   = int(Xw.shape[0])

                if getattr(args, "shuffle_windows", False) and Nw > 1:
                    perm = torch.randperm(Nw, device=Xw.device)
                    Xw, YMw, YAw, YMwP, YAwP = Xw[perm], YMw[perm], YAw[perm], YMwP[perm], YAwP[perm]

                micro_size = min(mb_size, Nw)
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
                            log.warning("[train/teacher] non-finite INPUT minibatch: %s", _tsum(xb, "xb"))
                        # Skip this microbatch
                        del xb, yb_m, yb_a
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        continue

                    preds = model(xb)  # [B,365,nm+na] (deltas if --delta)
                    
                    if not torch.isfinite(preds).all():
                        if log:
                            log.warning("[train/teacher] non-finite PRED minibatch: %s | %s",
                                        _tsum(xb, "xb"), _tsum(preds, "preds"))
                        del preds, xb, yb_m, yb_a
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        continue

                    dc = (rollout_cfg or {}).get("delta_ctx", None)
                    if (dc is not None) and getattr(dc, "enabled", False):
                        nm = int(yb_m.shape[-1]); na = int(yb_a.shape[-1])
                        # Grab matching prev anchors for this slice
                        yb_m_prev_last = YMwP[s:e]  # [B,1,nm]
                        yb_a_prev      = YAwP[s:e]  # [B,1,na]
                        preds = dc.reconstruct_groups_daily_segmentwise(
                            preds=preds, nm=nm, na=na,
                            month_slices = (rollout_cfg or {}).get("month_slices", None),
                            mode="teacher",
                            yb_m_prev_last=yb_m_prev_last,  # Dec of previous year
                            yb_a_prev=yb_a_prev,            # previous annual
                            out_m_idx=None, out_a_idx=None,
                            prev_monthly_state=None, prev_annual_state=None,
                        )
                    # now preds are daily ABSOLUTES (normalized); loss pools internally
                    loss = loss_func(preds, yb_m, yb_a)

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

            else:
                # ---------- carry rollout path ----------
                avg_batch_loss, global_opt_step = rollout_train_outer_batch(
                    model=model,
                    loss_func=loss_func,
                    opt=opt,
                    scheduler=scheduler,
                    inputs=inputs,
                    labels_m=labels_m,
                    labels_a=labels_a,
                    mb_size=mb_size,
                    eff_accum=eff_accum,
                    eff_clip=eff_clip,
                    history=history,
                    global_opt_step=global_opt_step,
                    device=model_device,
                    rollout_cfg=rollout_cfg,
                )
                train_losses.append(avg_batch_loss)
                history.add_batch(avg_batch_loss, global_batch_idx)
                global_batch_idx += 1

            # -------- in-epoch validation (optional) --------
            if val_plan is not None:
                veb = val_plan["validate_every_batches"]
                if (veb < len(train_dl)) and ((batch_idx + 1) % veb == 0):
                    total_val_batches = val_plan["val_total_batches"]

                    if "fixed_val_batch_ids" in val_plan:
                        batch_ids = list(val_plan["fixed_val_batch_ids"])
                    else:
                        use_k = max(1, int(round(val_plan["val_batches_to_use"])))
                        batch_ids = _sample_validation_batch_indices(total_val_batches, use_k)

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

                    interim_avg, interim_cnt = validate(
                        model, loss_func, valid_dl, device=model_device, batch_indices=batch_ids,
                        rollout_cfg=rollout_cfg,  
                    )
                    # Trigger mid epoch early stopping check
                    if early is not None:
                        early.step_on_check(interim_avg, epoch)

                        # optional: early exit during the epoch
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
                                    f"[Early stop mid-epoch] epoch {epoch+1}, "
                                    f"(best @ epoch {early.best_epoch+1}, best_val={early.best:.6f})"
                                )
                            stopped_early_flag = True
                            break  # breaks the outer-batch loop for this epoch

                    if log and is_main_fit:
                        log.info("[Validation @ batch %d] avg_loss=%.6f on %d selected val batches (windows=%d)",
                                 batch_idx + 1, interim_avg, len(batch_ids), interim_cnt)

                    history.add_val_batch(interim_avg, global_batch_idx)

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

                    if interim_avg < best_val and save_cb:
                        best_val = interim_avg
                        save_cb(epoch, best=True, val=best_val, history=history)

            # -------- periodic progress log --------
            if log and is_main_fit and (
                ((batch_idx + 1) % every_n == 0) or ((batch_idx + 1) == len(train_dl))
            ):
                total_outer_batches = max(1, epochs * len(train_dl))
                overall_done = epoch * len(train_dl) + (batch_idx + 1)
                percent_done = 100.0 * overall_done / total_outer_batches
                log.info("Progress %5.1f%% — epoch %d/%d, batch %d/%d, avg_batch_loss=%.6f",
                         percent_done, epoch + 1, epochs, batch_idx + 1, len(train_dl), train_losses[-1])

        # ---------------- end outer-batch loop ----------------
        history.close_epoch()
        
        # If we decided to stop mid-epoch, break out of the epoch loop now
        if stopped_early_flag:
            break

        # Epoch-level metrics, then DDP-reduce
        avg_train_loss = float(np.mean(train_losses)) if len(train_losses) else float("nan")
        avg_train_loss = ddp_mean_scalar(avg_train_loss, model_device)

        epoch_end_batch_indices = val_plan["fixed_val_batch_ids"] if (val_plan is not None) else None
        avg_val_loss, val_cnt = validate(
            model, loss_func, valid_dl, device=model_device,
            batch_indices=epoch_end_batch_indices,
            rollout_cfg=rollout_cfg,   
        )
        if val_cnt == 0:
            if log and is_main_fit:
                log.warning("Validation produced 0 windows — check batch_indices/filters.")
            avg_val_loss = float("inf")

        if log and is_main_fit:
            log.info("Epoch average train loss=%.6f, val loss=%.6f (val_cnt=%d)",
                     avg_train_loss, avg_val_loss, val_cnt)

        history.update(train_loss=avg_train_loss, val_loss=avg_val_loss)
        
        if early is not None:
            early.step_on_check(avg_val_loss, epoch)

        # ---- plots (best-effort) ----
        if log and is_main_fit and (run_dir is not None and args is not None and start_dt is not None):
            try:
                elapsed_seconds = (datetime.now() - start_dt).total_seconds()
                history.save_epoch_plots_overwrite(run_dir, args, start_dt, elapsed_seconds)
            except Exception as e:
                log.warning(f"Per-epoch plotting failed: {e}")

        # ---- snapshot early state before any saving ----
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

        # ---- checkpointing (best + rolling) ----
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            if save_cb:
                save_cb(epoch, best=True, val=best_val, history=history)
        if save_cb:
            save_cb(epoch, best=False, val=avg_val_loss, history=history)

        # ---- early stopping decision ----
        if early is not None:
            # NOTE: do NOT call early.step(...) here; it's already updated via step_on_check()
            stop_flag = torch.tensor(
                [1 if (is_main_fit and early.should_stop) else 0],
                device=model_device,
                dtype=torch.int32,
            )

            # broadcast main-rank decision to all ranks
            if ddp and dist.is_available() and dist.is_initialized():
                dist.broadcast(stop_flag, src=0)

            if int(stop_flag.item()) == 1:
                if log and is_main_fit:
                    log.info(
                        f"Early stopping triggered at epoch {epoch+1} "
                        f"(best @ epoch {early.best_epoch+1}, best_val={early.best:.6f})"
                    )
                stopped_early_flag = True
                break

    # DDP: make samples_seen global
    if ddp and dist.is_available() and dist.is_initialized():
        t = torch.tensor([history.samples_seen], device=next(model.parameters()).device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        history.samples_seen = int(t.item())

    return history, best_val, stopped_early_flag

# ------------------------------ validation -----------------------------------

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
    import random
    rng = random.Random(42)  # set a constant seed, or pass one into this function if you want variability
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
    """
    Randomly select k validation batch indices from total_batches.
    Only used if subsetting in order to make the validation distributed.

    Args:
        total_batches (int): Number of available validation batches.
        k (int): Number of indices to sample.

    Returns:
        list[int]: List of sampled batch indices (sorted).
    """
    k = min(max(0, int(k)), int(total_batches))

    if k == 0:
        return []
    if k >= total_batches:
        return list(range(total_batches))

    return random.sample(range(total_batches), k)

def validate(
    model: torch.nn.Module,
    loss_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    valid_dl: Iterable,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    batch_indices: Optional[Iterable[int]] = None,
    rollout_cfg: Optional[dict] = None,
) -> Tuple[float, int]:
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # carry flag (unchanged semantics)
    carry_on = False
    if rollout_cfg is not None:
        try:
            carry_on = float(rollout_cfg.get("carry_horizon", 0.0)) > 0.0
        except Exception:
            carry_on = False

    # month metadata
    month_lengths = (rollout_cfg or {}).get("month_lengths", [31,28,31,30,31,30,31,31,30,31,30,31])
    bounds = [0]
    for Lm in month_lengths:
        bounds.append(bounds[-1] + Lm)

    # delta flag
    dc = (rollout_cfg or {}).get("delta_ctx", None)
    delta_enabled = (dc is not None) and getattr(dc, "enabled", False)

    use_index_subset = batch_indices is not None
    index_set = set(batch_indices) if use_index_subset else None
    seen_selected = 0
    need_selected = len(index_set) if use_index_subset else None

    total_loss = 0.0
    total_cnt  = 0

    with torch.inference_mode():
        for b_idx, (batch_inputs, batch_monthly, batch_annual) in enumerate(valid_dl):
            if use_index_subset and (b_idx not in index_set):
                continue
            if (not use_index_subset) and (max_batches is not None) and (b_idx >= max_batches):
                break

            inputs   = batch_inputs.squeeze(0).float().to(device, non_blocking=True)
            labels_m = batch_monthly.squeeze(0).float().to(device, non_blocking=True)
            labels_a = batch_annual.squeeze(0).float().to(device, non_blocking=True)

            if carry_on:
                # delegate to the carry-eval kernel (it already implements the new delta logic)
                batch_sum_loss, n_windows = rollout_eval_outer_batch(
                    model=model,
                    loss_func=loss_func,
                    inputs=inputs,
                    labels_m=labels_m,
                    labels_a=labels_a,
                    device=device,
                    rollout_cfg=rollout_cfg,
                )
                total_loss += float(batch_sum_loss)
                total_cnt  += int(n_windows)

            else:
                # teacher-forced evaluation per (year, location) window
                n_locations = int(inputs.shape[2])
                n_years     = int(labels_a.shape[1])

                for loc in range(n_locations):
                    for year_idx in range(1, n_years):
                        x_win   = inputs[:, year_idx*365:(year_idx+1)*365, loc]              # [nin,365]
                        yM_win  = labels_m[:, year_idx*12:(year_idx+1)*12, loc]              # [nm,12]
                        yA_win  = labels_a[:, year_idx:(year_idx+1),        loc]              # [na,1]

                        xb   = x_win.T.unsqueeze(0)                                           # [1,365,nin]
                        yb_m = yM_win.T.unsqueeze(0)                                          # [1,12,nm]
                        yb_a = yA_win.T.unsqueeze(0)                                          # [1,1,na]

                        preds_daily = model(xb)                                               # [1,365,out]
                        if delta_enabled:
                            nm = int(yb_m.shape[-1])
                            na = int(yb_a.shape[-1])
                            dc = (rollout_cfg or {}).get("delta_ctx", None)

                            if dc is None:
                                preds_daily = preds_daily  # no-op safety
                            else:
                                # anchors from previous year labels for *this* location/window
                                dec_prev = labels_m[:, year_idx*12 - 1, loc].view(1, 1, nm)  # [1,1,nm]
                                ann_prev = labels_a[:, year_idx - 1,    loc].view(1, 1, na)  # [1,1,na]

                                preds_daily = dc.reconstruct_groups_daily_segmentwise(
                                    preds=preds_daily,               # [1,365,nm+na] deltas
                                    nm=nm, na=na,
                                    month_slices=(rollout_cfg or {}).get("month_slices", None),
                                    mode="teacher",
                                    yb_m_prev_last=dec_prev,         # Dec_{y-1}
                                    yb_a_prev=ann_prev,              # annual_{y-1}
                                    out_m_idx=None, out_a_idx=None,
                                    prev_monthly_state=None, prev_annual_state=None,
                                )

                        # loss on daily->pooled (custom_loss performs the pooling internally)
                        loss  = loss_func(preds_daily, yb_m, yb_a)
                        total_loss += float(loss.item())
                        total_cnt  += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if use_index_subset:
                seen_selected += 1
                if seen_selected >= need_selected:
                    break

    # DDP all-reduce
    if dist.is_available() and dist.is_initialized():
        dev = device
        t_sum = torch.tensor([total_loss], device=dev, dtype=torch.float64)
        t_cnt = torch.tensor([total_cnt],  device=dev, dtype=torch.float64)
        dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_cnt, op=dist.ReduceOp.SUM)
        total_loss = float(t_sum.item())
        total_cnt  = int(t_cnt.item())

    avg = (total_loss / total_cnt) if total_cnt > 0 else float("inf")
    return avg, total_cnt


def unwrap(model):
    """
    Return the underlying model when wrapped in DDP; otherwise return as-is.
    """
    return model.module if isinstance(model, DDP) else model

