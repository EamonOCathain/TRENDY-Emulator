from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.ticker import FixedLocator, FixedFormatter

class History:
    """
    Lightweight training history tracker.

    Tracks:
      - Per-batch training loss (for detailed curves).
      - Optional in-epoch validation probes (scattered points).
      - Per-epoch averaged train/val loss.
    """

    def __init__(self, model) -> None:
        # Keep a stringified snapshot of the model for provenance
        self.model_info: str = str(model)

        # Epoch-level metrics
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []

        # In-epoch validation probes (optional)
        self.val_loss_batches: List[float] = []
        self.val_loss_steps: List[int] = []

        # Batch-level training loss
        self.batch_loss: List[float] = []
        self.batch_step: List[int] = []

        # Indices in batch_loss where each epoch ends; starts at 0
        self.epoch_edges: List[int] = [0]
        
        self.samples_seen: int = 0
        
        # LR steps record
        self.lr_values: List[float] = []
        self.lr_steps:  List[int]   = []
        self.carry_stage_marks: list[dict] = []

    # -------------------------
    # Recording helpers
    # -------------------------
    def update(
        self,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,   # kept for API compatibility
        val_acc: Optional[float] = None,     # kept for API compatibility
    ) -> None:
        """Append epoch-level metrics (call once per epoch)."""
        if train_loss is not None:
            self.train_loss.append(float(train_loss))
        if val_loss is not None:
            self.val_loss.append(float(val_loss))

    def add_batch(self, loss: float, step: int) -> None:
        """Record a single batch's training loss."""
        self.batch_loss.append(float(loss))
        self.batch_step.append(int(step))

    def add_val_batch(self, loss: float, step: int) -> None:
        """Record an in-epoch validation probe at a given global batch step."""
        self.val_loss_batches.append(float(loss))
        self.val_loss_steps.append(int(step))

    def close_epoch(self) -> None:
        """Mark the end of the current epoch (for vertical separators in the plot)."""
        self.epoch_edges.append(len(self.batch_loss))
    
    def to_dict(self) -> dict:
        return {
            "model_info": self.model_info,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_loss_batches": self.val_loss_batches,
            "val_loss_steps": self.val_loss_steps,
            "batch_loss": self.batch_loss,
            "batch_step": self.batch_step,
            "epoch_edges": self.epoch_edges,
            "samples_seen": getattr(self, "samples_seen", 0),
            "lr_values": getattr(self, "lr_values", []),
            "lr_steps": getattr(self, "lr_steps", []),
            "carry_stage_steps":  [m["x_batch"]   for m in self.carry_stage_marks],
            "carry_stage_epochs": [m["epoch_idx"] for m in self.carry_stage_marks],
            "carry_stage_values": [m["carry"]     for m in self.carry_stage_marks],
        }

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        carry_steps = np.array([m["x_batch"] for m in self.carry_stage_marks], dtype=np.int64)
        carry_epochs = np.array([m["epoch_idx"] for m in self.carry_stage_marks], dtype=np.int64)
        carry_values = np.array([m["carry"] for m in self.carry_stage_marks], dtype=np.float64)

        np.savez_compressed(
            str(path),
            train_loss=np.array(self.train_loss, dtype=np.float64),
            val_loss=np.array(self.val_loss, dtype=np.float64),
            val_loss_batches=np.array(self.val_loss_batches, dtype=np.float64),
            val_loss_steps=np.array(self.val_loss_steps, dtype=np.int64),
            batch_loss=np.array(self.batch_loss, dtype=np.float64),
            batch_step=np.array(self.batch_step, dtype=np.int64),
            epoch_edges=np.array(self.epoch_edges, dtype=np.int64),

            # NEW: carry metadata
            carry_stage_steps=carry_steps,
            carry_stage_epochs=carry_epochs,
            carry_stage_values=carry_values,

            # keep this ONLY ONCE
            samples_seen=np.array([getattr(self, "samples_seen", 0)], dtype=np.int64),

            # (optional) LR traces
            lr_values=np.array(getattr(self, "lr_values", []), dtype=np.float64),
            lr_steps=np.array(getattr(self, "lr_steps", []), dtype=np.int64),
        )
        with open(path.with_suffix(".model.txt"), "w") as f:
            f.write(self.model_info)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def record_carry_stage(self, carry_years: float | int | None) -> None:
        if carry_years is None:
            return
        x_batch = int(self.batch_step[-1]) if self.batch_step else 0
        epoch_idx = len(self.train_loss)  # 0-based *completed* epochs at stage start
        if self.carry_stage_marks and \
        self.carry_stage_marks[-1]["x_batch"] == x_batch and \
        abs(float(self.carry_stage_marks[-1]["carry"]) - float(carry_years)) < 1e-9:
            return
        self.carry_stage_marks.append({
            "x_batch": x_batch,
            "epoch_idx": int(epoch_idx),
            "carry": float(carry_years),
        })
    
    def _draw_carry_stage_overlays(self, ax, *, coord: str = "epoch", top_axis_label: str = "Carry Length"):
        """Draw vertical lines and top axis for carry stages."""
        if not self.carry_stage_marks:
            return

        # choose x-positions
        xs = [m["epoch_idx"] if coord == "epoch" else m["x_batch"] for m in self.carry_stage_marks]
        if coord == "epoch":
            xs = [max(1, int(x)) for x in xs]

        # vertical dotted lines
        for x in xs:
            ax.axvline(x, color="gray", alpha=0.5, linestyle=":", linewidth=1.2)

        # top axis with carry labels
        labs = []
        for c in [m["carry"] for m in self.carry_stage_marks]:
            labs.append(f"{int(round(c))}" if abs(c - round(c)) < 1e-9 else f"{c:.3g}")

        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(FixedLocator(xs))
        ax_top.xaxis.set_major_formatter(FixedFormatter(labs))
        ax_top.set_xlabel(top_axis_label)

        ax_top.tick_params(axis='x', pad=6)  # optional: add space from axis label
        for label in ax_top.get_xticklabels():
            label.set_rotation(45)   # or 45
            label.set_ha("center")
            label.set_va("bottom")

    # -------------------------
    # Visualization
    # -------------------------
    def plot_batches(
        self,
        loss_type: str = "MSE",
        figsize: tuple[int, int] = (12, 5),
        save_dir: Optional[Path] = None,
        filename: str = "loss_epochs",
        show: bool = True,
    ) -> None:
        """
        Epoch-level loss plot:
        - Blue line: per-epoch averaged training loss (self.train_loss).
        - Orange line: per-epoch validation loss (self.val_loss), if present.
        """
        # Need at least epoch-level losses
        if not self.train_loss:
            return

        epochs = np.arange(1, len(self.train_loss) + 1, dtype=int)

        fig, ax = plt.subplots(figsize=figsize)

        # Train loss (per-epoch average)
        ax.plot(epochs, self.train_loss, linewidth=1.8, marker="o", label="Train loss (epoch avg)")

        # Val loss (per-epoch)
        if self.val_loss:
            n = min(len(self.val_loss), len(epochs))
            ax.plot(epochs[:n], np.asarray(self.val_loss[:n], dtype=float),
                    linewidth=1.8, marker="o", label="Val loss (epoch)")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Loss ({loss_type.upper()})")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")

        # CArry stage overlays
        self._draw_carry_stage_overlays(ax, coord="epoch", top_axis_label="Carry Length")
        
        # Save
        if save_dir is not None:
            save_path = Path(save_dir) / f"{filename}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show() if show else plt.close(fig)
            
    # Add this method to History
    def summarize_run(self, args, start_dt, elapsed_seconds: float) -> dict:
        """Package all run metadata + final metrics for plotting."""
        def _fmt_hms(s: float) -> str:
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            return f"{h:02d}:{m:02d}:{sec:02d}"

        # Flags
        early_stopping_on = bool(getattr(args, "early_stop", False))
        mass_balances_on  = bool(getattr(args, "use_mass_balances", False))

        # Mass balances list (with weights) only if enabled
        mb_list = []
        if mass_balances_on:
            if getattr(args, "water_balance_weight", 0) > 0:
                mb_list.append(f"ΔMRSO ({args.water_balance_weight:g})")
            if getattr(args, "npp_balance_weight", 0) > 0:
                mb_list.append(f"NPP ({args.npp_balance_weight:g})")
            if getattr(args, "nbp_balance_weight", 0) > 0:
                mb_list.append(f"NBP ({args.nbp_balance_weight:g})")
            if getattr(args, "carbon_partition_weight", 0) > 0:
                mb_list.append(f"cTotal=Veg+Litter+Soil ({args.carbon_partition_weight:g})")
            if getattr(args, "ctotal_mon_ann_weight", 0) > 0:
                mb_list.append(f"mean(cTotal_m)=cTotal_a ({args.ctotal_mon_ann_weight:g})")
            if getattr(args, "nbp_delta_ctotal_weight", 0) > 0:
                mb_list.append(f"ΔcTotal_m=NBP_m ({args.nbp_delta_ctotal_weight:g})")

        # Epochs summary
        epochs_done = len(self.train_loss)
        early_stopped = early_stopping_on and (epochs_done < getattr(args, "epochs", epochs_done))

        # Final losses
        final_train = self.train_loss[-1] if self.train_loss else float("nan")
        final_val   = self.val_loss[-1]   if self.val_loss   else float("nan")

        # Scheduler strings
        scheduler_name = "Cosine Annealing" if getattr(args, "scheduler", None) == "cosine_wr" else "None"
        if getattr(args, "scheduler", None) == "cosine_wr" and getattr(args, "n_cosine_cycles", None):
            cycles_str = f"{args.n_cosine_cycles} cycles over {args.epochs} epochs"
        elif getattr(args, "scheduler", None) == "cosine_wr":
            cycles_str = "planned over epochs"
        else:
            cycles_str = "N/A"

        return {
            # loss/info panel core fields (for plot_loss_with_info)
            "loss_type": getattr(args, "loss_type", "MSE"),
            "start_datetime_str": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "time_elapsed_str": _fmt_hms(elapsed_seconds),
            "epoch_done": epochs_done,
            "epoch_total": getattr(args, "epochs", epochs_done),
            "early_stopped": early_stopped,
            "samples_seen": getattr(self, "samples_seen", 0),

            # flags to control what to render
            "early_stopping_on": early_stopping_on,
            "mass_balances_on":  mass_balances_on,

            # configuration
            "learning_rate": getattr(args, "lr", None),
            "scheduler_name": scheduler_name,
            "scheduler_cycles_str": cycles_str,
            "scheduler_tmult": getattr(args, "sched_tmult", None),
            "scheduler_eta_min": getattr(args, "eta_min", None),
            "validation_frequency": getattr(args, "validation_frequency", None),
            "mass_balances": mb_list,  # already empty if mass_balances_on is False

            # early stop config (we’ll use only if early_stopping_on)
            "early_stop_patience": getattr(args, "early_stop_patience", None),
            "early_stop_min_delta": getattr(args, "early_stop_min_delta", None),
            "early_stop_warmup_epochs": getattr(args, "early_stop_warmup_epochs", None),

            # performance
            "final_train_loss": final_train,
            "final_val_loss": final_val,
        }
            
    def plot_loss_with_info(
        self,
        *,
        loss_type: str = "MSE",
        start_datetime_str: str,
        time_elapsed_str: str,
        epoch_done: int,
        epoch_total: int,
        early_stopped: bool,
        samples_seen: int,
        # --- flags ---
        early_stopping_on: bool,
        mass_balances_on: bool,
        # --- config ---
        learning_rate: float,
        scheduler_name: str,
        scheduler_cycles_str: str,
        scheduler_tmult: int | float | None = None,
        scheduler_eta_min: float | None = None,
        validation_frequency: float | str = "1.0",
        mass_balances: list[str] | None = None,
        early_stop_patience: int | None = None,
        early_stop_min_delta: float | None = None,
        early_stop_warmup_epochs: int | None = None,
        # --- performance ---
        final_train_loss: float | None = None,
        final_val_loss: float | None = None,
        # --- fig opts ---
        figsize: tuple[int, int] = (16, 6),
        save_dir: Optional[Path] = None,
        filename: str = "loss_and_info_epochs",
        show: bool = True,
        two_cols_right: bool = True,
    ) -> None:
        """
        Two-panel figure: left = epoch-level loss curves, right = run information summary.
        """
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, width_ratios=[2.2, 1.0], wspace=0.25, figure=fig)
        ax_loss = fig.add_subplot(gs[0, 0])

        # -------- Loss panel (epoch level) --------
        epochs = np.arange(1, len(self.train_loss) + 1, dtype=int)

        if self.train_loss:
            ax_loss.plot(epochs, self.train_loss,
                        linewidth=1.8, marker="o", color="blue", label="Train loss (epoch avg)")

        if self.val_loss:
            n = min(len(self.val_loss), len(epochs))
            ax_loss.plot(epochs[:n], np.asarray(self.val_loss[:n], dtype=float),
                        linewidth=1.8, marker="o", color="orange", label="Val loss (epoch)")

        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel(f"Loss ({loss_type.upper()})")
        ax_loss.grid(True, alpha=0.2)
        ax_loss.legend(loc="upper right")
        
        self._draw_carry_stage_overlays(ax_loss, coord="epoch", top_axis_label="Carry Length")

        # -------- Info panel --------
        if two_cols_right:
            sub = gs[0, 1].subgridspec(1, 2, wspace=0.15)
            ax_info_left  = fig.add_subplot(sub[0, 0])
            ax_info_right = fig.add_subplot(sub[0, 1])
            axes_info = [ax_info_left, ax_info_right]
        else:
            ax_info = fig.add_subplot(gs[0, 1])
            axes_info = [ax_info]

        for ax in axes_info:
            ax.axis("off")

        # Helper for text blocks
        def write_block(ax, title: str, items: list[str], y_start: float) -> float:
            base_fs = max(8, min(13, int(0.9 * figsize[1])))
            title_fs = base_fs + 1
            line_h = 0.06

            ax.text(0.0, y_start, title, fontsize=title_fs, fontweight="bold", va="top")
            y = y_start - line_h
            for s in items:
                ax.text(0.02, y, f"- {s}", fontsize=base_fs, va="top")
                y -= line_h
            return y - (line_h * 0.5)

        # Compose text
        epoch_str = f"{epoch_done} of {epoch_total} (early stopping)" if early_stopped else f"{epoch_done}"
        info_items = [
            f"Start: {start_datetime_str}",
            f"Elapsed: {time_elapsed_str}",
            f"Epoch: {epoch_str}",
            f"Samples seen: {samples_seen:,}",
        ]

        cfg_items = [
            f"Learning rate: {learning_rate:g}",
            f"Scheduler: {scheduler_name}",
            f"Scheduler cycles: {scheduler_cycles_str}",
            f"T_mult: {scheduler_tmult}",
            f"eta_min: {scheduler_eta_min}",
            f"Validation frequency: {validation_frequency}",
            f"Loss: {loss_type.upper()}",
            f"Early stopping: {'On' if early_stopping_on else 'Off'}",
        ]
        if early_stopping_on:
            cfg_items.extend([
                f"  Patience: {early_stop_patience}",
                f"  Min delta: {early_stop_min_delta}",
                f"  Warmup epochs: {early_stop_warmup_epochs}",
            ])

        cfg_items.append(f"Mass balances: {'On' if mass_balances_on else 'Off'}")
        if mass_balances_on:
            mb_list = mass_balances or []
            mb_str = "None" if len(mb_list) == 0 else ", ".join(mb_list)
            cfg_items.append(f"  Enabled: {mb_str}")

        perf_items = [
            f"Final train loss: {final_train_loss:.6f}" if final_train_loss is not None else "Final train loss: N/A",
            f"Final val loss: {final_val_loss:.6f}"     if final_val_loss   is not None else "Final val loss: N/A",
        ]

        if len(axes_info) == 1:
            y = 0.98
            y = write_block(axes_info[0], "Information",    info_items, y)
            y = write_block(axes_info[0], "Configuration",  cfg_items,  y)
            y = write_block(axes_info[0], "Performance",    perf_items, y)
        else:
            yL = 0.98
            yL = write_block(axes_info[0], "Information",   info_items, yL)
            yL = write_block(axes_info[0], "Configuration", cfg_items,  yL)

            yR = 0.98
            yR = write_block(axes_info[1], "Performance",   perf_items, yR)

        # Save and finish
        if save_dir is not None:
            save_path = Path(save_dir) / f"{filename}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show() if show else plt.close(fig)
        
    def save_epoch_plots_overwrite(self, run_dir: Path, args, start_dt, elapsed_seconds: float) -> None:
        """
        Overwrite rolling training plots each epoch. Each figure is guarded so one failure
        doesn't prevent the others from rendering.
        """
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")

        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        def _safe(fn, name: str):
            try:
                fn()
            except Exception as e:
                print(f"[plot warn] {name} failed: {e}")

        # 1) Per-batch loss curve
        _safe(
            lambda: self.plot_batches(
                loss_type=getattr(args, "loss_type", "mse"),
                save_dir=plots_dir,
                filename="loss_batches",
                show=False,
            ),
            "plot_batches",
        )

        # 3) Loss + info panel
        info = self.summarize_run(args, start_dt, elapsed_seconds)
        _safe(
            lambda: self.plot_loss_with_info(
                **info,
                figsize=(16, 6),
                save_dir=plots_dir,
                filename="loss_batches_with_info",
                show=False,
                two_cols_right=True,
            ),
            "plot_loss_with_info",
        )
        
    def save_test_results(
        self,
        run_dir,
        logger,
        test_out: dict,
        global_avg: float,
        best_val: float,
        world_size: int,
        is_main: bool = True,
        filename: str = "test_results.json",
    ):
        """
        Save and log test results (only on main rank), and stash them on History.

        Returns:
            Path | None: path to the written file on main, else None.
        """
        if not is_main:
            return None

        logger.info(
            "Test results: %d batches (per-rank), global_avg_loss=%.6f",
            test_out["num_batches"], global_avg
        )

        results = {
            "best_val_loss": best_val,
            "global_avg_test_loss": global_avg,
            "per_rank_num_batches": test_out["num_batches"],
            "world_size": world_size,
        }
        self.test_results = results

        path = Path(run_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Test results written to {path}")
        return path
