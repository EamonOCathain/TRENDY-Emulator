from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from pathlib import Path
from typing import List, Optional
import json

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class History:
    """
    Lightweight training history tracker.

    Tracks:
      • Per-batch training loss (for detailed curves).
      • Optional in-epoch validation probes (scattered points).
      • Per-epoch averaged train/val loss.
      • Samples seen (for reporting) and LR schedule points.
    """

    def __init__(self, model) -> None:
        # Stringified snapshot of the model for provenance/debug dumps
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

        # Additional run metadata
        self.samples_seen: int = 0  # updated by training loop
        self.lr_values: List[float] = []
        self.lr_steps: List[int] = []
        
        # Per-epoch mass-balance (weighted) breakdowns
        self.mb_train: dict[str, list[float]] = {}  # name -> [epoch values]
        self.mb_val:   dict[str, list[float]] = {}

    # -----------------------------------------------------------------------
    # Recording helpers
    # -----------------------------------------------------------------------

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
        
    def add_mass_balance_epoch(self, train_avgs: dict[str, float], val_avgs: dict[str, float]) -> None:
        """
        Append epoch-level weighted MB averages. Missing keys are filled with NaN
        so all series line up across epochs.
        """
        # union of keys so lengths stay aligned
        keys = set(self.mb_train.keys()) | set(self.mb_val.keys()) | set(train_avgs.keys()) | set(val_avgs.keys())

        for k in keys:
            # train
            if k not in self.mb_train:
                self.mb_train[k] = []
            self.mb_train[k].append(float(train_avgs.get(k, float("nan"))))
            # val
            if k not in self.mb_val:
                self.mb_val[k] = []
            self.mb_val[k].append(float(val_avgs.get(k, float("nan"))))

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable snapshot of history and metadata."""
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
            "mb_train": self.mb_train,
            "mb_val":   self.mb_val,
        }

    def save_npz(self, path: Path) -> None:
        """
        Save a compact .npz archive with arrays for losses/steps,
        plus a companion text file with the stringified model ('.model.txt').
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            str(path),
            train_loss=np.array(self.train_loss, dtype=np.float64),
            val_loss=np.array(self.val_loss, dtype=np.float64),
            val_loss_batches=np.array(self.val_loss_batches, dtype=np.float64),
            val_loss_steps=np.array(self.val_loss_steps, dtype=np.int64),
            batch_loss=np.array(self.batch_loss, dtype=np.float64),
            batch_step=np.array(self.batch_step, dtype=np.int64),
            epoch_edges=np.array(self.epoch_edges, dtype=np.int64),
            # keep this ONLY ONCE
            samples_seen=np.array([getattr(self, "samples_seen", 0)], dtype=np.int64),
            # (optional) LR traces
            lr_values=np.array(getattr(self, "lr_values", []), dtype=np.float64),
            lr_steps=np.array(getattr(self, "lr_steps", []), dtype=np.int64),
            # mass-balance epoch series (store per-key arrays)
            **{f"mb_train__{k}": np.array(v, dtype=np.float64) for k, v in self.mb_train.items()},
            **{f"mb_val__{k}":   np.array(v, dtype=np.float64) for k, v in self.mb_val.items()},
        )
        with open(path.with_suffix(".model.txt"), "w") as f:
            f.write(self.model_info)

    def save_json(self, path: Path) -> None:
        """Save history snapshot as JSON (human-readable, richer but larger)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------

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
          • Blue line: per-epoch averaged training loss.
          • Orange line: per-epoch validation loss (if present).
        """
        if not self.train_loss:
            return

        epochs = np.arange(1, len(self.train_loss) + 1, dtype=int)

        fig, ax = plt.subplots(figsize=figsize)

        # Train loss (per-epoch average)
        ax.plot(epochs, self.train_loss, linewidth=1.8, marker="o", label="Train loss (epoch avg)")

        # Validation loss (per-epoch)
        if self.val_loss:
            n = min(len(self.val_loss), len(epochs))
            ax.plot(
                epochs[:n],
                np.asarray(self.val_loss[:n], dtype=float),
                linewidth=1.8,
                marker="o",
                label="Val loss (epoch)",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Loss ({loss_type.upper()})")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")

        # Save
        if save_dir is not None:
            save_path = Path(save_dir) / f"{filename}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show() if show else plt.close(fig)

    # -----------------------------------------------------------------------
    # Info/summary helpers
    # -----------------------------------------------------------------------

    def summarize_run(self, args, start_dt, elapsed_seconds: float) -> dict:
        """
        Package run metadata + final metrics for annotation panels/plots.
        Returns a dict compatible with `plot_loss_with_info(**info)`.
        """
        def _fmt_hms(s: float) -> str:
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            return f"{h:02d}:{m:02d}:{sec:02d}"

        # Flags reflecting CLI args
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

        # Scheduler strings (friendly names)
        scheduler_name = "Cosine Annealing" if getattr(args, "scheduler", None) == "cosine_wr" else "None"
        if getattr(args, "scheduler", None) == "cosine_wr" and getattr(args, "n_cosine_cycles", None):
            cycles_str = f"{args.n_cosine_cycles} cycles over {args.epochs} epochs"
        elif getattr(args, "scheduler", None) == "cosine_wr":
            cycles_str = "planned over epochs"
        else:
            cycles_str = "N/A"

        return {
            # Core fields used by plot_loss_with_info
            "loss_type": getattr(args, "loss_type", "MSE"),
            "start_datetime_str": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "time_elapsed_str": _fmt_hms(elapsed_seconds),
            "epoch_done": epochs_done,
            "epoch_total": getattr(args, "epochs", epochs_done),
            "early_stopped": early_stopped,
            "samples_seen": getattr(self, "samples_seen", 0),

            # Flags
            "early_stopping_on": early_stopping_on,
            "mass_balances_on":  mass_balances_on,

            # Configuration summary
            "learning_rate": getattr(args, "lr", None),
            "scheduler_name": scheduler_name,
            "scheduler_cycles_str": cycles_str,
            "scheduler_tmult": getattr(args, "sched_tmult", None),
            "scheduler_eta_min": getattr(args, "eta_min", None),
            "validation_frequency": getattr(args, "validation_frequency", None),
            "mass_balances": mb_list,  # already empty if mass_balances_on is False

            # Early stop config (used only if early_stopping_on)
            "early_stop_patience": getattr(args, "early_stop_patience", None),
            "early_stop_min_delta": getattr(args, "early_stop_min_delta", None),
            "early_stop_warmup_epochs": getattr(args, "early_stop_warmup_epochs", None),

            # Performance
            "final_train_loss": final_train,
            "final_val_loss": final_val,
        }

    def plot_loss_with_info(
        self,
        *,
        # Left panel (loss)
        loss_type: str = "MSE",
        # Summary values (produced by summarize_run)
        start_datetime_str: str,
        time_elapsed_str: str,
        epoch_done: int,
        epoch_total: int,
        early_stopped: bool,
        samples_seen: int,
        # Flags
        early_stopping_on: bool,
        mass_balances_on: bool,
        # Config
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
        # Performance
        final_train_loss: float | None = None,
        final_val_loss: float | None = None,
        # Figure opts
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

        # ---- Loss panel (epoch level) ----
        epochs = np.arange(1, len(self.train_loss) + 1, dtype=int)

        if self.train_loss:
            ax_loss.plot(
                epochs,
                self.train_loss,
                linewidth=1.8,
                marker="o",
                color="blue",
                label="Train loss (epoch avg)",
            )

        if self.val_loss:
            n = min(len(self.val_loss), len(epochs))
            ax_loss.plot(
                epochs[:n],
                np.asarray(self.val_loss[:n], dtype=float),
                linewidth=1.8,
                marker="o",
                color="orange",
                label="Val loss (epoch)",
            )

        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel(f"Loss ({loss_type.upper()})")
        ax_loss.grid(True, alpha=0.2)
        ax_loss.legend(loc="upper right")

        # ---- Info panel ----
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

        # Compose info text blocks
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

        # Lay out the text panels
        if len(axes_info) == 1:
            y = 0.98
            y = write_block(axes_info[0], "Information",   info_items, y)
            y = write_block(axes_info[0], "Configuration", cfg_items,  y)
            y = write_block(axes_info[0], "Performance",   perf_items, y)
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
        
    def plot_mb_breakdown(
        self,
        *,
        loss_type: str = "MSE",
        figsize: tuple[int, int] = (12, 6),
        save_dir: Optional[Path] = None,
        filename: str = "loss_mb_breakdown",
        show: bool = True,
    ) -> None:
        """
        Overlay per-epoch Train/Val loss with each mass-balance (weighted) average.
        Each MB gets its own color; legend distinguishes them.
        Only draws if MB data exist.
        """
        if not self.train_loss:
            return
        if not (self.mb_train or self.mb_val):
            return

        epochs = np.arange(1, len(self.train_loss) + 1, dtype=int)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)

        # Main losses
        ax.plot(epochs, self.train_loss, linewidth=2.0, marker="o", label="Train (epoch avg)")
        if self.val_loss:
            n = min(len(self.val_loss), len(epochs))
            ax.plot(epochs[:n], np.asarray(self.val_loss[:n], dtype=float),
                    linewidth=2.0, marker="o", label="Val (epoch)")

        # MB series (weighted contributions)
        # Use stable sorted key order for legend stability
        keys = sorted(set(self.mb_train.keys()) | set(self.mb_val.keys()))
        for k in keys:
            y = self.mb_train.get(k, [])
            if y:
                ax.plot(epochs[:len(y)], np.asarray(y, dtype=float), linewidth=1.6, label=f"{k} (train)")
            yv = self.mb_val.get(k, [])
            if yv:
                ax.plot(epochs[:len(yv)], np.asarray(yv, dtype=float), linewidth=1.6, linestyle="--", label=f"{k} (val)")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Loss ({loss_type.upper()})")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", ncol=1)

        if save_dir is not None:
            save_path = Path(save_dir) / f"{filename}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show() if show else plt.close(fig)

    # -----------------------------------------------------------------------
    # Rolling per-epoch plot writer
    # -----------------------------------------------------------------------

    def save_epoch_plots_overwrite(self, run_dir: Path, args, start_dt, elapsed_seconds: float) -> None:
        """
        Overwrite rolling training plots each epoch. Each figure render is guarded,
        so a failure in one doesn’t stop the others.
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

        # 1) Per-batch/epoch loss curve (epoch-level)
        _safe(
            lambda: self.plot_batches(
                loss_type=getattr(args, "loss_type", "mse"),
                save_dir=plots_dir,
                filename="loss_batches",
                show=False,
            ),
            "plot_batches",
        )

        # 2) Loss + info panel
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
        
        # 3) Mass-balance overlay (only if enabled and data present)
        if getattr(args, "use_mass_balances", False) and (self.mb_train or self.mb_val):
            _safe(
                lambda: self.plot_mb_breakdown(
                    loss_type=getattr(args, "loss_type", "mse"),
                    save_dir=plots_dir,
                    filename="loss_mb_breakdown",
                    show=False,
                ),
                "plot_mb_breakdown",
            )

    # -----------------------------------------------------------------------
    # Test results helper
    # -----------------------------------------------------------------------

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
          Path | None: path to the written file on main rank, else None.
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
        self.test_results = results  # keep for quick later access

        path = Path(run_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Test results written to {path}")
        return path