#!/usr/bin/env python3
"""
logs2plots_single.py

- Parse ONE training log.
- Produce ONE CSV with epoch-level train/val losses.
- Produce TWO plots:
    * Plain train vs val loss over epochs
    * Plot with a right-hand info panel summarizing config + performance
- Print the best epoch (lowest val).

Hardcode paths in main() at the bottom (or add argparse if you prefer).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import gzip
import io
import re
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.gridspec import GridSpec

# ---------------------------
# Regex patterns
# ---------------------------

EM_DASH = "\u2014"  # '—'

BATCHES_PER_GPU_RE = re.compile(
    rf"Dataloader batches per GPU\s*[{EM_DASH}-]\s*train=(?P<train>\d+),\s*val=(?P<val>\d+),\s*test=(?P<test>\d+)",
    re.IGNORECASE,
)

EPOCH_START_RE = re.compile(
    r"\[Epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\]",
    re.IGNORECASE,
)

PROGRESS_RE = re.compile(
    rf".*?Progress\s+.*?[{EM_DASH}-]\s*epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+),\s*batch\s+(?P<batch>\d+)\/(?P<batches>\d+),\s*avg_batch_loss=(?P<loss>[-+]?[\d.]+(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)

# Standalone "avg_batch_loss=0.193793"
STANDALONE_BATCH_LOSS_RE = re.compile(
    r"\bavg_batch_loss=(?P<loss>[-+]?[\d.]+(?:e[-+]?\d+)?)\b"
)

MID_VAL_RE = re.compile(
    r"\[Validation\s*@\s*batch\s*(?P<val_step>\d+)\]\s*avg_loss=(?P<val_loss>[-+]?[\d.]+(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)

PREFLIGHT_VAL_RE = re.compile(
    r"\[Pre-flight\s+validation\]\s*avg_loss=(?P<val_loss>[-+]?[\d.]+(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)

EPOCH_SUMMARY_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+)\s+\[INFO\]\s+Epoch average train loss="
    r"(?P<train>[-+]?[\d.]+(?:e[-+]?\d+)?),\s*val loss=(?P<val>[-+]?[\d.]+(?:e[-+]?\d+)?)"
    r"(?:\s*\(val_cnt=(?P<cnt>\d+)\))?",
    re.IGNORECASE,
)

def _open_maybe_gz(path: Path) -> io.TextIOBase:
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

@dataclass
class HistoryLite:
    epoch: List[int] = field(default_factory=list)
    train_loss_epoch: List[float] = field(default_factory=list)
    val_loss_epoch: List[float] = field(default_factory=list)
    val_cnt_epoch: List[Optional[int]] = field(default_factory=list)
    epoch_timestamps: List[pd.Timestamp] = field(default_factory=list)

    # Optional step-wise series (not plotted here but parsed for completeness)
    batch_step: List[int] = field(default_factory=list)
    batch_loss: List[float] = field(default_factory=list)
    val_loss_steps: List[int] = field(default_factory=list)
    val_loss_batches: List[float] = field(default_factory=list)

    epoch_edges: List[int] = field(default_factory=lambda: [0])
    steps_per_epoch: Optional[int] = None

    args: dict = field(default_factory=dict)
    start_dt: Optional[pd.Timestamp] = None
    end_dt: Optional[pd.Timestamp] = None

    val_epoch_steps: List[int] = field(default_factory=list)
    val_epoch_losses: List[float] = field(default_factory=list)

    def to_epoch_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "epoch": self.epoch,
            "timestamp": self.epoch_timestamps,
            "train_loss": self.train_loss_epoch,
            "val_loss": self.val_loss_epoch,
            "val_cnt": self.val_cnt_epoch,
        })

def _fmt_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def _summarize_args_for_info_panel(args: dict, df: pd.DataFrame) -> dict:
    start_ts = df["timestamp"].min()
    end_ts   = df["timestamp"].max()
    elapsed_seconds = float((end_ts - start_ts).total_seconds()) if pd.notna(start_ts) and pd.notna(end_ts) else 0.0

    def _fmt_hms(s: float) -> str:
        h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"

    early_stopping_on = bool(args.get("early_stop", False))
    mass_balances_on  = bool(args.get("use_mass_balances", False))

    mbl = []
    def add(key, label):
        w = _fmt_float(args.get(key, 0.0) or 0.0)
        if (w or 0.0) > 0:
            mbl.append(f"{label} ({w:g})")
    if mass_balances_on:
        add("water_balance_weight", "ΔMRSO")
        add("npp_balance_weight", "NPP")
        add("nbp_balance_weight", "NBP")
        add("carbon_partition_weight", "cTotal=Veg+Litter+Soil")
        add("ctotal_mon_ann_weight", "mean(cTotal_m)=cTotal_a")
        add("nbp_delta_ctotal_weight", "ΔcTotal_m=NBP_m")

    epochs_done = int(df["epoch"].max()) if len(df) else 0
    epochs_cfg  = int(args.get("epochs", epochs_done or 0))
    early_stopped = early_stopping_on and (epochs_done < epochs_cfg)

    final_train = float(df["train_loss"].iloc[-1]) if len(df) else None
    final_val   = float(df["val_loss"].iloc[-1])   if len(df) else None

    sched = args.get("scheduler", None)
    scheduler_name = "Cosine Annealing" if sched == "cosine_wr" else "None"
    if sched == "cosine_wr" and args.get("n_cosine_cycles") is not None:
        scheduler_cycles_str = f"{args['n_cosine_cycles']} cycles over {epochs_cfg} epochs"
    elif sched == "cosine_wr":
        scheduler_cycles_str = "planned over epochs"
    else:
        scheduler_cycles_str = "N/A"

    start_str = start_ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(start_ts, pd.Timestamp) else "N/A"

    return dict(
        loss_type=str(args.get("loss_type", "MSE")),
        start_datetime_str=start_str,
        time_elapsed_str=_fmt_hms(elapsed_seconds),
        epoch_done=epochs_done,
        epoch_total=epochs_cfg,
        early_stopped=early_stopped,
        samples_seen=int(args.get("samples_seen", 0)),
        early_stopping_on=early_stopping_on,
        mass_balances_on=mass_balances_on,
        learning_rate=_fmt_float(args.get("lr")),
        scheduler_name=scheduler_name,
        scheduler_cycles_str=scheduler_cycles_str,
        scheduler_tmult=args.get("sched_tmult"),
        scheduler_eta_min=_fmt_float(args.get("eta_min")),
        validation_frequency=args.get("validation_frequency", "1.0"),
        mass_balances=mbl,
        early_stop_patience=args.get("early_stop_patience"),
        early_stop_min_delta=_fmt_float(args.get("early_stop_min_delta")),
        early_stop_warmup_epochs=args.get("early_stop_warmup_epochs"),
        final_train_loss=final_train,
        final_val_loss=final_val,
    )

def plot_plain(df: pd.DataFrame, out_png: Path, title: str = "Train vs Val loss") -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["epoch"], df["train_loss"], label="Train_loss", linewidth=1.8, color="tab:blue")
    ax.plot(df["epoch"], df["val_loss"],   label="Val_loss",   linewidth=1.8, color="tab:orange")
    # best val
    if len(df) and df["val_loss"].notna().any():
        best_idx = df["val_loss"].idxmin()
        ax.scatter([df.loc[best_idx, "epoch"]], [df.loc[best_idx, "val_loss"]], s=70, zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_with_info(
    df: pd.DataFrame,
    info: dict,
    out_png: Path,
    title_left: str = "Train vs Val loss",
    two_cols_right: bool = True,
) -> None:
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 2, width_ratios=[2.2, 1.0], wspace=0.25, figure=fig)
    ax = fig.add_subplot(gs[0, 0])

    ax.plot(df["epoch"], df["train_loss"], label="Train_loss", linewidth=1.8, color="tab:blue")
    ax.plot(df["epoch"], df["val_loss"],   label="Val_loss",   linewidth=1.8, color="tab:orange")

    best_ep = best_val = None
    if len(df) and df["val_loss"].notna().any():
        best_idx = df["val_loss"].idxmin()
        best_ep  = int(df.loc[best_idx, "epoch"])
        best_val = float(df.loc[best_idx, "val_loss"])
        ax.scatter([best_ep], [best_val], s=70, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss ({str(info.get('loss_type','MSE')).upper()})")
    ax.set_title(title_left)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    # Right panel
    if two_cols_right:
        sub = gs[0, 1].subgridspec(1, 2, wspace=0.15)
        ax_info_left  = fig.add_subplot(sub[0, 0])
        ax_info_right = fig.add_subplot(sub[0, 1])
        axes_info = [ax_info_left, ax_info_right]
    else:
        ax_info = fig.add_subplot(gs[0, 1])
        axes_info = [ax_info]

    for ax_i in axes_info:
        ax_i.axis("off")

    def write_block(ax_i, title, items, y_start):
        base_fs  = 12
        title_fs = base_fs + 1
        line_h   = 0.06
        ax_i.text(0.0, y_start, title, fontsize=title_fs, fontweight="bold", va="top")
        y = y_start - line_h
        for s in items:
            ax_i.text(0.02, y, f"- {s}", fontsize=base_fs, va="top")
            y -= line_h
        return y - (line_h * 0.5)

    # Information
    start_str = str(info.get("start_datetime_str", "N/A"))
    start_date = start_str.split(" ")[0] if start_str and start_str != "N/A" else "N/A"
    ep_done  = info.get("epoch_done", 0)
    ep_total = info.get("epoch_total", ep_done)
    info_items = [
        f"Start: {start_date}",
        f"{ep_done} of {ep_total}",
        f"Samples seen: {int(info.get('samples_seen', 0)):,}",
        f"Time elapsed: {info.get('time_elapsed_str','N/A')}",
    ]

    # Configuration
    cfg_items = [
        f"Learning rate: {info['learning_rate'] if info.get('learning_rate') is not None else 'N/A'}",
        f"Scheduler: {info.get('scheduler_name', 'None')}",
        f"Scheduler cycles: {info.get('scheduler_cycles_str', 'N/A')}",
        f"T_mult: {info.get('scheduler_tmult')}",
        f"eta_min: {info.get('scheduler_eta_min')}",
        f"Validation frequency: {info.get('validation_frequency', '1.0')}",
        f"Loss: {str(info.get('loss_type','MSE')).upper()}",
        f"Early stopping: {'On' if info.get('early_stopping_on') else 'Off'}",
    ]
    if info.get("early_stopping_on"):
        cfg_items.extend([
            f"  Patience: {info.get('early_stop_patience')}",
            f"  Min delta: {info.get('early_stop_min_delta')}",
            f"  Warmup epochs: {info.get('early_stop_warmup_epochs')}",
        ])
    cfg_items.append(f"Mass balances: {'On' if info.get('mass_balances_on') else 'Off'}")
    if info.get("mass_balances_on"):
        mb = info.get("mass_balances") or []
        cfg_items.append(f"  Enabled: {', '.join(mb) if mb else 'None'}")

    # Performance
    final_train = info.get("final_train_loss")
    final_val   = info.get("final_val_loss")
    perf_items = [
        f"Final train loss: {final_train:.6f}" if final_train is not None else "Final train loss: N/A",
        f"Final val loss: {final_val:.6f}"     if final_val   is not None else "Final val loss: N/A",
    ]
    if best_ep is not None and best_val is not None:
        perf_items.append(f"Best val: {best_val:.6f} @ epoch {best_ep}")

    # Render
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

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def parse_args_yaml(path: Path) -> dict:
    if path is None or not Path(path).exists():
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return dict(data or {})

def parse_log_build_history(log_path: Path) -> Tuple["HistoryLite", pd.DataFrame]:
    hist = HistoryLite()

    epoch_rows: List[tuple] = []  # (ts, train, val, cnt)
    batch_steps: List[int] = []
    batch_losses: List[float] = []
    val_steps: List[int] = []
    val_losses: List[float] = []
    epoch_edges: List[int] = [0]

    steps_per_epoch: Optional[int] = None
    current_epoch: Optional[int] = None
    first_epoch_seen: Optional[int] = None

    tmp_epoch_batch_pairs: List[Tuple[int, int, float]] = []
    tmp_midval_pairs: List[Tuple[int, int, float]] = []

    start_ts: Optional[pd.Timestamp] = None
    end_ts: Optional[pd.Timestamp] = None

    with _open_maybe_gz(log_path) as f:
        for raw in f:
            line = raw.strip()

            if steps_per_epoch is None:
                m_bpg = BATCHES_PER_GPU_RE.search(line)
                if m_bpg:
                    steps_per_epoch = int(m_bpg.group("train"))

            m_start = EPOCH_START_RE.search(line)
            if m_start:
                current_epoch = int(m_start.group("epoch"))
                if first_epoch_seen is None:
                    first_epoch_seen = current_epoch

            m_prog = PROGRESS_RE.match(line)
            if m_prog:
                ep = int(m_prog.group("epoch"))
                ba = int(m_prog.group("batch"))
                loss = float(m_prog.group("loss"))
                tmp_epoch_batch_pairs.append((ep, ba, loss))
                continue

            m_standalone = STANDALONE_BATCH_LOSS_RE.search(line)
            if m_standalone and steps_per_epoch is not None and current_epoch is not None:
                ba = 1
                loss = float(m_standalone.group("loss"))
                tmp_epoch_batch_pairs.append((current_epoch, ba, loss))
                continue

            m_mid = MID_VAL_RE.search(line)
            if m_mid:
                step_within_epoch = int(m_mid.group("val_step"))
                val_loss = float(m_mid.group("val_loss"))
                ep = current_epoch if current_epoch is not None else (first_epoch_seen or 1)
                tmp_midval_pairs.append((ep, step_within_epoch, val_loss))
                continue

            m_pre = PREFLIGHT_VAL_RE.match(line)
            if m_pre:
                val_losses.append(float(m_pre.group("val_loss")))
                val_steps.append(0)
                continue

            m_epoch = EPOCH_SUMMARY_RE.match(line)
            if m_epoch:
                ts = pd.to_datetime(m_epoch.group("ts"), format="%Y-%m-%d %H:%M:%S,%f")
                if start_ts is None:
                    start_ts = ts
                end_ts = ts

                train = float(m_epoch.group("train"))
                val = float(m_epoch.group("val"))
                cnt = int(m_epoch.group("cnt")) if m_epoch.group("cnt") else None
                epoch_rows.append((ts, train, val, cnt))
                continue

    # Build epoch-level series
    n_epochs = len(epoch_rows)
    base = first_epoch_seen if first_epoch_seen is not None else 1
    epoch_numbers = list(range(base, base + n_epochs))

    for i, (ts, tr, va, cnt) in enumerate(epoch_rows):
        hist.epoch.append(epoch_numbers[i])
        hist.epoch_timestamps.append(ts)
        hist.train_loss_epoch.append(tr)
        hist.val_loss_epoch.append(va)
        hist.val_cnt_epoch.append(cnt)

    # Derive steps_per_epoch if not found
    if steps_per_epoch is None and tmp_epoch_batch_pairs:
        steps_per_epoch = max(b for (_, b, _) in tmp_epoch_batch_pairs)

    # Map (epoch, batch) -> global step
    if steps_per_epoch is not None and tmp_epoch_batch_pairs:
        tmp_epoch_batch_pairs.sort(key=lambda x: (x[0], x[1]))
        for (ep, b, loss) in tmp_epoch_batch_pairs:
            gstep = (ep - base) * steps_per_epoch + b
            batch_steps.append(gstep)
            batch_losses.append(loss)

        counts_per_epoch: dict[int, int] = {}
        for (ep, _, _) in tmp_epoch_batch_pairs:
            counts_per_epoch[ep] = counts_per_epoch.get(ep, 0) + 1

        cum = 0
        epoch_edges = [0]
        for ep in sorted(counts_per_epoch.keys()):
            cum += counts_per_epoch.get(ep, 0)
            epoch_edges.append(cum)

        for (ep, b, v) in tmp_midval_pairs:
            b = min(b, steps_per_epoch)
            g = (ep - base) * steps_per_epoch + b
            val_steps.append(g)
            val_losses.append(v)

        val_epoch_steps: List[int] = []
        val_epoch_losses: List[float] = []
        for i, ep in enumerate(epoch_numbers):
            end_step = (ep - base + 1) * steps_per_epoch
            val_epoch_steps.append(end_step)
            val_epoch_losses.append(hist.val_loss_epoch[i])

        hist.val_epoch_steps = val_epoch_steps
        hist.val_epoch_losses = val_epoch_losses

    # Attach
    hist.batch_step = batch_steps
    hist.batch_loss = batch_losses
    hist.val_loss_steps = val_steps
    hist.val_loss_batches = val_losses
    hist.epoch_edges = epoch_edges
    hist.steps_per_epoch = steps_per_epoch
    hist.start_dt = start_ts
    hist.end_dt = end_ts

    epoch_df = hist.to_epoch_df()
    return hist, epoch_df

# ---------------------------
# Single-run output
# ---------------------------

def main():
    # Hardcode your run here (or make these argparse params)
    run_dir = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/training/runs/2025-09-29/3534659_full_run_with_mrso")
    log_path  = run_dir / "logs/full_run_with_mrso.log"
    args_path = run_dir / "info/args.yaml"

    out_dir       = run_dir / "parsed"
    out_csv       = out_dir / "epoch_losses.csv"
    out_png_plain = out_dir / "train_val.png"
    out_png_info  = out_dir / "train_val_with_info.png"

    # Parse
    cfg = parse_args_yaml(args_path)
    hist, df = parse_log_build_history(log_path)
    hist.args = cfg

    # Save CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Plots
    plot_plain(df, out_png_plain)
    info = _summarize_args_for_info_panel(cfg, df)
    plot_with_info(df, info, out_png_info)

    # Best epoch
    best_idx = df["val_loss"].idxmin()
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val   = float(df.loc[best_idx, "val_loss"])

    print("[OK] CSV         ->", out_csv)
    print("[OK] Plain plot  ->", out_png_plain)
    print("[OK] Info plot   ->", out_png_info)
    print(f"[BEST] epoch {best_epoch} (val={best_val:.6f})")

if __name__ == "__main__":
    main()