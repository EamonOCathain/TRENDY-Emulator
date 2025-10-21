
import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Torch runtime flags (determinism / precision)
# -----------------------------------------------------------------------------#
torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------#
# Project paths and environment bootstrap
# -----------------------------------------------------------------------------#
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

# Run start timestamp (string form used in downstream metadata/logging)
start_dt = datetime.now()
start_dt_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------------------------------------------------------#
# Local imports (after sys.path updated)
# -----------------------------------------------------------------------------#
from src.paths.paths import *                      # noqa: F401,F403
from src.dataset.variables import *                # noqa: F401,F403

from src.models.custom_transformer import YearProcessor
from src.training.checkpoints import save_cb, extract_state_dict_for_foundation
from src.training.history import History
from src.training.loss import build_loss_fn
from src.training.trainer import fit, plan_validation
from src.dataset.dataloader import get_train_val_test, get_data
from src.training.distributed import init_distributed
from src.training.logging import save_args, setup_logging
from src.training.scheduler import build_cosine_wr_scheduler
from src.training.stats import get_split_stats, set_seed, load_and_filter_standardisation
from src.dataset.dataset import base, get_subset
from src.training.tester import (
    run_and_save_test_suite,
    run_and_save_metrics_csv,
    run_and_save_scatter_grids,
)
from src.training.varschema import VarSchema

# -----------------------------------------------------------------------------#
# Global run configuration
# -----------------------------------------------------------------------------#
training_dir = train_pipeline_dir  # Overwrite training dir root from paths module

# SLURM resources
workers = max(1, int(os.getenv("SLURM_CPUS_PER_TASK", "4")))
slurm_id = os.getenv("SLURM_JOB_ID", "no_slurm_id")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # --- Training loop ---
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .pt (epoch*.pt or best.pt) to resume training")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--mb_size", type=int, default=1,  # windows per microbatch
                        help="Windows per microbatch")
    parser.add_argument("--accum_steps", type=int, default=None,
                        help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=None,
                        help="Gradient norm clip (None disables)")
    parser.add_argument("--shuffle_windows", action="store_true",
                        help="Shuffle windows in the dataloader (disabled if carry>0)")
    parser.add_argument("--subset_frac", type=float, default=None,
                        help="Subsample fraction for train/val splits")
    parser.add_argument("--exclude_vars", nargs="*", default=[],
                        help="Variable names to exclude from inputs")
    parser.add_argument("--use_foundation", type=str, default=None,
                        help=("Path to checkpoint to initialize model weights from "
                              "(optimizer/scheduler/history/epoch are ignored)"))
    parser.add_argument("--train_only", action="store_true",
                        help="Skip testing after training.")
    parser.add_argument("--test_only", action="store_true",
                        help="Run only the testing suite (skip training/validation).")

    # --- Optimiser & scheduler ---
    parser.add_argument("--lr", type=float, default=9e-5)
    parser.add_argument("--scheduler", type=str, default="cosine_wr",
                        choices=["none", "cosine_wr"])
    parser.add_argument("--sched_t0", type=str, default="epoch",
                        help='Cosine WR T0. Use "epoch" to match steps/epoch, or provide an integer (steps).')
    parser.add_argument("--sched_tmult", type=int, default=2)
    parser.add_argument("--eta_min", type=float, default=9e-6)

    # --- Validation ---
    parser.add_argument("--val_freq", type=float, default=1.0,
                        help="Validate every N epochs (can be fractional)")
    parser.add_argument("--val_frac", type=float, default=1.0,
                        help="Fraction of validation loader to use per validation pass")

    # --- Testing ---
    parser.add_argument("--test_frac", type=float, default=None,
                        help=("Subset the TEST set by a fraction RELATIVE to --subset_frac. "
                              "Effective test fraction = (subset_frac if set else 1.0) * (test_frac if set else 1.0)."))

    # --- Data loading ---
    parser.add_argument("--num_workers", type=int, default=8)

    # --- Misc / logging ---
    parser.add_argument("--job_name", type=str, default="transformer_run")
    parser.add_argument("--log_batches_per_rank", action="store_true")

    # --- Loss ---
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "mae"])
    parser.add_argument("--use_mass_balances", action="store_true",
                        help="Enable mass-balance penalties in the loss (off by default)")
    parser.add_argument("--water_balance_weight", type=float, default=0.0)
    parser.add_argument("--npp_balance_weight", type=float, default=0.0)
    parser.add_argument("--nbp_balance_weight", type=float, default=0.0)
    parser.add_argument("--carbon_partition_weight", type=float, default=0.0)
    parser.add_argument("--nbp_delta_ctotal_weight", type=float, default=0.0)

    # --- Early Stopping ---
    parser.add_argument("--early_stop", action="store_true",
                        help="Enable early stopping (off by default)")
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--early_stop_warmup_epochs", type=int, default=0)

    # --- Checkpointing ---
    parser.add_argument("--ckpt_every_epochs", type=int, default=1,
                        help="Save rolling checkpoint every N epochs (0=disable)")
    parser.add_argument("--keep_last", type=int, default=5,
                        help="Keep last K rolling checkpoints")

    # --- Carry years ---
    parser.add_argument("--carry_years", type=str, default="0",
                        help='Carry horizon across years. Single numeric value: "0", a float ("2.5"), or "3/12".')
    parser.add_argument("--carry_granularity", type=str, default="annual", choices=["monthly", "annual"],
                        help=("Carry coupling across years: 'monthly' (last-month state to next Jan) "
                              "or 'annual' (annual-mean state broadcast across all days of next year)."))

    # --- Other ---
    parser.add_argument("--scan_finite", action="store_true",
                        help="Scan datasets for non-finite values before training (warns if any found)")

    return parser.parse_args()


# -----------------------------------------------------------------------------#
# Parse + top-level guards
# -----------------------------------------------------------------------------#
args = parse_args()

if args.resume and args.use_foundation:
    raise SystemExit("Use either --resume or --use_foundation, not both.")

if args.train_only and args.test_only:
    raise SystemExit("Cannot use --train_only and --test_only together.")


def _check_frac(name, val):
    """Guard helper for fraction-type flags."""
    if val is None:
        return
    if not (0.0 < float(val) <= 1.0):
        raise SystemExit(f"--{name} must be in (0,1], got {val}")


_check_frac("subset_frac", args.subset_frac)
_check_frac("test_frac", args.test_frac)

# =============================================================================
# Carry-years normalization (single value only)
# =============================================================================
def _to_float(token: str) -> float:
    """Parse a numeric or fractional string (e.g., '3/12') into a float."""
    token = token.strip().lower()
    if "/" in token:
        num, den = token.split("/", 1)
        return float(num) / float(den)
    return float(token)


try:
    carry_value = _to_float(args.carry_years)
except Exception as e:
    raise SystemExit(f"Invalid --carry_years value '{args.carry_years}': {e}")

if carry_value < 0:
    raise SystemExit("--carry_years must be >= 0")

# In annual mode: clamp 0 < carry < 1 to 1.0 (keep exact 0.0)
if args.carry_granularity == "annual" and 0.0 < carry_value < 1.0:
    logging.getLogger(args.job_name).warning(
        f"[carry] annual granularity: clamped carry years {carry_value} -> 1.0 (no fractional carry)"
    )
    carry_value = 1.0

# If any positive carry is used, disable window shuffling
if carry_value > 0.0 and getattr(args, "shuffle_windows", False):
    args.shuffle_windows = False


# =============================================================================
# Utility: non-finite scan (optional)
# =============================================================================
@torch.no_grad()
def scan_for_nonfinite(dl, max_batches=None):
    """
    Iterate over a dataloader (optionally capped) and report if any tensor contains non-finite values.
    Returns (idx, (x_ok, m_ok, a_ok)) or (None, (True, True, True)) if all good.
    """
    for bi, (x, m, a) in enumerate(dl):
        if max_batches and bi >= max_batches:
            break
        bad = (not torch.isfinite(x).all()
               or not torch.isfinite(m).all()
               or not torch.isfinite(a).all())
        if bad:
            return bi, (
                torch.isfinite(x).all().item(),
                torch.isfinite(m).all().item(),
                torch.isfinite(a).all().item(),
            )
    return None, (True, True, True)

# =============================================================================
# Load and filter with standardisation stats
# =============================================================================
# This removes any variables which have S.D or mean 0

std_dict, pruned = load_and_filter_standardisation(
    standardisation_path= std_dict_path,
    all_vars=all_vars,
    daily_vars=daily_vars,
    monthly_vars=monthly_vars,
    annual_vars=annual_vars,
    monthly_states=monthly_states,
    annual_states=annual_states,
    exclude_vars=set(getattr(args, "exclude_vars", [])),
)

# Unpack back into variable lists
all_vars       = pruned["all_vars"]
daily_vars     = pruned["daily_vars"]
monthly_vars   = pruned["monthly_vars"]
annual_vars    = pruned["annual_vars"]
monthly_states = pruned["monthly_states"]
annual_states  = pruned["annual_states"]


# =============================================================================
# Main
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    set_seed(42)

    # -------------------------------------------------------------------------
    # Distributed init & device selection
    # -------------------------------------------------------------------------
    global DEVICE
    ddp, device, LOCAL_RANK, WORLD_SIZE, RANK = init_distributed()
    DEVICE = device
    is_main = (RANK == 0)

    # Per-rank worker count (respect SLURM CPU allocation)
    cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "4"))
    workers_per_rank = min(8, cpus)

    # -------------------------------------------------------------------------
    # Run/output directories & logging
    # -------------------------------------------------------------------------
    today = datetime.now().strftime("%Y-%m-%d")
    run_leaf = f"{slurm_id}_{args.job_name}"
    run_dir = training_dir / "runs" / today / run_leaf

    # Ensure expected subfolders exist
    (run_dir / "saves").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "info").mkdir(parents=True, exist_ok=True)

    # Logger: main rank emits, others stay quiet (no propagation to root)
    log = setup_logging(run_dir, args.job_name) if is_main else logging.getLogger(args.job_name)
    if not is_main:
        log.propagate = False

    # Gate early stopping flags if disabled
    if not args.early_stop:
        args.early_stop_patience = None
        args.early_stop_min_delta = 0.0
        args.early_stop_warmup_epochs = 0

    # Gate mass-balance penalties if disabled
    if not args.use_mass_balances:
        args.water_balance_weight = 0.0
        args.npp_balance_weight = 0.0
        args.nbp_balance_weight = 0.0
        args.carbon_partition_weight = 0.0
        args.nbp_delta_ctotal_weight = 0.0

    if is_main:
        log.info(f"Early stopping: {'ON' if args.early_stop else 'OFF'}")
        log.info(f"Mass balances: {'ON' if args.use_mass_balances else 'OFF'}")
        log.info(f"Using device: {DEVICE} | world_size={WORLD_SIZE} rank={RANK} local_rank={LOCAL_RANK}")

        info_file = save_args(run_dir, args)
        log.info(f"Saved run arguments to {info_file}")

    # -------------------------------------------------------------------------
    # Dataset loading (train/val/test splits)
    # -------------------------------------------------------------------------
    ds_dict = get_train_val_test(std_dict, block_locs=70, carry_years=carry_value)

    if is_main:
        print("Number of location chunks in full dataset:",
              len(ds_dict["train"]), "val len:", len(ds_dict["val"]), "test len:", len(ds_dict["test"]))

    # Optional subsetting of train/val
    if args.subset_frac is None:
        train_ds = ds_dict["train"]
        val_ds = ds_dict["val"]
    else:
        train_ds = get_subset(ds_dict["train"], frac=args.subset_frac, seed=42)
        val_ds = get_subset(ds_dict["val"], frac=args.subset_frac, seed=1337)

    # Optional subsetting of test (multiplicative with subset_frac)
    base_test_frac = args.subset_frac if args.subset_frac is not None else 1.0
    mult_test_frac = args.test_frac if args.test_frac is not None else 1.0
    effective_t_frac = float(base_test_frac) * float(mult_test_frac)
    if not (0.0 < effective_t_frac <= 1.0):
        raise SystemExit(f"Effective test fraction must be in (0,1], got {effective_t_frac}")
    test_ds = get_subset(ds_dict["test"], frac=effective_t_frac, seed=999) if effective_t_frac < 1.0 else ds_dict["test"]

    if is_main:
        print("Datasets after subsetting:",
              "train", len(train_ds), "val", len(val_ds),
              f"test {len(test_ds)} (effective_test_frac={effective_t_frac})")

    # -------------------------------------------------------------------------
    # Dataloaders
    # -------------------------------------------------------------------------
    train_dl, valid_dl, test_dl = get_data(
        train_ds, val_ds, test_ds,
        bs=1,
        num_workers=(args.num_workers if args.num_workers is not None else workers_per_rank),
        ddp=ddp,
    )

    # Optional non-finite scan across splits
    if args.scan_finite and is_main:
        log.info("[preflight] scanning test data for non-finite values in first 5000 test batches...")
        bi, fins = scan_for_nonfinite(test_dl, max_batches=None)
        if bi is not None:
            log.warning(f"[preflight] first non-finite at test batch {bi} "
                        f"(x={fins[0]}, m={fins[1]}, a={fins[2]})")
        else:
            log.info("[preflight] no non-finite values detected in test batches.")

        log.info("[preflight] scanning train/val/test data for non-finite values...")
        for name, dl in [("train", train_dl), ("val", valid_dl), ("test", test_dl)]:
            bi, fins = scan_for_nonfinite(dl, max_batches=None)
            if bi is not None:
                log.warning(f"[preflight/{name}] first non-finite at batch {bi} "
                            f"(x={fins[0]}, m={fins[1]}, a={fins[2]})")
            else:
                log.info(f"[preflight/{name}] no non-finite in batches")

    # Per-rank dataloader stats (handy for debugging DDP)
    if args.log_batches_per_rank and ddp and dist.is_initialized():
        dist.barrier()
        log.info(f"[Rank {RANK}] train_ds={len(train_ds)}, val_ds={len(val_ds)}, test_ds={len(test_ds)}")
        log.info(f"[Rank {RANK}] train_batches={len(train_dl)}, val_batches={len(valid_dl)}, test_batches={len(test_dl)}")
        dist.barrier()
    else:
        if is_main:
            log.info(f"Dataset sizes — train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
            log.info(f"Dataloader batches per GPU — train={len(train_dl)}, val={len(valid_dl)}, test={len(test_dl)}")

    # -------------------------------------------------------------------------
    # Split stats & validation planning
    # -------------------------------------------------------------------------
    stats = get_split_stats(train_dl, valid_dl, test_dl, accum_steps=args.accum_steps)
    if is_main:
        log.info(
            "Data stats – train: %d batches, steps/epoch=%d (eff_accum=%d); val: %d; test: %d",
            stats['train']['batches'], stats['train']['steps_per_epoch'], stats['train']['eff_accum'],
            stats['val']['batches'], stats['test']['batches']
        )

    val_plan = plan_validation(
        train_stats=stats['train'],
        valid_dl=valid_dl,
        validation_frequency=args.val_freq,
        validation_size=args.val_frac,
    )
    if is_main:
        log.info(
            "Validation plan – validate_every_batches=%d, val_batches_to_use=%d (train_batches=%d; val_total_batches=%d)",
            val_plan["validate_every_batches"], val_plan["val_batches_to_use"],
            val_plan["train_batches"], val_plan["val_total_batches"]
        )
        if "fixed_val_batch_ids" in val_plan:
            log.info("Fixed val batch ids (first 10): %s", val_plan["fixed_val_batch_ids"][:10])

    # -------------------------------------------------------------------------
    # Schema + model I/O mapping (canonicalized from dataset)
    # -------------------------------------------------------------------------
    base_train = base(train_ds)
    if hasattr(base_train, "schema"):
        schema = base_train.schema  # dataset-provided schema (preferred)
    else:
        # Legacy path: synthesize schema from dataset var_names
        schema = VarSchema(
            daily_forcing=sorted(base_train.var_names['daily_forcing']),
            monthly_forcing=sorted(base_train.var_names['monthly_forcing']),
            monthly_states=sorted(base_train.var_names['monthly_states']),
            annual_forcing=sorted(base_train.var_names['annual_forcing']),
            annual_states=sorted(base_train.var_names['annual_states']),
            monthly_fluxes=sorted(base_train.var_names['monthly_fluxes']),
        )

    INPUT_ORDER = schema.input_order()
    OUTPUT_ORDER = schema.output_order()
    dims = schema.dims()
    input_dim = dims["input_dim"]
    output_dim = dims["output_dim"]
    schema_sig = schema.signature()
    schema_dims = dims

    # Minimal sanity
    assert len(INPUT_ORDER) > 0 and len(INPUT_ORDER) == len(set(INPUT_ORDER)), "INPUT_ORDER empty or has duplicates"
    assert len(OUTPUT_ORDER) > 0 and len(OUTPUT_ORDER) == len(set(OUTPUT_ORDER)), "OUTPUT_ORDER empty or has duplicates"

    # Names per head (used for loss/metrics)
    monthly_names = schema.out_monthly_names()  # monthly head: fluxes + states
    annual_names = schema.out_annual_names()    # annual head: states
    output_names = monthly_names + annual_names

    # Indices for loss heads (local to the combined vector)
    idx_monthly = list(range(len(monthly_names)))
    idx_annual = list(range(len(monthly_names), len(output_names)))

    # For balance penalties & diagnostics: name -> combined-index
    out_idx = {name: i for i, name in enumerate(output_names)}

    # Stable snapshot for checkpoint metadata
    if hasattr(base_train, "var_names"):
        varnames_snapshot = {k: sorted(list(v)) for k, v in base_train.var_names.items()}
    else:
        varnames_snapshot = {
            "daily_forcing": list(schema.daily_forcing),
            "monthly_forcing": list(schema.monthly_forcing),
            "monthly_states": list(schema.monthly_states),
            "annual_forcing": list(schema.annual_forcing),
            "annual_states": list(schema.annual_states),
            "monthly_fluxes": list(schema.monthly_fluxes),
        }

    # -------------------------------------------------------------------------
    # Optional mass-balance variable index mapping
    # -------------------------------------------------------------------------
    mb_var_idx = None
    if args.use_mass_balances:
        needed_vars = [
            "mrso", "pre", "mrro", "evapotrans",
            "npp", "gpp", "ra",
            "nbp", "rh", "fFire", "fLuc",
            "cTotal_monthly", "cTotal_annual", "cVeg", "cLitter", "cSoil"
        ]
        mb_var_idx = {}
        missing = []
        for v in needed_vars:
            if v in out_idx:
                mb_var_idx[v] = out_idx[v]
            else:
                missing.append(v)
        if missing and is_main:
            print("[mass-balance] missing variables:", missing)

    # -------------------------------------------------------------------------
    # Rollout configuration (carry behavior + calendar)
    # -------------------------------------------------------------------------
    rollout_cfg = {
        "in_monthly_state_idx": schema.in_monthly_state_idx(),
        "in_annual_state_idx": schema.in_annual_state_idx(),
        "out_monthly_state_idx": schema.out_monthly_state_idx_local(),
        "out_annual_state_idx": schema.out_annual_state_idx_local(),
        "out_monthly_names": monthly_names,
        "out_annual_names": annual_names,
        "month_lengths": schema.month_lengths,
        "carry_horizon": float(carry_value),
        "carry_granularity": str(args.carry_granularity),
        "output_order": OUTPUT_ORDER,
        "schema_sig": schema_sig,
    }

    if is_main:
        nm = len(rollout_cfg["out_monthly_names"])
        na = len(rollout_cfg["out_annual_names"])
        log.info(f"[rollout_cfg] nm={nm} monthly names, na={na} annual names")
        log.info(f"[rollout_cfg] out_monthly_state_idx (local) = {rollout_cfg['out_monthly_state_idx']}")
        log.info(f"[rollout_cfg] out_annual_state_idx  (local) = {rollout_cfg['out_annual_state_idx']}")
        log.info(f"[schema] sig={schema.signature()} | input_dim={input_dim} | output_dim={output_dim} "
                 f"| nm={nm} | na={na}")
    
    # -------------------------------------------------------------------------
    # Loss function (base + optional balance penalties)
    # -------------------------------------------------------------------------
    # Set the variable weights to 1 for all vars, can be customized later
    monthly_weights = [1.0 for _ in monthly_names]
    annual_weights  = [1.0 for _ in annual_names]
    
    # Build a list of mu/sd to pass for output denormalization
    mu_out = [float(std_dict.get(name, {}).get("mean", 0.0)) for name in output_names]
    sd_out = [float(std_dict.get(name, {}).get("std",  1.0))  for name in output_names]
    
    
    loss_fn = build_loss_fn(
        idx_monthly=idx_monthly,
        idx_annual=idx_annual,
        use_mass_balances=args.use_mass_balances,  # from your CLI
        loss_type=args.loss_type,
        monthly_weights=monthly_weights,
        annual_weights=annual_weights,
        # the following only matter if use_mass_balances=True:
        mb_var_idx=mb_var_idx,
        water_balance_weight=args.water_balance_weight,
        npp_balance_weight=args.npp_balance_weight,
        nbp_balance_weight=args.nbp_balance_weight,
        nbp_delta_ctotal_weight=args.nbp_delta_ctotal_weight,
        carbon_partition_weight=args.carbon_partition_weight,
        mu_out=mu_out, sd_out=sd_out,
    )

    # -------------------------------------------------------------------------
    # Model + optional DDP wrapper
    # -------------------------------------------------------------------------
    model = YearProcessor(
        input_dim=input_dim,
        output_dim=output_dim,
        in_monthly_state_idx=rollout_cfg["in_monthly_state_idx"],
        out_monthly_state_idx=rollout_cfg["out_monthly_state_idx"],
        month_lengths=rollout_cfg["month_lengths"],
        d=128, h=1024, g=256, num_layers=4, nhead=8, dropout=0.1,
        transformer_kwargs={"max_len": 31},
        mode="batch_months",
    ).float().to(DEVICE)

    def _ckpt_io_dims(ckpt: dict) -> tuple[int, int]:
        """Helper to read io dims from a training-style checkpoint (optional)."""
        return int(ckpt.get("input_dim", -1)), int(ckpt.get("output_dim", -1))

    # Optional: initialize from foundation checkpoint (weights only)
    if args.use_foundation:
        if is_main:
            print(f"[FOUNDATION] Loading foundation weights from {args.use_foundation}")

        ckpt_f = torch.load(args.use_foundation, map_location="cpu", weights_only=True)

        # Accept either a full state_dict or a wrapped training checkpoint dict
        if isinstance(ckpt_f, dict) and any(k.startswith(("module.", "inner.")) for k in ckpt_f.keys()):
            sd_f = ckpt_f
        else:
            try:
                sd_f = extract_state_dict_for_foundation(ckpt_f)
            except Exception as e:
                raise SystemExit(f"Failed to read foundation checkpoint: {e}")

        f_in, f_out = _ckpt_io_dims(ckpt_f)
        if f_in and f_in != input_dim:
            print(f"[FOUNDATION][WARN] input_dim mismatch: ckpt={f_in}, model={input_dim}")
        if f_out and f_out != output_dim:
            print(f"[FOUNDATION][WARN] output_dim mismatch: ckpt={f_out}, model={output_dim}")

        target = model.module if isinstance(model, DDP) else model
        missing, unexpected = target.load_state_dict(sd_f, strict=False)
        if is_main:
            msg = f"Loaded foundation weights from {args.use_foundation}"
            if missing or unexpected:
                msg += f" (missing={len(missing)}, unexpected={len(unexpected)})"
            print(msg)

    # DDP wrapping (after any weight loading)
    if ddp:
        model = DDP(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            find_unused_parameters=False,
        )

    # -------------------------------------------------------------------------
    # Optimizer & LR scheduler
    # -------------------------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler, sched_info = build_cosine_wr_scheduler(args, opt, stats['train'], log=log if is_main else None)
    if sched_info and is_main:
        info_clean = {k: v for k, v in sched_info.items() if k != "total_windows"}
        log.info(f"scheduler_info: {info_clean}")

    # -------------------------------------------------------------------------
    # Checkpointing helpers
    # -------------------------------------------------------------------------
    real_model = model.module if isinstance(model, DDP) else model

    base_save_cb = (
        lambda epoch, best, val, history: save_cb(
            epoch, best, val, history,
            args=args, run_dir=run_dir, model=real_model, opt=opt, scheduler=scheduler,
            input_dim=input_dim, output_dim=output_dim,
            input_order=INPUT_ORDER, output_order=OUTPUT_ORDER,
            var_names_snapshot=varnames_snapshot,
            schema_sig=schema_sig,
            schema_dims=schema_dims,
        )
    )

    def save_cb_main(epoch, best, val, history):
        """Main-rank-only save callback wrapper."""
        if not is_main:
            return
        base_save_cb(epoch, best, val, history)

    # GPU memory tracking (peaks) — best to reset before training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

    # -------------------------------------------------------------------------
    # Optional: resume full training state (model/opt/scheduler/history)
    # -------------------------------------------------------------------------
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

        real_model = model.module if isinstance(model, DDP) else model
        real_model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        if ckpt.get("sched_state") is not None and scheduler is not None:
            scheduler.load_state_dict(ckpt["sched_state"])

        hist_ckpt = ckpt.get("history", {}) or {}

        early_state_prev = ckpt.get("early_state", None)
        if early_state_prev is not None:
            hist_ckpt["_early_state"] = early_state_prev

        raw_epoch = int(ckpt.get("epoch", 0))
        start_epoch = 1 if (raw_epoch == 0 and len(hist_ckpt.get("train_loss", [])) > 0) else raw_epoch
        best_val_so_far = float(ckpt.get("best_val", float("inf")))
        samples_seen_prev = int(hist_ckpt.get("samples_seen", 0) or 0)
    else:
        start_epoch = 0
        best_val_so_far = float("inf")
        samples_seen_prev = 0
        hist_ckpt = {}

    resume_payload = {
        "start_epoch": start_epoch,
        "best_val_init": best_val_so_far,
        "history_seed": hist_ckpt,
        "samples_seen_seed": samples_seen_prev,
    }
    resume_pending = args.resume is not None

    history = None
    best_val = float("inf")

    # Best checkpoint reloader (used before test/plots)
    BEST_PATH = run_dir / "checkpoints" / "best.pt"

    def reload_best_weights() -> bool:
        """Load checkpoints/best.pt into `model` if present. Returns True if loaded."""
        try:
            if not BEST_PATH.exists():
                return False
            ck = torch.load(BEST_PATH, map_location="cpu", weights_only=True)
            state = ck.get("model_state", ck) if isinstance(ck, dict) else ck
            target = model.module if isinstance(model, DDP) else model
            missing, unexpected = target.load_state_dict(state, strict=False)
            if is_main:
                msg = "Reloaded best weights"
                if missing or unexpected:
                    msg += f" (missing={len(missing)}, unexpected={len(unexpected)})"
                log.info(msg)
            return True
        except Exception as e:
            if is_main:
                log.warning(f"Failed to reload best weights: {e}")
            return False

    history = None
    best_val = float("inf")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    try:
        if not args.test_only:
            
            # We define a wrapper to run a single carry years stage - if we wanted mutliple stages in a single run this would make it easier later
            def run_one_stage(carry_val: float):
                """
                Single-stage training wrapper.
                Applies (optional) resume payload and forwards to fit(...).
                """
                nonlocal resume_pending, resume_payload
                rollout_cfg["carry_horizon"] = float(carry_val)

                if resume_pending:
                    se = resume_payload.get("start_epoch", 0)
                    bvi = resume_payload.get("best_val_init", float("inf"))
                    hs = resume_payload.get("history_seed", None)
                    sss = resume_payload.get("samples_seen_seed", 0)
                    resume_pending = False
                else:
                    se, bvi, hs, sss = 0, float("inf"), None, 0

                # Main Training Core Function
                return fit(
                    args.epochs, model, loss_fn, opt, train_dl, valid_dl,
                    log=(log if is_main else None),
                    save_cb=save_cb_main,
                    accum_steps=args.accum_steps,
                    grad_clip=args.grad_clip,
                    scheduler=scheduler,
                    val_plan=val_plan,
                    mb_size=args.mb_size,
                    ddp=ddp,
                    early_stop_patience=args.early_stop_patience if args.early_stop else None,
                    early_stop_min_delta=args.early_stop_min_delta,
                    early_stop_warmup_epochs=args.early_stop_warmup_epochs,
                    start_dt=start_dt,
                    run_dir=run_dir,
                    args=args,
                    start_epoch=se,
                    best_val_init=bvi,
                    history_seed=hs,
                    samples_seen_seed=sss,
                    rollout_cfg=rollout_cfg
                )

            # Single carry value path
            history, _, _ = run_one_stage(float(carry_value))

    except KeyboardInterrupt:
        if is_main:
            log.warning("Interrupted during training; proceeding to test & save.")
    except Exception:
        if is_main:
            log.exception("Exception during training; proceeding to test & save.")

    # -------------------------------------------------------------------------
    # Post-training: persist history + final weights; run tests/plots
    # -------------------------------------------------------------------------
    finally:
        real_model = model.module if isinstance(model, DDP) else model

        if is_main:
            # Persist training curves
            hist_dir = run_dir / "info"
            h = history if history is not None else History(real_model)
            h.save_json(hist_dir / "loss_history.json")
            h.save_npz(hist_dir / "loss_history.npz")
            log.info(f"Saved loss history to {hist_dir}")

            # Persist final weights snapshot (independent of best)
            final_path = run_dir / "saves" / "model_final_state_dict.pt"
            torch.save(real_model.state_dict(), final_path)
            log.info(f"Model weights saved to {final_path}")

        # Keep ranks aligned before evaluation
        if ddp and dist.is_initialized():
            dist.barrier()

        # Testing + metrics + plots (unless train-only)
        if not args.train_only:

            def _safe(fn, name: str):
                """Run helper to avoid aborting the pipeline if a plot step fails."""
                try:
                    return fn()
                except Exception as e:
                    if is_main and log:
                        log.warning(f"[plots] {name} failed but continuing: {e}", exc_info=True)
                    return None

            reload_best_weights()

            if ddp and dist.is_initialized():
                dist.barrier()

            _ = run_and_save_test_suite(
                model=model,
                loss_func=loss_fn,
                test_dl=test_dl,
                device=DEVICE,
                logger=log,
                rollout_cfg=rollout_cfg,
                run_dir=run_dir,
                is_main=is_main,
                ddp=ddp,
                world_size=WORLD_SIZE,
            )

            if is_main:
                _ = run_and_save_metrics_csv(
                    model=model,
                    test_dl=test_dl,
                    device=DEVICE,
                    rollout_cfg=rollout_cfg,
                    run_dir=run_dir,
                    logger=log,
                )

                log.info("[plots] starting full plots (subsample=200k)")
                _safe(lambda: run_and_save_scatter_grids(
                    model=model,
                    test_dl=test_dl,
                    device=DEVICE,
                    rollout_cfg=rollout_cfg,
                    run_dir=run_dir,
                    logger=log,
                    subsample_points=200_000,
                ), "full_plots")


# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    main()