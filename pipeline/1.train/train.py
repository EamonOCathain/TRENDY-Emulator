import os, sys, argparse, time, logging
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime

torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------ Environment Setup -----------------------------------
# set project root
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

# start time
start_dt = datetime.now() 
start_dt_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

# import paths and variables
from src.paths.paths import *
from src.dataset.variables import *

# import training modules
from src.models.custom_transformer import YearProcessor
from src.training.checkpoints import save_cb, extract_state_dict_for_foundation
from src.training.history import History
from src.training.loss import custom_loss
from src.training.trainer import fit, plan_validation
from src.dataset.dataloader import get_train_val_test, get_data
from src.training.distributed import  init_distributed
from src.training.logging import save_args, setup_logging 
from src.training.scheduler import build_cosine_wr_scheduler
from src.training.stats import get_split_stats, set_seed, load_and_filter_standardisation
from src.dataset.dataset import base, get_subset
from src.training.carry import next_progressive_carry, progressive_train, month_slices_from_lengths, parse_carry_years_flag
from src.training.tester import run_and_save_test_suite, run_and_save_metrics_csv, run_and_save_scatter_grids
from src.training.varschema import VarSchema
from src.training.delta import build_delta_ctx

# Overwrite training dir
training_dir = train_pipeline_dir

# Slurm stuff
workers = max(1, int(os.getenv("SLURM_CPUS_PER_TASK", "4")))
slurm_id = os.getenv("SLURM_JOB_ID", "no_slurm_id")

# ------------------------------ Distributed Setup -----------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------ Args -----------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # --- Training loop ---
    parser.add_argument("--resume", type=str, default=None,
    help="Path to a checkpoint .pt (epoch*.pt or best.pt) to resume training")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--mb_size", type=int, default=1)       
    parser.add_argument("--accum_steps", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--shuffle_windows", action="store_true")
    parser.add_argument("--subset_frac", type=float, default=None)
    parser.add_argument("--exclude_vars", nargs="*", default=[],
        help="Variable names to exclude from inputs (space-separated e.g --exclude_vars var1 var2)")
    parser.add_argument("--use_foundation", type=str, default=None,
        help="Path to a checkpoint to initialize model weights from. "
            "This does NOT resume training: optimizer/scheduler/history/epoch are ignored.")
    parser.add_argument("--train_only", action="store_true",
        help="Skip testing after training.")
    parser.add_argument("--test_only", action="store_true",
        help="Run only the testing suite (skips all training and validation).")

    # --- Optimiser & scheduler ---
    parser.add_argument("--lr", type=float, default=9e-5)
    parser.add_argument("--scheduler", type=str, default="cosine_wr",
                        choices=["none", "cosine_wr"])
    parser.add_argument("--sched_t0", type=str, default="epoch",
                    help='Cosine WR T0. Use "epoch" to match steps/epoch, '
                         'or provide an integer (steps).')
    parser.add_argument("--sched_tmult", type=int, default=2)
    parser.add_argument("--eta_min", type=float, default=9e-6)

    # --- Validation ---
    parser.add_argument("--val_freq", type=float, default=1.0)
    parser.add_argument("--val_frac", type=float, default=1.0)
    
    # Testing
    parser.add_argument("--test_frac", type=float, default=None,
        help=("Subset the TEST set by a fraction RELATIVE to --subset_frac. "
              "Effective test fraction = (subset_frac if set else 1.0) * (test_frac if set else 1.0)."))

    # --- Data loading ---
    parser.add_argument("--num_workers", type=int, default=8)

    # --- Misc / logging ---
    parser.add_argument("--job_name", type=str, default="transformer_run")
    parser.add_argument("--log_batches_per_rank", action="store_true")
    
    # --- Loss ---
    parser.add_argument("--loss_type", type=str, default="mse",
                        choices=["mse", "mae"],
                        help="Base loss to use for supervision: 'mse' or 'mae'")
    parser.add_argument("--use_mass_balances", action="store_true",
                    help="Enable mass-balance penalties in the loss (off by default)")
    parser.add_argument("--water_balance_weight", type=float, default=0.0,
                        help="Weight for water balance penalty (Δmrso vs pr-mrro-evapotrans), set to 0 to disable")
    parser.add_argument("--npp_balance_weight", type=float, default=0.0,
                        help="Weight for NPP balance penalty (NPP vs GPP-ra), set to 0 to disable")
    parser.add_argument("--nbp_balance_weight", type=float, default=0.0,
                        help="Weight for NBP balance penalty (NBP vs NPP-rh-fFire-fLuc), set to 0 to disable")
    parser.add_argument("--carbon_partition_weight", type=float, default=0.0,
                    help="Weight for cTotal_annual = cVeg + cLitter + cSoil (annual means)")
    parser.add_argument("--nbp_delta_ctotal_weight", type=float, default=0.0,
                        help="Weight for Δ cTotal_monthly = NBP at monthly scale")
    
    # --- Early Stopping ---
    parser.add_argument("--early_stop", action="store_true",
                        help="Enable early stopping (off by default)")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Patience in epochs (used only if --early_stop is set)")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                        help="Minimum absolute improvement to reset patience (used only if --early_stop is set)")
    parser.add_argument("--early_stop_warmup_epochs", type=int, default=0,
                        help="Warmup epochs before monitoring (used only if --early_stop is set)")
    
    # --- Checkpointing ---
    parser.add_argument("--ckpt_every_epochs", type=int, default=1,
                    help="Save rolling checkpoint every N epochs (0=disable)")
    parser.add_argument("--keep_last", type=int, default=5,
                        help="Keep last K rolling checkpoints")
    
    # Delta
    parser.add_argument("--delta", action="store_true",
                        help="Train in delta mode: model daily outputs are normalized daily deltas which are reconstructed to normalized absolutes before loss.")
    # Carry years
    parser.add_argument("--carry_years", nargs="+", default=["0"], help=('Carry horizon. Accepts: "0", a float ("2.5"), a fraction ("3/12"), "progressive", or multiple values like: 1 2 3 6 9'))
    parser.add_argument("--carry_granularity", type=str, default="annual", choices=["monthly", "annual"], help="Carry coupling across years: 'monthly' (last-month state to next Jan) "
                         "or 'annual' (annual-mean state broadcast across all days of next year).")
    
    # Other
    parser.add_argument("--scan_finite", action="store_true", help = "Scan datasets for non-finite values before training (warns if any found)")

    return parser.parse_args()

# Parse them
args = parse_args()

# Argument Guards 
if args.resume and args.use_foundation:
    raise SystemExit("Use either --resume or --use_foundation, not both.")

if args.train_only and args.test_only:
    raise SystemExit("Cannot use --train_only and --test_only together.")

def _check_frac(name, val):
    if val is None: return
    if not (0.0 < float(val) <= 1.0):
        raise SystemExit(f"--{name} must be in (0,1], got {val}")

_check_frac("subset_frac", args.subset_frac)
_check_frac("test_frac", args.test_frac)

# ------------------------------ Carry Years Setup -----------------------------------
carry_mode, carry_values = parse_carry_years_flag(args.carry_years)  # ("static"|"multi"|"progressive", [floats])

# basic guard
if carry_mode == "static" and carry_values and carry_values[0] < 0:
    raise SystemExit("--carry_years must be >= 0")

# Annual mode: clamp any 0<c<1 to 1.0 (keep exact 0.0 as-is, since it means no-carry)
if args.carry_granularity == "annual":
    clamped = []
    any_clamped = False
    for c in carry_values:
        if c == 0.0:
            clamped.append(0.0)
        else:
            new_c = max(1.0, float(c))
            any_clamped |= (new_c != c)
            clamped.append(new_c)
    if any_clamped:
        logging.getLogger(args.job_name).warning(
            "[carry] annual granularity: clamped carry years %s -> %s (no fractional carry)",
            carry_values, clamped
        )
    carry_values = clamped

# For the dataloader helper that expects a string flag (back-compat)
if isinstance(args.carry_years, (list, tuple)):
    carry_flag_for_ds = " ".join(args.carry_years)
else:
    carry_flag_for_ds = str(args.carry_years)

# If any carry coupling is used (progressive or any positive carry), disable window shuffling
if (carry_mode == "progressive") or any(c > 0.0 for c in carry_values):
    if getattr(args, "shuffle_windows", False):
        args.shuffle_windows = False

# Non Finite Check
@torch.no_grad()
def scan_for_nonfinite(dl, max_batches=None):
    for bi, (x, m, a) in enumerate(dl):
        if max_batches and bi >= max_batches: break
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

# ------------------------------ Load Std. Dict and Filter Variables -----------------------------------
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

# ------------------------------ Main -----------------------------------
def main():
    # --- Reproducibility ---
    set_seed(42)

    # --- Distributed init & device selection ---
    global DEVICE
    # ddp is the 
    ddp, device, LOCAL_RANK, WORLD_SIZE, RANK = init_distributed() 
    DEVICE = device
    is_main = (RANK == 0)

    # Per-rank worker count (respect SLURM layout)
    cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "4"))
    workers_per_rank = min(8, cpus)

    # --- Run/output directories & logging ---
    today = datetime.now().strftime("%Y-%m-%d")
    run_leaf = f"{slurm_id}_{args.job_name}"
    run_dir = training_dir / "runs" / today / run_leaf
    (run_dir / "saves").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "info").mkdir(parents=True, exist_ok=True)

    log = setup_logging(run_dir, args.job_name) if is_main else logging.getLogger(args.job_name)
    if not is_main:
        # Don’t forward non-main rank logs to root logger
        log.propagate = False
    
    # ---------- Gate early stopping ----------
    if not args.early_stop:
        args.early_stop_patience = None
        args.early_stop_min_delta = 0.0
        args.early_stop_warmup_epochs = 0

    # ---------- Gate mass-balance penalties ----------
    if not args.use_mass_balances:
        args.water_balance_weight = 0.0
        args.npp_balance_weight = 0.0
        args.nbp_balance_weight = 0.0
        args.carbon_partition_weight = 0.0
        args.nbp_delta_ctotal_weight = 0.0

    if is_main:
        log.info(f"Early stopping: {'ON' if args.early_stop else 'OFF'}")
        log.info(f"Mass balances: {'ON' if args.use_mass_balances else 'OFF'}")

    if is_main:
        log.info(f"Using device: {DEVICE} | world_size={WORLD_SIZE} rank={RANK} local_rank={LOCAL_RANK}")
        info_file = save_args(run_dir, args)
        log.info(f"Saved run arguments to {info_file}")

    # Get the data
    ds_dict = get_train_val_test(std_dict, block_locs=70, carry_years_flag=carry_flag_for_ds)

    if is_main:
        print("Number of location chunks in full dataset:",
            len(ds_dict["train"]), "val len:", len(ds_dict["val"]), "test len:", len(ds_dict["test"]))

    # --- Train/Val subsetting (unchanged) ---
    if args.subset_frac is None:
        train_ds = ds_dict["train"]
        val_ds   = ds_dict["val"]
    else:
        train_ds = get_subset(ds_dict["train"], frac=args.subset_frac, seed=42)
        val_ds   = get_subset(ds_dict["val"],   frac=args.subset_frac, seed=1337)

    # --- Test subsetting (multiplicative) ---
    base_test_frac   = args.subset_frac if args.subset_frac is not None else 1.0
    mult_test_frac   = args.test_frac   if args.test_frac   is not None else 1.0
    effective_t_frac = float(base_test_frac) * float(mult_test_frac)

    # Clamp to (0,1] and short-circuit 1.0 to avoid extra copy
    if not (0.0 < effective_t_frac <= 1.0):
        raise SystemExit(f"Effective test fraction must be in (0,1], got {effective_t_frac}")

    test_ds = (get_subset(ds_dict["test"], frac=effective_t_frac, seed=999)
            if effective_t_frac < 1.0 else ds_dict["test"])

    if is_main:
        print("Datasets after subsetting:",
            "train", len(train_ds), "val", len(val_ds),
            f"test {len(test_ds)} (effective_test_frac={effective_t_frac})")

    # -----------------------------------------------------------
    
    # --- DataLoaders ---
    train_dl, valid_dl, test_dl = get_data(
        train_ds, val_ds, test_ds,
        bs=1,
        num_workers=(args.num_workers if args.num_workers is not None else workers_per_rank),
        ddp=ddp,
    )
    
    # --- Optional pre-flight dataset scan for NaNs or Infs ---
    if args.scan_finite:
        if is_main:
            log.info("[preflight] scanning test data for non-finite values in first 5000 test batches...")
            bi, fins = scan_for_nonfinite(test_dl, max_batches=None)
            if bi is not None:
                log.warning(f"[preflight] first non-finite at test batch {bi} "
                            f"(x={fins[0]}, m={fins[1]}, a={fins[2]})")
            else:
                log.info("[preflight] no non-finite values detected in test batches.")
            
            log.info("[preflight] scanning train and val data for non-finite values in first 5000 test batches...")
            for name, dl in [("train", train_dl), ("val", valid_dl), ("test", test_dl)]:
                bi, fins = scan_for_nonfinite(dl, max_batches=None)
                if bi is not None:
                    log.warning(f"[preflight/{name}] first non-finite at batch {bi} (x={fins[0]}, m={fins[1]}, a={fins[2]})")
                else:
                    log.info(f"[preflight/{name}] no non-finite in train and val batches")

    # Optional per-rank dataloader/debug logging
    if args.log_batches_per_rank and ddp and dist.is_initialized():
        dist.barrier()
        log.info(f"[Rank {RANK}] train_ds={len(train_ds)}, val_ds={len(val_ds)}, test_ds={len(test_ds)}")
        log.info(f"[Rank {RANK}] train_batches={len(train_dl)}, val_batches={len(valid_dl)}, test_batches={len(test_dl)}")
        dist.barrier()
    else:
        if is_main:
            log.info(f"Dataset sizes — train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
            log.info(f"Dataloader batches per GPU — train={len(train_dl)}, val={len(valid_dl)}, test={len(test_dl)}")

    # --- Split stats & validation plan (for in-epoch validation cadence) ---
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

    # --- Map dataset variables to model I/O dimensions & loss indices ---
    base_train = base(train_ds)

    # --- Build canonical schema from the dataset (works for both datasets) ---
    if hasattr(base_train, "schema"):
        # CarryBlockDataset path — trust the dataset-provided schema
        schema = base_train.schema
    else:
        # Legacy CustomDataset path — synthesize schema from var_names
        schema = VarSchema(
            daily_forcing   = sorted(base_train.var_names['daily_forcing']),
            monthly_forcing = sorted(base_train.var_names['monthly_forcing']),
            monthly_states  = sorted(base_train.var_names['monthly_states']),
            annual_forcing  = sorted(base_train.var_names['annual_forcing']),
            annual_states   = sorted(base_train.var_names['annual_states']),
            monthly_fluxes  = sorted(base_train.var_names['monthly_fluxes']),
        )

    # Canonical orders & dims (source of truth = schema)
    INPUT_ORDER   = schema.input_order()
    OUTPUT_ORDER  = schema.output_order()
    dims          = schema.dims()
    input_dim     = dims["input_dim"]
    output_dim    = dims["output_dim"]
    schema_sig    = schema.signature()
    schema_dims   = dims

    # Minimal internal sanity (no dependency on base_train.var_names here)
    assert len(INPUT_ORDER)  > 0 and len(INPUT_ORDER)  == len(set(INPUT_ORDER)),  "INPUT_ORDER empty or has duplicates"
    assert len(OUTPUT_ORDER) > 0 and len(OUTPUT_ORDER) == len(set(OUTPUT_ORDER)), "OUTPUT_ORDER empty or has duplicates"

    # Head-local names for loss/metrics
    monthly_names = schema.out_monthly_names()   # fluxes + states (monthly head)
    annual_names  = schema.out_annual_names()    # states (annual head)
    output_names  = monthly_names + annual_names

    # Indices for the loss heads (local to the combined monthly+annual loss vector)
    idx_monthly = list(range(len(monthly_names)))
    idx_annual  = list(range(len(monthly_names), len(output_names)))

    # Name -> index for balances/diagnostics (over the same output_names)
    out_idx = {name: i for i, name in enumerate(output_names)}

    # Snapshot for checkpoint (keep API stable)
    if hasattr(base_train, "var_names"):
        # Legacy dataset: snapshot its dict (sorted) for backwards-compat saves
        varnames_snapshot = {k: sorted(list(v)) for k, v in base_train.var_names.items()}
    else:
        # Carry dataset: derive snapshot directly from schema
        varnames_snapshot = {
            "daily_forcing":   list(schema.daily_forcing),
            "monthly_forcing": list(schema.monthly_forcing),
            "monthly_states":  list(schema.monthly_states),
            "annual_forcing":  list(schema.annual_forcing),
            "annual_states":   list(schema.annual_states),
            "monthly_fluxes":  list(schema.monthly_fluxes),
        }
    
    # Set mass balance variable index to none if not needed
    mb_var_idx = None
    if args.use_mass_balances:
        # Pick only the variables needed for balances
        needed_vars = [
            "mrso", "pre", "mrro", "evapotrans",  
            "npp", "gpp", "ra",                 
            "nbp", "rh", "fFire", "fLuc",   
            "cTotal_monthly", "cTotal_annual", "cVeg", "cLitter",
            "cSoil" 
        ]

        # Check that any are missing and build the needed vars 
        mb_var_idx = {}
        missing = []
        for v in needed_vars:
            if v in out_idx:
                mb_var_idx[v] = out_idx[v]
            else:
                missing.append(v)

        if missing and is_main:
            print("[mass-balance] missing variables:", missing)
            
    # Build roll out config for carry years
    rollout_cfg = {
        "in_monthly_state_idx":   schema.in_monthly_state_idx(),
        "in_annual_state_idx":    schema.in_annual_state_idx(),
        "out_monthly_state_idx":  schema.out_monthly_state_idx_local(),
        "out_annual_state_idx":   schema.out_annual_state_idx_local(),
        "out_monthly_names":      monthly_names,
        "out_annual_names":       annual_names,
        "month_lengths":          schema.month_lengths,
        "carry_horizon":          float(carry_values[0] if carry_values else 0.0),
        "carry_granularity":      str(args.carry_granularity),
        "output_order":           OUTPUT_ORDER,
        "schema_sig":             schema_sig,
    }

    if is_main:
        nm = len(rollout_cfg["out_monthly_names"])
        na = len(rollout_cfg["out_annual_names"])
        log.info(f"[rollout_cfg] nm={nm} monthly names, na={na} annual names")
        log.info(f"[rollout_cfg] out_monthly_state_idx (local) = {rollout_cfg['out_monthly_state_idx']}")
        log.info(f"[rollout_cfg] out_annual_state_idx  (local) = {rollout_cfg['out_annual_state_idx']}")
        log.info(f"[schema] sig={schema.signature()} | input_dim={input_dim} | output_dim={output_dim} "
             f"| nm={nm} | na={na}")
    
    # Delta context
    month_slices = month_slices_from_lengths(rollout_cfg["month_lengths"])
    rollout_cfg["month_slices"] = month_slices
    delta_ctx = build_delta_ctx(
        enabled=bool(getattr(args, "delta", False)),
        month_slices=month_slices,
    )
    rollout_cfg["delta_ctx"] = delta_ctx
        
    # --- Loss ---
    loss_fn = custom_loss(
        idx_monthly, idx_annual,
        loss_type=str(args.loss_type),
        monthly_weights=[1.0]*len(idx_monthly),
        annual_weights=[1.0]*len(idx_annual),
        mb_var_idx=mb_var_idx,
        water_balance_weight=args.water_balance_weight,
        npp_balance_weight=args.npp_balance_weight,
        nbp_balance_weight=args.nbp_balance_weight,
        carbon_partition_weight=args.carbon_partition_weight,
        nbp_delta_ctotal_weight=args.nbp_delta_ctotal_weight,
    )

    # --- Model & DDP wrapper (if enabled) ---
    model = YearProcessor(
        input_dim=input_dim,
        output_dim=output_dim,
        in_monthly_state_idx=rollout_cfg["in_monthly_state_idx"],
        out_monthly_state_idx=rollout_cfg["out_monthly_state_idx"],
        month_lengths=rollout_cfg["month_lengths"],
        d=128, h=1024, g=256, num_layers=4, nhead=8, dropout=0.1,
        transformer_kwargs={"max_len": 31},
        mode="batch_months",   # fast pretrain path by default
    ).float().to(DEVICE)   
    
    # Helper: read io dims from a training-style checkpoint (optional)
    def _ckpt_io_dims(ckpt: dict) -> tuple[int, int]:
        return int(ckpt.get("input_dim", -1)), int(ckpt.get("output_dim", -1))

    # --- Load foundation weights if requested ---
    if args.use_foundation:
        if is_main:
            print(f"[FOUNDATION] Loading foundation weights from {args.use_foundation}")

        ckpt_f = torch.load(args.use_foundation, map_location="cpu", weights_only=True)

        # Handle either a full checkpoint or a bare state_dict
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

        # Load into current model (allowing head differences)
        target = (model.module if isinstance(model, DDP) else model)
        missing, unexpected = target.load_state_dict(sd_f, strict=False)

        if is_main:
            msg = f"Loaded foundation weights from {args.use_foundation}"
            if missing or unexpected:
                msg += f" (missing={len(missing)}, unexpected={len(unexpected)})"
            print(msg)

    if ddp:
        model = DDP(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            find_unused_parameters=False
        )

    # --- Optimizer & LR scheduler ---
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler, sched_info = build_cosine_wr_scheduler(
        args, opt, stats['train'], log=log if is_main else None
    )
    if sched_info and is_main:
        info_clean = {k: v for k, v in sched_info.items() if k != "total_windows"}
        log.info(f"scheduler_info: {info_clean}")

    # CHECKPOINTING
    # Save-callback that only runs on main rank
    real_model = (model.module if isinstance(model, DDP) else model)
    
    base_save_cb = (
        lambda epoch, best, val, history:
            save_cb(
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
        if not is_main:
            return
        base_save_cb(epoch, best, val, history)

    # start time (before training)
    start_ts = time.time() 
    
    # Start memory tracking (before training)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())
    
    # Resume from checkpoint if specified
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

        real_model = (model.module if isinstance(model, DDP) else model)
        real_model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        if ckpt.get("sched_state") is not None and scheduler is not None:
            scheduler.load_state_dict(ckpt["sched_state"])

        hist_ckpt = ckpt.get("history", {}) or {}

        # If older checkpoints didn’t store early_state, leave it missing; if present, inject
        early_state_prev = ckpt.get("early_state", None)
        if early_state_prev is not None:
            hist_ckpt["_early_state"] = early_state_prev

        # 1-based stored; resume at this number as the next 0-based epoch
        raw_epoch = int(ckpt.get("epoch", 0))
        # Back-compat: some very old (pre-fix) saves might have been 0- or 1-based.
        # Heuristic: if raw_epoch == 0 and you clearly have non-empty train_loss,
        # assume we completed at least 1 epoch and bump once. Otherwise trust it.
        if raw_epoch == 0 and len(hist_ckpt.get("train_loss", [])) > 0:
            start_epoch = 1
        else:
            start_epoch = raw_epoch

        best_val_so_far   = float(ckpt.get("best_val", float("inf")))
        samples_seen_prev = int(hist_ckpt.get("samples_seen", 0) or 0)

    else:
        start_epoch = 0
        best_val_so_far = float("inf")
        samples_seen_prev = 0
        hist_ckpt = {}
        
    # --- Resume bundle to apply only once (first stage) ---
    resume_payload = {
        "start_epoch": start_epoch,
        "best_val_init": best_val_so_far,
        "history_seed": hist_ckpt,
        "samples_seen_seed": samples_seen_prev,
    }
    resume_pending = args.resume is not None
    
    # ---- Train ----
    history = None
    best_val = float("inf")

    BEST_PATH = run_dir / "checkpoints" / "best.pt"
    
    def reload_best_weights() -> bool:
        """Load checkpoints/best.pt into `model` if present. Returns True if loaded."""
        try:
            if not BEST_PATH.exists():
                return False
            ck = torch.load(BEST_PATH, map_location="cpu", weights_only=True)
            state = ck.get("model_state", ck) if isinstance(ck, dict) else ck
            target = (model.module if isinstance(model, DDP) else model)
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

    try:
        if not args.test_only:

            def run_one_stage(carry_val: float):
                nonlocal resume_pending, resume_payload
                rollout_cfg["carry_horizon"] = float(carry_val)

                if resume_pending:
                    se  = resume_payload.get("start_epoch", 0)
                    bvi = resume_payload.get("best_val_init", float("inf"))
                    hs  = resume_payload.get("history_seed", None)
                    sss = resume_payload.get("samples_seen_seed", 0)
                    resume_pending = False
                else:
                    se, bvi, hs, sss = 0, float("inf"), None, 0

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
                    rollout_cfg=rollout_cfg,
                    stage_carry=float(carry_val),
                )

            # ---- choose training path (static / multi / progressive) ----
            has_pos_carry = any(float(c) > 0 for c in carry_values)

            # optional: one clear, final log line
            if is_main:
                log.info(f"[carry/final] mode={carry_mode}, values={carry_values}, "
                        f"granularity={args.carry_granularity}, has_pos_carry={has_pos_carry}")

            if carry_mode == "static" or not has_pos_carry:
                # true zero-carry path: single stage, horizon=0.0, no progressive logging
                history, best_val, _ = run_one_stage(0.0)

            elif carry_mode == "multi":
                history, best_val = progressive_train(
                    mode="multi",
                    carry_values=[float(c) for c in carry_values],
                    fit_fn=run_one_stage,
                    reload_best_fn=reload_best_weights,
                    next_carry_fn=next_progressive_carry,  # unused in multi
                    log=(log if is_main else None),
                    max_cap=86.0,
                    carry_granularity=rollout_cfg.get("carry_granularity", "monthly"),
                )

            else:  # "progressive"
                default_start = [1.0/12.0] if args.carry_granularity == "monthly" else [1.0]
                start_seq = carry_values if has_pos_carry else default_start
                history, best_val = progressive_train(
                    mode="progressive",
                    carry_values=[float(start_seq[0])],
                    fit_fn=run_one_stage,
                    reload_best_fn=reload_best_weights,
                    next_carry_fn=next_progressive_carry,
                    log=(log if is_main else None),
                    max_cap=86.0,
                    carry_granularity=rollout_cfg.get("carry_granularity", "monthly"),
                )
                
    except KeyboardInterrupt:
        if is_main:
            log.warning("Interrupted during training; proceeding to test & save.")
    except Exception:
        if is_main:
            log.exception("Exception during training; proceeding to test & save.")
    finally:
        # --- Persist loss history & final weights (whatever state we have) ---
        real_model = (model.module if isinstance(model, DDP) else model)
        if is_main:
            hist_dir = run_dir / "info"
            h = history if history is not None else History(real_model)
            h.save_json(hist_dir / "loss_history.json")
            h.save_npz(hist_dir / "loss_history.npz")
            log.info(f"Saved loss history to {hist_dir}")

            final_path = run_dir / "saves" / "model_final_state_dict.pt"
            torch.save(real_model.state_dict(), final_path)
            log.info(f"Model weights saved to {final_path}")

        # Sync before evaluation (only if DDP still active)
        if ddp and dist.is_initialized():
            dist.barrier()

        if not args.train_only:
            # Helper: make plotting best-effort so failures don't abort testing
            def _safe(fn, name: str):
                try:
                    return fn()
                except Exception as e:
                    if is_main and log:
                        log.warning(f"[plots] {name} failed but continuing: {e}", exc_info=True)
                    return None

            # --- Load best weights from checkpoints/best.pt (if available) ---
            reload_best_weights()

            # Optional sync so ranks start test after smoke plotting
            if ddp and dist.is_initialized():
                dist.barrier()

            # --- (B) Test suite & metrics ---
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

                # --- (C) Full plots (best-effort, larger subsample) ---
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

if __name__ == "__main__":
    main()