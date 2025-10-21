#!/usr/bin/env python3
import argparse, re, shlex, subprocess, sys, yaml
from pathlib import Path

JOB_RE = re.compile(r"Submitted batch job (\d+)")

def run(cmd, dry=False, cwd=None, capture=False):
    print("[CMD]", cmd if isinstance(cmd, str) else " ".join(map(shlex.quote, cmd)))
    if dry:
        return 0, ""
    res = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        cwd=cwd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True,
        check=False,
    )
    if capture:
        print(res.stdout, end="")
        return res.returncode, res.stdout
    return res.returncode, ""

def submit_sbatch(script, args, slurm, dep=None, dry=False, cwd=None):
    sb = ["sbatch"]
    if slurm.get("partition"): sb += ["-p", slurm["partition"]]
    if slurm.get("qos"):       sb += ["--qos", slurm["qos"]]
    if slurm.get("time"):      sb += ["--time", slurm["time"]]
    if slurm.get("mem"):       sb += ["--mem", slurm["mem"]]
    if slurm.get("cpus"):      sb += ["--cpus-per-task", str(slurm["cpus"])]
    if dep:                    sb += [f"--dependency=afterok:{dep}"]
    sb += [script] + args

    code, out = run(sb, dry=dry, cwd=cwd, capture=True)
    if code != 0:
        sys.exit(f"sbatch failed for {script}")
    m = JOB_RE.search(out or "")
    if not m:
        sys.exit(f"Could not parse job ID from sbatch output:\n{out}")
    jobid = m.group(1)
    print(f"[INFO] Submitted {Path(script).name} as job {jobid}")
    return jobid

# ----------------- argv builders -----------------

def add_kv(argv, key, val, flag=None):
    """Append '--key val' if val is not None; use custom flag name if given."""
    if val is None:
        return argv
    name = flag or f"--{key.replace('_','-')}"
    # boolean args (not store_true) still pass a value
    if isinstance(val, bool):
        return argv + [name, str(val).lower()]
    return argv + [name, str(val)]

def add_flag(argv, key, cfg, flag=None):
    """Append '--flag' only if cfg[key] is truthy (store_true)."""
    if cfg.get(key):
        name = flag or f"--{key.replace('_','-')}"
        argv.append(name)
    return argv

def build_train_args(train_cfg: dict) -> list[str]:
    args = []
    # --- Training loop ---
    args = add_kv(args, "resume", train_cfg.get("resume"))
    args = add_kv(args, "epochs", train_cfg.get("epochs"))
    args = add_kv(args, "mb_size", train_cfg.get("mb_size"))
    args = add_kv(args, "accum_steps", train_cfg.get("accum_steps"))
    args = add_kv(args, "grad_clip", train_cfg.get("grad_clip"))
    args = add_flag(args, "shuffle_windows", train_cfg, "--shuffle-windows")
    args = add_kv(args, "subset_frac", train_cfg.get("subset_frac"))
    excl = train_cfg.get("exclude_vars") or []
    if excl:
        args += ["--exclude_vars", *map(str, excl)]
    args = add_kv(args, "use_foundation", train_cfg.get("use_foundation"))

    # --- Optimiser & scheduler ---
    args = add_kv(args, "lr", train_cfg.get("lr"))
    args = add_kv(args, "scheduler", train_cfg.get("scheduler"))
    args = add_kv(args, "sched_t0", train_cfg.get("sched_t0"))
    args = add_kv(args, "sched_tmult", train_cfg.get("sched_tmult"))
    args = add_kv(args, "eta_min", train_cfg.get("eta_min"))

    # --- Validation ---
    args = add_kv(args, "validation_frequency", train_cfg.get("validation_frequency"))
    args = add_kv(args, "validation_size", train_cfg.get("validation_size"))

    # --- Data loading ---
    args = add_kv(args, "num_workers", train_cfg.get("num_workers"))

    # --- Misc / logging ---
    args = add_kv(args, "job_name", train_cfg.get("job_name"))
    args = add_flag(args, "log_batches_per_rank", train_cfg, "--log-batches-per-rank")

    # --- Loss ---
    args = add_kv(args, "loss_type", train_cfg.get("loss_type"))
    args = add_flag(args, "use_mass_balances", train_cfg, "--use-mass-balances")
    args = add_kv(args, "water_balance_weight", train_cfg.get("water_balance_weight"))
    args = add_kv(args, "npp_balance_weight", train_cfg.get("npp_balance_weight"))
    args = add_kv(args, "nbp_balance_weight", train_cfg.get("nbp_balance_weight"))
    args = add_kv(args, "carbon_partition_weight", train_cfg.get("carbon_partition_weight"))
    args = add_kv(args, "ctotal_mon_ann_weight", train_cfg.get("ctotal_mon_ann_weight"))
    args = add_kv(args, "nbp_delta_ctotal_weight", train_cfg.get("nbp_delta_ctotal_weight"))

    # --- Early Stopping ---
    args = add_flag(args, "early_stop", train_cfg, "--early-stop")
    args = add_kv(args, "early_stop_patience", train_cfg.get("early_stop_patience"))
    args = add_kv(args, "early_stop_min_delta", train_cfg.get("early_stop_min_delta"))
    args = add_kv(args, "early_stop_warmup_epochs", train_cfg.get("early_stop_warmup_epochs"))

    # --- Checkpointing ---
    args = add_kv(args, "ckpt_every_epochs", train_cfg.get("ckpt_every_epochs"))
    args = add_kv(args, "keep_last", train_cfg.get("keep_last"))

    # Extra passthrough
    args += train_cfg.get("extra_args", [])
    return args

def build_predict_args(cfg: dict) -> list[str]:
    args = []
    # --- Required paths ---
    args = add_kv(args, "job_name",     cfg.get("job_name"))
    args = add_kv(args, "forcing_dir",  cfg.get("forcing_dir"))
    args = add_kv(args, "weights",      cfg.get("weights"))
    args = add_kv(args, "out_dir",      cfg.get("out_dir"))

    # --- Optional identifiers ---
    args = add_kv(args, "scenario",     cfg.get("scenario"))
    args = add_kv(args, "array_name",   cfg.get("array_name"))

    # --- Periods ---
    args = add_kv(args, "store_period", cfg.get("store_period"))
    args = add_kv(args, "write_period", cfg.get("write_period"))

    # --- Device ---
    args = add_kv(args, "device",       cfg.get("device", "cuda"))

    # --- store_true flags ---
    for flag in ["export_nc", "overwrite_skeleton", "overwrite_data", "repair_coords"]:
        args = add_flag(args, flag, cfg)

    # --- SLURM / tiling ---
    for key in ["tile_index", "shards", "shard_id", "tile_h", "tile_w"]:
        args = add_kv(args, key, cfg.get(key))

    # --- Nudging ---
    args = add_kv(args, "nudge_mode",   cfg.get("nudge_mode", "none"))
    args = add_kv(args, "nudge_lambda", cfg.get("nudge_lambda"))

    # --- Carry-forward states (boolean arg expecting value) ---
    if "carry_forward_states" in cfg:
        args = add_kv(args, "carry_forward_states", cfg["carry_forward_states"])

    # --- Passthrough ---
    args += cfg.get("extra_args", [])
    return args

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, default="launch/config.yml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--local", action="store_true",
                    help="Run stages locally (no SLURM), sequentially.")
    ap.add_argument("--stages", nargs="+", default=["train","predict","bench"],
                    choices=["train","predict","bench"])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    workdir = Path(cfg["env"]["workdir"])

    # Build per-stage arg lists (strings) for your stage scripts
    train_args   = build_train_args(cfg["train"])
    predict_args = build_predict_args(cfg["predict"])
    bench_args   = [
        "--ilamb-root", cfg["bench"]["ilamb_root"],
        "--config",     cfg["bench"]["config"],
        "--regions",    cfg["bench"]["regions"],
        *cfg["bench"].get("extra_args", []),
    ]

    stage_map = {
        "train":   (cfg["paths"]["train_script"],   train_args),
        "predict": (cfg["paths"]["predict_script"], predict_args),
        "bench":   (cfg["paths"]["bench_script"],   bench_args),
    }

    if args.local:
        # Sequential local execution (stage scripts should be executable & env-aware)
        for s in args.stages:
            script, sargs = stage_map[s]
            code, _ = run([script, *sargs], dry=args.dry_run, cwd=workdir)
            if code != 0:
                sys.exit(f"{s} failed")
        return

    # SLURM mode: submit with dependencies train -> predict -> bench
    slurm = cfg.get("slurm", {})
    last_job = None
    for s in args.stages:
        script, sargs = stage_map[s]
        last_job = submit_sbatch(script, sargs, slurm, dep=last_job,
                                 dry=args.dry_run, cwd=workdir)
    print(f"[INFO] Pipeline submitted. Final stage depends on job {last_job}.")

if __name__ == "__main__":
    main()