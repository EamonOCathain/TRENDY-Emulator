#!/usr/bin/env python3
"""
Tile-wise daily inference for a single scenario (no scenario arg needed).
Refactored to use:
  - src.inference.make_stores : store/tiles creation + init lock
  - src.inference.make_preds  : prediction helpers and per-tile processing
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard Library
# ---------------------------------------------------------------------------
import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Third-Party
# ---------------------------------------------------------------------------
import numpy as np
import xarray as xr
import zarr
import torch
import yaml

# ---------------------------------------------------------------------------
# Project Imports / Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.paths.paths import predictions_dir, std_dict_path, masks_dir
from src.dataset.variables import var_names, nfert
from src.models.custom_transformer import YearProcessor

# --- use new stats loader (no filtering here) ---
from src.training.stats import load_standardisation_only
from src.utils.tools import slurm_shard

from src.inference.make_stores import *
from src.inference.make_preds import (
    InferenceSpec,
    years_in_range,
    open_forcing_stores,
    _extract_state_dict,
    _extract_dims_and_cfg,
    _check_dims_or_die,
    _vector_from_std_dict,
    _load_ocean_only_indices,
    is_tile_done,
    mark_tile_done,
    clear_tile_done,
    process_one_tile,
    export_monthly_netcdf_for_scenario,   
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # -----------------------------------------------------------------------
    # CLI / Arguments
    # -----------------------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_name",     required=True)
    ap.add_argument("--scenario",     default=None)
    ap.add_argument("--forcing_dir",  required=True, type=Path)
    ap.add_argument("--weights",      required=True, type=Path)
    ap.add_argument("--out_dir",      required=True, type=Path)
    ap.add_argument("--array_name",   default=None, help="Optional name nested under job_name/scenario; e.g. variants")
    ap.add_argument("--store_period", default=None,
                    help="YYYY-MM-DD:YYYY-MM-DD span used to CREATE/VALIDATE store shapes.")
    ap.add_argument("--write_period", default=None,
                    help="YYYY-MM-DD:YYYY-MM-DD span actually READ/WRITTEN by this run.")
    ap.add_argument("--device",       default="cuda")
    ap.add_argument("--export_nc",    action="store_true",
                    help="Export monthly & annual Zarr variables to per-variable NetCDFs under <out_dir>/netcdf/<scenario>/.")
    ap.add_argument("--export_nc_only", action="store_true",
                help="Skip predictions; just wait for tiles, consolidate once, and export NetCDFs.")

    # Filtering
    ap.add_argument("--exclude_vars", type=str, default="",
                    help="Comma-separated list of variable names to exclude from prediction (applied before building spec).")

    # Slurm arguments
    ap.add_argument("--tile_index",   type=int, default=None,
                    help="If set, process only this tile and exit.")
    ap.add_argument("--shards",       type=int, default=None,
                    help="Total workers for round-robin distribution (e.g., 8).")
    ap.add_argument("--shard_id",     type=int, default=None,
                    help="This worker id [0..shards-1]. Defaults to $SLURM_ARRAY_TASK_ID when --shards is set.")
    ap.add_argument("--tile_h",       type=int, default=30)
    ap.add_argument("--tile_w",       type=int, default=30)

    # Overwrite / repair
    ap.add_argument("--overwrite_skeleton", action="store_true",
                    help="Delete and recreate the Zarr stores and tiles JSON before running.")
    ap.add_argument("--overwrite_data", action="store_true",
                    help="Reprocess tiles even if a per-tile done flag exists.")
    ap.add_argument("--repair_coords", action="store_true",
                    help="Repair the coordinates of an existing store.")

    # Nudging
    ap.add_argument("--nudge_mode", default="none",
                    choices=["none", "original", "z_shrink", "z_mirror", "z_adaptive"],
                    help="How to nudge state vars when nudge_lambda > 0.")
    ap.add_argument("--nudge_lambda", type=float, default=None,
                    help="λ for state nudging (0–1). If unset or 0, nudging is disabled.")

    # Carrying
    ap.add_argument("--carry_forward_states", type=lambda x: str(x).lower() == "true",
                    default=False, help="Whether to use predicted states from year t-1 (True/False). Year 0 always pins.")
    ap.add_argument("--sequential_months",    type=lambda x: str(x).lower() == "true",
                    default=False, help="If True, use within-year monthly-state passing.")
    
    # Transfer-learning variables / renames
    ap.add_argument("--tl_vars", nargs="*", default=[], help=("Optional TL variable selections that rename base variables in the schema. Supported now: 'lai_avh15c1' (replaces 'lai'), 'lai_modis' (replaces 'lai'). You cannot pass both together."),)
    ap.add_argument("--tl_initial_state", type=int, default=None, help=("Year from which to copy monthly *state* values for TL variables to seed the simulation start (e.g., 1982 for LAI). Required when --tl_vars is provided."),)
    
    args = ap.parse_args()
    
    # export-only runs should not touch the model or GPU
    if args.export_nc_only:
        args.export_nc = True          # ensure export happens
        args.device = "cpu"            # avoid any CUDA init

    # -----------------------------------------------------------------------
    # Run Metadata / Args dump
    # -----------------------------------------------------------------------
    info_dir = args.out_dir / args.job_name / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    args_path = info_dir / "args.yaml"

    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(args_path, "w") as f:
        yaml.safe_dump(args_dict, f, sort_keys=False)
    print(f"[INFO] Saved run arguments to {args_path}")

    # -----------------------------------------------------------------------
    # Periods / Validation
    # -----------------------------------------------------------------------
    if not args.store_period and not args.write_period:
        raise SystemExit("You must provide --store_period and/or --write_period.")

    store_period = args.store_period or args.write_period
    write_period = args.write_period or args.store_period

    sp0, sp1 = store_period.split(":")
    wp0, wp1 = write_period.split(":")

    store_start_year = int(sp0[:4]); store_end_year = int(sp1[:4])
    write_start_year = int(wp0[:4]); write_end_year = int(wp1[:4])

    if not (store_start_year <= write_start_year <= store_end_year and
            store_start_year <= write_end_year   <= store_end_year and
            write_start_year <= write_end_year):
        raise SystemExit(
            f"--write_period {write_period} must be within --store_period {store_period} and increasing."
        )

    years = years_in_range(wp0, wp1)

    # -----------------------------------------------------------------------
    # TL Vars / Renames (existing) + NEW seeding requirement
    # -----------------------------------------------------------------------
    tl_vars_set = set(args.tl_vars or [])
    if {"lai_avh15c1", "lai_modis"} <= tl_vars_set or (("lai_avh15c1" in tl_vars_set) and ("lai_modis" in tl_vars_set)):
        raise SystemExit("You cannot pass both 'lai_avh15c1' and 'lai_modis' to --tl_vars at the same time.")

    # If TL vars are specified, tl_initial_state is REQUIRED.
    if tl_vars_set and (args.tl_initial_state is None):
        raise SystemExit("--tl_initial_state <YEAR> is required when --tl_vars is provided.")

    # If provided, the seed year must be within the store period so we can read that year's states.
    if args.tl_initial_state is not None:
        if not (store_start_year <= int(args.tl_initial_state) <= store_end_year):
            raise SystemExit(
                f"--tl_initial_state {args.tl_initial_state} must lie within the store period "
                f"[{store_start_year}..{store_end_year}] to seed initial states."
            )

    rename_map: Dict[str, str] = {}
    # Supported mappings today (extend later as needed)
    if "lai_avh15c1" in tl_vars_set:
        rename_map["lai"] = "lai_avh15c1"
    if "lai_modis" in tl_vars_set:
        rename_map["lai"] = "lai_modis"

    if tl_vars_set:
        known = {"lai_avh15c1", "lai_modis"}
        unknown = sorted(list(tl_vars_set - known))
        if unknown:
            print(f"[INFO] Ignoring unknown --tl_vars entries: {unknown}")
        if rename_map:
            print(f"[INFO] Applying TL renames: {rename_map}")

    def _apply_renames(names: List[str]) -> List[str]:
        """Replace base names using rename_map (order-preserving, no dups)."""
        out: List[str] = []
        for v in names:
            actual = rename_map.get(v, v)
            if actual not in out:
                out.append(actual)
        return out

    # -----------------------------------------------------------------------
    # Variable Lists & Standardisation
    # -----------------------------------------------------------------------
    # Parse exclude_vars into a set from CLI
    exclude_set = set()
    if args.exclude_vars:
        exclude_set = {v.strip() for v in args.exclude_vars.split(",") if v.strip()}
        print(f"[INFO] Excluding variables from inference: {sorted(exclude_set)}")

    # Load standardisation stats (no filtering done here)
    std_dict, invalid_vars = load_standardisation_only(std_dict_path)
    if invalid_vars:
        ex_preview = sorted(list(invalid_vars))[:10]
        print(f"[INFO] Dropping {len(invalid_vars)} var(s) with invalid stats. e.g., {ex_preview}")

    # Helper: final keep rule (mirror dataset class; apply after rename)
    def _keep(name: str) -> bool:
        return (name not in exclude_set) and (name not in invalid_vars)

    # Build filtered group lists with TL renames first, then filter & sort
    DAILY_FORCING   = sorted([v for v in _apply_renames(var_names["daily_forcing"])     if _keep(v)])
    MONTHLY_FORCING = sorted([v for v in _apply_renames(var_names["monthly_forcing"])   if _keep(v)])
    MONTHLY_STATES  = sorted([v for v in _apply_renames(var_names["monthly_states"])    if _keep(v)])
    ANNUAL_FORCING  = sorted([v for v in _apply_renames(var_names["annual_forcing"])    if _keep(v)])
    ANNUAL_STATES   = sorted([v for v in _apply_renames(var_names["annual_states"])     if _keep(v)])
    monthly_fluxes_all = sorted([v for v in _apply_renames(var_names["monthly_fluxes"]) if _keep(v)])

    # OUTPUT order = [monthly_fluxes | monthly_states | annual_states]
    OUTPUT_ORDER = monthly_fluxes_all + MONTHLY_STATES + ANNUAL_STATES

    # Variables written to stores
    MONTHLY_OUT_VARS = monthly_fluxes_all + MONTHLY_STATES
    ANNUAL_OUT_VARS  = ANNUAL_STATES

    # Sanity checks
    missing_monthly = set(MONTHLY_OUT_VARS) - set(OUTPUT_ORDER)
    if missing_monthly:
        raise ValueError("Some monthly output variables are not present in OUTPUT_ORDER: "
                         f"{sorted(missing_monthly)}")

    missing_annual = set(ANNUAL_OUT_VARS) - set(OUTPUT_ORDER)
    if missing_annual:
        raise ValueError("Some annual output variables are not present in OUTPUT_ORDER: "
                         f"{sorted(missing_annual)}")

    # Spec (before logging nin)
    spec = InferenceSpec(
        DAILY_FORCING=DAILY_FORCING,
        MONTHLY_FORCING=MONTHLY_FORCING,
        MONTHLY_STATES=MONTHLY_STATES,
        ANNUAL_FORCING=ANNUAL_FORCING,
        ANNUAL_STATES=ANNUAL_STATES,
        OUTPUT_ORDER=OUTPUT_ORDER,
    )

    print(f"[INFO] Inference input_dim={spec.nin} (should match training)")
    print(f"[INFO] Inference output_dim={len(OUTPUT_ORDER)}")
    
    def resolve_roots_for(scen: str) -> tuple[Path, Path]:
        """
        Return (zarr_root, nc_root) for this scenario.
        zarr_root: <out_dir>/<job_name>/zarr/<scenario>/<array_name?>
        nc_root:   <out_dir>/<job_name>/netcdf/<scenario>/<array_name?>
        """
        base = args.out_dir / args.job_name
        zarr_root = base / "zarr" / scen
        nc_root   = base / "netcdf" / scen
        if args.array_name:
            zarr_root = zarr_root / args.array_name
            nc_root   = nc_root   / args.array_name
        return zarr_root, nc_root

    # Store S3 zarr root for TL Initial states
    s3_zarr_root, _ = resolve_roots_for("S3")
    s3_monthly_path = s3_zarr_root / "monthly.zarr"
    
    # Build TL seeding config (targets limited to variables that are indeed monthly states)
    tl_seed_cfg: Optional[dict] = None
    if tl_vars_set:
        mapped = [rename_map.get("lai", None)]
        state_targets = sorted([m for m in mapped if (m is not None and m in MONTHLY_STATES)])
        if not state_targets:
            print("[WARN] --tl_vars provided but none of the mapped variables are in MONTHLY_STATES; seeding will be a no-op.")
        tl_seed_cfg = {
            "seed_year": int(args.tl_initial_state),
            "state_map": rename_map.copy(),
            "state_targets": state_targets,
            "seed_source_scenario": "S3",
            "seed_source_monthly": str(s3_monthly_path),
        }
        if not s3_monthly_path.exists():
            print(f"[INFO] S3 monthly store not found yet at {s3_monthly_path}. "
                "That's fine if S3 hasn't completed—this run will wait until tiles are done before finalizing.")
        with open(info_dir / "tl_seed_cfg.json", "w") as f:
            json.dump(tl_seed_cfg, f, indent=2, sort_keys=True)

    # -----------------------------------------------------------------------
    # Scenario Resolution / Paths
    # -----------------------------------------------------------------------
    if args.scenario == "all":
        scenarios = ['S0', 'S1', 'S2', 'S3']
    elif args.scenario in ['S0', 'S1', 'S2', 'S3']:
        scenarios = [args.scenario]
    else:
        raise SystemExit("Scenario argument not recognised")

    # -----------------------------------------------------------------------
    # Per-Scenario Processing
    # -----------------------------------------------------------------------
    for scenario in scenarios:
        zarr_root, nc_root = resolve_roots_for(scenario)
        run_root = zarr_root
        
        print(f"Writing Zarr into {run_root}")

        # Optionally blow away existing Zarr skeleton before re-init
        if args.overwrite_skeleton:
            print(f"[WARN] --overwrite_skeleton requested: removing {run_root} (if exists)")
            if run_root.exists():
                shutil.rmtree(run_root)

        # -------------------------------------------------------------------
        # Store Creation / Opening (with lock)
        # -------------------------------------------------------------------
        lock_file = run_root / ".init.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with simple_lock(lock_file):
            daily_path, monthly_path, annual_path, tiles_json_path, meta = ensure_all_stores(
                run_root=run_root,
                period=(sp0, sp1),
                daily_vars=OUTPUT_ORDER,          # daily holds all outputs
                monthly_vars=OUTPUT_ORDER,        # monthly also holds all outputs (monthly means)
                annual_vars=[],                   # no annual outputs
                tile_h=args.tile_h,
                tile_w=args.tile_w,
            )

        # Optional coordinate repair
        if args.repair_coords:
            print("[INFO] Repairing coord arrays in prediction stores …")
            repair_coords(daily_path,   time_res="daily")
            repair_coords(monthly_path, time_res="monthly")
            repair_coords(annual_path,  time_res="annual")
            zarr.consolidate_metadata(zarr.DirectoryStore(str(daily_path)))
            zarr.consolidate_metadata(zarr.DirectoryStore(str(monthly_path)))
            zarr.consolidate_metadata(zarr.DirectoryStore(str(annual_path)))
            print("[INFO] repaired coords and exiting without processing tiles.")
            return

        # Open store groups
        g_daily   = zarr.open_group(store=zarr.DirectoryStore(str(daily_path)),   mode="a")
        g_monthly = zarr.open_group(store=zarr.DirectoryStore(str(monthly_path)), mode="a")
        g_annual  = zarr.open_group(store=zarr.DirectoryStore(str(annual_path)),  mode="a")

        with open(tiles_json_path) as f:
            tiles_json = json.load(f)
        ntiles = len(tiles_json["tiles"])

        # Ocean-only skip list
        ocean_json = masks_dir / "tiles_ocean_only.json"
        ocean_only = _load_ocean_only_indices(ocean_json)

        # Run info
        print(f"[INFO] Store period:  {sp0}–{sp1} (years {store_start_year}–{store_end_year})")
        print(f"[INFO] Write period:  {wp0}–{wp1} (years {write_start_year}–{write_end_year})")
        print(f"[INFO] Predicting for {len(years)} years in write window.")
        forc = open_forcing_stores(args.forcing_dir / scenario)

        # Gate predictions with export-only mode
        if not args.export_nc_only:
            # -------------------------------------------------------------------
            # Device / Checkpoint / Model
            # -------------------------------------------------------------------
            device = torch.device(args.device if args.device in ["cpu", "cuda"] else "cpu")
            print(f"[INFO] Using device: {device}")

            print(f"[INFO] Loading checkpoint (with metadata) from {args.weights}")
            ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
            ck_input_dim, ck_output_dim, ck_model_cfg = _extract_dims_and_cfg(ckpt)
            print(f"[INFO] Checkpoint dims: input_dim={ck_input_dim}, output_dim={ck_output_dim}")

            _check_dims_or_die(
                ck_input_dim=ck_input_dim,
                ck_output_dim=ck_output_dim,
                nin_expected=spec.nin,
                out_names_expected=OUTPUT_ORDER,
            )

            # Indices for monthly states (input and output)
            input_names = (DAILY_FORCING + MONTHLY_FORCING + MONTHLY_STATES + ANNUAL_FORCING + ANNUAL_STATES)
            in_monthly_state_idx  = [input_names.index(v) for v in MONTHLY_STATES]
            out_monthly_state_idx = [OUTPUT_ORDER.index(v) for v in MONTHLY_STATES]

            model = YearProcessor(
                input_dim=ck_input_dim,
                output_dim=ck_output_dim,
                in_monthly_state_idx=in_monthly_state_idx if in_monthly_state_idx else None,
                out_monthly_state_idx=out_monthly_state_idx if out_monthly_state_idx else None,
                **ck_model_cfg,
            ).float().to(device)

            mode = "sequential_months" if args.sequential_months else "batch_months"
            model.set_mode(mode)
            print(f"[INFO] Model runtime mode: {mode}")
            print(
                "[INFO] Carry policy: "
                + (
                    "annual_broadcast (all days)"
                    if (args.carry_forward_states and not args.sequential_months)
                    else (
                        "dec_to_jan_seed (Jan only)"
                        if (args.carry_forward_states and args.sequential_months)
                        else "disabled"
                    )
                )
                + " | nudging & carry are applied in PHYSICAL space"
            )

            state_dict = _extract_state_dict(ckpt if isinstance(ckpt, dict) else {})
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print("[WARN] load_state_dict(strict=False) differences:")
                if missing:    print("  Missing:", missing)
                if unexpected: print("  Unexpected:", unexpected)

            model.eval()
            
            if tl_seed_cfg and scenario != "S3":
                print(f"[INFO] TL seeding will read initial states from S3 monthly store: {s3_monthly_path} "
                    f"(year={tl_seed_cfg['seed_year']})")

            # -------------------------------------------------------------------
            # Standardisation Vectors
            # -------------------------------------------------------------------
            input_vars = input_names
            mu_in,  sd_in  = _vector_from_std_dict(input_vars,   std_dict)
            mu_out, sd_out = _vector_from_std_dict(OUTPUT_ORDER, std_dict)

            def _broadcast(mu_vec, sd_vec):
                return mu_vec.reshape(1, 1, 1, -1), sd_vec.reshape(1, 1, 1, -1)

            MU_IN_B,  SD_IN_B  = _broadcast(mu_in,  sd_in)
            MU_OUT_B, SD_OUT_B = _broadcast(mu_out, sd_out)

            # -------------------------------------------------------------------
            # Tile Distribution / Skips
            # -------------------------------------------------------------------
            if args.tile_index is not None:
                # Single-tile mode
                idxs = [args.tile_index]
                assigned_land  = [i for i in idxs if i not in ocean_only]
                assigned_ocean = [i for i in idxs if i in ocean_only]
                shard_id = None
                total_shards = None
            else:
                if not args.shards:
                    raise SystemExit("Either provide --tile_index or set --shards for round-robin distribution.")
                shard_id = args.shard_id
                if shard_id is None:
                    shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
                if not (0 <= shard_id < args.shards):
                    raise SystemExit(f"Bad shard_id {shard_id} for shards={args.shards}")

                all_tiles   = list(range(ntiles))
                land_tiles  = [i for i in all_tiles if i not in ocean_only]
                ocean_tiles = [i for i in all_tiles if i in ocean_only]

                assigned_land  = [t for j, t in enumerate(land_tiles)  if (j % args.shards) == shard_id]
                assigned_ocean = [t for j, t in enumerate(ocean_tiles) if (j % args.shards) == shard_id]

                total_shards = args.shards

            todo    = assigned_land
            skipped = assigned_ocean

            if not args.overwrite_data:
                todo_before = len(todo)
                todo = [t for t in todo if not is_tile_done(run_root, t)]
                skipped_done = todo_before - len(todo)
                if skipped_done > 0:
                    print(f"[INFO] Skipping {skipped_done} tiles already marked done (use --overwrite_data to reprocess).")
            else:
                for t in todo:
                    clear_tile_done(run_root, t)

            if args.tile_index is not None:
                if skipped:
                    print(f"[INFO] Tile {args.tile_index} is ocean-only; it will be skipped.")
                else:
                    print(f"[INFO] Tile {args.tile_index} is land; it will be processed.")
            else:
                print(
                    f"[INFO] Shard {shard_id}/{total_shards}: "
                    f"{len(assigned_land)} land assigned, {len(assigned_ocean)} ocean assigned "
                    f"(of total {len([i for i in range(ntiles) if i not in ocean_only])} land / "
                    f"{len([i for i in range(ntiles) if i in ocean_only])} ocean)."
                )
            if skipped:
                print(f"[INFO] Skipping {len(skipped)} ocean-only tiles per {ocean_json}")

            # -------------------------------------------------------------------
            # Tile Loop
            # -------------------------------------------------------------------
            failures: List[int] = []
            ntiles_total = len(todo)
            tiles_done = 0

            for ti in todo:
                try:
                    process_one_tile(
                        spec=spec,
                        tile_index=ti,
                        tiles_json=tiles_json,
                        forc=forc,
                        g_daily=g_daily,
                        g_monthly=g_monthly,
                        g_annual=g_annual,                 # will be unused, harmless
                        monthly_vars=OUTPUT_ORDER,         # <-- aggregate ALL outputs to monthly means
                        annual_vars=[],                    # <-- no annual aggregation
                        years=years,
                        model=model,
                        device=device,
                        MU_IN_B=MU_IN_B,
                        SD_IN_B=SD_IN_B,
                        MU_OUT_B=MU_OUT_B,
                        SD_OUT_B=SD_OUT_B,
                        tiles_done_before=tiles_done,
                        ntiles_total=ntiles_total,
                        nfert=nfert,
                        store_start_year=store_start_year,
                        std_dict=std_dict,
                        nudge_lambda=args.nudge_lambda,
                        nudge_mode=args.nudge_mode,
                        carry_forward_states=args.carry_forward_states,
                        sequential_months=args.sequential_months,
                        tl_seed_cfg=tl_seed_cfg,
                    )
                    tiles_done += 1
                    mark_tile_done(run_root, ti, years=years)
                except Exception as e:
                    print(f"[ERR] tile {ti} failed: {e}")
                    failures.append(ti)

            if failures:
                print(f"[DONE] Completed with {len(failures)} failures: {failures}")
                raise SystemExit(1)

            print("[DONE] All assigned tiles completed.")
        else: 
            print("[INFO] --export_nc_only: skipping model init and tile processing")

        # -------------------------------------------------------------------
        # Finalize (leader only): wait for all tiles, consolidate, optional export
        # -------------------------------------------------------------------
        is_leader = True
        if args.tile_index is None and args.shards:
            is_leader = (shard_id == 0)

        all_land_tiles = [i for i in range(ntiles) if i not in ocean_only]
        n_total_land   = len(all_land_tiles)

        if is_leader:
            print(f"[FINALIZE] Leader waiting for all {n_total_land} land tiles to finish…")
            while True:
                n_done = sum(1 for t in all_land_tiles if is_tile_done(run_root, t))
                if n_done >= n_total_land:
                    break
                print(f"[FINALIZE] {n_done}/{n_total_land} tiles done; sleeping 30s …")
                time.sleep(30)

            finalize_lock = run_root / ".export.lock"   # reuse lock
            try:
                with simple_lock(finalize_lock, timeout=600):
                    # re-check after lock
                    n_done = sum(1 for t in all_land_tiles if is_tile_done(run_root, t))
                    if n_done < n_total_land:
                        print(f"[FINALIZE][SKIP] Re-check failed ({n_done}/{n_total_land}); another leader will retry.")
                    else:
                        # ---- consolidate once (idempotent guard) ----
                        consolidation_marker = run_root / ".consolidated.ok"
                        if not consolidation_marker.exists():
                            try:
                                print("[FINALIZE] Consolidating Zarr metadata (daily/monthly/annual) …")
                                for p in (daily_path, monthly_path, annual_path):
                                    if p.exists():
                                        zarr.consolidate_metadata(zarr.DirectoryStore(str(p)))
                                consolidation_marker.write_text("ok\n")
                                print("[FINALIZE] Consolidation complete.")
                            except Exception as e:
                                print(f"[FINALIZE][WARN] Consolidation failed: {e} (continuing)")
                        else:
                            print("[FINALIZE] Consolidation marker present; skipping consolidate.")

                        # ---- NetCDF export if requested ----
                        if args.export_nc:
                            try:
                                export_monthly_netcdf_for_scenario(
                                    scenario=scenario,
                                    nc_root=nc_root,
                                    monthly_zarr=monthly_path,
                                    overwrite=bool(args.overwrite_data),
                                )
                            except Exception as e:
                                print(f"[FINALIZE][ERR] NetCDF export failed for {scenario}: {e}")
                                raise
            except TimeoutError as e:
                print(f"[FINALIZE][INFO] Could not acquire finalize lock: {e}. Another task likely finalizing.")
        else:
            print("[FINALIZE] Non-leader shard; skipping consolidation/export.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        print("[BOOT] starting predict.py")
        main()
        print("[BOOT] finished predict.py")
    except Exception as e:
        import traceback
        print("[FATAL] Unhandled exception in predict.py:", repr(e))
        traceback.print_exc()
        sys.exit(1)