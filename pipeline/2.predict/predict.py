#!/usr/bin/env python3
"""
Tile-wise daily inference for one scenario.

Uses:
  - src.inference.make_stores : store/tiles creation + init lock
  - src.inference.make_preds  : prediction helpers + per-tile processing/export
"""

from __future__ import annotations

# --- Stdlib -----------------------------------------------------------------
import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional


# --- Third-party -------------------------------------------------------------
import zarr
import torch
import yaml

# --- Project paths & imports -------------------------------------------------
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.paths.paths import std_dict_path, masks_dir
from src.dataset.variables import var_names, nfert
from src.models.custom_transformer import YearProcessor
from src.training.stats import load_standardisation_only

# Store & inference helpers
from src.inference.make_stores import *  # simple_lock, ensure_all_stores, repair_coords, etc.
from src.inference.make_preds import (
    InferenceSpec,
    years_in_range,
    open_forcing_stores,
    _extract_state_dict,
    _extract_dims_and_cfg,
    _check_dims_or_die,
    _load_ocean_only_indices,
    is_tile_done,
    mark_tile_done,
    clear_tile_done,
    process_one_tile,
    export_netcdf_sharded,
    mu_sd_from_std,
)


# ============================================================================ #
# Main
# ============================================================================ #
def main():
    # -----------------------------------------------------------------------
    # CLI
    # -----------------------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_name", required=True)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--forcing_dir", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--array_name", default=None, help="Optional subfolder under job_name/scenario")
    ap.add_argument("--store_period", default=None, help="YYYY-MM-DD:YYYY-MM-DD used to CREATE/VALIDATE store shapes")
    ap.add_argument("--write_period", default=None, help="YYYY-MM-DD:YYYY-MM-DD actually READ/WRITTEN by this run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--number_tiles", type=int, default=1, help="Concurrent tiles per shard (default: 1)")

    # Export
    ap.add_argument("--export_nc", action="store_true",
                    help="Export per-variable NetCDFs into <out_dir>/netcdf/<scenario>/")
    ap.add_argument("--export_nc_only", action="store_true",
                    help="Skip predictions; wait for tiles, consolidate once, then export NetCDFs only")
    ap.add_argument("--ilamb_dir_global", type=Path, default=None,
                    help="Copy full-period NetCDFs to this directory")
    ap.add_argument("--ilamb_dir_test", type=Path, default=None,
                    help="Copy test-period NetCDFs (early/late) into this directory")

    # Filtering
    ap.add_argument("--exclude_vars", type=str, default="",
                    help="Comma-separated list of variables to exclude before building spec")

    # Slurm / tiling
    ap.add_argument("--tile_index", type=int, default=None,
                    help="If set, process only this tile index")
    ap.add_argument("--shards", type=int, default=None,
                    help="Total shards for round-robin tile distribution")
    ap.add_argument("--shard_id", type=int, default=None,
                    help="This worker id [0..shards-1]; defaults to $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--tile_h", type=int, default=30)
    ap.add_argument("--tile_w", type=int, default=30)

    # Overwrite / repair
    ap.add_argument("--overwrite_skeleton", action="store_true",
                    help="Delete and recreate stores and tiles JSON before running")
    ap.add_argument("--overwrite_data", action="store_true",
                    help="Reprocess tiles even if per-tile .done exists")
    ap.add_argument("--repair_coords", action="store_true",
                    help="Repair coordinates in existing stores and exit")

    # Nudging
    ap.add_argument("--nudge_mode", default="none",
                    choices=["none", "original", "z_shrink", "z_mirror", "z_adaptive"],
                    help="Nudging strategy for state vars when nudge_lambda > 0")
    ap.add_argument("--nudge_lambda", type=float, default=None,
                    help="λ in [0,1] for nudging; disabled if unset or 0")

    # Carrying
    ap.add_argument("--carry_forward_states", type=lambda x: str(x).lower() == "true",
                    default=False, help="Use predicted states from year t-1")
    ap.add_argument("--sequential_months", type=lambda x: str(x).lower() == "true",
                    default=False, help="Within-year monthly-state passing")

    # TL variables / renames
    ap.add_argument("--tl_vars", nargs="*", default=[],
                    help="Supported: 'lai_avh15c1' or 'lai_modis' (mutually exclusive)")
    ap.add_argument("--tl_initial_state", type=int, default=None,
                    help="Year to seed monthly *state* values for TL vars (e.g., 1982 for LAI)")

    # Counterfactual forcing offsets
    ap.add_argument("--forcing_offsets", type=str, default="",
                    help="Comma list 'scope:var=value' where scope ∈ {daily,monthly,annual}. "
                         "e.g., 'annual:co2=121.54,monthly:lai=0.3,daily:pre=-20%'")

    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Shard normalization (applies to all modes)
    # -----------------------------------------------------------------------
    # Default to single-shard semantics; force (1,0) for --tile_index runs.
    if args.shards is None:
        args.shards = 1
    if args.tile_index is not None:
        args.shards = 1
        args.shard_id = 0

    safe_shard_id = args.shard_id
    if safe_shard_id is None:
        safe_shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if not (0 <= safe_shard_id < int(args.shards)):
        raise SystemExit(f"Bad shard_id {safe_shard_id} for shards={args.shards}")
    is_leader = (safe_shard_id == 0)

    # -----------------------------------------------------------------------
    # Parse forcing offsets "scope:var=value" (percent or additive)
    # -----------------------------------------------------------------------
    def parse_forcing_offsets(spec: str) -> dict:
        out = {"daily": {}, "monthly": {}, "annual": {}}
        if not spec:
            return out
        for entry in spec.split(","):
            entry = entry.strip()
            if not entry:
                continue
            try:
                scope, rest = entry.split(":", 1)
                var, val = rest.split("=", 1)
                scope, var, val = scope.strip().lower(), var.strip(), val.strip()
                if scope not in out:
                    print(f"[WARN] Unknown forcing offset scope '{scope}' (ignored)")
                    continue
                if val.endswith("%"):
                    out[scope][var] = {"mode": "percent", "value": float(val[:-1])}
                else:
                    out[scope][var] = {"mode": "add", "value": float(val)}
            except ValueError:
                print(f"[WARN] Could not parse forcing offset entry: '{entry}'")
        return out

    FORCING_OFFSETS = parse_forcing_offsets(args.forcing_offsets)
    print(f"[INFO] Parsed forcing offsets: {FORCING_OFFSETS}")

    # Export-only runs: skip any CUDA init
    if args.export_nc_only:
        args.export_nc = True
        args.device = "cpu"

    # Keep torch from spawning many threads per process
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    
    
    # -----------------------------------------------------------------------
    # Run metadata
    # -----------------------------------------------------------------------
    info_dir = args.out_dir / args.job_name / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    args_path = info_dir / "args.yaml"
    with open(args_path, "w") as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                       f, sort_keys=False)
    print(f"[INFO] Saved run arguments to {args_path}")

    # -----------------------------------------------------------------------
    # Periods / validation
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
        raise SystemExit(f"--write_period {write_period} must be within --store_period {store_period} and increasing.")

    years = years_in_range(wp0, wp1)

    # -----------------------------------------------------------------------
    # TL vars / renames and seeding requirement
    # -----------------------------------------------------------------------
    tl_vars_set = set(args.tl_vars or [])
    if {"lai_avh15c1", "lai_modis"} <= tl_vars_set:
        raise SystemExit("You cannot pass both 'lai_avh15c1' and 'lai_modis' to --tl_vars.")

    if tl_vars_set and (args.tl_initial_state is None):
        raise SystemExit("--tl_initial_state <YEAR> is required when --tl_vars is provided.")

    if args.tl_initial_state is not None:
        if not (store_start_year <= int(args.tl_initial_state) <= store_end_year):
            raise SystemExit(f"--tl_initial_state {args.tl_initial_state} must lie within [{store_start_year}..{store_end_year}].")

    rename_map: Dict[str, str] = {}
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
        """Replace base names using rename_map (order-preserving; avoid dups)."""
        out: List[str] = []
        for v in names:
            actual = rename_map.get(v, v)
            if actual not in out:
                out.append(actual)
        return out

    # -----------------------------------------------------------------------
    # Variable lists & standardisation stats
    # -----------------------------------------------------------------------
    exclude_set = {v.strip() for v in args.exclude_vars.split(",") if v.strip()} if args.exclude_vars else set()

    std_dict, invalid_vars = load_standardisation_only(std_dict_path)
    if invalid_vars:
        ex_preview = sorted(list(invalid_vars))[:10]
        print(f"[INFO] Dropping {len(invalid_vars)} var(s) with invalid stats. e.g., {ex_preview}")

    def _keep(name: str) -> bool:
        return (name not in exclude_set) and (name not in invalid_vars)

    DAILY_FORCING   = sorted([v for v in _apply_renames(var_names["daily_forcing"])     if _keep(v)])
    MONTHLY_FORCING = sorted([v for v in _apply_renames(var_names["monthly_forcing"])   if _keep(v)])
    MONTHLY_STATES  = sorted([v for v in _apply_renames(var_names["monthly_states"])    if _keep(v)])
    ANNUAL_FORCING  = sorted([v for v in _apply_renames(var_names["annual_forcing"])    if _keep(v)])
    ANNUAL_STATES   = sorted([v for v in _apply_renames(var_names["annual_states"])     if _keep(v)])
    monthly_fluxes_all = sorted([v for v in _apply_renames(var_names["monthly_fluxes"]) if _keep(v)])

    # OUTPUT = [monthly_fluxes | monthly_states | annual_states]
    OUTPUT_ORDER = monthly_fluxes_all + MONTHLY_STATES + ANNUAL_STATES
    MONTHLY_OUT_VARS = monthly_fluxes_all + MONTHLY_STATES
    ANNUAL_OUT_VARS  = ANNUAL_STATES

    # Sanity
    missing_monthly = set(MONTHLY_OUT_VARS) - set(OUTPUT_ORDER)
    if missing_monthly:
        raise ValueError(f"Monthly outputs missing from OUTPUT_ORDER: {sorted(missing_monthly)}")
    missing_annual = set(ANNUAL_OUT_VARS) - set(OUTPUT_ORDER)
    if missing_annual:
        raise ValueError(f"Annual outputs missing from OUTPUT_ORDER: {sorted(missing_annual)}")

    spec = InferenceSpec(
        DAILY_FORCING=DAILY_FORCING,
        MONTHLY_FORCING=MONTHLY_FORCING,
        MONTHLY_STATES=MONTHLY_STATES,
        ANNUAL_FORCING=ANNUAL_FORCING,
        ANNUAL_STATES=ANNUAL_STATES,
        OUTPUT_ORDER=OUTPUT_ORDER,
    )

    print(f"[INFO] Inference input_dim={spec.nin} | output_dim={len(OUTPUT_ORDER)}")

    # TL seeding provenance blob (optional)
    tl_seed_cfg: Optional[dict] = None
    if tl_vars_set:
        mapped = [rename_map.get("lai", None)]
        state_targets = sorted([m for m in mapped if (m is not None and m in MONTHLY_STATES)])
        if not state_targets:
            print("[WARN] --tl_vars provided but none mapped into MONTHLY_STATES; seeding will be a no-op.")
        tl_seed_cfg = {
            "seed_year": int(args.tl_initial_state),
            "state_map": rename_map.copy(),
            "state_targets": state_targets,
        }
        with open(info_dir / "tl_seed_cfg.json", "w") as f:
            json.dump(tl_seed_cfg, f, indent=2, sort_keys=True)

    # -----------------------------------------------------------------------
    # Scenario resolution
    # -----------------------------------------------------------------------
    if args.scenario == "all":
        scenarios = ['S0', 'S1', 'S2', 'S3']
    elif args.scenario in ['S0', 'S1', 'S2', 'S3']:
        scenarios = [args.scenario]
    else:
        raise SystemExit("Scenario argument not recognised")

    def resolve_roots_for(scen: str) -> tuple[Path, Path]:
        base = args.out_dir / args.job_name
        zarr_root = base / "zarr" / scen
        nc_root   = base / "netcdf" / scen
        if args.array_name:
            zarr_root = zarr_root / args.array_name
            nc_root   = nc_root   / args.array_name
        return zarr_root, nc_root

    # ======================================================================
    # Per-scenario processing
    # ======================================================================
    for scenario in scenarios:
        zarr_root, nc_root = resolve_roots_for(scenario)
        run_root = zarr_root
        print(f"Writing Zarr into {run_root}")

        # (Re)create stores under lock
        if args.overwrite_skeleton and run_root.exists():
            print(f"[WARN] --overwrite_skeleton requested: removing {run_root}")
            shutil.rmtree(run_root)

        lock_file = run_root / ".init.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with simple_lock(lock_file):
            daily_path, monthly_path, annual_path, tiles_json_path, _ = ensure_all_stores(
                run_root=run_root,
                period=(sp0, sp1),
                daily_vars=OUTPUT_ORDER,     # daily holds all outputs (daily means)
                monthly_vars=OUTPUT_ORDER,   # monthly holds all outputs (monthly means)
                annual_vars=[],              # no dedicated annual outputs store
                tile_h=args.tile_h,
                tile_w=args.tile_w,
            )

        # Optional coordinate repair
        if args.repair_coords:
            print("[INFO] Repairing coord arrays …")
            repair_coords(daily_path,   time_res="daily")
            repair_coords(monthly_path, time_res="monthly")
            repair_coords(annual_path,  time_res="annual")
            for p in (daily_path, monthly_path, annual_path):
                if p.exists():
                    zarr.consolidate_metadata(zarr.DirectoryStore(str(p)))
            print("[INFO] Repaired coords. Exiting.")
            return

        # Open store groups
        g_daily   = zarr.open_group(store=zarr.DirectoryStore(str(daily_path)),   mode="a")
        g_monthly = zarr.open_group(store=zarr.DirectoryStore(str(monthly_path)), mode="a")
        g_annual  = zarr.open_group(store=zarr.DirectoryStore(str(annual_path)),  mode="a")

        with open(tiles_json_path) as f:
            tiles_json = json.load(f)
        ntiles = len(tiles_json["tiles"])

        ocean_json = masks_dir / "tiles_ocean_only.json"
        ocean_only = _load_ocean_only_indices(ocean_json)

        print(f"[INFO] Store period: {sp0}–{sp1} (years {store_start_year}–{store_end_year})")
        print(f"[INFO] Write period: {wp0}–{wp1} (years {write_start_year}–{write_end_year})")
        print(f"[INFO] Years to predict in write window: {len(years)}")

        # ---------------- Predictions ----------------
        if not args.export_nc_only:
            forc = open_forcing_stores(args.forcing_dir / scenario)

            device = torch.device(args.device if args.device in ["cpu", "cuda"] else "cpu")
            print(f"[INFO] Using device: {device}")

            print(f"[INFO] Loading checkpoint from {args.weights}")
            ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
            ck_input_dim, ck_output_dim, ck_model_cfg = _extract_dims_and_cfg(ckpt)
            print(f"[INFO] Checkpoint dims: input={ck_input_dim}, output={ck_output_dim}")

            _check_dims_or_die(
                ck_input_dim=ck_input_dim,
                ck_output_dim=ck_output_dim,
                nin_expected=spec.nin,
                out_names_expected=OUTPUT_ORDER,
            )

            # Indices for monthly states (input/output)
            input_names = (DAILY_FORCING + MONTHLY_FORCING + MONTHLY_STATES + ANNUAL_FORCING + ANNUAL_STATES)
            in_monthly_state_idx  = [input_names.index(v) for v in MONTHLY_STATES]
            out_monthly_state_idx = [OUTPUT_ORDER.index(v) for v in MONTHLY_STATES]

            model = YearProcessor(
                input_dim=ck_input_dim,
                output_dim=ck_output_dim,
                in_monthly_state_idx=in_monthly_state_idx or None,
                out_monthly_state_idx=out_monthly_state_idx or None,
                **ck_model_cfg,
            ).float().to(device)

            # Runtime mode
            mode = "sequential_months" if args.sequential_months else "batch_months"
            model.set_mode(mode)
            print(f"[INFO] Model runtime mode: {mode}")
            print(
                "[INFO] Carry policy: "
                + (
                    "annual_broadcast"
                    if (args.carry_forward_states and not args.sequential_months)
                    else ("dec_to_jan_seed" if (args.carry_forward_states and args.sequential_months) else "disabled")
                )
                + " | nudging/carry happen in PHYSICAL space"
            )

            # Load weights (non-strict to allow benign shape/key drift)
            state_dict = _extract_state_dict(ckpt if isinstance(ckpt, dict) else {})
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print("[WARN] load_state_dict(strict=False) differences:")
                if missing:    print("  Missing:", missing)
                if unexpected: print("  Unexpected:", unexpected)
            model.eval()

            # Standardisation vectors (broadcasted to [1,1,1,C])
            mu_in,  sd_in  = mu_sd_from_std(input_names,   std_dict)
            mu_out, sd_out = mu_sd_from_std(OUTPUT_ORDER,  std_dict)

            MU_IN_B  = mu_in.reshape(1, 1, 1, -1)
            SD_IN_B  = sd_in.reshape(1, 1, 1, -1)
            MU_OUT_B = mu_out.reshape(1, 1, 1, -1)
            SD_OUT_B = sd_out.reshape(1, 1, 1, -1)

            # Tile distribution
            if args.tile_index is not None:
                idxs = [args.tile_index]
                assigned_land  = [i for i in idxs if i not in ocean_only]
                assigned_ocean = [i for i in idxs if i in ocean_only]
                total_shards = None
            else:
                all_tiles   = list(range(ntiles))
                land_tiles  = [i for i in all_tiles if i not in ocean_only]
                ocean_tiles = [i for i in all_tiles if i in ocean_only]
                assigned_land  = [t for j, t in enumerate(land_tiles)  if (j % args.shards) == safe_shard_id]
                assigned_ocean = [t for j, t in enumerate(ocean_tiles) if (j % args.shards) == safe_shard_id]
                total_shards = args.shards

            todo    = assigned_land
            skipped = assigned_ocean

            # Done-flag filtering
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
                land_total   = len([i for i in range(ntiles) if i not in ocean_only])
                ocean_total  = len([i for i in range(ntiles) if i in ocean_only])
                print(f"[INFO] Shard {safe_shard_id}/{total_shards}: "
                      f"{len(assigned_land)} land assigned, {len(assigned_ocean)} ocean assigned "
                      f"(of total {land_total} land / {ocean_total} ocean).")
            if skipped:
                print(f"[INFO] Skipping {len(skipped)} ocean-only tiles per {ocean_json}")

            # Tile loop (batch concurrency via threads)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            failures: List[int] = []
            ntiles_total = len(todo)
            tiles_done = 0
            concurrency = max(1, int(args.number_tiles))

            def _run_one_tile(ti: int) -> tuple[int, Optional[Exception]]:
                try:
                    process_one_tile(
                        spec=spec,
                        tile_index=ti,
                        tiles_json=tiles_json,
                        forc=forc,
                        g_daily=g_daily,
                        g_monthly=g_monthly,
                        g_annual=g_annual,
                        monthly_vars=OUTPUT_ORDER,
                        annual_vars=[],
                        years=years,
                        model=model,
                        device=device,
                        MU_IN_B=MU_IN_B,
                        SD_IN_B=SD_IN_B,
                        MU_OUT_B=MU_OUT_B,
                        SD_OUT_B=SD_OUT_B,
                        tiles_done_before=tiles_done,  # informational
                        ntiles_total=ntiles_total,     # informational
                        nfert=nfert,
                        store_start_year=store_start_year,
                        std_dict=std_dict,
                        nudge_lambda=args.nudge_lambda,
                        nudge_mode=args.nudge_mode,
                        carry_forward_states=args.carry_forward_states,
                        sequential_months=args.sequential_months,
                        tl_seed_cfg=tl_seed_cfg,
                        forcing_offsets=FORCING_OFFSETS,
                    )
                    mark_tile_done(run_root, ti, years=years)
                    return (ti, None)
                except Exception as e:
                    return (ti, e)

            if ntiles_total == 0:
                print("[INFO] No land tiles assigned; skipping tile loop.")
            else:
                print(f"[INFO] Processing {ntiles_total} tile(s) with concurrency={concurrency}")
                for start in range(0, ntiles_total, concurrency):
                    batch = todo[start:start + concurrency]
                    with ThreadPoolExecutor(max_workers=concurrency) as ex:
                        futures = {ex.submit(_run_one_tile, ti): ti for ti in batch}
                        for fut in as_completed(futures):
                            ti = futures[fut]
                            ti_, err = fut.result()
                            if err is not None:
                                print(f"[ERR] tile {ti} failed: {err}")
                                failures.append(ti)
                            else:
                                tiles_done += 1

            if failures:
                print(f"[DONE] Completed with {len(failures)} failures: {sorted(failures)}")
                raise SystemExit(1)

            print("[DONE] All assigned tiles completed.")

        # -------------------------------------------------------------------
        # Finalize: wait for all tiles, consolidate once, then export
        # -------------------------------------------------------------------
        all_land_tiles = [i for i in range(ntiles) if i not in ocean_only]
        n_total_land   = len(all_land_tiles)

        print(f"[FINALIZE] Waiting for all {n_total_land} land tiles to finish…")
        while True:
            n_done = sum(1 for t in all_land_tiles if is_tile_done(run_root, t))
            if n_done >= n_total_land:
                break
            print(f"[FINALIZE] {n_done}/{n_total_land} tiles done; sleeping 30s …")
            time.sleep(30)

        finalize_lock = run_root / ".export.lock"
        try:
            if is_leader:
                with simple_lock(finalize_lock, timeout=600):
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
            else:
                print("[FINALIZE] Non-leader shard; skipping consolidation (leader will do it).")
        except TimeoutError as e:
            print(f"[FINALIZE][INFO] Could not acquire finalize lock: {e}. Another task is finalizing.")

        # -------------------------
        # NetCDF export (sharded)
        # -------------------------
        if args.export_nc:
            print(f"Exporting NetCDF into {nc_root}")
            print(f"[FINALIZE] Sharded NetCDF export: shard {safe_shard_id}/{args.shards}")
            export_netcdf_sharded(
                monthly_zarr=monthly_path,
                nc_root=nc_root,
                shards=int(args.shards),
                shard_id=int(safe_shard_id or 0),
                overwrite=bool(args.overwrite_data),
                var_order=OUTPUT_ORDER,
                annual_vars=ANNUAL_OUT_VARS, 
            )

            # Mark this shard's export completion (for ILAMB copy barrier)
            def _mark_export_nc_done(root: Path, shard_id: Optional[int]) -> None:
                d = root / ".nc.done"
                d.mkdir(parents=True, exist_ok=True)
                name = "solo.ok" if shard_id is None else f"shard_{int(shard_id)}.ok"
                (d / name).write_text("ok\n")

            _mark_export_nc_done(run_root, shard_id=safe_shard_id)

            # Optional ILAMB copies (leader only, after all shards exported)
            if args.ilamb_dir_global or args.ilamb_dir_test:

                def _wait_for_all_exports(root: Path, shards: Optional[int], timeout_s: int = 36000) -> None:
                    if not shards:
                        return
                    d = root / ".nc.done"
                    deadline = time.time() + timeout_s
                    while True:
                        done = len(list(d.glob("shard_*.ok")))
                        if done >= int(shards):
                            break
                        if time.time() > deadline:
                            raise TimeoutError(f"Timed out waiting for {shards} shard export markers in {d}")
                        print(f"[FINALIZE] Waiting for sharded NetCDF exports: {done}/{shards} done; sleeping 15s …")
                        time.sleep(15)

                def _copy_nc_dir(src_dir: Path, dst_dir: Path, overwrite: bool = False) -> int:
                    if not src_dir.exists():
                        print(f"[ILAMB] Source missing, skip: {src_dir}")
                        return 0
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    n = 0
                    for p in sorted(src_dir.glob("*.nc")):
                        q = dst_dir / p.name
                        if q.exists() and not overwrite:
                            continue
                        shutil.copy2(p, q)
                        n += 1
                    print(f"[ILAMB] Copied {n} file(s) from {src_dir} → {dst_dir}{' (overwrite)' if overwrite else ''}")
                    return n

                if is_leader:
                    try:
                        _wait_for_all_exports(run_root, shards=args.shards)
                    except TimeoutError as e:
                        print(f"[FINALIZE][WARN] ILAMB copy barrier timed out: {e}. Proceeding may yield partial copies.")
                    overwrite_copy = bool(args.overwrite_data)

                    # Global copy: <ilamb_dir>/<job>/<scenario>/
                    if args.ilamb_dir_global:
                        dst_global = Path(args.ilamb_dir_global) / args.job_name / scenario
                        _copy_nc_dir(nc_root / "full", dst_global, overwrite=overwrite_copy)

                    # Test copy: <ilamb_dir_test>/<job>/<scenario>/early and /late
                    if args.ilamb_dir_test:
                        test_root = Path(args.ilamb_dir_test) / args.job_name / scenario
                        _copy_nc_dir(nc_root / "test" / "early", test_root / "early", overwrite=overwrite_copy)
                        _copy_nc_dir(nc_root / "test" / "late",  test_root / "late",  overwrite=overwrite_copy)
                else:
                    print("[FINALIZE] Non-leader shard; skipping ILAMB copies (leader will copy after export barrier).")
        else:
            print("[FINALIZE] Export disabled (args.export_nc is False)")


# ============================================================================ #
# Entrypoint
# ============================================================================ #
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