#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Subset

# ---------------- Project imports ---------------- #
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.paths.paths import std_dict_path
from src.training.stats import load_standardisation_only, set_seed
from src.dataset.dataloader import get_train_val_test, get_data
from src.dataset.dataset import base
from src.training.loss import build_loss_fn
# ---------------- Helpers ---------------- #

def expand_monthly_to_daily_torch(monthly: torch.Tensor) -> torch.Tensor:
    device = monthly.device
    month_lengths = torch.tensor([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], device=device)
    day_to_month = torch.repeat_interleave(torch.arange(12, device=device), month_lengths)
    return monthly[:, day_to_month, :]

def build_mb_var_idx_exact(
    output_names: list[str],
    *,
    available_extra: set[str] | frozenset[str] = frozenset(),
    strict: bool = False,
    debug: bool = True
) -> dict[str, int]:
    name_to_idx = {n: i for i, n in enumerate(output_names)}

    # Variables we expect among OUTPUTS (exclude cTotal_annual entirely; 'pre' comes from extras)
    canonical_vars = [
        # water balance
        "mrso", "mrro", "evapotrans",
        # npp balance
        "npp", "gpp", "ra",
        # nbp balance
        "nbp", "rh", "fFire", "fLuc",
        # carbon / totals used by December partition + ΔcTotal check
        "cTotal_monthly", "cVeg", "cLitter", "cSoil",
    ]

    found = {v: name_to_idx[v] for v in canonical_vars if v in name_to_idx}

    # Don't call out vars as "missing" if they are satisfied via extras (e.g., 'pre')
    missing_outputs = [
        v for v in canonical_vars
        if (v not in name_to_idx) and (v not in available_extra)
    ]

    # Only the balances we actually support/print now
    required = {
        "water_balance": {"mrso", "pre", "mrro", "evapotrans"},          # 'pre' may be provided via extras
        "npp_balance": {"npp", "gpp", "ra"},
        "nbp_balance": {"nbp", "npp", "rh", "fFire", "fLuc"},
        "nbp_vs_delta_ctotal_monthly": {"cTotal_monthly", "nbp"},
        "carbon_partition_december": {"cTotal_monthly", "cVeg", "cLitter", "cSoil"},
    }

    have_outputs = set(found.keys())
    have_all = have_outputs | set(available_extra)

    enabled = {k: req.issubset(have_all) for k, req in required.items()}

    if debug:
        print(f"[mass-balance] present in OUTPUTS: {sorted(have_outputs)}")
        if available_extra:
            print(f"[mass-balance] available from EXTRAS: {sorted(set(available_extra))}")
        if missing_outputs:
            print(f"[mass-balance] missing in OUTPUTS: {sorted(missing_outputs)}")
        print(f"[mass-balance] balances enabled (considering outputs ∪ extras): {enabled}")

    if strict:
        disabled = {k: sorted(list(req - have_all)) for k, req in required.items() if not req.issubset(have_all)}
        if disabled:
            msgs = [f"{k}: missing {v}" for k, v in disabled.items()]
            raise RuntimeError("Required variables missing for some balances: " + "; ".join(msgs))

    return found

def make_pseudo_preds_for_year(
    monthly_year: torch.Tensor,  # [B,12,Cm]
    annual_year: torch.Tensor,   # [B, 1,Ca]
    out_order: list[str],
) -> torch.Tensor:
    daily_monthly = expand_monthly_to_daily_torch(monthly_year)  # [B,365,Cm]
    daily_annual  = annual_year.repeat(1, 365, 1)                # [B,365,Ca]
    preds = torch.cat([daily_monthly, daily_annual], dim=-1)     # [B,365,Cout]
    return preds

def compute_balance_residuals(
    preds: torch.Tensor,
    idx: dict,
    *,
    month_mask: torch.Tensor,
    month_lengths: torch.Tensor,
    extra_daily: dict[str, torch.Tensor] | None = None,
    **_ignored,
) -> dict[str, torch.Tensor]:
    device = preds.device
    mmask = month_mask.to(device)      # [12,365]
    mlen  = month_lengths.to(device)   # [12]

    out: dict[str, torch.Tensor] = {}

    def monthly_avg_from_daily(d: torch.Tensor) -> torch.Tensor:
        # d: [B,365] -> [B,12]
        return torch.einsum("bc,mc->bm", d, mmask) / mlen

    def monthly_avg(chan_idx: int) -> torch.Tensor:
        x = preds[:, :, chan_idx]                               # [B,365]
        return monthly_avg_from_daily(x)

    def annual_mean(chan_idx: int) -> torch.Tensor:
        return preds[:, :, chan_idx].mean(dim=1)

    def get_daily_series(name: str) -> torch.Tensor | None:
        if extra_daily is not None and name in extra_daily:
            x = extra_daily[name]
            if x.dim() == 3 and x.size(-1) == 1:
                x = x.squeeze(-1)
            return x.to(device)
        if name in idx:
            return preds[:, :, idx[name]]
        return None

    # ---------- Water balance (ΔS ≈ ∫(P - R - E) dt) ----------
    mrso_d = get_daily_series("mrso")
    pre_d  = get_daily_series("pre")
    mrro_d = get_daily_series("mrro")
    evap_d = get_daily_series("evapotrans")
    if all(x is not None for x in (mrso_d, pre_d, mrro_d, evap_d)):
        mrso_m = monthly_avg_from_daily(mrso_d)   # [B,12]
        pre_m  = monthly_avg_from_daily(pre_d)
        mrro_m = monthly_avg_from_daily(mrro_d)
        evap_m = monthly_avg_from_daily(evap_d)

        seconds_per_day = torch.tensor(86400.0, device=device)
        seconds_in_month = (month_lengths * seconds_per_day).view(1, 12)
        flux_int = (pre_m - mrro_m - evap_m) * seconds_in_month  # [B,12]

        d_mrso = mrso_m[:, 1:] - mrso_m[:, :-1]                  # [B,11]
        out["water_balance"] = d_mrso - flux_int[:, 1:]          # [B,11]

    # NPP = GPP - Ra
    if all(k in idx for k in ("npp", "gpp", "ra")):
        npp = monthly_avg(idx["npp"])
        gpp = monthly_avg(idx["gpp"])
        ra  = monthly_avg(idx["ra"])
        out["npp_balance"] = npp - (gpp - ra)     # [B,12]

    # NBP = NPP - Rh - fFire - fLuc
    if all(k in idx for k in ("nbp", "npp", "rh", "fFire", "fLuc")):
        nbp   = monthly_avg(idx["nbp"])
        npp   = monthly_avg(idx["npp"])
        rh    = monthly_avg(idx["rh"])
        fFire = monthly_avg(idx["fFire"])
        fLuc  = monthly_avg(idx["fLuc"])
        out["nbp_balance"] = nbp - (npp - rh - fFire - fLuc)  # [B,12]

    # Δ cTotal_monthly == NBP (monthly)
    if all(k in idx for k in ("cTotal_monthly", "nbp")):
        cTot = monthly_avg(idx["cTotal_monthly"])  # [B,12]
        nbp  = monthly_avg(idx["nbp"])             # [B,12]
        out["nbp_vs_delta_ctotal_monthly"] = (cTot[:, 1:] - cTot[:, :-1]) - nbp[:, 1:]  # [B,11]

    # cTotal_annual == cVeg + cLitter + cSoil
    if all(k in idx for k in ("cTotal_annual", "cVeg", "cLitter", "cSoil")):
        cTotA = annual_mean(idx["cTotal_annual"])  # [B]
        cVeg  = annual_mean(idx["cVeg"])           # [B]
        cLit  = annual_mean(idx["cLitter"])        # [B]
        cSoil = annual_mean(idx["cSoil"])          # [B]
        out["carbon_partition"] = cTotA - (cVeg + cLit + cSoil)   # [B]

    # December cTotal_monthly == cVeg + cLitter + cSoil (annual means)
    if all(k in idx for k in ("cTotal_monthly", "cVeg", "cLitter", "cSoil")):
        cTot_m_all = monthly_avg(idx["cTotal_monthly"])  # [B,12]
        cTot_dec   = cTot_m_all[:, 11]                   # [B]
        cVeg  = annual_mean(idx["cVeg"])                 # [B]
        cLit  = annual_mean(idx["cLitter"])              # [B]
        cSoil = annual_mean(idx["cSoil"])                # [B]
        out["carbon_partition_december"] = cTot_dec - (cVeg + cLit + cSoil)  # [B]

    return out

def save_npz_residuals(path: Path, residuals_accum: dict[str, list[np.ndarray]], meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.concatenate(v, axis=0) for k, v in residuals_accum.items() if v}
    np.savez_compressed(path, **arrays)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

# ---------------- Main ---------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train","val","test","all"], default="all",
                   help="Which split to process. Use 'all' to process train/val/test (no sharding).")
    p.add_argument("--n_shards", type=int, default=1,
                   help="Number of shards for the chosen split (only used if split != 'all').")
    p.add_argument("--shard_id", type=int, default=0,
                   help="This task's shard id in [0..n_shards-1].")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--out_dir", type=Path, default=Path(
        "/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/training/other/mass_balance_labels/shards"
    ))
    args, _ = p.parse_known_args()  # tolerant to launcher extras

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load standardisation stats (current API)
    std_dict, invalid_vars = load_standardisation_only(std_dict_path)

    # 2) Build datasets (align with current get_train_val_test signature)
    #    Use defaults consistent with train.py; adjust if you need TL/delta_luh, etc.
    ds_dict = get_train_val_test(
        std_dict=std_dict,
        block_locs=70,
        exclude_vars=set(),
        tl_activated=False,
        tl_start=1981,
        tl_end=2019,
        replace_map=None,
        delta_luh=False,
    )

    # 3) DataLoaders
    if args.split == "all":
        train_dl, val_dl, test_dl = get_data(
            ds_dict["train"], ds_dict["val"], ds_dict["test"],
            bs=args.batch_size, num_workers=args.num_workers, ddp=False
        )
        split_loaders = [("train", train_dl), ("val", val_dl), ("test", test_dl)]
    else:
        base_ds = ds_dict[args.split]
        if args.n_shards > 1:
            N = len(base_ds)
            if not (0 <= args.shard_id < args.n_shards):
                raise SystemExit(f"shard_id must be in [0,{args.n_shards-1}]")
            starts = np.linspace(0, N, args.n_shards + 1, dtype=int)
            lo, hi = starts[args.shard_id], starts[args.shard_id + 1]
            base_ds = Subset(base_ds, range(lo, hi))
            print(f"[{args.split}] shard {args.shard_id}/{args.n_shards}: indices [{lo}:{hi}) -> {len(base_ds)} samples")

        dl, _, _ = get_data(base_ds, base_ds, base_ds,
                            bs=args.batch_size, num_workers=args.num_workers, ddp=False)
        split_loaders = [(args.split, dl)]

    # 4) Recover OUTPUT_ORDER and daily forcing order from the dataset schema
    base_train = base(ds_dict["train"])
    if hasattr(base_train, "schema") and base_train.schema is not None:
        schema = base_train.schema
        monthly_names = schema.out_monthly_names()
        annual_names  = schema.out_annual_names()
        OUTPUT_ORDER  = monthly_names + annual_names
        # daily forcing
        try:
            daily_forcing_order = sorted(list(schema.daily_forcing))
        except Exception:
            daily_forcing_order = sorted(list(base_train.var_names["daily_forcing"]))
    else:
        # Fallback to var_names with sorted categories (as older code did)
        monthly_fluxes_order = sorted(base_train.var_names["monthly_fluxes"])
        monthly_states_order = sorted(base_train.var_names["monthly_states"])
        annual_states_order  = sorted(base_train.var_names["annual_states"])
        OUTPUT_ORDER = monthly_fluxes_order + monthly_states_order + annual_states_order
        daily_forcing_order = sorted(base_train.var_names["daily_forcing"])
        
    # We use the same std_dict loaded above (canonical source of truth).
    # This is a list of the mean and std dev in the correct order of the var heads in the model.
    mu_out = torch.tensor([float(std_dict[name]["mean"]) for name in OUTPUT_ORDER],
                        dtype=torch.float32, device=device)
    sd_out = torch.tensor([float(std_dict[name]["std"])  for name in OUTPUT_ORDER],
                        dtype=torch.float32, device=device)

    def make_loss_fn(idx_m, idx_a, *, wb=0.0, npp=0.0, nbp=0.0, cp=0.0, nbpdc=0.0, mb_var_idx=None):
        # zeros → supervised terms disabled; only physics penalties contribute
        monthly_zero = [0.0] * idx_m
        annual_zero  = [0.0] * idx_a
        return build_loss_fn(
            idx_monthly=list(range(idx_m)),
            idx_annual=list(range(idx_m, idx_m + idx_a)),
            use_mass_balances=True,               
            loss_type="mse",
            monthly_weights=monthly_zero,
            annual_weights=annual_zero,
            mb_var_idx=mb_var_idx,
            water_balance_weight=wb,
            npp_balance_weight=npp,
            nbp_balance_weight=nbp,
            carbon_partition_weight=cp,
            nbp_delta_ctotal_weight=nbpdc,
            mu_out=mu_out,
            sd_out=sd_out,
        )

    # Month meta tensors
    month_lengths = torch.tensor([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=torch.float32, device=device)
    day_to_month = torch.repeat_interleave(torch.arange(12, device=device), month_lengths.to(torch.long))
    month_mask = torch.nn.functional.one_hot(day_to_month, num_classes=12).T.float()  # [12,365]

    # 5) Mass-balance var index map (in OUTPUT space)
    mb_var_idx = build_mb_var_idx_exact(
        OUTPUT_ORDER,
        available_extra={"pre"},   # <-- tell it we will supply pre from inputs
        strict=False,
        debug=True
    )

    # Optional: which balances will run (informative only)
    def _will_run(balance: str, have: set[str]) -> bool:
        reqs = {
            "water_balance": {"mrso", "pre", "mrro", "evapotrans"},
            "npp_balance": {"npp", "gpp", "ra"},
            "nbp_balance": {"nbp", "npp", "rh", "fFire", "fLuc"},
            "nbp_vs_delta_ctotal_monthly": {"cTotal_monthly", "nbp"},
            "carbon_partition_december": {"cTotal_monthly", "cVeg", "cLitter", "cSoil"},
        }
        return reqs.get(balance, set()).issubset(have)

    # include extras in enabled-check
    extras = {"pre"}  # keep in sync with available_extra above
    have_all = set(mb_var_idx.keys()) | extras

    print("[mass-balance] balances enabled:",
        {k: _will_run(k, have_all) for k in
        ["water_balance", "npp_balance", "nbp_balance",
            "nbp_vs_delta_ctotal_monthly", "carbon_partition_december"]})

    if not mb_var_idx:
        print("[WARN] No overlap with OUTPUT_ORDER. Nothing to score.")
        return

    # Loss builders
    # If we used schema above, monthly/annual sizes are:
    if hasattr(base_train, "schema") and base_train.schema is not None:
        idx_m = len(schema.out_monthly_names())
        idx_a = len(schema.out_annual_names())
    else:
        idx_m = len([*sorted(base_train.var_names["monthly_fluxes"]), *sorted(base_train.var_names["monthly_states"])])
        idx_a = len(sorted(base_train.var_names["annual_states"]))

    loss_fns = {
        "water_balance":              make_loss_fn(idx_m, idx_a, wb=1, mb_var_idx=mb_var_idx),
        "npp_balance":                make_loss_fn(idx_m, idx_a, npp=1, mb_var_idx=mb_var_idx),
        "nbp_balance":                make_loss_fn(idx_m, idx_a, nbp=1, mb_var_idx=mb_var_idx),
        "carbon_partition_december":  make_loss_fn(idx_m, idx_a, cp=1, mb_var_idx=mb_var_idx),
        "nbp_delta_ctotal":           make_loss_fn(idx_m, idx_a, nbpdc=1, mb_var_idx=mb_var_idx),
        "all_on":                     make_loss_fn(idx_m, idx_a, wb=1, npp=1, nbp=1, cp=1, nbpdc=1, mb_var_idx=mb_var_idx),
    }
    balance_keys = list(loss_fns.keys())

    # Accumulators
    sums_by_split: dict[str, dict[str, float]] = {s: {k: 0.0 for k in balance_keys} for s,_ in split_loaders}
    counts_by_split: dict[str, int] = {s: 0 for s,_ in split_loaders}

    # Residuals saver
    def save_residuals(split_name, residuals_accum, split_sum_map, split_n,
                       out_dir_base: Path, shard_id: int, n_shards: int):
        shard_dir = out_dir_base / split_name / f"shard_{shard_id:03d}_of_{n_shards:03d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        out_npz = shard_dir / f"residuals_{split_name}_shard{shard_id:03d}-of-{n_shards:03d}.npz"
        meta_npz = {
            "split": split_name,
            "year_windows": split_n,
            "shard_id": shard_id,
            "n_shards": n_shards,
        }
        save_npz_residuals(out_npz, residuals_accum, meta_npz)

        means = {k: (split_sum_map[k] / max(1, split_n)) for k in split_sum_map.keys()}
        metrics = {
            "split": split_name,
            "year_windows": split_n,
            "shard_id": shard_id,
            "n_shards": n_shards,
            "means_per_balance": means,
            "mean_all_on": means.get("all_on", float("nan")),
        }
        with open(shard_dir / "metrics_balances.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[{split_name}][shard {shard_id}/{n_shards}] means: "
              + " | ".join(f"{k}={means[k]:.6f}" for k in balance_keys))

    # Main loop
    for split_name, loader in split_loaders:
        residuals_accum = defaultdict(list)
        split_sums = {k: 0.0 for k in balance_keys}
        split_n   = 0

        for inputs, labels_m, labels_a in loader:
            device_here = device
            inputs   = inputs.to(device_here)
            labels_m = labels_m.to(device_here)
            labels_a = labels_a.to(device_here)

            B, Cm, Tm, L = labels_m.shape
            _, Ca, Ta, _ = labels_a.shape
            if Tm % 12 != 0:
                raise ValueError(f"Monthly labels T={Tm} not divisible by 12.")
            n_years = Tm // 12
            if Ta != n_years:
                raise ValueError(f"Annual labels years {Ta} != {n_years} derived from monthly.")

            # Reshape to [B*L, T, C]
            lbl_m_bl_tc = labels_m.permute(0, 3, 2, 1).reshape(B * L, Tm, Cm)
            lbl_a_bl_ta = labels_a.permute(0, 3, 2, 1).reshape(B * L, Ta, Ca)

            for y in range(n_years):
                my = lbl_m_bl_tc[:, y*12:(y+1)*12, :]   # [B*L, 12, Cm]
                ay = lbl_a_bl_ta[:, y:y+1, :]           # [B*L,  1, Ca]

                preds = make_pseudo_preds_for_year(my, ay, OUTPUT_ORDER)  # [B*L, 365, Cm+Ca]

                # Zero labels (so only penalty terms contribute)
                dummy_m = torch.zeros((B * L, 12, Cm), device=device_here, dtype=preds.dtype)
                dummy_a = torch.zeros((B * L, 1,  Ca), device=device_here, dtype=preds.dtype)
                
                # ---- pull daily precipitation from inputs for this year (DESTANDARDIZED) ----
                extra_daily = {}
                try:
                    B_in, Cin, Td, L_in = inputs.shape
                    assert B_in == B and L_in == L, "inputs shape mismatch vs labels"
                    inp_bl_tc = inputs.permute(0, 3, 2, 1).reshape(B * L, Td, Cin)   # [B*L, Td, Cin]

                    # year slice (assumes 365 days per year)
                    td0, td1 = y * 365, (y + 1) * 365
                    inp_y = inp_bl_tc[:, td0:td1, :]                                 # [B*L, 365, Cin]

                    daily_force_order = sorted(base_train.var_names["daily_forcing"])
                    if "pre" in daily_force_order:
                        pre_idx = daily_force_order.index("pre")
                        pre_daily_norm = inp_y[:, :, pre_idx]                         # normalized
                        mu = float(std_dict["pre"]["mean"])
                        sd = float(std_dict["pre"]["std"])
                        pre_daily_phys = pre_daily_norm * sd + mu                    
                        extra_daily["pre"] = pre_daily_phys
                except Exception:
                    pass

                # Evaluate losses
                with torch.no_grad():
                    for k, lf in loss_fns.items():
                        v = lf(preds, dummy_m, dummy_a, extra_daily=extra_daily).item()
                        split_sums[k] += v

                    # destandardize preds to physical units to match how penalties are computed
                    preds_phys = preds * sd_out.view(1, 1, -1) + mu_out.view(1, 1, -1)

                    # residuals for plotting/saving (now in physical units)
                    res = compute_balance_residuals(
                        preds_phys, mb_var_idx, month_mask=month_mask, month_lengths=month_lengths,
                        extra_daily=extra_daily
                    )
                    
                    for key, tensor in res.items():
                        residuals_accum[key].append(tensor.reshape(-1).detach().cpu().numpy())

                split_n += 1

        # Save per split
        sums_by_split[split_name] = split_sums
        counts_by_split[split_name] = split_n
        save_residuals(split_name, residuals_accum, split_sums, split_n,
                       args.out_dir, args.shard_id, args.n_shards)

    # Overall summary
    total_n = sum(counts_by_split.values())
    if total_n > 0:
        totals = {k: 0.0 for k in balance_keys}
        for split_name in counts_by_split.keys():
            for k in balance_keys:
                totals[k] += sums_by_split[split_name][k]
        means = {k: totals[k] / total_n for k in balance_keys}
        print("[ALL] mean penalties: " + " | ".join(f"{k}={means[k]:.6f}" for k in balance_keys))

if __name__ == "__main__":
    main()