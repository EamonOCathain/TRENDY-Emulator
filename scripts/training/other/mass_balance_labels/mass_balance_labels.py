#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
import sys
import torch
import numpy as np
import argparse
from torch.utils.data import Subset
import json
from collections import defaultdict

# ---------------- Project imports ---------------- #
PROJECT_ROOT = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(PROJECT_ROOT))

from src.paths.paths import std_dict_path
from src.training.stats import load_and_filter_standardisation, set_seed
from src.dataset.variables import *
from src.dataset.dataset import base
from src.training.loss import custom_loss
from src.dataset.dataloader import get_train_val_test, get_data

# ---------------- Helpers ---------------- #

def expand_monthly_to_daily_torch(monthly: torch.Tensor) -> torch.Tensor:
    device = monthly.device
    month_lengths = torch.tensor([31,28,31,30,31,30,31,31,30,31,30,31], device=device)
    day_to_month = torch.repeat_interleave(torch.arange(12, device=device), month_lengths)
    return monthly[:, day_to_month, :]

def build_mb_var_idx_exact(
    output_names: list[str],
    *,
    available_extra: set[str] | frozenset[str] = frozenset(),
    strict: bool = False,   # set True to raise if a balance is missing vars
    debug: bool = True      # prints which balances are enabled/missing
) -> dict[str, int]:
    """
    EXACT name matching for mass-balance variables.
    Returns a dict {canonical_name -> index_in_OUTPUT_ORDER} for names that exist
    in outputs. For debug/enabling, also considers variables you will supply from
    inputs (e.g., daily 'pre') via `available_extra`.
    """

    name_to_idx = {n: i for i, n in enumerate(output_names)}

    # Canonical names (EXACT) from your variable lists
    canonical_vars = [
        # water balance
        "mrso", "pre", "mrro", "evapotrans",
        # npp balance
        "npp", "gpp", "ra",
        # nbp balance
        "nbp", "rh", "fFire", "fLuc",
        # carbon / totals
        "cTotal_monthly", "cTotal_annual", "cVeg", "cLitter", "cSoil",
    ]

    # What exists in outputs
    found = {v: name_to_idx[v] for v in canonical_vars if v in name_to_idx}
    missing_outputs = [v for v in canonical_vars if v not in name_to_idx]

    # Balance requirements
    required = {
        "water_balance": {"mrso", "pre", "mrro", "evapotrans"},
        "npp_balance": {"npp", "gpp", "ra"},
        "nbp_balance": {"nbp", "npp", "rh", "fFire", "fLuc"},
        "ctotal_monthly_vs_annual": {"cTotal_monthly", "cTotal_annual"},
        "nbp_vs_delta_ctotal_monthly": {"cTotal_monthly", "nbp"},
        "carbon_partition": {"cTotal_annual", "cVeg", "cLitter", "cSoil"},
        # Optional alternative check using December of cTotal_monthly:
        "carbon_partition_december": {"cTotal_monthly", "cVeg", "cLitter", "cSoil"},
    }

    have_outputs = set(found.keys())
    have_all = have_outputs | set(available_extra)

    # Enabled considering outputs ∪ extras (extras are typically daily forcings like 'pre')
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

    # Return only the output-index map (extras are not in outputs, so no index)
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
    **_ignored,   # swallow legacy kwargs like extra_monthly/extra_annual
) -> dict[str, torch.Tensor]:
    """
    Compute residual tensors for each balance that’s feasible with provided vars.
    Residual shapes:
      - monthly identities: [B, 12]  (or [B, 11] for month-to-month deltas)
      - annual identities:  [B]

    Notes:
      * For water balance, daily series may be pulled from `extra_daily` when
        not present in `preds` outputs (e.g., daily 'pre').
      * Adds optional 'carbon_partition_december': December cTotal_monthly vs
        (annual mean of cVeg + cLitter + cSoil).
    """
    device = preds.device
    mmask = month_mask.to(device)      # [12,365]
    mlen  = month_lengths.to(device)   # [12]

    out: dict[str, torch.Tensor] = {}

    def monthly_avg_from_daily(d: torch.Tensor) -> torch.Tensor:
        # d: [B,365] -> [B,12]
        return torch.einsum("bc,mc->bm", d, mmask) / mlen

    def monthly_avg(chan_idx: int) -> torch.Tensor:
        # from outputs daily channel -> [B,12]
        x = preds[:, :, chan_idx]                               # [B,365]
        return monthly_avg_from_daily(x)

    def annual_mean(chan_idx: int) -> torch.Tensor:
        # from outputs daily channel -> [B]
        return preds[:, :, chan_idx].mean(dim=1)

    def get_daily_series(name: str) -> torch.Tensor | None:
        # Try extras first (e.g., 'pre' from inputs), then outputs
        if extra_daily is not None and name in extra_daily:
            x = extra_daily[name]
            # Expect [B,365]; if [B,365,1], squeeze last dim
            if x.dim() == 3 and x.size(-1) == 1:
                x = x.squeeze(-1)
            return x.to(device)
        if name in idx:
            return preds[:, :, idx[name]]
        return None

    # ---------- Water balance (ΔS = P - R - E) ----------
    need = ("mrso", "pre", "mrro", "evapotrans")
    mrso_d = get_daily_series("mrso")
    pre_d  = get_daily_series("pre")
    mrro_d = get_daily_series("mrro")
    evap_d = get_daily_series("evapotrans")
    if all(x is not None for x in (mrso_d, pre_d, mrro_d, evap_d)):
        mrso_m = monthly_avg_from_daily(mrso_d)   # [B,12]
        pre_m  = monthly_avg_from_daily(pre_d)
        mrro_m = monthly_avg_from_daily(mrro_d)
        evap_m = monthly_avg_from_daily(evap_d)
        d_mrso = mrso_m[:, 1:] - mrso_m[:, :-1]   # [B,11]
        rhs    = pre_m[:, 1:] - mrro_m[:, 1:] - evap_m[:, 1:]
        out["water_balance"] = d_mrso - rhs       # [B,11]

    # ---------- NPP = GPP - Ra ----------
    need = ("npp", "gpp", "ra")
    if all(k in idx for k in need):
        npp = monthly_avg(idx["npp"])
        gpp = monthly_avg(idx["gpp"])
        ra  = monthly_avg(idx["ra"])
        out["npp_balance"] = npp - (gpp - ra)     # [B,12]

    # ---------- NBP = NPP - Rh - fFire - fLuc ----------
    need = ("nbp", "npp", "rh", "fFire", "fLuc")
    if all(k in idx for k in need):
        nbp   = monthly_avg(idx["nbp"])
        npp   = monthly_avg(idx["npp"])
        rh    = monthly_avg(idx["rh"])
        fFire = monthly_avg(idx["fFire"])
        fLuc  = monthly_avg(idx["fLuc"])
        out["nbp_balance"] = nbp - (npp - rh - fFire - fLuc)  # [B,12]

    # ---------- mean(monthly cTotal) == annual cTotal (if both exist) ----------
    need = ("cTotal_monthly", "cTotal_annual")
    if all(k in idx for k in need):
        cTot_m = monthly_avg(idx["cTotal_monthly"]).mean(dim=1)   # [B]
        cTot_a = annual_mean(idx["cTotal_annual"])                # [B]
        out["ctotal_monthly_vs_annual"] = cTot_m - cTot_a         # [B]

    # ---------- Δ cTotal_monthly == NBP (monthly) ----------
    need = ("cTotal_monthly", "nbp")
    if all((k in idx) for k in need):
        cTot = monthly_avg(idx["cTotal_monthly"])   # [B,12]
        nbp  = monthly_avg(idx["nbp"])              # [B,12]
        out["nbp_vs_delta_ctotal_monthly"] = (cTot[:, 1:] - cTot[:, :-1]) - nbp[:, 1:]  # [B,11]

    # ---------- cTotal_annual == cVeg + cLitter + cSoil ----------
    need = ("cTotal_annual", "cVeg", "cLitter", "cSoil")
    if all(k in idx for k in need):
        cTotA = annual_mean(idx["cTotal_annual"])  # [B]
        cVeg  = annual_mean(idx["cVeg"])           # [B]
        cLit  = annual_mean(idx["cLitter"])        # [B]
        cSoil = annual_mean(idx["cSoil"])          # [B]
        out["carbon_partition"] = cTotA - (cVeg + cLit + cSoil)   # [B]

    # ---------- December cTotal_monthly == cVeg + cLitter + cSoil (annual means) ----------
    # Use this when you DON'T have cTotal_annual; compares Dec cTotal_monthly to annual pools.
    need = ("cTotal_monthly", "cVeg", "cLitter", "cSoil")
    if all(k in idx for k in need):
        cTot_m_all = monthly_avg(idx["cTotal_monthly"])  # [B,12]
        cTot_dec   = cTot_m_all[:, 11]                   # [B], December (0-based index)
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

def make_loss_fn(idx_m: int, idx_a: int, *, wb=0.0, npp=0.0, nbp=0.0, cp=0.0, nbpdc=0.0, mb_var_idx=None):
    return custom_loss(
        idx_monthly=list(range(idx_m)),
        idx_annual=list(range(idx_m, idx_m + idx_a)),
        monthly_weights=[0.0] * idx_m,
        annual_weights=[0.0] * idx_a,
        loss_type="mse",
        mb_var_idx=mb_var_idx,
        water_balance_weight=wb,
        npp_balance_weight=npp,
        nbp_balance_weight=nbp,
        carbon_partition_weight=cp,
        nbp_delta_ctotal_weight=nbpdc,
    )

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
    args = p.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load std + prune lists
    std_dict, pruned = load_and_filter_standardisation(
        standardisation_path=std_dict_path,
        all_vars=all_vars,
        daily_vars=daily_vars,
        monthly_vars=monthly_vars,
        annual_vars=annual_vars,
        monthly_states=monthly_states,
        annual_states=annual_states,
        exclude_vars=set(),
    )

    # 2) Datasets
    ds_dict = get_train_val_test(std_dict)

    # 3) Build loaders
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

    # 4) Recover OUTPUT_ORDER
    base_train = base(ds_dict["train"])
    monthly_fluxes_order = sorted(base_train.var_names["monthly_fluxes"])
    monthly_states_order = sorted(base_train.var_names["monthly_states"])
    annual_states_order  = sorted(base_train.var_names["annual_states"])
    OUTPUT_ORDER = monthly_fluxes_order + monthly_states_order + annual_states_order

    # Month meta (tensors) for residuals (compute once)
    month_lengths = torch.tensor(
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        dtype=torch.float32,
        device=device,
    )
    day_to_month = torch.repeat_interleave(torch.arange(12, device=device), month_lengths.to(torch.long))
    month_mask = torch.nn.functional.one_hot(day_to_month, num_classes=12).T.float()  # [12,365]
    
    # 5) Mass-balance var index map
    mb_var_idx = build_mb_var_idx_exact(OUTPUT_ORDER, strict=False, debug=True)
    
    
    # Optional: print which balances will be active
    def _will_run(balance: str, have: set[str]) -> bool:
        reqs = {
            "water_balance": {"mrso","pre","mrro","evapotrans"},
            "npp_balance": {"npp","gpp","ra"},
            "nbp_balance": {"nbp","npp","rh","fFire","fLuc"},
            "ctotal_monthly_vs_annual": {"cTotal_monthly","cTotal_annual"},
            "nbp_vs_delta_ctotal_monthly": {"cTotal_monthly","nbp"},
            "carbon_partition": {"cTotal_annual","cVeg","cLitter","cSoil"},
        }
        return reqs.get(balance, set()).issubset(have)

    have_set = set(mb_var_idx.keys())
    print("[mass-balance] balances enabled:",
        {k: _will_run(k, have_set) for k in
        ["water_balance","npp_balance","nbp_balance","ctotal_monthly_vs_annual",
            "nbp_vs_delta_ctotal_monthly","carbon_partition"]})
    
    if not mb_var_idx:
        print("[WARN] No overlap with OUTPUT_ORDER. Nothing to score.")
        return

    # Loss builders
    idx_m = len(monthly_fluxes_order) + len(monthly_states_order)
    idx_a = len(annual_states_order)
    loss_fns = {
        "water_balance":              make_loss_fn(idx_m, idx_a, wb=1, mb_var_idx=mb_var_idx),
        "npp_balance":                make_loss_fn(idx_m, idx_a, npp=1, mb_var_idx=mb_var_idx),
        "nbp_balance":                make_loss_fn(idx_m, idx_a, nbp=1, mb_var_idx=mb_var_idx),
        "carbon_partition":           make_loss_fn(idx_m, idx_a, cp=1, mb_var_idx=mb_var_idx),
        "nbp_delta_ctotal":           make_loss_fn(idx_m, idx_a, nbpdc=1, mb_var_idx=mb_var_idx),
        "all_on":                     make_loss_fn(idx_m, idx_a, wb=1, npp=1, nbp=1, cp=1, nbpdc=1, mb_var_idx=mb_var_idx),
    }
    balance_keys = list(loss_fns.keys())

    # Accumulators
    sums_by_split: dict[str, dict[str, float]] = {s: {k: 0.0 for k in balance_keys} for s,_ in split_loaders}
    counts_by_split: dict[str, int] = {s: 0 for s,_ in split_loaders}

    # Residuals (for plotting later) — computed once (independent of weights)
    def save_residuals(split_name, residuals_accum, split_sum_map, split_n,
                    out_dir_base: Path, shard_id: int, n_shards: int):
        shard_dir = out_dir_base / split_name / f"shard_{shard_id:03d}_of_{n_shards:03d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        # Save residuals NPZ (per-shard)
        out_npz = shard_dir / f"residuals_{split_name}_shard{shard_id:03d}-of-{n_shards:03d}.npz"
        meta_npz = {
            "split": split_name,
            "year_windows": split_n,
            "shard_id": shard_id,
            "n_shards": n_shards,
        }
        save_npz_residuals(out_npz, residuals_accum, meta_npz)

        # Save metrics JSON (per-shard)
        balance_keys = list(split_sum_map.keys())
        means = {k: (split_sum_map[k] / max(1, split_n)) for k in balance_keys}
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

                # Build pseudo predictions in OUTPUT_ORDER (monthly + annual outputs)
                preds = make_pseudo_preds_for_year(my, ay, OUTPUT_ORDER)  # [B*L, 365, Cm+Ca]

                # Zero labels (so only penalty terms contribute)
                dummy_m = torch.zeros((B * L, 12, Cm), device=device_here, dtype=preds.dtype)
                dummy_a = torch.zeros((B * L, 1,  Ca), device=device_here, dtype=preds.dtype)

                # ---- pull daily precipitation from inputs for this year ----
                # inputs shape typically [B, Cin, Td, L]  -> flatten to [B*L, Td, Cin]
                extra_daily = {}
                try:
                    B_in, Cin, Td, L_in = inputs.shape
                    assert B_in == B and L_in == L, "inputs shape mismatch vs labels"
                    inp_bl_tc = inputs.permute(0, 3, 2, 1).reshape(B * L, Td, Cin)   # [B*L, Td, Cin]

                    # year slice (assumes 365 days per year)
                    td0, td1 = y * 365, (y + 1) * 365
                    inp_y = inp_bl_tc[:, td0:td1, :]                                   # [B*L, 365, Cin]

                    # find 'pre' channel in daily forcing order
                    daily_force_order = sorted(base_train.var_names["daily_forcing"])
                    if "pre" in daily_force_order:
                        pre_idx = daily_force_order.index("pre")
                        pre_daily = inp_y[:, :, pre_idx]                                # [B*L, 365]

                        # OPTIONAL: de-standardize if your inputs are standardized; best-effort try/catch
                        try:
                            mu_pre = torch.tensor(std_dict["daily"]["means"]["pre"], device=device_here, dtype=pre_daily.dtype)
                            sd_pre = torch.tensor(std_dict["daily"]["stds"]["pre"],  device=device_here, dtype=pre_daily.dtype)
                            pre_daily = pre_daily * sd_pre + mu_pre
                        except Exception:
                            # If std_dict structure differs, skip; make sure units are physical before trusting penalties.
                            pass

                        extra_daily["pre"] = pre_daily
                except Exception:
                    # If anything fails, extra_daily stays empty; water balance will be skipped.
                    pass

                # ---- evaluate losses (now physics can use daily 'pre') ----
                with torch.no_grad():
                    for k, lf in loss_fns.items():
                        v = lf(preds, dummy_m, dummy_a, extra_daily=extra_daily).item()
                        split_sums[k] += v

                    # residuals for plotting (optional; unchanged)
                    res = compute_balance_residuals(
                        preds, mb_var_idx, month_mask=month_mask, month_lengths=month_lengths,
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

    # Overall summary across requested splits
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