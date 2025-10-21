import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Callable, Dict, Optional

# ---------- helpers ----------
def check_loss_shapes(
    preds: Tensor,
    labels_monthly: Tensor,
    labels_annual: Tensor,
    idx_monthly: List[int],
    idx_annual: List[int],
) -> None:
    """
    preds           : [B, 365, Cout]
    labels_monthly  : [B, 12, n_monthly]
    labels_annual   : [B, 1,  n_annual]
    idx_monthly     : indices into Cout for monthly targets
    idx_annual      : indices into Cout for annual targets
    """
    if preds.dim() != 3:
        raise ValueError(f"preds must be [B,365,out_dim], got {tuple(preds.shape)}")

    B, D, Cout = preds.shape
    n_monthly = len(idx_monthly)
    n_annual = len(idx_annual)

    if D != 365:
        raise ValueError(f"preds second dim must be 365, got {D}")

    if max(idx_monthly, default=-1) >= Cout:
        raise ValueError("idx_monthly out of range")
    if max(idx_annual, default=-1) >= Cout:
        raise ValueError("idx_annual out of range")

    if len(set(idx_monthly)) != n_monthly:
        raise ValueError("idx_monthly contains duplicates")
    if len(set(idx_annual)) != n_annual:
        raise ValueError("idx_annual contains duplicates")

    if labels_monthly.dim() != 3:
        raise ValueError(f"labels_monthly must be [B,12,{n_monthly}]")
    if labels_monthly.size(0) != B:
        raise ValueError("labels_monthly batch size mismatch")
    if labels_monthly.size(1) != 12:
        raise ValueError("labels_monthly second dim must be 12")
    if labels_monthly.size(2) != n_monthly:
        raise ValueError("labels_monthly channels mismatch")

    if labels_annual.dim() != 3:
        raise ValueError(f"labels_annual must be [B,1,{n_annual}]")
    if labels_annual.size(0) != B:
        raise ValueError("labels_annual batch size mismatch")
    if labels_annual.size(1) != 1:
        raise ValueError("labels_annual second dim must be 1")
    if labels_annual.size(2) != n_annual:
        raise ValueError("labels_annual channels mismatch")


def base_loss(pred: Tensor, target: Tensor, *,
              loss_type: str = "mse",
              reduction: str = "mean") -> Tensor:
    lt = loss_type.lower()
    if lt == "mse":
        return F.mse_loss(pred, target, reduction=reduction)
    elif lt == "mae":
        return F.l1_loss(pred, target, reduction=reduction)
    else:
        raise ValueError(f"loss_type must be 'mse' or 'mae', got {loss_type!r}")


def _monthly_avg_from_daily(daily: Tensor, month_mask: Tensor, month_lengths: Tensor) -> Tensor:
    """
    daily: [B, 365, K] -> monthly averages: [B, 12, K]
    """
    xT = daily.permute(0, 2, 1)                           # [B,K,365]
    x_sum = torch.einsum("bkc,mc->bkm", xT, month_mask)   # [B,K,12]
    x_avg = (x_sum / month_lengths.view(1, 1, 12)).permute(0, 2, 1)
    return x_avg                                          # [B,12,K]

# ---------- mass-balance/consistency penalties ----------
def water_balance_penalty(
    preds: Tensor,
    idx: Dict[str, int],
    *,
    month_mask: Tensor,
    month_lengths: Tensor,
    base_loss_fn,                    # callable(pred, target) -> scalar tensor
    reduction: str = "mean",
) -> Optional[Tensor]:
    """
    Water balance (monthly):
        Δmrso_m = mrso_m - mrso_{m-1}
        Constraint: Δmrso_m = pr_m - mrro_m - evapotrans_m
    Uses monthly averages. Evaluates months 2..12.
    Requires idx keys: 'mrso','pre','mrro','evapotrans'.
    """
    needed = ("mrso", "pre", "mrro", "evapotrans")
    if not all(k in idx for k in needed):
        return None

    device = preds.device
    mmask = month_mask.to(device)
    mlen  = month_lengths.to(device)

    Kvars = torch.stack([preds[:, :, idx[k]] for k in needed], dim=-1)  # [B,365,4]
    monthly = _monthly_avg_from_daily(Kvars, mmask, mlen)               # [B,12,4]
    mrso, pr, mrro, evap = [monthly[..., i] for i in range(4)]          # [B,12]

    d_mrso = mrso[:, 1:] - mrso[:, :-1]                                  # [B,11]
    rhs    = pr[:, 1:] - mrro[:, 1:] - evap[:, 1:]                       # [B,11]
    resid  = d_mrso - rhs

    pen = base_loss_fn(resid, torch.zeros_like(resid))
    if reduction == "none":
        pen = pen.mean()
    return pen


def npp_balance_penalty(
    preds: Tensor,
    idx: Dict[str, int],
    *,
    month_mask: Tensor,
    month_lengths: Tensor,
    base_loss_fn,
    reduction: str = "mean",
) -> Optional[Tensor]:
    """
    NPP balance (monthly averages):
        NPP = GPP - ra
    Requires idx keys: 'npp','gpp','ra'
    """
    needed = ("npp", "gpp", "ra")
    if not all(k in idx for k in needed):
        return None

    device = preds.device
    mmask = month_mask.to(device)
    mlen  = month_lengths.to(device)

    Kvars = torch.stack([preds[:, :, idx[k]] for k in needed], dim=-1)  # [B,365,3]
    monthly = _monthly_avg_from_daily(Kvars, mmask, mlen)               # [B,12,3]
    NPP, GPP, ra = [monthly[..., i] for i in range(3)]                  # [B,12]

    resid = NPP - (GPP - ra)
    pen = base_loss_fn(resid, torch.zeros_like(resid))
    if reduction == "none":
        pen = pen.mean()
    return pen


def nbp_balance_penalty(
    preds: Tensor,
    idx: Dict[str, int],
    *,
    month_mask: Tensor,
    month_lengths: Tensor,
    base_loss_fn,
    reduction: str = "mean",
) -> Optional[Tensor]:
    """
    NBP balance (monthly averages):
        NBP = NPP - rh - fFire - fLuc
    Requires idx keys: 'nbp','npp','rh','fFire','fLuc'
    """
    needed = ("nbp", "npp", "rh", "fFire", "fLuc")
    if not all(k in idx for k in needed):
        return None

    device = preds.device
    mmask = month_mask.to(device)
    mlen  = month_lengths.to(device)

    Kvars = torch.stack([preds[:, :, idx[k]] for k in needed], dim=-1)  # [B,365,5]
    monthly = _monthly_avg_from_daily(Kvars, mmask, mlen)               # [B,12,5]
    NBP, NPP, rh, fFire, fLuc = [monthly[..., i] for i in range(5)]     # [B,12]

    resid = NBP - (NPP - rh - fFire - fLuc)
    pen = base_loss_fn(resid, torch.zeros_like(resid))
    if reduction == "none":
        pen = pen.mean()
    return pen


def carbon_partition_penalty(
    preds: Tensor,
    idx: Dict[str, int],
    *,
    base_loss_fn,
    reduction: str = "mean",
) -> Optional[Tensor]:
    """
    Annual partition (using annual means of daily predictions):
        cTotal_annual = cVeg + cLitter + cSoil
    Requires idx keys: 'cTotal_annual','cVeg','cLitter','cSoil'
    """
    needed = ("cTotal_annual", "cVeg", "cLitter", "cSoil")
    if not all(k in idx for k in needed):
        return None

    cTotA = preds[:, :, idx["cTotal_annual"]].mean(dim=1)  # [B]
    cVeg  = preds[:, :, idx["cVeg"]].mean(dim=1)
    cLit  = preds[:, :, idx["cLitter"]].mean(dim=1)
    cSoil = preds[:, :, idx["cSoil"]].mean(dim=1)

    resid = cTotA - (cVeg + cLit + cSoil)
    pen = base_loss_fn(resid, torch.zeros_like(resid))
    if reduction == "none":
        pen = pen.mean()
    return pen

def nbp_vs_delta_ctotal_monthly_penalty(
    preds: Tensor,
    idx: Dict[str, int],
    *,
    month_mask: Tensor,
    month_lengths: Tensor,
    base_loss_fn,
    reduction: str = "mean",
) -> Optional[Tensor]:
    """
    Stock–flux linkage (monthly):
        Δ cTotal_monthly(m) = NBP(m), evaluated for months 2..12.
    Requires idx keys: 'cTotal_monthly','nbp'
    """
    needed = ("cTotal_monthly", "nbp")
    if not all(k in idx for k in needed):
        return None

    device = preds.device
    mmask = month_mask.to(device)      # [12,365]
    mlen  = month_lengths.to(device)   # [12]

    cTotDaily = preds[:, :, idx["cTotal_monthly"]]                      # [B,365]
    nbpDaily  = preds[:, :, idx["nbp"]]                                 # [B,365]

    cTot_m = (torch.einsum("bc,mc->bm", cTotDaily, mmask) / mlen)       # [B,12]
    nbp_m  = (torch.einsum("bc,mc->bm", nbpDaily,  mmask) / mlen)       # [B,12]

    d_cTot = cTot_m[:, 1:] - cTot_m[:, :-1]                             # [B,11]
    resid  = d_cTot - nbp_m[:, 1:]                                      # [B,11]

    pen = base_loss_fn(resid, torch.zeros_like(resid))
    if reduction == "none":
        pen = pen.mean()
    return pen

# ---------- unified custom_loss ----------
def custom_loss(
    idx_monthly: List[int],
    idx_annual: List[int],
    monthly_weight: float = 1.0,
    annual_weight: float = 1.0,
    *,
    loss_type: str = "mse",
    reduction: str = "mean",
    # mass-balance knobs
    mb_var_idx: Optional[Dict[str, int]] = None,   # name -> output channel index
    water_balance_weight: float = 0.0,
    npp_balance_weight: float = 0.0,
    nbp_balance_weight: float = 0.0,
    # carbon knobs (names match train.py flags)
    carbon_partition_weight: float = 0.0,
    ctotal_mon_ann_weight: float = 0.0,            # mean(monthly cTotal) == annual cTotal
    nbp_delta_ctotal_weight: float = 0.0,          # Δ cTotal_monthly == NBP
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """
    Supervised monthly/annual loss with optional physics/consistency penalties.
    Pass channel indices for balance variables via `mb_var_idx` and set *_weight > 0 to activate.
    """
    # Month metadata (constructed once, captured in closure)
    month_lengths = torch.tensor([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=torch.float32)
    day_to_month = torch.repeat_interleave(torch.arange(12, dtype=torch.long), month_lengths.to(torch.long))
    month_mask = torch.nn.functional.one_hot(day_to_month, num_classes=12).T.float()  # [12, 365]

    def loss_fn(preds: Tensor, labels_monthly: Tensor, labels_annual: Tensor) -> Tensor:
        # shape/index validation
        check_loss_shapes(preds, labels_monthly, labels_annual, idx_monthly, idx_annual)

        device = preds.device
        mmask = month_mask.to(device)
        mlen  = month_lengths.to(device)

        # Select supervised channels
        pm = preds[:, :, idx_monthly]  # [B,365,nm]
        pa = preds[:, :, idx_annual]   # [B,365,na]

        # Monthly aggregation -> [B,12,nm]
        pm_T  = pm.permute(0, 2, 1)
        pm_sum = torch.einsum("bnc,mc->bnm", pm_T, mmask)
        pm_avg = (pm_sum / mlen.view(1, 1, 12)).permute(0, 2, 1)

        # Annual aggregation -> [B,1,na]
        pa_avg = pa.mean(dim=1, keepdim=True)

        # Supervised losses
        def _bl(pred, targ):
            return base_loss(pred, targ, loss_type=loss_type, reduction=reduction)

        l_m = _bl(pm_avg, labels_monthly)
        l_a = _bl(pa_avg, labels_annual)
        if reduction == "none":
            l_m = l_m.mean()
            l_a = l_a.mean()

        total = monthly_weight * l_m + annual_weight * l_a

        # Optional penalties
        if mb_var_idx is not None:
            if water_balance_weight > 0.0:
                pw = water_balance_penalty(
                    preds, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen,
                    base_loss_fn=_bl, reduction=reduction
                )
                if pw is not None:
                    total = total + water_balance_weight * pw

            if npp_balance_weight > 0.0:
                pnpp = npp_balance_penalty(
                    preds, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen,
                    base_loss_fn=_bl, reduction=reduction
                )
                if pnpp is not None:
                    total = total + npp_balance_weight * pnpp

            if nbp_balance_weight > 0.0:
                pnbp = nbp_balance_penalty(
                    preds, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen,
                    base_loss_fn=_bl, reduction=reduction
                )
                if pnbp is not None:
                    total = total + nbp_balance_weight * pnbp

            # Carbon penalties
            if carbon_partition_weight > 0.0:
                p_cp = carbon_partition_penalty(
                    preds, mb_var_idx, base_loss_fn=_bl, reduction=reduction
                )
                if p_cp is not None:
                    total = total + carbon_partition_weight * p_cp

            if ctotal_mon_ann_weight > 0.0:
                p_cc = ctotal_monthly_vs_annual_penalty(
                    preds, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen,
                    base_loss_fn=_bl, reduction=reduction
                )
                if p_cc is not None:
                    total = total + ctotal_mon_ann_weight * p_cc

            if nbp_delta_ctotal_weight > 0.0:
                p_sf = nbp_vs_delta_ctotal_monthly_penalty(
                    preds, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen,
                    base_loss_fn=_bl, reduction=reduction
                )
                if p_sf is not None:
                    total = total + nbp_delta_ctotal_weight * p_sf

        return total

    return loss_fn