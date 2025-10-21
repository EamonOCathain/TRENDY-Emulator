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
    """
    DO NOT CHANGE: supervised/base loss exactly as before.
    """
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


def _annual_mean_from_daily(daily: Tensor) -> Tensor:
    """
    daily: [B, 365, K] -> annual mean: [B, K]
    """
    return daily.mean(dim=1)  # [B,K]


def _destandardize_preds(preds: Tensor, mu_out: Optional[Tensor], sd_out: Optional[Tensor]) -> Tensor:
    """
    De-standardise daily predictions per channel if mu_out/sd_out provided.
    preds : [B,365,C]
    mu_out, sd_out : [C] or broadcastable to [1,1,C]
    """
    if (mu_out is None) or (sd_out is None):
        return preds
    # Ensure broadcast shape [1,1,C]
    if mu_out.dim() == 1: mu_out = mu_out.view(1, 1, -1)
    if sd_out.dim() == 1: sd_out = sd_out.view(1, 1, -1)
    return preds * sd_out + mu_out


# ---------- mass-balance/consistency penalties ----------
def water_balance_penalty(
    preds_phys: Tensor,
    idx: Dict[str, int],
    *,
    month_mask: Tensor,
    month_lengths: Tensor,
    base_loss_fn,                    # callable(pred, target) -> scalar tensor
) -> Optional[Tensor]:
    """
    Water balance (monthly means + flux integration):
        ΔMRso(m) = MRso(m) - MRso(m-1)
        ΔMRso(m) = ∫ Pre dt - ∫ Evapotrans dt - ∫ MRro dt  over month m
    Units: kg m^-2
    """
    needed = ("mrso", "pre", "mrro", "evapotrans")
    if not all(k in idx for k in needed):
        return None

    # Monthly means (states & fluxes)
    mrso = _monthly_avg_from_daily(preds_phys[:, :, idx["mrso"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)  # [B,12]
    pr   = _monthly_avg_from_daily(preds_phys[:, :, idx["pre"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)   # [B,12]
    mrro = _monthly_avg_from_daily(preds_phys[:, :, idx["mrro"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)  # [B,12]
    evap = _monthly_avg_from_daily(preds_phys[:, :, idx["evapotrans"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)  # [B,12]

    # Convert flux means (kg m^-2 s^-1) -> monthly integrals (kg m^-2)
    seconds_per_day = 86400.0
    seconds_in_month = (month_lengths * seconds_per_day).view(1, 12)  # [1,12]
    flux_int = (pr - mrro - evap) * seconds_in_month  # [B,12]

    # Δ state (months 2..12)
    delta_mrso = mrso[:, 1:] - mrso[:, :-1]  # [B,11]
    rhs = flux_int[:, 1:]                    # [B,11]

    resid = delta_mrso - rhs
    return base_loss_fn(resid, torch.zeros_like(resid))


def npp_balance_penalty(
    preds_phys: Tensor,
    idx: Dict[str, int],
    *,
    month_mask: Tensor,
    month_lengths: Tensor,
    base_loss_fn,
) -> Optional[Tensor]:
    """
    NPP balance on monthly means (flux vs flux):
        NPP(m) = GPP(m) - Ra(m)
    Units: kg m^-2 s^-1
    """
    needed = ("npp", "gpp", "ra")
    if not all(k in idx for k in needed):
        return None

    npp = _monthly_avg_from_daily(preds_phys[:, :, idx["npp"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    gpp = _monthly_avg_from_daily(preds_phys[:, :, idx["gpp"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    ra  = _monthly_avg_from_daily(preds_phys[:, :, idx["ra"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)

    resid = npp - (gpp - ra)
    return base_loss_fn(resid, torch.zeros_like(resid))


def nbp_balance_penalty(
    preds_phys: Tensor,
    idx: Dict[str, int],
    *,
    month_mask: Tensor,
    month_lengths: Tensor,
    base_loss_fn,
) -> Optional[Tensor]:
    """
    NBP balance on monthly means (flux vs flux):
        NBP(m) = NPP(m) - Rh(m) - fFire(m) - fLuc(m)
    Units: kg m^-2 s^-1
    """
    needed = ("nbp", "npp", "rh", "fFire", "fLuc")
    if not all(k in idx for k in needed):
        return None

    nbp   = _monthly_avg_from_daily(preds_phys[:, :, idx["nbp"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    npp   = _monthly_avg_from_daily(preds_phys[:, :, idx["npp"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    rh    = _monthly_avg_from_daily(preds_phys[:, :, idx["rh"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    fFire = _monthly_avg_from_daily(preds_phys[:, :, idx["fFire"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    fLuc  = _monthly_avg_from_daily(preds_phys[:, :, idx["fLuc"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)

    resid = nbp - (npp - rh - fFire - fLuc)
    return base_loss_fn(resid, torch.zeros_like(resid))


def nbp_vs_delta_ctotal_monthly_penalty(
    preds_phys: Tensor,
    idx: Dict[str, int],
    *,
    month_mask: Tensor,
    month_lengths: Tensor,
    base_loss_fn,
) -> Optional[Tensor]:
    """
    Stock–flux linkage on monthly means:
        Δ cTotal(m) = ∫ NBP dt  (months 2..12)
    Units: kg m^-2
    """
    needed = ("cTotal_monthly", "nbp")
    if not all(k in idx for k in needed):
        return None

    cTot_m = _monthly_avg_from_daily(preds_phys[:, :, idx["cTotal_monthly"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)  # [B,12]
    nbp_m  = _monthly_avg_from_daily(preds_phys[:, :, idx["nbp"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)             # [B,12]

    seconds_per_day = 86400.0
    seconds_in_month = (month_lengths * seconds_per_day).view(1, 12)  # [1,12]
    nbp_integral = nbp_m * seconds_in_month                           # [B,12]

    delta_cTot = cTot_m[:, 1:] - cTot_m[:, :-1]  # [B,11]
    resid = delta_cTot - nbp_integral[:, 1:]     # [B,11]
    return base_loss_fn(resid, torch.zeros_like(resid))


def carbon_partition_penalty(
    preds_phys: Tensor,
    idx: Dict[str, int],
    *,
    base_loss_fn,
) -> Optional[Tensor]:
    """
    Annual partition on ANNUAL MEANS (states only):
        cTotal_annual = cVeg + cLitter + cSoil
    Units: kg m^-2
    """
    needed = ("cTotal_annual", "cVeg", "cLitter", "cSoil")
    if not all(k in idx for k in needed):
        return None

    cTotA = _annual_mean_from_daily(preds_phys[:, :, idx["cTotal_annual"]].unsqueeze(-1)).squeeze(-1)  # [B]
    cVeg  = _annual_mean_from_daily(preds_phys[:, :, idx["cVeg"]].unsqueeze(-1)).squeeze(-1)
    cLit  = _annual_mean_from_daily(preds_phys[:, :, idx["cLitter"]].unsqueeze(-1)).squeeze(-1)
    cSoil = _annual_mean_from_daily(preds_phys[:, :, idx["cSoil"]].unsqueeze(-1)).squeeze(-1)

    resid = cTotA - (cVeg + cLit + cSoil)
    return base_loss_fn(resid, torch.zeros_like(resid))


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
    nbp_delta_ctotal_weight: float = 0.0,
    carbon_partition_weight: float = 0.0,
    # de-standardisation (for penalties only; supervised stays normalised)
    mu_out: Optional[Tensor] = None,
    sd_out: Optional[Tensor] = None,
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """
    Supervised monthly/annual loss (unchanged) + physics penalties (in real units).

    - Supervised (unchanged): monthly/annual MSE on NORMALISED preds vs labels.
    - Penalties: computed on DE-STANDARDISED preds (physical units), with:
        * monthly means for water/NPP/NBP/ΔcTotal=∫NBPdt (flux->integral via seconds-in-month)
        * annual means for cTotal = cVeg + cLitter + cSoil
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

        # ---------------- Supervised component (unchanged; normalised space) ----------------
        pm = preds[:, :, idx_monthly]  # [B,365,nm]
        pa = preds[:, :, idx_annual]   # [B,365,na]

        # Monthly aggregation -> [B,12,nm]
        pm_T  = pm.permute(0, 2, 1)
        pm_sum = torch.einsum("bnc,mc->bnm", pm_T, mmask)
        pm_avg = (pm_sum / mlen.view(1, 1, 12)).permute(0, 2, 1)

        # Annual aggregation -> [B,1,na]
        pa_avg = pa.mean(dim=1, keepdim=True)

        def _bl(pred, targ):
            return base_loss(pred, targ, loss_type=loss_type, reduction=reduction)

        l_m = _bl(pm_avg, labels_monthly)
        l_a = _bl(pa_avg, labels_annual)
        if reduction == "none":
            l_m = l_m.mean()
            l_a = l_a.mean()

        total = monthly_weight * l_m + annual_weight * l_a

        # ---------------- Physics-based penalties (de-standardised preds) ----------------
        if mb_var_idx is not None and (len(mb_var_idx) > 0):
            preds_phys = _destandardize_preds(preds, mu_out, sd_out)

            if water_balance_weight > 0.0:
                pw = water_balance_penalty(
                    preds_phys, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen,
                    base_loss_fn=_bl
                )
                if pw is not None:
                    total = total + water_balance_weight * pw

            if npp_balance_weight > 0.0:
                pnpp = npp_balance_penalty(
                    preds_phys, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen,
                    base_loss_fn=_bl
                )
                if pnpp is not None:
                    total = total + npp_balance_weight * pnpp

            if nbp_balance_weight > 0.0:
                pnbp = nbp_balance_penalty(
                    preds_phys, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen,
                    base_loss_fn=_bl
                )
                if pnbp is not None:
                    total = total + nbp_balance_weight * pnbp

            if nbp_delta_ctotal_weight > 0.0:
                pdc = nbp_vs_delta_ctotal_monthly_penalty(
                    preds_phys, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen,
                    base_loss_fn=_bl
                )
                if pdc is not None:
                    total = total + nbp_delta_ctotal_weight * pdc

            if carbon_partition_weight > 0.0:
                pcp = carbon_partition_penalty(
                    preds_phys, mb_var_idx,
                    base_loss_fn=_bl
                )
                if pcp is not None:
                    total = total + carbon_partition_weight * pcp

        return total

    return loss_fn