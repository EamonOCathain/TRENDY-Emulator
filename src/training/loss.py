import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Callable, Dict, Optional

# ---------- helpers ----------
def _monthly_avg_from_daily(daily: Tensor, month_mask: Tensor, month_lengths: Tensor) -> Tensor:
    """
    daily: [B, 365, K] -> monthly averages: [B, 12, K]
    """
    xT = daily.permute(0, 2, 1)                           # [B,K,365]
    x_sum = torch.einsum("bkc,mc->bkm", xT, month_mask)   # [B,K,12]
    x_avg = (x_sum / month_lengths.view(1, 1, 12)).permute(0, 2, 1)
    return x_avg                                          # [B,12,K]

def _annual_mean_from_daily(daily: Tensor) -> Tensor:
    """daily: [B, 365, K] -> annual mean: [B, K]"""
    return daily.mean(dim=1)

def _destandardize_preds(preds: Tensor, mu_out: Optional[Tensor], sd_out: Optional[Tensor]) -> Tensor:
    if (mu_out is None) or (sd_out is None):
        return preds
    if mu_out.dim() == 1: mu_out = mu_out.view(1, 1, -1)
    if sd_out.dim() == 1: sd_out = sd_out.view(1, 1, -1)
    return preds * sd_out + mu_out

def _elementwise_loss(a: Tensor, b: Tensor, loss_type: str) -> Tensor:
    lt = loss_type.lower()
    if lt == "mse": return (a - b) ** 2
    if lt == "mae": return (a - b).abs()
    raise ValueError(f"loss_type must be 'mse' or 'mae', got {loss_type!r}")

# ---------- physics penalties (can use extra_daily) ----------
def _get_daily_series(name: str, preds_phys: Tensor, idx: Dict[str,int], extra_daily: Optional[Dict[str,Tensor]]) -> Optional[Tensor]:
    """
    Return daily series [B,365] for `name` if available from preds or extra_daily.
    """
    if extra_daily is not None and name in extra_daily:
        x = extra_daily[name]
        return x  # expect [B,365]
    if name in idx:
        return preds_phys[:, :, idx[name]]  # [B,365]
    return None

def water_balance_penalty(preds_phys, idx, *, month_mask, month_lengths, base_loss_fn, extra_daily=None):
    needed = ("mrso", "pre", "mrro", "evapotrans")
    if not all((k in idx) or (extra_daily is not None and k in extra_daily) for k in needed):
        return None

    seconds_per_day = 86400.0
    seconds_in_month = (month_lengths * seconds_per_day).view(1, 12)

    mrso_d = _get_daily_series("mrso", preds_phys, idx, extra_daily)
    pre_d  = _get_daily_series("pre",  preds_phys, idx, extra_daily)
    mrro_d = _get_daily_series("mrro", preds_phys, idx, extra_daily)
    evap_d = _get_daily_series("evapotrans", preds_phys, idx, extra_daily)
    if any(x is None for x in (mrso_d, pre_d, mrro_d, evap_d)):
        return None

    mrso = _monthly_avg_from_daily(mrso_d.unsqueeze(-1), month_mask, month_lengths).squeeze(-1)  # [B,12]
    pr   = _monthly_avg_from_daily(pre_d .unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    mrro = _monthly_avg_from_daily(mrro_d.unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    evap = _monthly_avg_from_daily(evap_d.unsqueeze(-1), month_mask, month_lengths).squeeze(-1)

    flux_int   = (pr - mrro - evap) * seconds_in_month             # [B,12]
    delta_mrso = mrso[:, 1:] - mrso[:, :-1]                        # [B,11]
    resid      = delta_mrso - flux_int[:, 1:]                      # [B,11]
    return base_loss_fn(resid, torch.zeros_like(resid))

def npp_balance_penalty(preds_phys, idx, *, month_mask, month_lengths, base_loss_fn, extra_daily=None):
    needed = ("npp", "gpp", "ra")
    if not all(k in idx for k in needed): return None
    npp = _monthly_avg_from_daily(preds_phys[:, :, idx["npp"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    gpp = _monthly_avg_from_daily(preds_phys[:, :, idx["gpp"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    ra  = _monthly_avg_from_daily(preds_phys[:, :, idx["ra" ]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    resid = npp - (gpp - ra)
    return base_loss_fn(resid, torch.zeros_like(resid))

def nbp_balance_penalty(preds_phys, idx, *, month_mask, month_lengths, base_loss_fn, extra_daily=None):
    needed = ("nbp", "npp", "rh", "fFire", "fLuc")
    if not all(k in idx for k in needed): return None
    nbp   = _monthly_avg_from_daily(preds_phys[:, :, idx["nbp"]].unsqueeze(-1),   month_mask, month_lengths).squeeze(-1)
    npp   = _monthly_avg_from_daily(preds_phys[:, :, idx["npp"]].unsqueeze(-1),   month_mask, month_lengths).squeeze(-1)
    rh    = _monthly_avg_from_daily(preds_phys[:, :, idx["rh"] ].unsqueeze(-1),   month_mask, month_lengths).squeeze(-1)
    fFire = _monthly_avg_from_daily(preds_phys[:, :, idx["fFire"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    fLuc  = _monthly_avg_from_daily(preds_phys[:, :, idx["fLuc"] ].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    resid = nbp - (npp - rh - fFire - fLuc)
    return base_loss_fn(resid, torch.zeros_like(resid))

def nbp_vs_delta_ctotal_monthly_penalty(preds_phys, idx, *, month_mask, month_lengths, base_loss_fn, extra_daily=None):
    needed = ("cTotal_monthly", "nbp")
    if not all(k in idx for k in needed): return None
    seconds_per_day = 86400.0
    seconds_in_month = (month_lengths * seconds_per_day).view(1, 12)

    cTot_m = _monthly_avg_from_daily(preds_phys[:, :, idx["cTotal_monthly"]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)
    nbp_m  = _monthly_avg_from_daily(preds_phys[:, :, idx["nbp"          ]].unsqueeze(-1), month_mask, month_lengths).squeeze(-1)

    nbp_integral = nbp_m * seconds_in_month
    delta_cTot   = cTot_m[:, 1:] - cTot_m[:, :-1]
    resid        = delta_cTot - nbp_integral[:, 1:]
    return base_loss_fn(resid, torch.zeros_like(resid))

def carbon_partition_penalty(
    preds_phys: Tensor,
    idx: Dict[str, int],
    *,
    month_mask: Tensor,
    month_lengths: Tensor,
    base_loss_fn: Callable[[Tensor, Tensor], Tensor],
    extra_daily: Optional[Dict[str, Tensor]] = None,
) -> Optional[Tensor]:
    """
    Enforce: cTotal_monthly(December) == cVeg_annual + cLitter_annual + cSoil_annual

    - cTotal_monthly: take monthly averages then pick December (idx=11)
    - cVeg, cLitter, cSoil: annual mean from daily
    """
    needed = ("cTotal_monthly", "cVeg", "cLitter", "cSoil")
    if not all(k in idx for k in needed):
        return None

    # December from monthly averages of daily series
    cTot_m_monthly = _monthly_avg_from_daily(
        preds_phys[:, :, idx["cTotal_monthly"]].unsqueeze(-1),
        month_mask,
        month_lengths
    ).squeeze(-1)                                   # [B,12]
    cTot_dec = cTot_m_monthly[:, 11]               # [B]  (December)

    # Annual means for the partitions
    cVeg  = _annual_mean_from_daily(preds_phys[:, :, idx["cVeg"   ]].unsqueeze(-1)).squeeze(-1)  # [B]
    cLit  = _annual_mean_from_daily(preds_phys[:, :, idx["cLitter"]].unsqueeze(-1)).squeeze(-1)  # [B]
    cSoil = _annual_mean_from_daily(preds_phys[:, :, idx["cSoil"  ]].unsqueeze(-1)).squeeze(-1)  # [B]

    resid = cTot_dec - (cVeg + cLit + cSoil)       # [B]
    return base_loss_fn(resid, torch.zeros_like(resid))

# ---------- custom_loss with extra_daily ----------
def custom_loss(
    idx_monthly: List[int],
    idx_annual: List[int],
    *,
    loss_type: str = "mse",
    monthly_weights: Optional[Tensor | List[float]] = None,
    annual_weights:  Optional[Tensor | List[float]] = None,
    mb_var_idx: Optional[Dict[str, int]] = None,
    water_balance_weight: float = 0.0,
    npp_balance_weight: float = 0.0,
    nbp_balance_weight: float = 0.0,
    nbp_delta_ctotal_weight: float = 0.0,
    carbon_partition_weight: float = 0.0,
    mu_out: Optional[Tensor] = None,
    sd_out: Optional[Tensor] = None,
) -> Callable[[Tensor, Tensor, Tensor, Optional[Dict[str,Tensor]]], Tensor]:
    """
    Returns a function (preds, labels_monthly, labels_annual, extra_daily=None) -> scalar loss.
    Pass extra_daily with entries like {"pre": daily_tensor[B,365]} so physics penalties
    can use forcing variables that are not in preds.
    """
    month_lengths = torch.tensor([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=torch.float32)
    day_to_month  = torch.repeat_interleave(torch.arange(12, dtype=torch.long), month_lengths.to(torch.long))
    month_mask    = torch.nn.functional.one_hot(day_to_month, num_classes=12).T.float()  # [12,365]

    n_monthly = len(idx_monthly)
    n_annual  = len(idx_annual)
    if monthly_weights is None:
        monthly_weights = torch.ones(n_monthly, dtype=torch.float32)
    else:
        monthly_weights = torch.tensor(monthly_weights, dtype=torch.float32)
        assert monthly_weights.numel() == n_monthly
    if annual_weights is None:
        annual_weights = torch.ones(n_annual, dtype=torch.float32)
    else:
        annual_weights = torch.tensor(annual_weights, dtype=torch.float32)
        assert annual_weights.numel() == n_annual

    def loss_fn(preds: Tensor, labels_monthly: Tensor, labels_annual: Tensor, extra_daily: Optional[Dict[str,Tensor]] = None) -> Tensor:
        device = preds.device
        mmask  = month_mask.to(device)
        mlen   = month_lengths.to(device)
        w_m    = monthly_weights.to(device)
        w_a    = annual_weights.to(device)

        # ---- supervised monthly/annual from daily preds ----
        pm = preds[:, :, idx_monthly]  # [B,365,nm]
        pa = preds[:, :, idx_annual]   # [B,365,na]

        pm_T   = pm.permute(0, 2, 1)                                 # [B,nm,365]
        pm_sum = torch.einsum("bnc,mc->bnm", pm_T, mmask)             # [B,nm,12]
        pm_avg = (pm_sum / mlen.view(1, 1, 12)).permute(0, 2, 1)      # [B,12,nm]

        pa_avg = pa.mean(dim=1, keepdim=True)                         # [B,1,na]

        l_m_elem = _elementwise_loss(pm_avg, labels_monthly, loss_type)
        l_a_elem = _elementwise_loss(pa_avg, labels_annual,  loss_type)

        Lm_per_var = l_m_elem.mean(dim=1).mean(dim=0)   # [nm]
        La_per_var = l_a_elem.mean(dim=1).mean(dim=0)   # [na]

        total = (w_m * Lm_per_var).sum() + (w_a * La_per_var).sum()

        # ---- physics penalties on de-standardised preds + extra_daily ----
        if mb_var_idx:
            preds_phys = _destandardize_preds(preds, mu_out, sd_out)

            def _bl(pred, targ):
                return F.mse_loss(pred, targ, reduction="mean") if loss_type == "mse" \
                       else F.l1_loss(pred, targ, reduction="mean")

            if water_balance_weight > 0.0:
                pw = water_balance_penalty(preds_phys, mb_var_idx, month_mask=mmask, month_lengths=mlen, base_loss_fn=_bl, extra_daily=extra_daily)
                if pw is not None: total = total + water_balance_weight * pw

            if npp_balance_weight > 0.0:
                pnpp = npp_balance_penalty(preds_phys, mb_var_idx, month_mask=mmask, month_lengths=mlen, base_loss_fn=_bl, extra_daily=extra_daily)
                if pnpp is not None: total = total + npp_balance_weight * pnpp

            if nbp_balance_weight > 0.0:
                pnbp = nbp_balance_penalty(preds_phys, mb_var_idx, month_mask=mmask, month_lengths=mlen, base_loss_fn=_bl, extra_daily=extra_daily)
                if pnbp is not None: total = total + nbp_balance_weight * pnbp

            if nbp_delta_ctotal_weight > 0.0:
                pdc = nbp_vs_delta_ctotal_monthly_penalty(preds_phys, mb_var_idx, month_mask=mmask, month_lengths=mlen, base_loss_fn=_bl, extra_daily=extra_daily)
                if pdc is not None: total = total + nbp_delta_ctotal_weight * pdc

            if carbon_partition_weight > 0.0:
                pcp = carbon_partition_penalty(
                    preds_phys, mb_var_idx,
                    month_mask=mmask,
                    month_lengths=mlen,
                    base_loss_fn=_bl,
                    extra_daily=extra_daily
                )
                if pcp is not None:
                    total = total + carbon_partition_weight * pcp

        return total

    return loss_fn