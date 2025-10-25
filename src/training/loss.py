import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Callable, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Month/annual helpers
# ---------------------------------------------------------------------------

def _make_month_tensors() -> Tuple[Tensor, Tensor]:
    """
    Returns:
      month_lengths: [12] float32
      month_mask:    [12,365] float32 one-hot rows
    """
    month_lengths = torch.tensor(
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        dtype=torch.float32,
    )
    day_to_month = torch.repeat_interleave(
        torch.arange(12, dtype=torch.long),
        month_lengths.to(torch.long),
    )
    month_mask = F.one_hot(day_to_month, num_classes=12).T.float()  # [12,365]
    return month_lengths, month_mask


def _aggregate_month_from_daily(
    daily: Tensor, month_mask: Tensor, month_lengths: Tensor, use_delta: bool
) -> Tensor:
    """
    Aggregate daily → monthly.

    Args:
      daily:         [B, 365, K]
      month_mask:    [12, 365] one-hot month selector (rows sum to month lengths)
      month_lengths: [12] number of days in each month
      use_delta:     True → return SUMS; False → return AVERAGES

    Returns:
      [B, 12, K]
    """
    xT = daily.permute(0, 2, 1)                           # [B,K,365]
    x_sum = torch.einsum("bkc,mc->bkm", xT, month_mask)   # [B,K,12]
    x_sum = x_sum.permute(0, 2, 1)                        # [B,12,K]
    '''if use_delta:
        return x_sum
    else:
        return x_sum / month_lengths.view(1, 12, 1)'''
    return x_sum / month_lengths.view(1, 12, 1)


def _aggregate_year_from_daily(daily: Tensor, use_delta: bool) -> Tensor:
    """
    Aggregate daily → annual.

    Args:
      daily:     [B, 365, K]
      use_delta: True → return SUMS; False → return AVERAGES

    Returns:
      [B, K]
    """
    #return daily.sum(dim=1) if use_delta else daily.mean(dim=1)
    return daily.mean(dim=1)


def _destandardize_preds_normal(preds: Tensor, mu_out: Optional[Tensor], sd_out: Optional[Tensor]) -> Tensor:
    """
    Apply destandardization: preds * sd_out + mu_out.

    Accepts broadcasting with shapes:
      - preds:  [B, 1, D]
      - mu_out: [D] or [1,1,D]
      - sd_out: [D] or [1,1,D]
      
    Where B is batch size, D is number of output dims, and 365 is days in year.
    """
    if (mu_out is None) or (sd_out is None):
        return preds
    if mu_out.dim() == 1:
        mu_out = mu_out.view(1, 1, -1)
    if sd_out.dim() == 1:
        sd_out = sd_out.view(1, 1, -1)
    return preds * sd_out + mu_out

def _destandardize_preds_delta(z_t: Tensor, mu: Optional[Tensor], sigma: Optional[Tensor]) -> Tensor:
    """
    Apply destandardization: y_t = y_t-1 + (z_t * sigma).

    Accepts broadcasting with shapes:
      - z_t:  [B, 1, D]
      - y_t-1: [B, 1]
      - mu: [D] or [1,1,D]
      - sigma: [D] or [1,1,D]
      
    Where B is batch size, D is number of output dims, and 365 is days in year.
    """


def _destandardize_preds(preds: Tensor, mu_out: Optional[Tensor], sd_out: Optional[Tensor], delta_labels: bool) -> Tensor:
    """
    Apply normal or delta destandardization based on arg.
    """
    if delta_labels:
        return _destandardize_preds_delta(preds, mu_out, sd_out)
    else:
        return _destandardize_preds_normal(preds, mu_out, sd_out)


def _elementwise_loss(a: Tensor, b: Tensor, loss_type: str) -> Tensor:
    """
    Return elementwise residual loss (no reduction).

    Supports:
      - "mse": (a - b)^2
      - "mae": |a - b|
    """
    lt = loss_type.lower()
    if lt == "mse":
        return (a - b) ** 2
    if lt == "mae":
        return (a - b).abs()
    raise ValueError(f"loss_type must be 'mse' or 'mae', got {loss_type!r}")

# ---------------------------------------------------------------------------
# Standard Supervised Loss (now uses helpers)
# ---------------------------------------------------------------------------

def _supervised_terms_only(
    preds: Tensor,
    labels_monthly: Tensor,
    labels_annual: Tensor,
    idx_monthly: List[int],
    idx_annual: List[int],
    loss_type: str,
    month_lengths: Tensor,
    month_mask: Tensor,
    monthly_weights: Tensor,
    annual_weights: Tensor,
    *,
    delta_labels: bool,
) -> Tensor:
    """
    Compute the supervised monthly/annual loss terms (no physics penalties),
    using helper aggregations (sums if delta_labels=True, else averages).
    """
    # Select supervised channels from daily preds
    pm = preds[:, :, idx_monthly]  # [B,365,nm]
    pa = preds[:, :, idx_annual]   # [B,365,na]

    # Monthly aggregation
    pm_agg = _aggregate_month_from_daily(pm, month_mask, month_lengths, delta_labels)  # [B,12,nm]

    # Annual aggregation
    pa_agg = _aggregate_year_from_daily(pa, delta_labels).unsqueeze(1)                 # [B,1,na]

    # Elementwise residuals (no reduction)
    l_m_elem = _elementwise_loss(pm_agg, labels_monthly, loss_type)  # [B,12,nm]
    l_a_elem = _elementwise_loss(pa_agg, labels_annual,  loss_type)  # [B,1, na]

    # Mean across batch and time → per-variable
    Lm_per_var = l_m_elem.mean(dim=1).mean(dim=0)  # [nm]
    La_per_var = l_a_elem.mean(dim=1).mean(dim=0)  # [na]

    # Weighted sum
    total = (monthly_weights * Lm_per_var).sum() + (annual_weights * La_per_var).sum()
    return total


def make_supervised_loss(
    idx_monthly: List[int],
    idx_annual: List[int],
    *,
    loss_type: str = "mse",
    monthly_weights: Optional[Tensor | List[float]] = None,
    annual_weights:  Optional[Tensor | List[float]] = None,
    delta_labels: bool = False,
) -> Callable[[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]], Tensor]:
    """
    Factory: build loss(preds, labels_monthly, labels_annual, extra_daily=None) that
    computes ONLY the supervised monthly/annual terms (no mass-balance penalties),
    aggregating with sums if delta_labels=True, else averages.
    """
    month_lengths, month_mask = _make_month_tensors()

    # Normalize weights
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

    def loss_fn(preds: Tensor, labels_monthly: Tensor, labels_annual: Tensor, extra_daily: Optional[Dict[str, Tensor]] = None) -> Tensor:  # noqa: ARG001
        device = preds.device
        return _supervised_terms_only(
            preds, labels_monthly, labels_annual,
            idx_monthly, idx_annual, loss_type,
            month_lengths.to(device), month_mask.to(device),
            monthly_weights.to(device), annual_weights.to(device),
            delta_labels=delta_labels,
        )

    return loss_fn

# ---------------------------------------------------------------------------
# Physics penalties (now use helpers too)
# ---------------------------------------------------------------------------

def _get_daily_series(
    name: str,
    preds_phys: Tensor,
    idx: Dict[str, int],
    extra_daily: Optional[Dict[str, Tensor]],
) -> Optional[Tensor]:
    """
    Return daily series [B,365] for `name` either from preds_phys or extra_daily.
    """
    if extra_daily is not None and name in extra_daily:
        return extra_daily[name]  # expected [B,365]
    if name in idx:
        return preds_phys[:, :, idx[name]]  # [B,365]
    return None


def water_balance_penalty(
    preds_phys: Tensor,
    idx: Dict[str, int],
    *,
    month_mask: Tensor,
    month_lengths: Tensor,
    base_loss_fn: Callable[[Tensor, Tensor], Tensor],
    extra_daily: Optional[Dict[str, Tensor]] = None,
    delta_labels: bool,
) -> Optional[Tensor]:
    """
    Enforce (monthly, except Jan): Δmrso ≈ ∫(pr - mrro - evapotrans) dt.

    We aggregate daily to monthly via helpers:
      - sums if delta_labels=True
      - averages if delta_labels=False (then multiply by seconds_in_month as before)
    """
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

    mrso = _aggregate_month_from_daily(mrso_d.unsqueeze(-1), month_mask, month_lengths, delta_labels).squeeze(-1)  # [B,12]
    pr   = _aggregate_month_from_daily(pre_d .unsqueeze(-1), month_mask, month_lengths, delta_labels).squeeze(-1)
    mrro = _aggregate_month_from_daily(mrro_d.unsqueeze(-1), month_mask, month_lengths, delta_labels).squeeze(-1)
    evap = _aggregate_month_from_daily(evap_d.unsqueeze(-1), month_mask, month_lengths, delta_labels).squeeze(-1)

    flux_int   = (pr - mrro - evap) * seconds_in_month   # [B,12]
    delta_mrso = mrso[:, 1:] - mrso[:, :-1]              # [B,11]
    resid      = delta_mrso - flux_int[:, 1:]            # [B,11]
    return base_loss_fn(resid, torch.zeros_like(resid))


def npp_balance_penalty(
    preds_phys: Tensor,
    idx: Dict[str, int],
    *,
    month_mask: Tensor,
    month_lengths: Tensor,
    base_loss_fn: Callable[[Tensor, Tensor], Tensor],
    extra_daily: Optional[Dict[str, Tensor]] = None,
    delta_labels: bool,
) -> Optional[Tensor]:
    """Enforce: npp ≈ gpp - ra (monthly)."""
    needed = ("npp", "gpp", "ra")
    if not all(k in idx for k in needed):
        return None
    npp = _aggregate_month_from_daily(preds_phys[:, :, idx["npp"]].unsqueeze(-1), month_mask, month_lengths, delta_labels).squeeze(-1)
    gpp = _aggregate_month_from_daily(preds_phys[:, :, idx["gpp"]].unsqueeze(-1), month_mask, month_lengths, delta_labels).squeeze(-1)
    ra  = _aggregate_month_from_daily(preds_phys[:, :, idx["ra" ]].unsqueeze(-1), month_mask, month_lengths, delta_labels).squeeze(-1)
    resid = npp - (gpp - ra)
    return base_loss_fn(resid, torch.zeros_like(resid))


def nbp_balance_penalty(
    preds_phys: Tensor,
    idx: Dict[str, int],
    *,
    month_mask: Tensor,
    month_lengths: Tensor,
    base_loss_fn: Callable[[Tensor, Tensor], Tensor],
    extra_daily: Optional[Dict[str, Tensor]] = None,
    delta_labels: bool,
) -> Optional[Tensor]:
    """Enforce: nbp ≈ npp - rh - fFire - fLuc (monthly)."""
    needed = ("nbp", "npp", "rh", "fFire", "fLuc")
    if not all(k in idx for k in needed):
        return None
    nbp   = _aggregate_month_from_daily(preds_phys[:, :, idx["nbp"]].unsqueeze(-1),   month_mask, month_lengths, delta_labels).squeeze(-1)
    npp   = _aggregate_month_from_daily(preds_phys[:, :, idx["npp"]].unsqueeze(-1),   month_mask, month_lengths, delta_labels).squeeze(-1)
    rh    = _aggregate_month_from_daily(preds_phys[:, :, idx["rh"] ].unsqueeze(-1),   month_mask, month_lengths, delta_labels).squeeze(-1)
    fFire = _aggregate_month_from_daily(preds_phys[:, :, idx["fFire"]].unsqueeze(-1), month_mask, month_lengths, delta_labels).squeeze(-1)
    fLuc  = _aggregate_month_from_daily(preds_phys[:, :, idx["fLuc"] ].unsqueeze(-1), month_mask, month_lengths, delta_labels).squeeze(-1)
    resid = nbp - (npp - rh - fFire - fLuc)
    return base_loss_fn(resid, torch.zeros_like(resid))


def nbp_vs_delta_ctotal_monthly_penalty(
    preds_phys: Tensor,
    idx: Dict[str, int],
    *,
    month_mask: Tensor,
    month_lengths: Tensor,
    base_loss_fn: Callable[[Tensor, Tensor], Tensor],
    extra_daily: Optional[Dict[str, Tensor]] = None,
    delta_labels: bool,
) -> Optional[Tensor]:
    """Enforce: Δ cTotal_monthly ≈ ∫ nbp dt (monthly, except Jan)."""
    needed = ("cTotal_monthly", "nbp")
    if not all(k in idx for k in needed):
        return None

    seconds_per_day = 86400.0
    seconds_in_month = (month_lengths * seconds_per_day).view(1, 12)

    cTot_m = _aggregate_month_from_daily(
        preds_phys[:, :, idx["cTotal_monthly"]].unsqueeze(-1), month_mask, month_lengths, delta_labels
    ).squeeze(-1)  # [B,12]
    nbp_m  = _aggregate_month_from_daily(
        preds_phys[:, :, idx["nbp"]].unsqueeze(-1), month_mask, month_lengths, delta_labels
    ).squeeze(-1)

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
    delta_labels: bool,
) -> Optional[Tensor]:
    """
    Enforce: cTotal_monthly(December) == cVeg_annual + cLitter_annual + cSoil_annual.
    """
    needed = ("cTotal_monthly", "cVeg", "cLitter", "cSoil")
    if not all(k in idx for k in needed):
        return None

    # December value from monthly aggregation
    cTot_m_monthly = _aggregate_month_from_daily(
        preds_phys[:, :, idx["cTotal_monthly"]].unsqueeze(-1),
        month_mask,
        month_lengths,
        delta_labels,
    ).squeeze(-1)                               # [B,12]
    cTot_dec = cTot_m_monthly[:, 11]            # [B]

    # Annual aggregation for partitions
    cVeg  = _aggregate_year_from_daily(preds_phys[:, :, idx["cVeg"   ]].unsqueeze(-1), delta_labels).squeeze(-1)  # [B]
    cLit  = _aggregate_year_from_daily(preds_phys[:, :, idx["cLitter"]].unsqueeze(-1), delta_labels).squeeze(-1)  # [B]
    cSoil = _aggregate_year_from_daily(preds_phys[:, :, idx["cSoil"  ]].unsqueeze(-1), delta_labels).squeeze(-1)  # [B]

    resid = cTot_dec - (cVeg + cLit + cSoil)    # [B]
    return base_loss_fn(resid, torch.zeros_like(resid))

# ---------------------------------------------------------------------------
# Loss with mass balances factory
# ---------------------------------------------------------------------------

def make_supervised_plus_mass_balance_loss(
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
    delta_labels: bool = False,
) -> Callable[[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]], Tensor]:
    """
    Factory: build loss(...) that includes the supervised terms + optional mass-balance penalties.
    Aggregation uses sums if delta_labels=True, else averages.
    """
    month_lengths, month_mask = _make_month_tensors()

    # Normalize weights
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

    def loss_fn(
        preds: Tensor,
        labels_monthly: Tensor,
        labels_annual: Tensor,
        extra_daily: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        device = preds.device
        mlen   = month_lengths.to(device)
        mmask  = month_mask.to(device)
        w_m    = monthly_weights.to(device)
        w_a    = annual_weights.to(device)

        # 1) Supervised terms
        L_sup = _supervised_terms_only(
            preds, labels_monthly, labels_annual,
            idx_monthly, idx_annual, loss_type,
            mlen, mmask, w_m, w_a,
            delta_labels=delta_labels,
        )
        total = L_sup

        # --- publish breakdown dicts
        raw_mb: Dict[str, float] = {}
        w_mb:   Dict[str, float] = {}

        # 2) Mass-balance penalties (on destandardized preds)
        if mb_var_idx:
            preds_phys = _destandardize_preds(preds, mu_out, sd_out)

            def _bl(pred, targ):
                return F.mse_loss(pred, targ, reduction="mean") if loss_type.lower() == "mse" \
                    else F.l1_loss(pred, targ, reduction="mean")

            if water_balance_weight > 0.0:
                pw = water_balance_penalty(
                    preds_phys, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen, base_loss_fn=_bl,
                    extra_daily=extra_daily, delta_labels=delta_labels,
                )
                if pw is not None:
                    v = float(pw.detach().item())
                    raw_mb["water_balance"] = v
                    w_mb["water_balance"]   = water_balance_weight * v
                    total = total + water_balance_weight * pw

            if npp_balance_weight > 0.0:
                pnpp = npp_balance_penalty(
                    preds_phys, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen, base_loss_fn=_bl,
                    extra_daily=extra_daily, delta_labels=delta_labels,
                )
                if pnpp is not None:
                    v = float(pnpp.detach().item())
                    raw_mb["npp_balance"] = v
                    w_mb["npp_balance"]   = npp_balance_weight * v
                    total = total + npp_balance_weight * pnpp

            if nbp_balance_weight > 0.0:
                pnbp = nbp_balance_penalty(
                    preds_phys, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen, base_loss_fn=_bl,
                    extra_daily=extra_daily, delta_labels=delta_labels,
                )
                if pnbp is not None:
                    v = float(pnbp.detach().item())
                    raw_mb["nbp_balance"] = v
                    w_mb["nbp_balance"]   = nbp_balance_weight * v
                    total = total + nbp_balance_weight * pnbp

            if nbp_delta_ctotal_weight > 0.0:
                pdc = nbp_vs_delta_ctotal_monthly_penalty(
                    preds_phys, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen, base_loss_fn=_bl,
                    extra_daily=extra_daily, delta_labels=delta_labels,
                )
                if pdc is not None:
                    v = float(pdc.detach().item())
                    raw_mb["nbp_vs_delta_ctotal_monthly"] = v
                    w_mb["nbp_vs_delta_ctotal_monthly"]   = nbp_delta_ctotal_weight * v
                    total = total + nbp_delta_ctotal_weight * pdc

            if carbon_partition_weight > 0.0:
                pcp = carbon_partition_penalty(
                    preds_phys, mb_var_idx,
                    month_mask=mmask, month_lengths=mlen, base_loss_fn=_bl,
                    extra_daily=extra_daily, delta_labels=delta_labels,
                )
                if pcp is not None:
                    v = float(pcp.detach().item())
                    raw_mb["carbon_partition_december"] = v
                    w_mb["carbon_partition_december"]   = carbon_partition_weight * v
                    total = total + carbon_partition_weight * pcp

        try:
            loss_fn.last_breakdown = {
                "supervised": float(L_sup.detach().item()),
                "weighted":   dict(w_mb),
                "raw":        dict(raw_mb),
            }
        except Exception:
            pass

        return total

    return loss_fn

# ---------------------------------------------------------------------------
# Selector (thread the aggregation mode)
# ---------------------------------------------------------------------------

def build_loss_fn(
    *,
    idx_monthly: List[int],
    idx_annual: List[int],
    use_mass_balances: bool,
    # common args
    loss_type: str = "mse",
    monthly_weights: Optional[Tensor | List[float]] = None,
    annual_weights:  Optional[Tensor | List[float]] = None,
    # mass-balance extras (ignored when use_mass_balances=False)
    mb_var_idx: Optional[Dict[str, int]] = None,
    water_balance_weight: float = 0.0,
    npp_balance_weight: float = 0.0,
    nbp_balance_weight: float = 0.0,
    nbp_delta_ctotal_weight: float = 0.0,
    carbon_partition_weight: float = 0.0,
    mu_out: Optional[Tensor] = None,
    sd_out: Optional[Tensor] = None,
    delta_labels: bool = False,
) -> Callable[[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]], Tensor]:
    """
    Chooses loss function to use based on flags.
    Pass delta_labels=True to aggregate with sums; False for averages.
    """
    if use_mass_balances:
        return make_supervised_plus_mass_balance_loss(
            idx_monthly, idx_annual,
            loss_type=loss_type,
            monthly_weights=monthly_weights,
            annual_weights=annual_weights,
            mb_var_idx=mb_var_idx,
            water_balance_weight=water_balance_weight,
            npp_balance_weight=npp_balance_weight,
            nbp_balance_weight=nbp_balance_weight,
            nbp_delta_ctotal_weight=nbp_delta_ctotal_weight,
            carbon_partition_weight=carbon_partition_weight,
            mu_out=mu_out,
            sd_out=sd_out,
            delta_labels=delta_labels,
        )
    else:
        return make_supervised_loss(
            idx_monthly, idx_annual,
            loss_type=loss_type,
            monthly_weights=monthly_weights,
            annual_weights=annual_weights,
            delta_labels=delta_labels,
        )