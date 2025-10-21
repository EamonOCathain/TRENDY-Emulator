# src/training/delta.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List
import torch

"""
Delta-mode helpers (generalised for any group).

Design (this version)
---------------------
- The model outputs NORMALISED daily *deltas* for each group:
      Δŷ̃_day ∈ R^{C}
  but these are interpreted as *segment-level* (month / year) deltas:
  within each segment, ALL days share the same anchor (previous segment's
  absolute), and each day's prediction is an independent estimate of the
  segment delta.

- Reconstruction to NORMALISED daily absolutes is therefore:
      ŷ̃_day = Ã_segment + Δŷ̃_day,
  where Ã_segment is the normalised absolute of the **previous** segment.
  After reconstruction, supervision is applied to the pooled
  (monthly/annual) averages of ŷ̃_day (your `custom_loss` already does this).

Anchors
-------
- Teacher-forced:
    * Monthly group:   initial anchor = last month of the previous year label
                      (Dec_{y-1}); then for each month m in year y, we set the
                      next anchor to that month’s label (i.e. label_{m} is the
                      "previous" for month m+1 during teacher training).
                      We only *need* the very first anchor for the year to
                      reconstruct; updating the anchor internally uses the
                      reconstructed segment mean (not labels) to avoid extra
                      I/O. You may alternatively feed labels; we avoid it here.
    * Annual  group:   initial anchor = previous year's annual label.

- Carry rollout:
    * Monthly group:   initial anchor = previous predicted monthly absolute
                      (last month of previous year); then updated from the
                      *predicted* mean absolute of the current month.
    * Annual  group:   initial anchor = previous predicted annual absolute.

Implementation notes
--------------------
- No cumulative sum across days. Reconstruction is purely:
      y_abs[d] = anchor_seg + delta[d]
  with a piecewise-constant anchor per segment and sequential anchor updates.

Public API
----------
- DeltaContext(enabled, month_slices)
- DeltaContext.reconstruct_groups_daily_segmentwise(...)
- build_delta_ctx(enabled, month_slices)

All logic is contained here to keep call sites clean.
"""


# -------------------------- config --------------------------

@dataclass
class DeltaConfig:
    enabled: bool = False


class DeltaContext:
    def __init__(self, *, enabled: bool, month_slices: Sequence[Tuple[int, int]]):
        """
        month_slices: list of 12 (start, end) tuples that cover 365 (noleap).
        """
        self.cfg = DeltaConfig(enabled=bool(enabled))
        self.month_slices: List[Tuple[int, int]] = list(month_slices)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    # -------------------------- core piecewise reconstruction --------------------------

    @staticmethod
    def _segmentwise_additive_reconstruct(
        *,
        preds_daily_delta: torch.Tensor,   # [B, T, C] (normalised daily deltas)
        segment_slices: Sequence[Tuple[int, int]],
        initial_anchor: torch.Tensor,      # [B, C] (normalised absolute of previous segment)
        update_anchor_from_segment_mean: bool = True,
    ) -> torch.Tensor:
        """
        Reconstruct NORMALISED daily absolutes from NORMALISED daily deltas by
        treating each segment independently with a constant anchor:

            for segment s with days t ∈ [start_s, end_s):
                y_abs[t] = A_s + delta[t]
                mu_s     = mean_t y_abs[t]
                A_{s+1}  = mu_s   (if update_anchor_from_segment_mean)

        Args
        ----
        preds_daily_delta: [B, T, C]
        segment_slices:    list of (start, end) slices that tile T
        initial_anchor:    [B, C] previous segment absolute (normalised)
        update_anchor_from_segment_mean:
                           if True, the next segment anchor is set to the
                           reconstructed mean absolute of the current segment.

        Returns
        -------
        y_abs: [B, T, C] normalised daily absolutes.
        """
        B, T, C = preds_daily_delta.shape
        y_abs = torch.empty_like(preds_daily_delta)

        # current anchor (normalised)
        A = initial_anchor  # [B, C]

        for (s, e) in segment_slices:
            seg_delta = preds_daily_delta[:, s:e, :]           # [B, L, C]
            seg_abs   = A.unsqueeze(1) + seg_delta             # [B, L, C]
            y_abs[:, s:e, :] = seg_abs

            if update_anchor_from_segment_mean:
                # mean absolute over the segment to carry forward
                mu = seg_abs.mean(dim=1)                       # [B, C]
                A = mu

        return y_abs

    # -------------------------- anchor utilities --------------------------

    @staticmethod
    def dense_anchor_from_state_dims(
        *,
        total_dim: int,
        state_local_idx: Sequence[int],
        prev_state_vector: Optional[torch.Tensor],  # [B, len(state_local_idx)] or None
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build a [B, total_dim] anchor, filling only the provided indices from
        prev_state_vector and leaving others at 0 (unknown).
        """
        B = 1 if prev_state_vector is None else prev_state_vector.shape[0]
        anchor = torch.zeros((B, total_dim), device=device, dtype=dtype)
        if (prev_state_vector is not None) and len(state_local_idx) > 0:
            anchor[:, state_local_idx] = prev_state_vector
        return anchor

    @staticmethod
    def teacher_initial_anchor_monthly_from_prev_last(
        yb_m_prev_last: torch.Tensor,   # [B, 1, C] previous year's last-month absolute (normalised)
    ) -> torch.Tensor:
        return yb_m_prev_last[:, 0, :]  # [B, C]

    @staticmethod
    def teacher_initial_anchor_annual_from_prev(
        yb_a_prev: torch.Tensor,        # [B, 1, C] previous year's annual absolute (normalised)
    ) -> torch.Tensor:
        return yb_a_prev[:, 0, :]       # [B, C]

    # -------------------------- public: group reconstructor --------------------------

    def reconstruct_groups_daily_segmentwise(
        self,
        *,
        preds: torch.Tensor,                # [B, 365, nm+na] daily deltas (normalised)
        nm: int,
        na: int,
        month_slices: Sequence[Tuple[int, int]],
        mode: str,                          # "carry" or "teacher"
        # carry anchors (state-only → densified)
        out_m_idx: Optional[Sequence[int]] = None,
        out_a_idx: Optional[Sequence[int]] = None,
        prev_monthly_state: Optional[torch.Tensor] = None,  # [B, len(out_m_idx)] or None
        prev_annual_state:  Optional[torch.Tensor] = None,  # [B, len(out_a_idx)] or None
        # teacher-forced initial anchors
        yb_m_prev_last: Optional[torch.Tensor] = None,      # [B,1,nm] (Dec of previous year)
        yb_a_prev:      Optional[torch.Tensor] = None,      # [B,1,na]
    ) -> torch.Tensor:
        """
        Reconstruct NORMALISED daily absolutes for both groups, concatenated.

        Monthly group:
          - piecewise (12 segments from `month_slices`)
          - initial anchor:
              * carry   : densified from prev_monthly_state (pred)
              * teacher : last-month label of previous year (yb_m_prev_last)
          - next-segment anchor = mean of reconstructed current segment

        Annual group:
          - single segment of full year [(0, T)]
          - initial anchor:
              * carry   : densified from prev_annual_state (pred)
              * teacher : previous year's annual label (yb_a_prev)
        """
        if not self.enabled:
            return preds  # pass-through when delta-mode is OFF

        B, T, Cout = preds.shape
        assert nm + na == Cout, "nm + na must equal output channels"
        assert T == sum(e - s for (s, e) in month_slices), "month_slices must tile T"

        preds_m_delta = preds[..., :nm]           # [B, T, nm]
        preds_a_delta = preds[..., nm:nm+na]      # [B, T, na]

        # ----- Monthly group -----
        if mode == "carry":
            init_anchor_m = self.dense_anchor_from_state_dims(
                total_dim=nm,
                state_local_idx=out_m_idx or [],
                prev_state_vector=prev_monthly_state,
                device=preds.device,
                dtype=preds.dtype,
            )
        elif mode == "teacher":
            assert yb_m_prev_last is not None, \
                "teacher mode requires yb_m_prev_last for monthly initial anchor"
            init_anchor_m = self.teacher_initial_anchor_monthly_from_prev_last(yb_m_prev_last)
        else:
            raise ValueError(f"Unknown mode {mode!r}")

        preds_m_abs = self._segmentwise_additive_reconstruct(
            preds_daily_delta=preds_m_delta,
            segment_slices=month_slices,
            initial_anchor=init_anchor_m,
            update_anchor_from_segment_mean=True,
        )  # [B, T, nm]

        # ----- Annual group (single segment covering the full year) -----
        ann_slices = [(0, T)]
        if mode == "carry":
            init_anchor_a = self.dense_anchor_from_state_dims(
                total_dim=na,
                state_local_idx=out_a_idx or [],
                prev_state_vector=prev_annual_state,
                device=preds.device,
                dtype=preds.dtype,
            )
        else:  # teacher
            assert yb_a_prev is not None, "teacher mode requires yb_a_prev for annual initial anchor"
            init_anchor_a = self.teacher_initial_anchor_annual_from_prev(yb_a_prev)

        preds_a_abs = self._segmentwise_additive_reconstruct(
            preds_daily_delta=preds_a_delta,
            segment_slices=ann_slices,
            initial_anchor=init_anchor_a,
            update_anchor_from_segment_mean=True,  # next-year carry would use this mean
        )  # [B, T, na]

        return torch.cat([preds_m_abs, preds_a_abs], dim=-1)


# -------------------------- simple builder --------------------------

def build_delta_ctx(*, enabled: bool, month_slices: Sequence[Tuple[int, int]]) -> DeltaContext:
    """
    Create a DeltaContext. Reconstruction uses segmentwise anchors and
    pools to targets; no mean-correction is applied here.
    """
    return DeltaContext(enabled=enabled, month_slices=month_slices)