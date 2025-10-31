import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

# ------------------------------ Positional Encoding -----------------------------------
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        # Create positional encoding matrix
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model)
        )
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0)  # → [1, max_len, dim_model]
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding):
        # token_embedding: [batch, seq_len, dim_model]
        seq_len = token_embedding.size(1)
        return self.dropout(token_embedding + self.pos_encoding[:, :seq_len, :])


# ------------------------------ Transformer Architecture -----------------------------------
class CustomTransformer(nn.Module):
    """Custom transformer architecture for time series processing"""
    def __init__(self, input_dim, output_dim, d=128, h=1024, g=256,
                 num_layers=4, nhead=8, dropout=0.1, max_len=1000):
        super().__init__()
        self.d = d

        # Pre-processing convolutional layers
        self.pre_conv = nn.Sequential(
            nn.Conv1d(input_dim, h, kernel_size=1),
            nn.PReLU(),
            nn.Conv1d(h, 3 * d, kernel_size=1),
            nn.BatchNorm1d(3 * d)
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(dim_model=d, dropout_p=dropout, max_len=max_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=4*d,
            dropout=dropout, batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # Post-processing convolutional layers
        self.post_conv = nn.Sequential(
            nn.Conv1d(d, g, kernel_size=1),
            nn.PReLU(),
            nn.Conv1d(g, output_dim, kernel_size=1)
        )

    def forward(self, x, key_padding_mask: torch.Tensor | None = None):
        """
        x: [batch, n_days, input_dim]
        key_padding_mask: [batch, n_days] bool, True = PAD (ignored by attention)
        """
        x = x.permute(0, 2, 1)         # [B, in_dim, n_days]
        x = self.pre_conv(x)           # [B, 3*d, n_days]

        x = x.view(x.size(0), 3, self.d, x.size(2)).mean(dim=1)  # [B, d, n_days]
        x = x.permute(0, 2, 1)                                   # [B, n_days, d]

        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # <— use the mask

        x = x.permute(0, 2, 1)          # [B, d, n_days]
        x = self.post_conv(x)           # [B, out_dim, n_days]
        return x.permute(0, 2, 1)       # [B, n_days, out_dim]

# ------------------------------ Year Processor -----------------------------------
# ------------------------------ Year Processor -----------------------------------
class YearProcessor(nn.Module):
    """
    One module, two forward modes:

      mode="batch_months":
        - Processes all 12 months in parallel (each padded to 31 days)
        - No within-year carry; fastest path (good for pretraining)

      mode="sequential_months":
        - Processes months m=0..11 in order (each padded to 31)
        - After finishing month m, computes the mean of the predicted
          monthly-state channels and injects them into the input channels
          for month m+1 (days d of that month)
        - This matches your carry training/inference needs

    Input : x  of shape [B, 365, in_dim]
    Output: out of shape [B, 365, out_dim]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        # Needed for sequential carry (ignored in batch mode)
        in_monthly_state_idx: Optional[List[int]] = None,
        out_monthly_state_idx: Optional[List[int]] = None,
        month_lengths: Optional[List[int]] = None,
        # inner transformer config
        d: int = 128, h: int = 1024, g: int = 256,
        num_layers: int = 4, nhead: int = 8, dropout: float = 0.1,
        transformer_kwargs: Optional[dict] = None,
        # default mode
        mode: str = "batch_months",
    ):
        super().__init__()
        self.input_dim  = int(input_dim)
        self.output_dim = int(output_dim)

        # calendar definition (noleap by default)
        if month_lengths is None:
            month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        assert sum(month_lengths) == 365, "Month lengths must sum to 365 for noleap."
        self.month_lengths = list(month_lengths)

        # Build month→day index map and padding mask
        month_day_idx, month_pad_mask = self._build_month_index_and_mask(self.month_lengths)
        self.register_buffer("month_day_idx", month_day_idx, persistent=False)     # [12, 31] in [0..364] or -1
        self.register_buffer("month_pad_mask", month_pad_mask, persistent=False)   # [12, 31] (True = pad)

        # Carry index lists (only used in sequential mode)
        self.in_monthly_state_idx  = None if in_monthly_state_idx  is None else torch.as_tensor(in_monthly_state_idx,  dtype=torch.long)
        self.out_monthly_state_idx = None if out_monthly_state_idx is None else torch.as_tensor(out_monthly_state_idx, dtype=torch.long)

        # inner monthly transformer (shared for all months)
        tfm_cfg = dict(d=d, h=h, g=g, num_layers=num_layers, nhead=nhead, dropout=dropout, max_len=31)
        if transformer_kwargs:
            tfm_cfg.update(transformer_kwargs)
        tfm_cfg["max_len"] = max(31, int(tfm_cfg.get("max_len", 31)))

        self.inner = CustomTransformer(input_dim=input_dim, output_dim=output_dim, **tfm_cfg)

        # runtime mode
        assert mode in ("batch_months", "sequential_months")
        self._mode = mode

    # ---------- public control ----------
    @property
    def mode(self) -> str:
        return self._mode

    @torch.no_grad()
    def set_mode(self, mode: str) -> None:
        """
        Switches runtime behavior without changing parameters.
        """
        assert mode in ("batch_months", "sequential_months")
        self._mode = mode

    # ---------- helpers ----------
    @staticmethod
    def _build_month_index_and_mask(month_lengths: List[int]) -> tuple[torch.Tensor, torch.Tensor]:
        starts = []
        s = 0
        for L in month_lengths:
            starts.append(s)
            s += L

        day_idx = torch.full((12, 31), -1, dtype=torch.long)
        for m, (st, L) in enumerate(zip(starts, month_lengths)):
            day_idx[m, :L] = torch.arange(st, st + L, dtype=torch.long)
        pad_mask = day_idx.eq(-1)
        return day_idx, pad_mask

    def _gather_months(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather all months from [B, 365, C] into a [B, 12, 31, C] tensor with padding mask.
        Returns (Xm, mask) where mask is [B, 12, 31] (True = pad).
        """
        B, _, C = x.shape
        idx31 = self.month_day_idx.clamp_min(0)                          # [12, 31]
        idx_expanded = idx31.view(1, 12, 31, 1).expand(B, 12, 31, C)     # [B, 12, 31, C]
        x_exp = x.unsqueeze(2).expand(B, 365, 31, C)                     # [B, 365, 31, C]
        Xm = torch.gather(x_exp, dim=1, index=idx_expanded)              # [B, 12, 31, C]
        mask = self.month_pad_mask.unsqueeze(0).expand(B, -1, -1)        # [B, 12, 31]
        return Xm, mask

    @staticmethod
    def _monthly_mean(preds_month: torch.Tensor) -> torch.Tensor:
        # preds_month: [B, n_days, C] -> [B, C]
        return preds_month.mean(dim=1)

    # ---------- forward paths ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 365, in_dim]  ->  out: [B, 365, out_dim]
        """
        if x.dim() != 3 or x.size(1) != 365 or x.size(2) != self.input_dim:
            raise ValueError(f"Expected [B, 365, {self.input_dim}], got {tuple(x.shape)}")

        if self._mode == "batch_months":
            return self._forward_batch_months(x)
        else:
            # safety: require indices for sequential carry
            if (self.in_monthly_state_idx is None) or (self.out_monthly_state_idx is None):
                raise RuntimeError("sequential_months mode requires in_monthly_state_idx and out_monthly_state_idx")
            return self._forward_sequential_months(x)

    def _forward_batch_months(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast path: all 12 months in parallel, no within-year carry.
        """
        B, _, _ = x.shape
        Xm, mask = self._gather_months(x)                       # [B, 12, 31, in_dim], [B, 12, 31]

        Xm = Xm.reshape(B * 12, 31, self.input_dim)             # was: .view(...)
        mask_flat = mask.contiguous().reshape(B * 12, 31)

        Ym = self.inner(Xm, key_padding_mask=mask_flat)         # [B*12, 31, out_dim]
        Ym = Ym.reshape(B, 12, 31, self.output_dim)             # was: .view(...)

        out = x.new_zeros(B, 365, self.output_dim)
        for m, L in enumerate(self.month_lengths):
            day_ids = self.month_day_idx[m, :L]
            out[:, day_ids, :] = Ym[:, m, :L, :]
        return out

    def _forward_sequential_months(self, x: torch.Tensor) -> torch.Tensor:
        """
        Month→month carry path with DETACHED carry:
        - after finishing month m, compute the mean over the monthly-state outputs,
            detach it from the graph, and inject into month m+1 inputs under no_grad.
        - result: months are independent for autograd; only shared params get grads.
        """
        B, _, _ = x.shape
        device = x.device
        out = x.new_zeros(B, 365, self.output_dim)

        in_m_idx  = self.in_monthly_state_idx.to(device)
        out_m_idx = self.out_monthly_state_idx.to(device)

        # Work copy we’re free to overwrite. Keep it detached so writes don’t build graphs.
        # (Inputs typically don’t require grad anyway, but this makes the intent explicit.)
        x_work = x.detach().clone()

        prev_m_mean: torch.Tensor | None = None

        for m, L in enumerate(self.month_lengths):
            day_ids = self.month_day_idx[m, :L]  # [L] long

            # --- Inject previous-month carry into CURRENT month inputs (DETACHED) ---
            if (m > 0) and (prev_m_mean is not None) and (in_m_idx.numel() > 0):
                inject = prev_m_mean.view(B, 1, -1).expand(B, L, -1)  # [B, L, n_mstates]
                b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, L, in_m_idx.numel())
                r_idx = day_ids.view(1, L, 1).expand(B, L, in_m_idx.numel())
                c_idx = in_m_idx.view(1, 1, -1).expand(B, L, in_m_idx.numel())
                with torch.no_grad():
                    x_work.index_put_((b_idx, r_idx, c_idx), inject, accumulate=False)

            # --- run month m ---
            Xm = x_work[:, day_ids, :].contiguous()  # [B, L, in_dim]
            if L < 31:
                pad_len = 31 - L
                pad = torch.zeros(B, pad_len, self.input_dim, device=device, dtype=Xm.dtype)
                Xm31 = torch.cat([Xm, pad], dim=1)
                mask31 = torch.zeros(B, 31, dtype=torch.bool, device=device)
                mask31[:, L:] = True
            else:
                Xm31 = Xm
                mask31 = torch.zeros(B, 31, dtype=torch.bool, device=device)

            # This forward builds the graph only within month m
            Ym31 = self.inner(Xm31, key_padding_mask=mask31)  # [B,31,out_dim]
            out[:, day_ids, :] = Ym31[:, :L, :]

            # --- prepare DETACHED carry for NEXT month and inject (also DETACHED) ---
            if m < 11:
                # Compute monthly-state mean and DETACH so no grads flow month→month
                m_slice = Ym31[:, :L, :][:, :, out_m_idx]          # [B, L, n_mstates]
                prev_m_mean = m_slice.mean(dim=1).detach()         # [B, n_mstates]  (DETACHED)

                if in_m_idx.numel() > 0:
                    next_len = self.month_lengths[m + 1]
                    next_ids = self.month_day_idx[m + 1, :next_len].to(device)  # [next_len]
                    inject_next = prev_m_mean.view(B, 1, -1).expand(B, next_len, -1)
                    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, next_len, in_m_idx.numel())
                    r_idx = next_ids.view(1, next_len, 1).expand(B, next_len, in_m_idx.numel())
                    c_idx = in_m_idx.view(1, 1, -1).expand(B, next_len, in_m_idx.numel())
                    with torch.no_grad():
                        x_work.index_put_((b_idx, r_idx, c_idx), inject_next, accumulate=False)

        return out