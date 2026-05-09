"""
Multi-Token Prediction (MTP) Head.

Predicts token at t+1 and t+2 using hidden state at t.
2-layer MLP per step, weight-tied to embedding matrix for final projection.

L_total = L_main + mtp_weight × (L_mtp1 + L_mtp2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from model.components.rms_norm import RMSNorm


class MTPHead(nn.Module):
    """
    Multi-Token Prediction head.

    For each step t, predicts tokens at t+1, t+2, ..., t+mtp_steps.
    Each prediction uses a 2-layer MLP from hidden states +
    the embedding of the previous prediction target.

    Args:
        d_model: model dimension
        vocab_size: vocabulary size
        mtp_steps: number of future tokens to predict (default: 2)
        tie_output: whether to tie output projections to embed_weight
        embed_weight: shared embedding weight (for tied projections)
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        mtp_steps: int = 2,
        tie_output: bool = False,
        embed_weight: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.mtp_steps = mtp_steps
        self.tie_output = tie_output

        # Per-step prediction MLPs
        self.prediction_heads = nn.ModuleList()
        for _ in range(mtp_steps):
            self.prediction_heads.append(nn.Sequential(
                RMSNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model, bias=False),
                nn.SiLU(),
                RMSNorm(d_model),
                nn.Linear(d_model, d_model, bias=False),
            ))

        # Per-step output projections — NOT tied to embedding if tie_output=False.
        # If tied, we project using either a single tied linear or directly with F.linear.
        self.output_projs = None
        if not tie_output:
            self.output_projs = nn.ModuleList()
            for _ in range(mtp_steps):
                proj = nn.Linear(d_model, vocab_size, bias=False)
                self.output_projs.append(proj)
        else:
            self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
            if embed_weight is not None:
                self.output_proj.weight = embed_weight

        # Embedding projection for feeding back targets during training
        self.embed_proj = nn.Linear(d_model, d_model, bias=False)

        # Stats
        self._last_losses = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        embed_weight: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MTP loss.

        Args:
            hidden_states: (B, T, d_model) — final hidden states from the model
            labels: (B, T) — target token IDs (standard next-token labels)
            embed_weight: (vocab_size, d_model) — embedding table for lookup
            loss_mask: (B, T) — optional mask for selective loss

        Returns:
            total MTP loss (scalar)
        """
        B, T, d = hidden_states.shape
        total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        self._last_losses = []

        for step in range(self.mtp_steps):
            # Target: labels shifted by (step + 1) positions
            # For step=0: predict t+2 (main head predicts t+1)
            # For step=1: predict t+3
            shift = step + 1
            if shift >= T:
                break

            # Hidden states for positions that have valid targets
            h = hidden_states[:, :T - shift, :]  # (B, T-shift, d)

            # Get embeddings of the label at position t+step (the previous MTP target)
            # For step 0: embed labels at t+0 (the current target)
            if embed_weight is not None:
                prev_target_ids = labels[:, step:T - shift + step]  # (B, T-shift)
                prev_target_embed = F.embedding(prev_target_ids, embed_weight)  # (B, T-shift, d)
            else:
                prev_target_embed = torch.zeros_like(h)

            # Concatenate hidden state + previous target embedding
            mlp_input = torch.cat([h, self.embed_proj(prev_target_embed)], dim=-1)  # (B, T-shift, 2d)

            # Predict
            pred_hidden = self.prediction_heads[step](mlp_input)  # (B, T-shift, d)
            if self.tie_output:
                logits = F.linear(pred_hidden, self.output_proj.weight)
            else:
                logits = self.output_projs[step](pred_hidden)           # (B, T-shift, vocab)

            # Target labels for this step
            target = labels[:, shift:T]  # (B, T-shift)

            # Handle length mismatch
            min_len = min(logits.shape[1], target.shape[1])
            logits = logits[:, :min_len, :]
            target = target[:, :min_len]

            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                target.reshape(-1),
                reduction="none",
            ).view(B, min_len)

            # Apply loss mask if provided
            if loss_mask is not None:
                mask = loss_mask[:, shift:shift + min_len]
                loss = (loss * mask).sum() / (mask.sum() + 1e-10)
            else:
                loss = loss.mean()

            self._last_losses.append(loss.item())
            total_loss = total_loss + loss

        return total_loss

    def stats(self) -> dict:
        return {
            "mtp_steps": self.mtp_steps,
            "last_losses": self._last_losses,
            "params": sum(p.numel() for p in self.parameters()),
        }
