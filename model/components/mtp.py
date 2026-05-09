"""
Multi-Token Prediction (MTP) Head — DeepSeek-style.

Clean auxiliary loss heads: no embedding feedback, no autoregressive
entanglement, no per-step MLP towers.

Design: shared projection + per-step lightweight residual adapters.
    h_k = h + scale_k * SiLU(shared_proj(h)) + bias_k
    loss_k = CE(lm_head(embed_down(h_k[:, :-k])), labels[:, k:])

L_total = L_main + mtp_weight * sum(loss_k)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class MTPHead(nn.Module):
    """
    DeepSeek-style Multi-Token Prediction head.

    Conditionally independent given hidden state — no recursive
    token injection, no autoregressive entanglement.

    Args:
        d_model: model dimension
        embed_dim: embedding dimension (for factorized projection)
        vocab_size: vocabulary size
        mtp_steps: number of future tokens to predict (default: 2)
        tie_output: whether to tie output projection to embed_weight
        embed_weight: shared embedding weight (for tied projections)
    """

    def __init__(self, d_model: int, embed_dim: int, vocab_size: int,
                 mtp_steps: int = 2, tie_output: bool = True, embed_weight=None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.mtp_steps = mtp_steps
        self.tie_output = tie_output

        # Shared transform — single projection, not per-step MLP towers
        self.shared_proj = nn.Linear(d_model, d_model, bias=False)

        # Per-step lightweight residual adapters (scale + bias)
        self.step_scales = nn.ParameterList([
            nn.Parameter(torch.ones(d_model) * 0.1)
            for _ in range(mtp_steps)
        ])
        self.step_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model))
            for _ in range(mtp_steps)
        ])

        # Output projection (tied to embedding)
        if tie_output:
            self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
            if embed_weight is not None:
                self.output_proj.weight = embed_weight

        # Stats
        self._last_losses = []

    def forward(self, hidden_states, labels, loss_mask=None, embed_down=None):
        """
        Compute MTP loss — clean multi-shift supervision.

        Args:
            hidden_states: (B, T, d_model) — final hidden states
            labels: (B, T) — target token IDs
            loss_mask: (B, T) — optional mask for selective loss
            embed_down: Linear(d_model, embed_dim) — factorized projection

        Returns:
            total MTP loss (scalar)
        """
        B, T, d = hidden_states.shape
        total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)
        self._last_losses = []

        # Shared transform (computed once, reused across all steps)
        h_shared = F.silu(self.shared_proj(hidden_states))  # (B, T, d)

        for k in range(self.mtp_steps):
            shift = k + 1
            if shift >= T:
                break

            # Per-step residual refinement — NOT a full MLP tower
            h_k = hidden_states + self.step_scales[k] * h_shared + self.step_biases[k]

            # Slice: predict tokens at t+shift from hidden states at t
            pred_h = h_k[:, :-shift, :]       # (B, T-shift, d)
            target = labels[:, shift:]          # (B, T-shift)

            # Length alignment
            min_len = min(pred_h.shape[1], target.shape[1])
            pred_h = pred_h[:, :min_len, :]
            target = target[:, :min_len]

            # Selective loss masking
            if loss_mask is not None:
                mask = loss_mask[:, shift:shift + min_len]
                active_mask = (mask == 1).view(-1)
            else:
                active_mask = torch.ones(B * min_len, dtype=torch.bool, device=hidden_states.device)

            pred_flat = pred_h.reshape(B * min_len, -1)[active_mask]
            target_flat = target.reshape(-1)[active_mask]

            if pred_flat.shape[0] == 0:
                continue

            # Chunked cross-entropy to control peak memory (128 × 262k × 4B ≈ 134MB)
            chunk_size = 128
            step_loss = 0.0
            total_tokens = pred_flat.shape[0]

            for chunk_start in range(0, total_tokens, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_tokens)
                h_chunk = pred_flat[chunk_start:chunk_end]
                y_chunk = target_flat[chunk_start:chunk_end]

                def compute_chunk_loss(xc, yc):
                    if embed_down is not None:
                        xc = embed_down(xc)
                    if self.tie_output:
                        logits_chunk = F.linear(xc, self.output_proj.weight)
                    else:
                        logits_chunk = self.output_proj(xc)
                    logits_chunk = logits_chunk.float()
                    logits_chunk = logits_chunk.clamp(min=-65504.0, max=65504.0)
                    logits_chunk = torch.nan_to_num(logits_chunk, nan=0.0)
                    return F.cross_entropy(logits_chunk, yc, reduction="sum", ignore_index=-100)
                
                if h_chunk.requires_grad:
                    loss_chunk = torch.utils.checkpoint.checkpoint(
                        compute_chunk_loss, h_chunk, y_chunk, use_reentrant=False
                    )
                else:
                    loss_chunk = compute_chunk_loss(h_chunk, y_chunk)

                step_loss = step_loss + loss_chunk

            loss = step_loss / total_tokens
            self._last_losses.append(loss.item())
            total_loss = total_loss + loss

        return total_loss

    def stats(self) -> dict:
        return {
            "mtp_steps": self.mtp_steps,
            "last_losses": self._last_losses,
            "params": sum(p.numel() for p in self.parameters()),
        }
