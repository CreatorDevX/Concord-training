"""
MLP Router — 2-layer MLP with dual load balancing:
    1. Differentiable auxiliary loss (primary) — prevents early expert collapse
    2. Bias-based drift correction (secondary) — fine-grained late-training correction

Architecture:
    h = rms_norm(x)
    h = silu(W1 @ h)           # (B, T, router_hidden)
    logits = W2 @ h             # (B, T, n_experts)
    logits = logits + bias      # per-expert learned bias
    scores = softmax(logits)
    top_k_indices = topk(scores, k=n_routed)

Auxiliary Loss (Switch Transformer style):
    L_aux = α * n_experts * Σ_e (f_e * P_e)
    where f_e = fraction of tokens routed to expert e
          P_e = mean router probability for expert e
    This is differentiable and prevents expert collapse in early training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from model.components.rms_norm import RMSNorm


class MLPRouter(nn.Module):
    """
    2-layer MLP router with auxiliary load balancing loss + bias correction.

    Args:
        d_model: input dimension
        n_experts: number of experts to route to
        n_routed: top-k experts to select per token
        hidden_dim: hidden dimension of MLP
        bias_update_interval: steps between bias updates
        bias_lr: learning rate for bias updates
        aux_loss_coeff: coefficient for auxiliary load balancing loss
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        n_routed: int,
        hidden_dim: int = 384,
        bias_update_interval: int = 1000,
        bias_lr: float = 0.01,
        aux_loss_coeff: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_routed = n_routed
        self.bias_update_interval = bias_update_interval
        self.bias_lr = bias_lr
        self.aux_loss_coeff = aux_loss_coeff

        # 2-layer MLP
        self.norm = RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_experts, bias=False)

        # Load-balancing bias (not gradient-trained)
        self.register_buffer("expert_bias", torch.zeros(n_experts))

        # Load tracking
        self.register_buffer("expert_load_acc", torch.zeros(n_experts))
        self.register_buffer("token_count_acc", torch.zeros(1))
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))

        # Stats
        self._last_load_distribution = None
        self._last_aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            x: (B, T, d_model)

        Returns:
            indices: (B, T, n_routed) — expert indices
            scores: (B, T, n_routed) — softmax scores for selected experts
            aux_loss: scalar — differentiable load balancing loss
        """
        B, T, _ = x.shape

        # MLP router
        h = self.norm(x)
        h = F.silu(self.w1(h))           # (B, T, hidden_dim)
        logits = self.w2(h)              # (B, T, n_experts)

        # Add bias (not gradient-trained)
        logits = logits + self.expert_bias

        # Softmax scores
        scores = F.softmax(logits, dim=-1)  # (B, T, n_experts)

        # Top-k selection
        top_scores, top_indices = torch.topk(scores, self.n_routed, dim=-1)

        # Re-normalize selected scores
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-10)

        # ── Auxiliary load balancing loss (differentiable) ──────────────
        # Switch Transformer style: L_aux = α * n_experts * Σ_e (f_e * P_e)
        # f_e: fraction of tokens dispatched to expert e (from hard routing)
        # P_e: mean router probability for expert e (from soft scores — differentiable)
        # Must be fp32 — fp16 can underflow on the product sum
        aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        if self.training:
            with torch.no_grad():
                # f_e: fraction of tokens routed to each expert (non-differentiable)
                one_hot = F.one_hot(top_indices, self.n_experts).float()
                tokens_per_expert = one_hot.sum(dim=(0, 1, 2))
                total_tokens = B * T * self.n_routed
                f_e = tokens_per_expert / total_tokens

                # Load tracking for bias updates
                self.expert_load_acc += tokens_per_expert
                self.token_count_acc += total_tokens
                self.step_counter += 1
                self._last_load_distribution = f_e

            # P_e: mean probability per expert (differentiable through scores)
            P_e = scores.mean(dim=(0, 1))

            aux_loss = self.aux_loss_coeff * self.n_experts * (f_e * P_e).sum()
            # Sanitize: if router inputs contain NaN (from upstream fp16 overflow),
            # P_e will be NaN and aux_loss will be NaN. We zero it to prevent
            # the entire total_loss from becoming NaN.
            if torch.isnan(aux_loss).any() or torch.isinf(aux_loss).any():
                aux_loss = torch.zeros_like(aux_loss)
            self._last_aux_loss = aux_loss.item()

        return top_indices, top_scores, aux_loss

    def update_bias(self):
        """
        Update expert bias based on accumulated load statistics.
        Called every `bias_update_interval` steps.
        """
        if self.token_count_acc.item() == 0:
            return

        # Actual load fraction per expert
        actual_load = self.expert_load_acc / self.token_count_acc

        # Target: uniform distribution
        target_load = torch.ones_like(actual_load) / self.n_experts

        # Update bias
        self.expert_bias += self.bias_lr * (target_load - actual_load)

        # Reset accumulators
        self.expert_load_acc.zero_()
        self.token_count_acc.zero_()

    def should_update_bias(self) -> bool:
        """Check if it's time to update the bias."""
        return self.step_counter.item() > 0 and self.step_counter.item() % self.bias_update_interval == 0

    def stats(self) -> dict:
        result = {
            "bias_mean": self.expert_bias.mean().item(),
            "bias_std": self.expert_bias.std().item(),
            "bias_min": self.expert_bias.min().item(),
            "bias_max": self.expert_bias.max().item(),
            "step_counter": self.step_counter.item(),
            "aux_loss": self._last_aux_loss,
        }
        if self._last_load_distribution is not None:
            load = self._last_load_distribution
            result["load_mean"] = load.mean().item()
            result["load_std"] = load.std().item()
            result["load_max"] = load.max().item()
            result["load_min"] = load.min().item()
            result["load_max_over_mean"] = (load.max() / (load.mean() + 1e-10)).item()
        return result
