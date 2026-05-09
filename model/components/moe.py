"""
MoE Block — Router + Shared Expert + Routed Experts + dispatch_and_combine.

Architecture:
    shared_out  = shared_expert(x)
    indices, scores = router(x)
    routed_out  = dispatch_and_combine(x, indices, scores, experts)
    output      = shared_out + routed_out

dispatch_and_combine:
    Sort tokens by assigned expert index → batch matmul per expert →
    unsort and weight by router scores. No token dropping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from model.components.expert import ExpertFFN
from model.components.router import MLPRouter


def dispatch_and_combine(
    x: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    experts: nn.ModuleList,
) -> torch.Tensor:
    """
    Dispatch tokens to experts and combine results.

    Sort tokens by expert → batch process per expert → unsort and weight.
    No token dropping.

    Args:
        x: (B, T, d_model) — input tokens
        indices: (B, T, n_routed) — expert indices per token
        scores: (B, T, n_routed) — routing scores per token
        experts: ModuleList of ExpertFFN modules

    Returns:
        (B, T, d_model) — combined expert outputs
    """
    B, T, d = x.shape
    n_routed = indices.shape[-1]
    n_experts = len(experts)

    # Flatten batch and sequence dims
    x_flat = x.view(B * T, d)                          # (N, d) where N = B*T
    indices_flat = indices.view(B * T, n_routed)        # (N, n_routed)
    scores_flat = scores.view(B * T, n_routed)          # (N, n_routed)

    # Output accumulator
    output = torch.zeros_like(x_flat)                   # (N, d)

    # Process each routing slot
    for slot in range(n_routed):
        slot_indices = indices_flat[:, slot]             # (N,) — expert id per token
        slot_scores = scores_flat[:, slot]               # (N,) — score per token

        # Group tokens by expert
        for expert_idx in range(n_experts):
            mask = (slot_indices == expert_idx)
            if not mask.any():
                continue

            # Gather tokens for this expert
            expert_input = x_flat[mask]                  # (n_tokens, d)

            # Forward through expert
            expert_output = experts[expert_idx](expert_input)  # (n_tokens, d)

            # Weight by score and accumulate
            expert_scores = slot_scores[mask].unsqueeze(-1)    # (n_tokens, 1)
            output[mask] += expert_scores * expert_output

    return output.view(B, T, d)


def dispatch_and_combine_fast(
    x: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    experts: nn.ModuleList,
) -> torch.Tensor:
    """
    Fast dispatch: sort by expert, batch matmul, unsort.

    More efficient than naive loop — groups all tokens for each expert
    and processes them in a single forward pass.

    Args:
        x: (B, T, d_model)
        indices: (B, T, n_routed)
        scores: (B, T, n_routed)
        experts: ModuleList of ExpertFFN

    Returns:
        (B, T, d_model)
    """
    B, T, d = x.shape
    n_routed = indices.shape[-1]
    n_experts = len(experts)

    # Flatten
    x_flat = x.reshape(B * T, d)                       # (N, d)
    indices_flat = indices.reshape(B * T * n_routed)    # (N * n_routed,)
    scores_flat = scores.reshape(B * T * n_routed)      # (N * n_routed,)

    # Expand x for each routing slot
    x_expanded = x_flat.unsqueeze(1).expand(-1, n_routed, -1).reshape(B * T * n_routed, d)

    # Sort by expert index for batch processing
    sorted_idx = torch.argsort(indices_flat, stable=True)
    x_sorted = x_expanded[sorted_idx]
    indices_sorted = indices_flat[sorted_idx]
    scores_sorted = scores_flat[sorted_idx]

    # Find boundaries for each expert
    # Count tokens per expert
    expert_counts = torch.zeros(n_experts, dtype=torch.long, device=x.device)
    unique_experts, counts = torch.unique(indices_sorted, return_counts=True)
    expert_counts[unique_experts] = counts

    # Process each expert's batch
    output_sorted = torch.zeros_like(x_sorted)
    offset = 0
    for expert_idx in range(n_experts):
        count = expert_counts[expert_idx].item()
        if count == 0:
            continue

        expert_input = x_sorted[offset:offset + count]
        expert_output = experts[expert_idx](expert_input)
        output_sorted[offset:offset + count] = expert_output
        offset += count

    # Weight by scores
    output_sorted = output_sorted * scores_sorted.unsqueeze(-1)

    # Unsort
    output_unsorted = torch.zeros_like(output_sorted)
    output_unsorted[sorted_idx] = output_sorted

    # Reshape and sum over routing slots
    output = output_unsorted.view(B * T, n_routed, d).sum(dim=1)  # (N, d)

    return output.view(B, T, d)


class MoEBlock(nn.Module):
    """
    Mixture of Experts block.

    Combines a shared expert with routed experts via an MLP router.

    Args:
        d_model: model dimension
        n_experts: number of routed experts
        n_routed: top-k experts per token
        n_shared: number of shared experts (typically 1)
        expert_intermediate: FFN intermediate dim per expert
        router_hidden: router MLP hidden dim
        router_bias_update_interval: steps between bias updates
        expert_dtype: weight storage dtype ("fp8" or "bf16")
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        n_routed: int,
        n_shared: int = 1,
        expert_intermediate: int = 192,
        router_hidden: int = 384,
        router_bias_update_interval: int = 1000,
        expert_dtype: str = "fp8",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_routed = n_routed

        # Router
        self.router = MLPRouter(
            d_model=d_model,
            n_experts=n_experts,
            n_routed=n_routed,
            hidden_dim=router_hidden,
            bias_update_interval=router_bias_update_interval,
        )

        # Shared expert(s) — always in full precision
        self.shared_experts = nn.ModuleList([
            ExpertFFN(d_model, expert_intermediate, dtype="bf16")
            for _ in range(n_shared)
        ])

        # Routed experts
        self.routed_experts = nn.ModuleList([
            ExpertFFN(d_model, expert_intermediate, dtype=expert_dtype)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        # Shared expert(s)
        shared_out = sum(expert(x.view(-1, self.d_model)).view(x.shape)
                        for expert in self.shared_experts)

        # Route
        indices, scores = self.router(x)

        # Dispatch and combine routed experts
        routed_out = dispatch_and_combine_fast(x, indices, scores, self.routed_experts)

        return shared_out + routed_out

    def sync_fp8_weights(self):
        """Sync FP8 shadow copies after optimizer step."""
        for expert in self.routed_experts:
            expert.sync_fp8()

    def maybe_update_router_bias(self):
        """Update router bias if it's time."""
        if self.router.should_update_bias():
            self.router.update_bias()

    def stats(self) -> dict:
        router_stats = self.router.stats()
        expert_params = sum(p.numel() for e in self.routed_experts for p in e.parameters())
        shared_params = sum(p.numel() for e in self.shared_experts for p in e.parameters())
        return {
            "router": router_stats,
            "n_routed_experts": self.n_experts,
            "n_shared_experts": len(self.shared_experts),
            "expert_params": expert_params,
            "shared_params": shared_params,
            "total_params": expert_params + shared_params + sum(
                p.numel() for p in self.router.parameters()
            ),
        }
