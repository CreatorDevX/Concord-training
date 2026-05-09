"""
MoE Block — Router + Shared Expert + Routed Experts + dispatch_and_combine.

Architecture:
    shared_out  = shared_expert(x)
    indices, scores, aux_loss = router(x)
    routed_out  = dispatch_and_combine(x, indices, scores, experts)
    output      = shared_out + routed_out

dispatch_and_combine:
    Sort tokens by assigned expert index → batch matmul per expert →
    unsort and weight by router scores. No token dropping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Callable

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

    output = torch.zeros_like(x_flat)

    for slot in range(n_routed):
        slot_indices = indices_flat[:, slot]
        slot_scores = scores_flat[:, slot]

        for expert_idx in range(n_experts):
            mask = (slot_indices == expert_idx)
            if not mask.any():
                continue

            expert_input = x_flat[mask]
            expert_output = experts[expert_idx](expert_input)

            # Sanitize expert output: SwiGLU in fp16 can overflow on edge cases
            if expert_output.dtype == torch.float16 or expert_output.dtype == torch.bfloat16:
                expert_output = torch.where(
                    torch.isnan(expert_output) | torch.isinf(expert_output),
                    torch.zeros_like(expert_output),
                    expert_output
                )

            expert_scores = slot_scores[mask].unsqueeze(-1)
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

    output_sorted = torch.zeros_like(x_sorted)
    offset = 0
    for expert_idx in range(n_experts):
        count = expert_counts[expert_idx].item()
        if count == 0:
            if x.requires_grad:
                dummy_input = x_sorted[:1] * 0.0
                dummy_out = experts[expert_idx](dummy_input) * 0.0
                output_sorted[:1] = output_sorted[:1] + dummy_out
            continue

        expert_input = x_sorted[offset:offset + count]
        expert_output = experts[expert_idx](expert_input)
        if expert_output.dtype == torch.float16 or expert_output.dtype == torch.bfloat16:
            expert_output = torch.where(
                torch.isnan(expert_output) | torch.isinf(expert_output),
                torch.zeros_like(expert_output),
                expert_output
            )
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
    Now propagates auxiliary load balancing loss.
    Supports activation recomputation for memory-efficient training.

    Args:
        d_model: model dimension
        n_experts: number of routed experts
        n_routed: top-k experts per token
        n_shared: number of shared experts (typically 1)
        expert_intermediate: FFN intermediate dim per expert
        router_hidden: router MLP hidden dim
        router_bias_update_interval: steps between bias updates
        expert_dtype: weight storage dtype ("fp16" or "bf16")
        use_activation_recomputation: if True, checkpoint the routed computation
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
        expert_dtype: str = "fp16",
        use_activation_recomputation: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_routed = n_routed
        self.use_activation_recomputation = use_activation_recomputation

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
            ExpertFFN(d_model, expert_intermediate)
            for _ in range(n_shared)
        ])

        # Routed experts
        self.routed_experts = nn.ModuleList([
            ExpertFFN(d_model, expert_intermediate)
            for _ in range(n_experts)
        ])

    def _forward_routed(self, x: torch.Tensor, indices: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Dispatch and combine routed experts (wrapped for checkpointing).
        
        Uses the memory-efficient dispatch (no B*T*n_routed expansion) to
        avoid OOM on long sequences with large batch sizes.
        """
        return dispatch_and_combine(x, indices, scores, self.routed_experts)

    def _forward_with_checkpoint(
        self, x: torch.Tensor, indices: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """Routed forward with activation recomputation to save memory."""
        return torch.utils.checkpoint.checkpoint(
            self._forward_routed, x, indices, scores, use_reentrant=False
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            output: (B, T, d_model)
            aux_loss: scalar — router load balancing loss
        """
        # Shared expert(s) — lightweight, kept outside checkpoint
        shared_out = sum(expert(x.view(-1, self.d_model)).view(x.shape)
                        for expert in self.shared_experts)

        # Route
        indices, scores, aux_loss = self.router(x)

        # Dispatch and combine routed experts with optional activation recomputation
        if self.training and self.use_activation_recomputation:
            routed_out = self._forward_with_checkpoint(x, indices, scores)
        else:
            routed_out = self._forward_routed(x, indices, scores)

        # Sanitize: if expert ffn or router produced NaN from fp16 overflow,
        # zero them to prevent cascading NaN through the residual stream.
        if self.training and (torch.isnan(routed_out).any() or torch.isinf(routed_out).any()):
            routed_out = torch.zeros_like(routed_out)
        return shared_out + routed_out, aux_loss

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
