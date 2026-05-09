"""
Block Attention Residuals (AttnRes).

Replaces standard residual connections. Each layer selectively aggregates
earlier layer representations via softmax attention over depth, rather
than blindly summing the residual stream.

Block AttnRes: partition 24 layers into 6 blocks matching the 6 layer groups.
Within each block, standard residuals accumulate. At each layer, depth-wise
attention runs over the completed block representations + partial sum of
the current block.

Pseudo-queries initialized to zero → uniform attention at start → reduces
to equal-weight averaging at step 0. Critical for early training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict

from model.components.rms_norm import RMSNorm


class BlockAttnRes(nn.Module):
    """
    Block Attention Residuals.

    Args:
        d_model: model dimension
        n_layers: total number of layers (24)
        n_blocks: number of blocks / groups (6)
    """

    def __init__(self, d_model: int, n_layers: int, n_blocks: int):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_blocks = n_blocks

        # One pseudo-query per layer. Zero-init is critical for stability.
        self.pseudo_queries = nn.Parameter(torch.zeros(n_layers, d_model))

        # Shared key projection across all layers
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

        # Normalization for sources before attention
        self.norm = RMSNorm(d_model)

        # Scale factor
        self.scale = 1.0 / math.sqrt(d_model)

        # Stats tracking
        self._last_attn_weights = None

    def forward(
        self,
        layer_idx: int,
        block_reprs: List[torch.Tensor],
        partial_residual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention-weighted combination of source representations.

        Args:
            layer_idx: current layer index (0..n_layers-1)
            block_reprs: list of completed block representations [(B,T,d), ...]
            partial_residual: running sum within current block (B,T,d)

        Returns:
            (B,T,d) — attention-weighted combination of sources
        """
        # Collect sources: completed blocks + current partial
        sources = block_reprs + [partial_residual]
        n_sources = len(sources)

        # Stack sources: (B, T, n_sources, d)
        stacked = torch.stack(sources, dim=2)

        # Normalize
        stacked_norm = self.norm(stacked)

        # Project to keys: (B, T, n_sources, d)
        keys = self.k_proj(stacked_norm)

        # Get pseudo-query for this layer: (1, 1, 1, d)
        q = self.pseudo_queries[layer_idx].view(1, 1, 1, -1)

        # Attention scores: (B, T, n_sources)
        attn_logits = (q * keys).sum(dim=-1) * self.scale

        # Softmax over sources
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, T, n_sources)

        # Track for stats
        self._last_attn_weights = attn_weights.detach()

        # Weighted combination: (B, T, d)
        output = (attn_weights.unsqueeze(-1) * stacked).sum(dim=2)

        return output

    def stats(self) -> dict:
        result = {
            "pseudo_query_norm": self.pseudo_queries.data.float().norm().item(),
            "pseudo_query_mean": self.pseudo_queries.data.float().mean().item(),
            "k_proj_norm": next(self.k_proj.parameters()).data.float().norm().item(),
        }
        if self._last_attn_weights is not None:
            # Check uniformity: entropy of attention weights
            w = self._last_attn_weights.float()
            entropy = -(w * (w + 1e-10).log()).sum(dim=-1).mean().item()
            max_entropy = math.log(w.shape[-1])
            result["attn_entropy"] = entropy
            result["attn_max_entropy"] = max_entropy
            result["attn_uniformity"] = entropy / max_entropy if max_entropy > 0 else 1.0
            result["attn_weight_mean"] = w.mean(dim=(0, 1)).tolist()
        return result
