"""
Block Attention Residuals (AttnRes).

Replaces standard residual connections. Each layer selectively aggregates
completed block representations via softmax attention over depth.

Per the paper (arXiv:2603.15031v1):
- Layers are partitioned into N blocks.
- Within each block, standard residuals accumulate.
- b0 = h1 (the token embedding) is always included as the first source.
- Each completed block bn is produced by summing all layer outputs in that block.
- Cross-block attention runs over only the completed block representations.
- The intra-block partial sum flows via standard residual, NOT as an attention source.

Pseudo-queries initialized to zero → uniform attention at start → reduces
to equal-weight averaging at step 0. Critical for early training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

from model.components.rms_norm import RMSNorm


class BlockAttnRes(nn.Module):
    """
    Block Attention Residuals.

    Attends ONLY over completed block representations (no partial residual).
    Each block produces exactly one compressed tensor. The initial embedding
    (b0 = h1) is always the first source.

    Args:
        d_model: model dimension
        n_layers: total number of layers
        n_blocks: number of blocks / groups
    """

    def __init__(self, d_model: int, n_layers: int, n_blocks: int):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_blocks = n_blocks

        self.pseudo_queries = nn.Parameter(torch.zeros(n_layers, d_model))
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.scale = 1.0 / math.sqrt(d_model)

        self._last_attn_weights = None

    def forward(
        self,
        layer_idx: int,
        block_reprs: List[torch.Tensor],
        partial_residual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention-weighted combination over completed block representations.

        Attends ONLY over block_reprs (completed blocks + b0 = h1).
        partial_residual is accepted for API compatibility but is NOT included
        as an attention source — intra-block information flows via standard residual.

        Args:
            layer_idx: current layer index (0..n_layers-1)
            block_reprs: list of completed block representations [(B,T,d), ...]
                block_reprs[0] must be the token embedding (b0 = h1).
            partial_residual: ignored (kept for API compatibility)

        Returns:
            (B,T,d) — attention-weighted combination of completed blocks
        """
        sources = block_reprs
        n_sources = len(sources)

        if n_sources == 0:
            return torch.zeros(
                partial_residual.shape, device=partial_residual.device, dtype=partial_residual.dtype
            )

        B, T = partial_residual.shape[0], partial_residual.shape[1]

        # Validate invariants
        for s in sources:
            if s.shape != (B, T, self.d_model):
                raise ValueError(
                    f"Expected source shape ({B}, {T}, {self.d_model}), got {s.shape}"
                )
        if n_sources > self.n_blocks:
            raise ValueError(
                f"Number of sources ({n_sources}) exceeds n_blocks ({self.n_blocks}). "
                "Each block produces at most one representation; "
                "block_reprs must be compressed (1 per block, not per layer)."
            )

        # Incremental logit computation (avoids stacking full (B,T,n,d) tensor)
        q = self.pseudo_queries[layer_idx].view(1, 1, -1)

        attn_logits = []
        for s in sources:
            s_norm = self.norm(s)
            k = self.k_proj(s_norm)
            logit = (q * k).sum(dim=-1) * self.scale
            attn_logits.append(logit)

        attn_logits = torch.stack(attn_logits, dim=-1)
        attn_weights = F.softmax(attn_logits, dim=-1)

        self._last_attn_weights = attn_weights.detach()

        # Incremental weighted combination (avoids 4D stack)
        output = torch.zeros_like(sources[0])
        for i, w in enumerate(attn_weights.unbind(dim=-1)):
            output = output + w.unsqueeze(-1) * sources[i]

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
