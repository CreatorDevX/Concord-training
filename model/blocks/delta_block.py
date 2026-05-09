"""
DeltaBlock — DeltaNet + MoE + AttnRes.

Integration pattern per spec Section 3.5:
    x_res = attn_res(layer_idx, block_reprs, partial_residual)
    x = x_res + sublayer1(norm1(x_res))   # DeltaNet
    partial_residual = partial_residual + x
    x_res2 = attn_res(layer_idx, block_reprs, partial_residual)
    x = x_res2 + sublayer2(norm2(x_res2)) # MoE
    partial_residual = partial_residual + x
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from model.components.rms_norm import RMSNorm
from model.components.delta_net import GatedDeltaNet
from model.components.moe import MoEBlock
from model.components.attn_res import BlockAttnRes


class DeltaBlock(nn.Module):
    """
    DeltaNet block: DeltaNet sublayer + MoE sublayer + AttnRes integration.

    Args:
        config: ModelConfig
        layer_idx: global layer index (0..n_layers-1)
        attn_res: shared BlockAttnRes module
        shared_moe: if provided, use this MoE instead of creating one (for expert sharing)
    """

    def __init__(self, config, layer_idx: int, attn_res: BlockAttnRes,
                 shared_moe: Optional[MoEBlock] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_res = attn_res

        # Pre-norms
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        # Sublayer 1: Gated DeltaNet
        self.delta_net = GatedDeltaNet(
            d_model=config.d_model,
            n_v_heads=config.delta_v_heads,
            n_qk_heads=config.delta_qk_heads,
            head_dim=config.delta_head_dim,
            chunk_size=config.delta_chunk_size,
        )

        # Sublayer 2: MoE — shared across group or owned
        if shared_moe is not None:
            self.moe = shared_moe
            self._owns_moe = False
        else:
            self.moe = MoEBlock(
                d_model=config.d_model,
                n_experts=config.n_experts,
                n_routed=config.n_routed_delta,
                n_shared=config.n_shared,
                expert_intermediate=config.expert_intermediate,
                router_hidden=config.router_hidden,
                router_bias_update_interval=config.router_bias_update_interval,
                expert_dtype=config.expert_dtype,
            )
            self._owns_moe = True

    def forward(
        self,
        x: torch.Tensor,
        block_reprs: List[torch.Tensor],
        partial_residual: torch.Tensor,
        delta_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model) — input (unused in AttnRes formulation, kept for interface)
            block_reprs: list of completed block representations
            partial_residual: (B, T, d_model) running sum within block
            delta_state: optional recurrent state for DeltaNet

        Returns:
            x: (B, T, d_model) — output
            partial_residual: (B, T, d_model) — updated partial residual
            delta_state: updated recurrent state
        """
        # AttnRes → DeltaNet sublayer
        x_res = self.attn_res(self.layer_idx, block_reprs, partial_residual)
        delta_out, delta_state = self.delta_net(self.norm1(x_res))
        x = x_res + delta_out
        partial_residual = partial_residual + x

        # AttnRes → MoE sublayer
        x_res2 = self.attn_res(self.layer_idx, block_reprs, partial_residual)
        moe_out = self.moe(self.norm2(x_res2))
        x = x_res2 + moe_out
        partial_residual = partial_residual + x

        return x, partial_residual, delta_state

    def stats(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "owns_moe": self._owns_moe,
            "delta_net": self.delta_net.stats(),
            "moe": self.moe.stats(),
            "norm1": self.norm1.stats(),
            "norm2": self.norm2.stats(),
        }
