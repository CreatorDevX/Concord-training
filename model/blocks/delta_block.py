"""
DeltaBlock — DeltaNet + MoE + AttnRes.

Integration pattern:
    x_res = attn_res(layer_idx, block_reprs, partial_residual)
    x = x + x_res + sublayer1(norm1(x_res))   # DeltaNet
    partial_residual = partial_residual + sublayer1_out
    x = x + x_res + sublayer2(norm2(x_res))   # MoE
    partial_residual = partial_residual + sublayer2_out

Partial accumulates only sublayer outputs (compressed block representation
per the AttnRes paper). Standard residuals carry the full hidden state.
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
        # Store attn_res as a plain reference — NOT as a submodule.
        # attn_res is shared across all blocks; if registered as a child,
        # block.to(device) would move it, breaking pipeline-parallel splits.
        object.__setattr__(self, '_attn_res', attn_res)

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
            use_checkpoint=not config.grad_checkpoint,
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
                use_activation_recomputation=not config.grad_checkpoint,
            )
            self._owns_moe = True

    def forward(
        self,
        x: torch.Tensor,
        block_reprs: List[torch.Tensor],
        partial_residual: torch.Tensor,
        delta_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            aux_loss: MoE auxiliary load balancing loss
        """
        # Single AttnRes call — shared by both sublayers.
        x_res = self._attn_res(self.layer_idx, block_reprs, partial_residual)

        # AttnRes → DeltaNet sublayer
        delta_out, delta_state = self.delta_net(self.norm1(x_res))
        x = x + x_res + delta_out
        # Partial accumulates only sublayer outputs (compressed block representation)
        partial_residual = partial_residual + delta_out

        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            partial_residual = torch.where(
                torch.isnan(partial_residual) | torch.isinf(partial_residual),
                torch.zeros_like(partial_residual),
                partial_residual
            )

        # AttnRes → MoE sublayer (reuses same x_res)
        moe_out, aux_loss = self.moe(self.norm2(x_res))
        x = x + x_res + moe_out
        partial_residual = partial_residual + moe_out

        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
            partial_residual = torch.where(
                torch.isnan(partial_residual) | torch.isinf(partial_residual),
                torch.zeros_like(partial_residual),
                partial_residual
            )

        return x, partial_residual, delta_state, aux_loss

    def stats(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "owns_moe": self._owns_moe,
            "delta_net": self.delta_net.stats(),
            "moe": self.moe.stats(),
            "norm1": self.norm1.stats(),
            "norm2": self.norm2.stats(),
        }
