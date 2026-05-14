"""
AttnBlock — Attention (CSA or HCA) + MoE + AttnRes.

Same integration pattern as DeltaBlock but using Gated Attention instead of DeltaNet.
Even-indexed attention layers → CSA, odd → HCA.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict

from model.components.rms_norm import RMSNorm
from model.components.attention import GatedAttention
from model.components.moe import MoEBlock
from model.components.attn_res import BlockAttnRes


class AttnBlock(nn.Module):
    """
    Attention block: Gated Attention (CSA/HCA) + MoE + AttnRes.

    Args:
        config: ModelConfig
        layer_idx: global layer index (0..n_layers-1)
        attn_idx: attention layer index (0..n_groups-1), used to determine CSA vs HCA
        attn_res: shared BlockAttnRes module
        shared_moe: if provided, use this MoE instead of creating one (for expert sharing)
    """

    def __init__(self, config, layer_idx: int, attn_idx: int, attn_res: BlockAttnRes,
                 shared_moe: Optional[MoEBlock] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_idx = attn_idx
        # Store attn_res as a plain reference — NOT as a submodule.
        # attn_res is shared across all blocks; if registered as a child,
        # block.to(device) would move it, breaking pipeline-parallel splits.
        object.__setattr__(self, '_attn_res', attn_res)

        # Determine variant: even-indexed → CSA, odd → HCA
        variant = "csa" if attn_idx % 2 == 0 else "hca"

        # Pre-norms
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        # Sublayer 1: Gated Attention
        self.attention = GatedAttention(
            d_model=config.d_model,
            n_q_heads=config.attn_q_heads,
            n_kv_heads=config.attn_kv_heads,
            head_dim=config.attn_head_dim,
            rope_axis_dims=config.rope_axis_dims,
            variant=variant,
            csa_top_k=config.csa_top_k,
            csa_window=config.csa_window,
            csa_compress=config.csa_compress,
            hca_compress=config.hca_compress,
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
                n_routed=config.n_routed_attn,
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
        position_ids: Optional[torch.Tensor] = None,
        coords: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model) — input
            block_reprs: list of completed block representations
            partial_residual: (B, T, d_model) running sum within block
            position_ids: (B, T) optional position IDs for RoPE
            coords: optional dict of multimodal coordinates

        Returns:
            x: (B, T, d_model) — output
            partial_residual: (B, T, d_model) — updated partial residual
            aux_loss: MoE auxiliary load balancing loss
        """
        # Single AttnRes call — shared by both sublayers.
        x_res = self._attn_res(self.layer_idx, block_reprs, partial_residual)

        # AttnRes → Attention sublayer
        attn_out = self.attention(self.norm1(x_res), position_ids, coords=coords)
        x = x + x_res + attn_out
        partial_residual = partial_residual + x

        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            partial_residual = torch.where(
                torch.isnan(partial_residual) | torch.isinf(partial_residual),
                torch.zeros_like(partial_residual),
                partial_residual
            )

        # AttnRes → MoE sublayer (reuses same x_res)
        moe_out, aux_loss = self.moe(self.norm2(x_res))
        x = x + x_res + moe_out
        partial_residual = partial_residual + x

        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
            partial_residual = torch.where(
                torch.isnan(partial_residual) | torch.isinf(partial_residual),
                torch.zeros_like(partial_residual),
                partial_residual
            )

        return x, partial_residual, aux_loss

    def stats(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "attn_idx": self.attn_idx,
            "owns_moe": self._owns_moe,
            "variant": self.attention.variant,
            "attention": self.attention.stats(),
            "moe": self.moe.stats(),
        }
