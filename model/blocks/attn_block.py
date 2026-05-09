"""
AttnBlock — Attention (CSA or HCA) + MoE + AttnRes.

Same integration pattern as DeltaBlock but using Gated Attention instead of DeltaNet.
Even-indexed attention layers → CSA, odd → HCA.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

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
        self.attn_res = attn_res

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
            rope_dim=config.rope_dim,
            max_seq_len=config.max_seq_len,
            variant=variant,
            csa_top_k=config.csa_top_k,
            csa_window=config.csa_window,
            csa_compress=config.csa_compress,
            hca_compress=config.hca_compress,
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
            )
            self._owns_moe = True

    def forward(
        self,
        x: torch.Tensor,
        block_reprs: List[torch.Tensor],
        partial_residual: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model) — input
            block_reprs: list of completed block representations
            partial_residual: (B, T, d_model) running sum within block
            position_ids: (B, T) optional position IDs for RoPE

        Returns:
            x: (B, T, d_model) — output
            partial_residual: (B, T, d_model) — updated partial residual
        """
        # AttnRes → Attention sublayer
        x_res = self.attn_res(self.layer_idx, block_reprs, partial_residual)
        attn_out = self.attention(self.norm1(x_res), position_ids)
        x = x_res + attn_out
        partial_residual = partial_residual + x

        # AttnRes → MoE sublayer
        x_res2 = self.attn_res(self.layer_idx, block_reprs, partial_residual)
        moe_out = self.moe(self.norm2(x_res2))
        x = x_res2 + moe_out
        partial_residual = partial_residual + x

        return x, partial_residual

    def stats(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "attn_idx": self.attn_idx,
            "owns_moe": self._owns_moe,
            "variant": self.attention.variant,
            "attention": self.attention.stats(),
            "moe": self.moe.stats(),
        }
