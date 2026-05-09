"""
Expert FFN — Single SwiGLU expert.

SwiGLU:
    gate = silu(W_gate @ x)   # (B', d_ffn)
    up   = W_up @ x           # (B', d_ffn)
    out  = W_down @ (gate * up)  # (B', d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ExpertFFN(nn.Module):
    """Single SwiGLU FFN expert."""

    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn

        self.w_gate = nn.Parameter(torch.empty(d_ffn, d_model))
        self.w_up = nn.Parameter(torch.empty(d_ffn, d_model))
        self.w_down = nn.Parameter(torch.empty(d_model, d_ffn))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.w_gate)
        nn.init.kaiming_uniform_(self.w_up)
        nn.init.kaiming_uniform_(self.w_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B', d_model) — tokens routed to this expert
        Returns:
            (B', d_model)
        """
        # Cast input to match weight dtype (supports mixed precision:
        # fp32 model + fp16 expert weights)
        x = x.to(self.w_gate.dtype)
        gate = F.silu(F.linear(x, self.w_gate))   # (B', d_ffn)
        up = F.linear(x, self.w_up)                # (B', d_ffn)
        return F.linear(gate * up, self.w_down)    # (B', d_model)

    def stats(self) -> dict:
        return {
            "w_gate_norm": self.w_gate.data.float().norm().item(),
            "w_up_norm": self.w_up.data.float().norm().item(),
            "w_down_norm": self.w_down.data.float().norm().item(),
            "params": sum(p.numel() for p in self.parameters()),
        }
