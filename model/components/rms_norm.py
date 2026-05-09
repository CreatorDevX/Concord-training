"""
RMSNorm — Root Mean Square Layer Normalization.
No bias. Learnable scale per dimension.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return (x * self.scale).to(dtype)

    def stats(self) -> dict:
        return {
            "scale_mean": self.scale.data.mean().item(),
            "scale_std": self.scale.data.std().item(),
            "scale_min": self.scale.data.min().item(),
            "scale_max": self.scale.data.max().item(),
        }
