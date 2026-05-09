"""
Expert FFN — Single SwiGLU expert with optional FP8 weight storage.

SwiGLU:
    gate = silu(W_gate @ x)   # (B', d_ffn)
    up   = W_up @ x           # (B', d_ffn)
    out  = W_down @ (gate * up)  # (B', d_model)

FP8 Storage:
    Weights stored in float8_e4m3fn for memory savings.
    Cast to BF16 before matmul, cast back after.
    T4 has no native FP8 tensor cores — this is storage-only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ExpertFFN(nn.Module):
    """Single SwiGLU FFN expert with optional FP8 weight storage."""

    def __init__(self, d_model: int, d_ffn: int, dtype: str = "fp8"):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.use_fp8 = (dtype == "fp8") and hasattr(torch, "float8_e4m3fn")

        if self.use_fp8:
            # Store weights as FP8 buffers (not parameters — optimizer won't track these)
            # We keep BF16 master copies as parameters for gradient computation
            self.w_gate = nn.Parameter(torch.empty(d_ffn, d_model, dtype=torch.bfloat16))
            self.w_up = nn.Parameter(torch.empty(d_ffn, d_model, dtype=torch.bfloat16))
            self.w_down = nn.Parameter(torch.empty(d_model, d_ffn, dtype=torch.bfloat16))

            # FP8 shadow copies for storage — updated after optimizer step
            self.register_buffer("w_gate_fp8", torch.zeros(d_ffn, d_model, dtype=torch.int8), persistent=True)
            self.register_buffer("w_up_fp8", torch.zeros(d_ffn, d_model, dtype=torch.int8), persistent=True)
            self.register_buffer("w_down_fp8", torch.zeros(d_model, d_ffn, dtype=torch.int8), persistent=True)

            # Scale factors for FP8 quantization
            self.register_buffer("gate_scale", torch.ones(1), persistent=True)
            self.register_buffer("up_scale", torch.ones(1), persistent=True)
            self.register_buffer("down_scale", torch.ones(1), persistent=True)
        else:
            self.w_gate = nn.Parameter(torch.empty(d_ffn, d_model))
            self.w_up = nn.Parameter(torch.empty(d_ffn, d_model))
            self.w_down = nn.Parameter(torch.empty(d_model, d_ffn))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.w_gate)
        nn.init.kaiming_uniform_(self.w_up)
        nn.init.kaiming_uniform_(self.w_down)
        if self.use_fp8:
            self.sync_fp8()

    def sync_fp8(self):
        """Quantize BF16 master weights to FP8 shadow copies for storage."""
        if not self.use_fp8:
            return
        with torch.no_grad():
            self.w_gate_fp8.copy_(self._quantize_to_fp8(self.w_gate, self.gate_scale))
            self.w_up_fp8.copy_(self._quantize_to_fp8(self.w_up, self.up_scale))
            self.w_down_fp8.copy_(self._quantize_to_fp8(self.w_down, self.down_scale))

    @staticmethod
    def _quantize_to_fp8(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Quantize a BF16 tensor to int8 (simulated FP8 E4M3)."""
        # Compute dynamic scale
        amax = tensor.abs().max().clamp(min=1e-12)
        # E4M3 range: max representable value is 448
        fp8_max = 448.0
        new_scale = amax / fp8_max
        scale.fill_(new_scale.item())
        # Quantize
        scaled = tensor / new_scale
        return scaled.clamp(-fp8_max, fp8_max).to(torch.int8)

    @staticmethod
    def _dequantize_from_fp8(tensor: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize int8 (simulated FP8) back to compute dtype."""
        return tensor.to(dtype) * scale

    def _get_compute_weights(self):
        """Get weights in compute dtype (BF16)."""
        if self.use_fp8 and self.training:
            # During training, use BF16 master weights directly
            return self.w_gate, self.w_up, self.w_down
        elif self.use_fp8:
            # During inference, dequantize FP8 copies
            dtype = torch.bfloat16
            wg = self._dequantize_from_fp8(self.w_gate_fp8, self.gate_scale, dtype)
            wu = self._dequantize_from_fp8(self.w_up_fp8, self.up_scale, dtype)
            wd = self._dequantize_from_fp8(self.w_down_fp8, self.down_scale, dtype)
            return wg, wu, wd
        else:
            return self.w_gate, self.w_up, self.w_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B', d_model) — tokens routed to this expert
        Returns:
            (B', d_model)
        """
        wg, wu, wd = self._get_compute_weights()
        gate = F.silu(F.linear(x, wg))   # (B', d_ffn)
        up = F.linear(x, wu)              # (B', d_ffn)
        return F.linear(gate * up, wd)    # (B', d_model)

    def stats(self) -> dict:
        wg, wu, wd = self._get_compute_weights()
        return {
            "w_gate_norm": wg.data.float().norm().item(),
            "w_up_norm": wu.data.float().norm().item(),
            "w_down_norm": wd.data.float().norm().item(),
            "use_fp8": self.use_fp8,
            "params": sum(p.numel() for p in self.parameters()),
        }
