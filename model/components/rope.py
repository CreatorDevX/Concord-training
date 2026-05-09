"""
Partial Rotary Position Embedding (RoPE).

Applied only to the first `rope_dim` dimensions of each attention head.
Remaining dims are content-only (unrotated).
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PartialRoPE(nn.Module):
    """
    Partial RoPE: rotates the first `rope_dim` dimensions of each head,
    leaves the remaining `head_dim - rope_dim` dimensions unchanged.
    """

    def __init__(self, head_dim: int, rope_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        assert rope_dim <= head_dim, f"rope_dim ({rope_dim}) must be <= head_dim ({head_dim})"
        assert rope_dim % 2 == 0, f"rope_dim must be even, got {rope_dim}"
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies: (rope_dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache cos/sin for up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, rope_dim // 2)
        cos_cached = freqs.cos()  # (seq_len, rope_dim // 2)
        sin_cached = freqs.sin()  # (seq_len, rope_dim // 2)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs: [x0, x1, x2, x3, ...] → [-x1, x0, -x3, x2, ...]"""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply partial RoPE to q and k.

        Args:
            q: (B, n_heads, T, head_dim)
            k: (B, n_kv_heads, T, head_dim)
            position_ids: (B, T) or None (defaults to 0..T-1)

        Returns:
            q_rotated, k_rotated: same shapes as input
        """
        B, _, T, _ = q.shape

        # Extend cache if needed
        if T > self.cos_cached.shape[0]:
            self._build_cache(T)

        if position_ids is None:
            cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, rope_dim//2)
            sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cached[position_ids].unsqueeze(1)  # (B, 1, T, rope_dim//2)
            sin = self.sin_cached[position_ids].unsqueeze(1)

        # Split into rotated and pass-through parts
        q_rot, q_pass = q[..., :self.rope_dim], q[..., self.rope_dim:]
        k_rot, k_pass = k[..., :self.rope_dim], k[..., self.rope_dim:]

        # Apply rotation
        q_rot = q_rot * cos.repeat(1, 1, 1, 2) + self._rotate_half(q_rot) * sin.repeat(1, 1, 1, 2)
        k_rot = k_rot * cos.repeat(1, 1, 1, 2) + self._rotate_half(k_rot) * sin.repeat(1, 1, 1, 2)

        # Concat rotated + pass-through
        q_out = torch.cat([q_rot, q_pass], dim=-1)
        k_out = torch.cat([k_rot, k_pass], dim=-1)

        return q_out.to(q.dtype), k_out.to(k.dtype)

    def stats(self) -> dict:
        return {
            "rope_dim": self.rope_dim,
            "head_dim": self.head_dim,
            "max_seq_len_cached": self.cos_cached.shape[0],
        }
