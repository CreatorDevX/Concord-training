"""
AgentRoPE — Real-Temporal Multimodal Rotary Position Embedding.

4-axis rotary system:
    x (spatial col), y (spatial row), u (local order), w (wallclock age)

Each axis applies a standard rotary transform on a dedicated subspace
of the head dimension, so axes do not interfere.

Wallclock encoding uses signed log compression:
    w_i = sign(Δt) * log(1 + |Δt| / τ)

Also provides a TemporalDecayBias module for learned attention logit bias
based on wallclock gap.

Backward-compatible PartialRoPE wrapper is provided for DeltaNet blocks
that don't need full multimodal coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class RotaryAxis(nn.Module):
    """Single rotary axis operating on a subspace of the head dimension."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"Rotary axis dim must be even, got {dim}"
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding for one axis.

        Args:
            x:     (B, H, L, D_axis)
            coord: (B, L) — coordinate values for this axis

        Returns:
            (B, H, L, D_axis) — rotated tensor
        """
        # Always compute angles in FP32 for numerical stability
        # coord -> [B, 1, L, D_axis/2]
        angles = coord[:, None, :, None].float() * self.inv_freq[None, None, None, :].float()
        cos = angles.cos()
        sin = angles.sin()

        x_even = x[..., ::2].float()
        x_odd = x[..., 1::2].float()

        out = torch.empty_like(x)
        out[..., ::2] = (x_even * cos - x_odd * sin).to(x.dtype)
        out[..., 1::2] = (x_even * sin + x_odd * cos).to(x.dtype)
        return out


class TemporalMultimodalRoPE(nn.Module):
    """
    4-axis Temporal Multimodal RoPE (AgentRoPE).

    Axes: x (spatial col), y (spatial row), u (local order), w (wallclock).
    Each axis rotates a dedicated subspace of the head dimension.

    Args:
        axis_dims: dict mapping axis name -> subspace dimension (must be even)
        axis_bases: dict mapping axis name -> RoPE base frequency
        wallclock_scale_seconds: τ parameter for wallclock normalization
        use_log_wallclock: if True, use signed log compression for wallclock
        wallclock_warmup_gain: initial scaling for wallclock axis (0.1 = start small)
    """

    def __init__(
        self,
        axis_dims: Dict[str, int],
        axis_bases: Dict[str, float] = None,
        wallclock_scale_seconds: float = 60.0,
        use_log_wallclock: bool = True,
        wallclock_warmup_gain: float = 0.1,
    ):
        super().__init__()
        self.axis_names = list(axis_dims.keys())
        self.axis_dims = axis_dims
        self.wallclock_scale = wallclock_scale_seconds
        self.use_log_wallclock = use_log_wallclock

        # Learnable gain for wallclock axis — start small per training recipe
        if "w" in axis_dims:
            self.wallclock_gain = nn.Parameter(
                torch.tensor(wallclock_warmup_gain)
            )
        else:
            self.wallclock_gain = None

        axis_bases = axis_bases or {k: 10000.0 for k in axis_dims}
        self.axes = nn.ModuleDict({
            k: RotaryAxis(axis_dims[k], base=axis_bases[k]) for k in axis_dims
        })

        total = sum(axis_dims.values())
        assert total > 0, "Total axis dims must be > 0"
        for k, d in axis_dims.items():
            assert d % 2 == 0, f"Axis {k} dim must be even, got {d}"

    @property
    def total_dim(self) -> int:
        return sum(self.axis_dims.values())

    def encode_wallclock(
        self,
        t_ms: torch.Tensor,
        ref_ms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode wallclock timestamps as normalized age coordinates.

        Args:
            t_ms:   (B, L) event timestamps in milliseconds
            ref_ms: (B, 1) reference timestamp in milliseconds

        Returns:
            (B, L) normalized wallclock coordinate
        """
        dt = (t_ms.float() - ref_ms.float()) / 1000.0  # seconds
        if self.use_log_wallclock:
            dt = torch.sign(dt) * torch.log1p(dt.abs() / self.wallclock_scale)
        else:
            dt = dt / self.wallclock_scale
        return dt

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        coords: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multimodal rotary embeddings.

        Args:
            q, k: (B, H, L, D) where D = sum of all axis dims
            coords: dict with keys matching axis_names, each (B, L)

        Returns:
            q_rotated, k_rotated: same shapes as input
        """
        q_parts = torch.split(q, [self.axis_dims[n] for n in self.axis_names], dim=-1)
        k_parts = torch.split(k, [self.axis_dims[n] for n in self.axis_names], dim=-1)

        q_out, k_out = [], []
        for name, q_part, k_part in zip(self.axis_names, q_parts, k_parts):
            coord = coords[name].to(dtype=q.dtype)
            # Apply wallclock gain scaling
            if name == "w" and self.wallclock_gain is not None:
                coord = coord * self.wallclock_gain
            q_out.append(self.axes[name](q_part, coord))
            k_out.append(self.axes[name](k_part, coord))

        return torch.cat(q_out, dim=-1), torch.cat(k_out, dim=-1)


class TemporalDecayBias(nn.Module):
    """
    Learned temporal decay bias for attention logits.

    Adds b(Δw_ij) to attention scores, where b is a bucketed embedding
    over wallclock gaps. This gives the model an explicit staleness knob.

    Args:
        n_heads: number of attention heads
        n_buckets: number of log-spaced time buckets
        max_log_gap: maximum log gap value for bucketing
    """

    def __init__(self, n_heads: int, n_buckets: int = 32, max_log_gap: float = 10.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.max_log_gap = max_log_gap
        # Per-head bias embedding
        self.bias_embed = nn.Embedding(n_buckets, n_heads)
        nn.init.zeros_(self.bias_embed.weight)  # start with no bias

    def _bucketize(self, delta_w: torch.Tensor) -> torch.Tensor:
        """Map wallclock gaps to bucket indices."""
        # signed log gap
        log_gap = torch.sign(delta_w) * torch.log1p(delta_w.abs())
        # normalize to [0, n_buckets-1]
        normalized = (log_gap + self.max_log_gap) / (2 * self.max_log_gap)
        bucket_ids = (normalized * (self.n_buckets - 1)).clamp(0, self.n_buckets - 1).long()
        return bucket_ids

    def forward(self, w_q: torch.Tensor, w_k: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal bias matrix.

        Args:
            w_q: (B, T_q) wallclock coords for queries
            w_k: (B, T_k) wallclock coords for keys

        Returns:
            (B, n_heads, T_q, T_k) bias to add to attention logits
        """
        # Pairwise gaps: (B, T_q, T_k)
        delta_w = w_q.unsqueeze(-1) - w_k.unsqueeze(-2)
        bucket_ids = self._bucketize(delta_w)  # (B, T_q, T_k)

        # Look up bias: (B, T_q, T_k, n_heads)
        bias = self.bias_embed(bucket_ids)

        # Transpose to (B, n_heads, T_q, T_k)
        return bias.permute(0, 3, 1, 2)


class PartialRoPE(nn.Module):
    """
    Backward-compatible wrapper around TemporalMultimodalRoPE.

    For components that only need sequential position encoding (like DeltaNet
    blocks when no multimodal coordinates are available), this provides the
    same interface as the old PartialRoPE but uses the 'u' axis internally.

    When full coords are provided, delegates to TemporalMultimodalRoPE.

    Args:
        head_dim: total head dimension
        rope_dim: dimension to apply RoPE to (rest is content-only)
        max_seq_len: maximum sequence length for cache
        base: RoPE base frequency
    """

    def __init__(self, head_dim: int, rope_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        assert rope_dim <= head_dim, f"rope_dim ({rope_dim}) must be <= head_dim ({head_dim})"
        assert rope_dim % 2 == 0, f"rope_dim must be even, got {rope_dim}"
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Single-axis rotary for sequential positions
        self.rotary = RotaryAxis(rope_dim, base=base)

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

        if position_ids is None:
            position_ids = torch.arange(T, device=q.device, dtype=q.dtype).unsqueeze(0).expand(B, -1)

        # Split into rotated and pass-through parts
        q_rot, q_pass = q[..., :self.rope_dim], q[..., self.rope_dim:]
        k_rot, k_pass = k[..., :self.rope_dim], k[..., self.rope_dim:]

        # Apply rotation
        q_rot = self.rotary(q_rot, position_ids)
        k_rot = self.rotary(k_rot, position_ids)

        # Concat rotated + pass-through
        q_out = torch.cat([q_rot, q_pass], dim=-1)
        k_out = torch.cat([k_rot, k_pass], dim=-1)

        return q_out, k_out

    def stats(self) -> dict:
        return {
            "rope_dim": self.rope_dim,
            "head_dim": self.head_dim,
        }
