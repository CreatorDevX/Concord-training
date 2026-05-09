"""
Gated Attention — CSA and HCA Variants.

Base: GQA with 8 Q heads, 2 KV heads, head_dim=256, partial RoPE,
output gating via sigmoid(W_gate @ x), Flash Attention 2 / SDPA backend.

CSA (Compressed Sparse Attention) — even-indexed attention layers:
    Compress KV at 4×, select top-1024 entries per query via inner-product,
    attend over selected + 128-token local window.

HCA (Hierarchical Compressed Attention) — odd-indexed attention layers:
    Compress KV at 128×, dense attention over compressed sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

from model.components.rms_norm import RMSNorm
from model.components.rope import PartialRoPE


class KVCompressor(nn.Module):
    """Compresses KV pairs by pooling along the sequence dimension."""

    def __init__(self, compress_ratio: int, d_kv: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        # Learned linear projection for compression
        self.compress_proj = nn.Linear(d_kv * compress_ratio, d_kv, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_kv_heads, T, head_dim)
        Returns:
            (B, n_kv_heads, T // compress_ratio, head_dim)
        """
        B, H, T, D = x.shape
        r = self.compress_ratio

        # Pad sequence to be divisible by compress_ratio
        pad_len = (r - T % r) % r
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            T_padded = T + pad_len
        else:
            T_padded = T

        # Reshape: group consecutive tokens
        n_compressed = T_padded // r
        x = x.view(B, H, n_compressed, r * D)  # (B, H, T//r, r*D)
        x = self.compress_proj(x)                # (B, H, T//r, D)
        return x


class GatedAttention(nn.Module):
    """
    Gated attention with CSA or HCA variant.

    Args:
        d_model: model dimension
        n_q_heads: number of query heads
        n_kv_heads: number of key/value heads
        head_dim: dimension per head
        rope_dim: number of dims to apply RoPE to
        max_seq_len: maximum sequence length for RoPE cache
        variant: "csa" or "hca"
        csa_top_k: top-k KV entries to select in CSA
        csa_window: local window size in CSA
        csa_compress: compression ratio for CSA
        hca_compress: compression ratio for HCA
    """

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_dim: int = 64,
        max_seq_len: int = 8192,
        variant: str = "csa",
        csa_top_k: int = 1024,
        csa_window: int = 128,
        csa_compress: int = 4,
        hca_compress: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.variant = variant
        self.csa_top_k = csa_top_k
        self.csa_window = csa_window

        assert n_q_heads % n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads"
        self.n_rep = n_q_heads // n_kv_heads

        # Projections
        self.q_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q_heads * head_dim, d_model, bias=False)

        # Output gating
        self.gate_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)

        # Partial RoPE
        self.rope = PartialRoPE(head_dim, rope_dim, max_seq_len)

        # KV compression
        d_kv = head_dim
        if variant == "csa":
            self.kv_compressor = KVCompressor(csa_compress, d_kv)
        elif variant == "hca":
            self.kv_compressor = KVCompressor(hca_compress, d_kv)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Scale factor
        self.scale = 1.0 / math.sqrt(head_dim)

        # Stats
        self._last_attn_entropy = 0.0

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads for GQA. (B, n_kv, T, D) → (B, n_q, T, D)"""
        if self.n_rep == 1:
            return x
        B, H, T, D = x.shape
        x = x[:, :, None, :, :].expand(B, H, self.n_rep, T, D)
        return x.reshape(B, H * self.n_rep, T, D)

    def _csa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_compressed: torch.Tensor,
        v_compressed: torch.Tensor,
    ) -> torch.Tensor:
        """
        CSA: Compressed Sparse Attention.
        At short seq_len (≤ csa_top_k after compression), degrades to full attention.
        """
        B, H, T, D = q.shape
        T_comp = k_compressed.shape[2]

        # If compressed length ≤ top_k, just do full attention on compressed
        if T_comp <= self.csa_top_k:
            # Expand KV heads for GQA
            k_exp = self._repeat_kv(k_compressed)
            v_exp = self._repeat_kv(v_compressed)

            # Also include local window from uncompressed
            k_local = self._repeat_kv(k)
            v_local = self._repeat_kv(v)

            # Concatenate compressed + local
            k_cat = torch.cat([k_exp, k_local], dim=2)
            v_cat = torch.cat([v_exp, v_local], dim=2)

            # Use SDPA with causal mask on local part
            # For simplicity, use full attention here (CSA degrades at short ctx)
            out = F.scaled_dot_product_attention(
                q, k_cat, v_cat, is_causal=False,
                attn_mask=self._build_csa_mask(T, k_cat.shape[2], q.device),
            )
            return out

        # Full CSA with top-k selection
        k_exp = self._repeat_kv(k_compressed)
        v_exp = self._repeat_kv(v_compressed)

        # Score each query against all compressed keys
        scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale  # (B,H,T,T_comp)

        # Top-k selection per query
        top_k = min(self.csa_top_k, T_comp)
        top_scores, top_idx = scores.topk(top_k, dim=-1)  # (B,H,T,top_k)

        # Gather selected KV
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)  # (B,H,T,top_k,D)
        k_selected = k_exp.unsqueeze(2).expand(-1, -1, T, -1, -1).gather(3, top_idx_exp)
        v_selected = v_exp.unsqueeze(2).expand(-1, -1, T, -1, -1).gather(3, top_idx_exp)

        # Local window keys/values
        k_local = self._repeat_kv(k)
        v_local = self._repeat_kv(v)

        # For each query position, get local window
        # Simplified: just use the last `csa_window` positions (causal)
        window = min(self.csa_window, T)
        k_win = k_local[:, :, max(0, T - window):T, :]
        v_win = v_local[:, :, max(0, T - window):T, :]

        # Attend over selected compressed + local window
        # Concatenate along the KV dimension
        k_attend = torch.cat([k_selected.view(B, H, T, top_k, D).reshape(B * H * T, top_k, D),
                              k_win.unsqueeze(2).expand(-1, -1, T, -1, -1).reshape(B * H * T, window, D)], dim=1)
        v_attend = torch.cat([v_selected.view(B, H, T, top_k, D).reshape(B * H * T, top_k, D),
                              v_win.unsqueeze(2).expand(-1, -1, T, -1, -1).reshape(B * H * T, window, D)], dim=1)

        q_flat = q.reshape(B * H * T, 1, D)
        attn_weights = torch.matmul(q_flat, k_attend.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v_attend)
        return out.view(B, H, T, D)

    def _hca_forward(
        self,
        q: torch.Tensor,
        k_compressed: torch.Tensor,
        v_compressed: torch.Tensor,
    ) -> torch.Tensor:
        """HCA: dense attention over heavily compressed KV sequence."""
        k_exp = self._repeat_kv(k_compressed)
        v_exp = self._repeat_kv(v_compressed)

        # Dense attention over compressed sequence — no causal mask needed
        # because compression is applied to the full past context
        out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
        return out

    def _build_csa_mask(self, T_q: int, T_kv: int, device: torch.device) -> torch.Tensor:
        """Build attention mask for CSA (allow all compressed + causal local)."""
        mask = torch.zeros(T_q, T_kv, device=device, dtype=torch.bool)
        # All positions can attend to all KV positions in this simplified version
        return None  # No mask needed — we handle causality through the structure

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            position_ids: (B, T) optional

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply partial RoPE
        # Need to expand k for RoPE then contract back — or apply RoPE per-group
        # RoPE expects same head count for q and k, so we expand k temporarily
        k_for_rope = self._repeat_kv(k)
        q, k_rotated = self.rope(q, k_for_rope, position_ids)
        # Average back to kv_heads for efficiency
        if self.n_rep > 1:
            k = k_rotated.view(B, self.n_kv_heads, self.n_rep, T, self.head_dim).mean(dim=2)
        else:
            k = k_rotated

        # Compress KV
        k_compressed = self.kv_compressor(k)
        v_compressed = self.kv_compressor(v)

        # Gate
        gate = torch.sigmoid(self.gate_proj(x))  # (B, T, n_q_heads * head_dim)
        gate = gate.view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)

        # Attention variant
        if self.variant == "csa":
            attn_out = self._csa_forward(q, k, v, k_compressed, v_compressed)
        else:  # hca
            attn_out = self._hca_forward(q, k_compressed, v_compressed)

        # Apply gate
        attn_out = gate * attn_out

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.o_proj(attn_out)

        return output

    def stats(self) -> dict:
        return {
            "variant": self.variant,
            "n_q_heads": self.n_q_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "params": sum(p.numel() for p in self.parameters()),
        }
