"""
Gated Attention — CSA and HCA Variants with Causal Compression.

Base: GQA with 8 Q heads, 2 KV heads, head_dim=256, partial RoPE,
output gating via sigmoid(W_gate @ x), Flash Attention 2 / SDPA backend.

CSA (Compressed Sparse Attention) — even-indexed attention layers:
    Compress KV at 4×, select top-1024 entries per query via inner-product,
    attend over selected + 128-token local window.

HCA (Hierarchical Compressed Attention) — odd-indexed attention layers:
    Compress KV at 128×, dense attention over compressed sequence.

FIXES:
    - KVCompressor now applies CAUSAL masking: compressed block j only
      summarizes tokens [j*r, ..., (j+1)*r - 1], and queries at position t
      can only attend to blocks where (j+1)*r - 1 <= t (fully past).
    - CSA fallback (short sequences) now applies proper causal+window mask.
    - Temporal decay bias integrated for wallclock-aware attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

from model.components.rms_norm import RMSNorm
from model.components.rope import PartialRoPE, TemporalDecayBias


class KVCompressor(nn.Module):
    """Compresses KV pairs by pooling along the sequence dimension (causal-aware)."""

    def __init__(self, compress_ratio: int, d_kv: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        # Learned linear projection for compression
        self.compress_proj = nn.Linear(d_kv * compress_ratio, d_kv, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Causal compression: each compressed slot j summarizes tokens
        [j*r, ..., (j+1)*r - 1]. We only compress COMPLETE groups.
        Incomplete trailing tokens are discarded (handled by the local
        window in CSA, or are negligible for HCA's high compression).

        Args:
            x: (B, n_kv_heads, T, head_dim)
        Returns:
            (B, n_kv_heads, T_comp, head_dim) where T_comp = T // compress_ratio
        """
        B, H, T, D = x.shape
        r = self.compress_ratio

        # Only compress complete groups — drop trailing tokens
        # This ensures each compressed slot is fully causal:
        # slot j summarizes exactly [j*r, ..., (j+1)*r - 1], all past.
        n_complete = T // r
        if n_complete == 0:
            # Sequence too short to compress at all — return empty
            return x.new_zeros(B, H, 0, D)

        usable_len = n_complete * r
        x_usable = x[:, :, :usable_len, :]

        # Reshape: group consecutive tokens
        x_grouped = x_usable.reshape(B, H, n_complete, r * D)  # (B, H, T_comp, r*D)
        compressed = self.compress_proj(x_grouped)               # (B, H, T_comp, D)
        return compressed


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
        rope_axis_dims: Dict[str, int],
        variant: str = "csa",
        csa_top_k: int = 1024,
        csa_window: int = 128,
        csa_compress: int = 4,
        hca_compress: int = 128,
        use_checkpoint: bool = True,
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

        # AgentRoPE (Temporal Multimodal RoPE)
        from model.components.rope import TemporalMultimodalRoPE
        self.rope = TemporalMultimodalRoPE(
            axis_dims=rope_axis_dims,
            wallclock_scale_seconds=60.0,
            use_log_wallclock=True,
        )

        # Temporal decay bias for wallclock-aware attention
        self.temporal_bias = TemporalDecayBias(n_q_heads, n_buckets=32)

        # KV compression
        d_kv = head_dim
        if variant == "csa":
            self.kv_compressor = KVCompressor(csa_compress, d_kv)
            self.compress_ratio = csa_compress
        elif variant == "hca":
            self.kv_compressor = KVCompressor(hca_compress, d_kv)
            self.compress_ratio = hca_compress
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Scale factor
        self.scale = 1.0 / math.sqrt(head_dim)
        self.use_checkpoint = use_checkpoint

        # Stats
        self._last_attn_entropy = 0.0

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads for GQA. (B, n_kv, T, D) → (B, n_q, T, D)"""
        if self.n_rep == 1:
            return x
        B, H, T, D = x.shape
        x = x[:, :, None, :, :].expand(B, H, self.n_rep, T, D)
        return x.reshape(B, H * self.n_rep, T, D)

    def _build_causal_compress_mask(
        self, T_q: int, T_comp: int, compress_ratio: int, device: torch.device,
        q_start: int = 0,
    ) -> torch.Tensor:
        """
        Build causal mask for compressed attention.

        Query at position i can attend to compressed block j
        only if (j+1) * compress_ratio - 1 <= i, i.e., the ENTIRE
        compressed block is in the past.

        Args:
            q_start: absolute position offset for chunked query processing.

        Returns:
            (T_q, T_comp) bool mask — True = allowed, False = masked
        """
        # The last original token in compressed block j is at (j+1)*r - 1
        block_end_positions = (torch.arange(T_comp, device=device) + 1) * compress_ratio - 1
        # Use absolute query positions when processing chunks
        query_positions = torch.arange(T_q, device=device) + q_start

        # query_positions (T_q, 1) >= block_end_positions (1, T_comp)
        mask = query_positions.unsqueeze(1) >= block_end_positions.unsqueeze(0)
        return mask

    def _csa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_compressed: torch.Tensor,
        v_compressed: torch.Tensor,
        w_q: Optional[torch.Tensor] = None,
        w_k: Optional[torch.Tensor] = None,
        q_start: int = 0,
    ) -> torch.Tensor:
        """
        CSA: Compressed Sparse Attention with proper causal masking.

        Args:
            q_start: absolute position offset for chunked query processing.
                     Used to correctly compute the local window mask when q is a subset.
        """
        B, H, T, D = q.shape
        T_comp = k_compressed.shape[2]

        if T_comp == 0:
            # No compressed keys — fall back to causal local attention
            k_local = self._repeat_kv(k)
            v_local = self._repeat_kv(v)
            out = F.scaled_dot_product_attention(q, k_local, v_local, is_causal=True)
            return out

        # If compressed length <= top_k, use all compressed + local with proper mask
        if T_comp <= self.csa_top_k:
            k_exp = self._repeat_kv(k_compressed)
            v_exp = self._repeat_kv(v_compressed)
            k_local = self._repeat_kv(k)
            v_local = self._repeat_kv(v)

            # Concatenate compressed + local
            k_cat = torch.cat([k_exp, k_local], dim=2)   # (B, H, T_comp+T_full, D)
            v_cat = torch.cat([v_exp, v_local], dim=2)

            # Build combined mask: causal-compress for first T_comp cols, causal for last T_full cols
            T_full = k.shape[2]
            comp_mask = self._build_causal_compress_mask(T, T_comp, self.compress_ratio, q.device, q_start=q_start)
            # Local mask uses absolute positions for correct window offset
            i_idx = (torch.arange(T, device=q.device) + q_start).unsqueeze(1)
            j_idx = torch.arange(T_full, device=q.device).unsqueeze(0)
            local_mask = (i_idx >= j_idx) & (i_idx - j_idx < self.csa_window)
            combined_mask = torch.cat([comp_mask, local_mask], dim=1)  # (T, T_comp+T_full)

            # Expand mask for SDPA: (1, 1, T, T_comp + T_full)
            attn_mask = combined_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
            float_mask = torch.where(attn_mask, torch.tensor(0.0, device=q.device, dtype=q.dtype),
                                     torch.tensor(float('-inf'), device=q.device, dtype=q.dtype))

            # Apply temporal bias if wallclock coords are available
            if w_q is not None and w_k is not None:
                w_k_comp = w_k[:, self.compress_ratio - 1 :: self.compress_ratio]
                if w_k_comp.shape[1] < T_comp:
                    pad_len = T_comp - w_k_comp.shape[1]
                    w_k_comp = torch.cat([w_k_comp, w_k_comp[:, -1:].expand(-1, pad_len)], dim=1)
                
                w_k_cat = torch.cat([w_k_comp, w_k], dim=1)
                bias = self.temporal_bias(w_q, w_k_cat)
                float_mask = float_mask + bias

            out = F.scaled_dot_product_attention(q, k_cat, v_cat, attn_mask=float_mask)
            return out

        # Full CSA with top-k selection + causal masking
        k_exp = self._repeat_kv(k_compressed)
        v_exp = self._repeat_kv(v_compressed)

        # Score each query against compressed keys
        scores_comp = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale  # (B,H,T,T_comp)

        # Apply causal mask to compressed scores
        comp_causal_mask = self._build_causal_compress_mask(T, T_comp, self.compress_ratio, q.device, q_start=q_start)
        scores_comp.masked_fill_(~comp_causal_mask.unsqueeze(0).unsqueeze(0), -float('inf'))

        # Mask out non-top-k scores (among causally valid ones)
        top_k = min(self.csa_top_k, T_comp)
        if top_k < T_comp:
            sanitized = scores_comp.masked_fill(
                scores_comp == float('-inf'), 
                torch.finfo(scores_comp.dtype).min
            )
            threshold = sanitized.topk(top_k, dim=-1, largest=True).values[..., -1:]
            scores_comp = scores_comp.masked_fill(scores_comp < threshold, -float('inf'))

        # Local window with causal masking (full KV for correct absolute positions)
        T_full = k.shape[2]
        k_local = self._repeat_kv(k)
        v_local = self._repeat_kv(v)
        scores_local = torch.matmul(q, k_local.transpose(-2, -1)) * self.scale  # (B,H,T,T_full)

        # Causal and window masking for local — absolute positions
        i_idx = (torch.arange(T, device=q.device) + q_start).unsqueeze(1)
        j_idx = torch.arange(T_full, device=q.device).unsqueeze(0)
        mask_local = (i_idx >= j_idx) & (i_idx - j_idx < self.csa_window)
        scores_local.masked_fill_(~mask_local, -float('inf'))

        # Apply temporal bias to both compressed and local segments
        if w_q is not None and w_k is not None:
            w_k_comp = w_k[:, self.compress_ratio - 1 :: self.compress_ratio]
            if w_k_comp.shape[1] < T_comp:
                pad_len = T_comp - w_k_comp.shape[1]
                w_k_comp = torch.cat([w_k_comp, w_k_comp[:, -1:].expand(-1, pad_len)], dim=1)
            
            bias_comp = self.temporal_bias(w_q, w_k_comp)
            bias_local = self.temporal_bias(w_q, w_k)
            
            scores_comp = scores_comp + bias_comp
            scores_local = scores_local + bias_local

        # Concatenate and softmax properly
        scores_all = torch.cat([scores_comp, scores_local], dim=-1)
        attn_weights = F.softmax(scores_all, dim=-1)

        attn_weights_comp = attn_weights[..., :T_comp]
        attn_weights_local = attn_weights[..., T_comp:]

        out = torch.matmul(attn_weights_comp, v_exp) + torch.matmul(attn_weights_local, v_local)
        return out

    def _hca_forward(
        self,
        q: torch.Tensor,
        k_compressed: torch.Tensor,
        v_compressed: torch.Tensor,
        w_q: Optional[torch.Tensor] = None,
        w_k: Optional[torch.Tensor] = None,
        q_start: int = 0,
    ) -> torch.Tensor:
        """
        HCA: dense attention over heavily compressed KV sequence
        with causal masking on compressed blocks.
        """
        B, H, T, D = q.shape
        T_comp = k_compressed.shape[2]

        if T_comp == 0:
            # Nothing to attend to — return zeros
            return torch.zeros_like(q)

        k_exp = self._repeat_kv(k_compressed)
        v_exp = self._repeat_kv(v_compressed)

        # Build causal mask for compressed blocks
        comp_causal_mask = self._build_causal_compress_mask(
            T, T_comp, self.compress_ratio, q.device, q_start=q_start
        )
        # Expand: (1, 1, T, T_comp)
        attn_mask = comp_causal_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        float_mask = torch.where(attn_mask, torch.tensor(0.0, device=q.device, dtype=q.dtype),
                                 torch.tensor(float('-inf'), device=q.device, dtype=q.dtype))

        # Apply temporal bias if wallclock coords are available
        if w_q is not None and w_k is not None:
            bias = self.temporal_bias(w_q, w_k)
            float_mask = float_mask + bias

        out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=float_mask)
        return out

    def _compute_attention_with_gate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        coords: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Full attention computation (RoPE + compress + attend + gate) wrapped
        as a single function for activation recomputation checkpointing.

        Queries are processed in chunks to avoid materializing giant
        attention score matrices (the main source of OOM at long context).
        """
        B, T = q.shape[0], q.shape[2]

        # Apply AgentRoPE (Temporal Multimodal RoPE)
        q, k = self.rope(q, k, coords)

        # Compress KV (causally safe)
        k_compressed = self.kv_compressor(k)
        v_compressed = self.kv_compressor(v)

        # Extract wallclock coords if present
        w_q = coords.get("w") if coords is not None else None
        w_k = coords.get("w") if coords is not None else None

        # Pre-process wallclock coords for HCA
        w_k_hca = None
        if self.variant == "hca" and w_k is not None:
            w_k_hca = w_k[:, self.compress_ratio - 1 :: self.compress_ratio]
            T_comp = k_compressed.shape[2]
            if w_k_hca.shape[1] < T_comp:
                pad_len = T_comp - w_k_hca.shape[1]
                w_k_hca = torch.cat([w_k_hca, w_k_hca[:, -1:].expand(-1, pad_len)], dim=1)

        # Full attention variant (no compression) — chunked for memory
        if self.variant == "full":
            attn_out = F.scaled_dot_product_attention(q, self._repeat_kv(k), self._repeat_kv(v), is_causal=True)
            attn_out = gate * attn_out
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
            return self.o_proj(attn_out)

        # Chunk queries to bound peak attention score memory
        # Each chunk processes T_queries × (T_comp + T_local) scores at a time
        q_chunk_size = 256
        output_chunks = []

        for start in range(0, T, q_chunk_size):
            end = min(start + q_chunk_size, T)
            q_chunk = q[:, :, start:end, :]
            gate_chunk = gate[:, :, start:end, :]

            if self.variant == "csa":
                w_q_chunk = w_q[:, start:end] if w_q is not None else None
                out_chunk = self._csa_forward(
                    q_chunk, k, v, k_compressed, v_compressed,
                    w_q=w_q_chunk, w_k=w_k, q_start=start,
                )
            else:  # hca
                out_chunk = self._hca_forward(
                    q_chunk, k_compressed, v_compressed,
                    w_q=w_q[:, start:end] if w_q is not None else None,
                    w_k=w_k_hca, q_start=start,
                )

            out_chunk = gate_chunk * out_chunk
            out_chunk = out_chunk.transpose(1, 2).contiguous().view(B, end - start, -1)
            output_chunks.append(out_chunk)

        output = torch.cat(output_chunks, dim=1)
        return self.o_proj(output)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        coords: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            position_ids: (B, T) optional
            coords: optional dict of multimodal coordinates for AgentRoPE

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Gate
        gate = torch.sigmoid(self.gate_proj(x))  # (B, T, n_q_heads * head_dim)
        gate = gate.view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)

        # Attention computation with optional activation recomputation
        if self.training and self.use_checkpoint:
            output = torch.utils.checkpoint.checkpoint(
                self._compute_attention_with_gate, q, k, v, gate, coords,
                use_reentrant=False,
            )
        else:
            output = self._compute_attention_with_gate(q, k, v, gate, coords)

        # Sanitize: attention matmuls can overflow fp16 in rare cases
        if self.training and (output.dtype == torch.float16 or output.dtype == torch.bfloat16):
            output = torch.where(
                torch.isnan(output) | torch.isinf(output),
                torch.zeros_like(output),
                output
            )

        return output

    def stats(self) -> dict:
        return {
            "variant": self.variant,
            "n_q_heads": self.n_q_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "params": sum(p.numel() for p in self.parameters()),
        }
