"""
Vision connectors and temporal video encoders for Multimodal adaptation.

Uses frozen TIPSv2-L14 patch embeddings.
- Image: VisionConnector (simple linear projection) -> 256 tokens.
- Video: SpatialPool -> Temporal DeltaNet -> Salient feature pooling -> 128 constant tokens.
"""

import torch
import torch.nn as nn
from typing import Optional

from model.config import ModelConfig
from model.components.rms_norm import RMSNorm
from model.components.delta_net import GatedDeltaNet


class VisionConnector(nn.Module):
    """
    Stage M1/M2 Connector for static images.
    Takes (B, N_patches, d_vision) and maps to (B, N_patches, d_model).
    TIPSv2-L14 outputs 1024d, so this is remarkably lightweight (effectively identity).
    """

    def __init__(self, config: ModelConfig, d_vision: int = 1024):
        super().__init__()
        self.d_vision = d_vision
        self.d_model = config.d_model

        # Only standard linear + norm to align residual streams
        self.proj = nn.Linear(d_vision, config.d_model, bias=False)
        self.norm = RMSNorm(config.d_model)

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeddings: (B, 256, 1024) from frozen TIPSv2-L14
        Returns:
            (B, 256, d_model) aligned to LM residual stream.
        """
        x = self.proj(patch_embeddings)
        return self.norm(x)


class SpatialPooler(nn.Module):
    """
    Learned 2D attention pooling.
    Compresses 256 patch tokens (16x16 grid) down to e.g., 64 tokens.
    """

    def __init__(self, d_vision: int = 1024, n_queries: int = 64):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_vision) * 0.02)
        
        self.q_proj = nn.Linear(d_vision, d_vision, bias=False)
        self.k_proj = nn.Linear(d_vision, d_vision, bias=False)
        self.v_proj = nn.Linear(d_vision, d_vision, bias=False)
        self.out_proj = nn.Linear(d_vision, d_vision, bias=False)
        self.norm = RMSNorm(d_vision)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*N_frames, 256, d_vision)
        Returns:
            (B*N_frames, n_queries, d_vision)
        """
        BF, T_in, d = x.shape
        q = self.q_proj(self.queries.expand(BF, -1, -1))  # (BF, 64, d)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Simple single-headed attention for pooling (scaling by sqrt(d))
        attn = torch.bmm(q, k.transpose(1, 2)) / (d ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)
        out = self.out_proj(out)
        return self.norm(out)


class SalientTokenPooler(nn.Module):
    """
    Pools the temporal sequence of tokens into a fixed K tokens.
    """
    
    def __init__(self, d_model: int, k_tokens: int = 128):
        super().__init__()
        self.k_tokens = k_tokens
        self.queries = nn.Parameter(torch.randn(1, k_tokens, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_frames * 64, d_model)
        Returns:
            (B, k_tokens, d_model)
        """
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out)


class TemporalVideoEncoder(nn.Module):
    """
    Stage M3 Temporal encoder for videos.
    Pipeline:
      1. Spatial Pool (256 -> 64 per frame)
      2. Flatten into temporal sequence (N_frames * 64)
      3. 2-layer DeltaNet over the temporal sequence
      4. Salient cross-attention to constant K=128 tokens
    """

    def __init__(self, config: ModelConfig, d_vision: int = 1024, spatial_tokens: int = 64, final_tokens: int = 128):
        super().__init__()
        self.spatial_pool = SpatialPooler(d_vision=d_vision, n_queries=spatial_tokens)
        
        # We project d_vision (1024) to d_model right before DeltaNet
        self.in_proj = nn.Linear(d_vision, config.d_model, bias=False)
        self.in_norm = RMSNorm(config.d_model)
        
        # 2-layer temporal DeltaNet (no MoE)
        self.temporal_delta1 = GatedDeltaNet(
            d_model=config.d_model,
            n_v_heads=config.delta_v_heads,
            n_qk_heads=config.delta_qk_heads,
            head_dim=config.delta_head_dim,
            chunk_size=config.delta_chunk_size,
        )
        self.delta_norm1 = RMSNorm(config.d_model)
        
        self.temporal_delta2 = GatedDeltaNet(
            d_model=config.d_model,
            n_v_heads=config.delta_v_heads,
            n_qk_heads=config.delta_qk_heads,
            head_dim=config.delta_head_dim,
            chunk_size=config.delta_chunk_size,
        )
        self.delta_norm2 = RMSNorm(config.d_model)

        # Salient features extractor
        self.salient_pool = SalientTokenPooler(d_model=config.d_model, k_tokens=final_tokens)

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_frames: (B, N_frames, 256, 1024) - patches from TIPSv2
        Returns:
            (B, 128, d_model) fixed-size salient tokens to prepend to text
        """
        B, N, P, D = video_frames.shape
        
        # Spatial compress independent frames
        x = video_frames.view(B * N, P, D)
        x = self.spatial_pool(x)  # (B*N, 64, D)
        
        # Flatten frames to temporal sequence
        x = x.view(B, N * 64, D)
        x = self.in_norm(self.in_proj(x))
        
        # Temporal DeltaNet process
        out1, _ = self.temporal_delta1(x, None)
        out1 = self.delta_norm1(out1)
        
        out2, _ = self.temporal_delta2(out1, None)
        out2 = self.delta_norm2(out2)
        
        # Final salient token extraction
        temporal_repr = self.salient_pool(out2)  # (B, 128, d_model)
        
        return temporal_repr
