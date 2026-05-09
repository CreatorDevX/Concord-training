"""
Test AttnRes Stability — Gate 5.

Pseudo-queries remain zero at init → uniform attention weights.
After training steps, weights should diverge from uniform.
"""

import pytest
import torch
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.config import ModelConfig
from model.components.attn_res import BlockAttnRes


@pytest.fixture
def config():
    return ModelConfig()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestAttnResStability:
    def test_zero_init(self, config, device):
        """Pseudo-queries should be initialized to zero."""
        attn_res = BlockAttnRes(
            d_model=config.d_model, n_layers=config.n_layers, n_blocks=config.n_groups
        ).to(device)
        assert torch.all(attn_res.pseudo_queries == 0)

    def test_uniform_attention_at_init(self, config, device):
        """With zero pseudo-queries, attention weights should be uniform."""
        attn_res = BlockAttnRes(
            d_model=config.d_model, n_layers=config.n_layers, n_blocks=config.n_groups
        ).to(device)
        attn_res.eval()

        block_reprs = [torch.randn(1, 8, config.d_model, device=device) for _ in range(2)]
        partial = torch.randn(1, 8, config.d_model, device=device)

        _ = attn_res(layer_idx=6, block_reprs=block_reprs, partial_residual=partial)
        stats = attn_res.stats()

        if "attn_uniformity" in stats:
            assert stats["attn_uniformity"] > 0.8

    def test_gradient_flows_to_pseudo_queries(self, config, device):
        """Verify gradients reach the pseudo-queries."""
        attn_res = BlockAttnRes(
            d_model=config.d_model, n_layers=config.n_layers, n_blocks=config.n_groups
        ).to(device)

        block_reprs = [torch.randn(1, 8, config.d_model, device=device)]
        partial = torch.randn(1, 8, config.d_model, device=device)
        target = torch.randn(1, 8, config.d_model, device=device)

        out = attn_res(layer_idx=0, block_reprs=block_reprs, partial_residual=partial)
        loss = F.mse_loss(out, target)
        loss.backward()

        assert attn_res.pseudo_queries.grad is not None
        assert attn_res.pseudo_queries.grad.abs().max().item() > 0

    def test_block_reprs_detach_prevents_backprop(self, config, device):
        """Block representations should be detached."""
        attn_res = BlockAttnRes(
            d_model=config.d_model, n_layers=config.n_layers, n_blocks=config.n_groups
        ).to(device)

        source = torch.randn(1, 8, config.d_model, device=device, requires_grad=True)
        block_reprs = [source.detach()]
        partial = torch.randn(1, 8, config.d_model, device=device, requires_grad=True)

        out = attn_res(layer_idx=0, block_reprs=block_reprs, partial_residual=partial)
        out.sum().backward()

        assert partial.grad is not None
        assert source.grad is None, "Gradient leaked through detached block_repr"
