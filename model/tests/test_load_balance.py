"""
Test Load Balance — Gate 4.

Tests router load distribution and bias correction.
Uses full-scale dimensions but tiny batch/seq.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.config import ModelConfig
from model.components.router import MLPRouter
from model.components.moe import MoEBlock


@pytest.fixture
def config():
    return ModelConfig()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestLoadBalance:
    def test_initial_routing_uses_many_experts(self, config, device):
        """At initialization, routing should use many experts."""
        router = MLPRouter(
            d_model=config.d_model, n_experts=config.n_experts,
            n_routed=config.n_routed_delta, hidden_dim=config.router_hidden
        ).to(device)
        router.eval()

        x = torch.randn(2, 32, config.d_model, device=device)
        indices, _ = router(x)
        unique_experts = indices.unique()
        # Should use a good fraction of experts
        assert len(unique_experts) >= config.n_experts // 4, (
            f"Only {len(unique_experts)}/{config.n_experts} experts used at init"
        )

    def test_bias_update_corrects_imbalance(self, config, device):
        """Bias updates should correct routing imbalance."""
        router = MLPRouter(
            d_model=config.d_model, n_experts=config.n_experts,
            n_routed=config.n_routed_delta, hidden_dim=config.router_hidden,
            bias_update_interval=10, bias_lr=0.1,
        ).to(device)
        router.train()

        for step in range(50):
            x = torch.randn(2, 16, config.d_model, device=device)
            router(x)
            if router.should_update_bias():
                router.update_bias()

        stats = router.stats()
        if stats.get("load_max") is not None:
            assert stats["load_max"] < 0.20, (
                f"Expert with {stats['load_max']:.1%} load — too concentrated"
            )

    def test_moe_block_no_token_drop(self, config, device):
        """MoE block should process all tokens."""
        moe = MoEBlock(
            d_model=config.d_model, n_experts=config.n_experts,
            n_routed=config.n_routed_delta,
            expert_intermediate=config.expert_intermediate,
            router_hidden=config.router_hidden,
            expert_dtype=config.expert_dtype,
        ).to(device)

        x = torch.randn(1, 8, config.d_model, device=device)
        out = moe(x)
        assert (out.abs().sum(dim=-1) > 0).all(), "Some tokens were dropped"
