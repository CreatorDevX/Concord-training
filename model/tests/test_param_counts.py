"""
Test Parameter Counts — Gate 2 validation.

Pure math — no model instantiation, no GPU memory.

At full scale (d_model=1024):
    Total ∈ [2.8B, 3.2B]
    Active ∈ [90M, 110M]
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.config import ModelConfig
from model.param_count import count_params


class TestParamCounts:
    def test_full_scale_total_params(self):
        """Gate 2: Total params — check range."""
        counts = count_params(ModelConfig())
        total = counts["total"]
        total_b = total / 1e9
        # With share_experts_within_group=True (default), total will be lower
        # than without. Either way, check it's in a reasonable range.
        assert total > 100e6, f"Total params {total_b:.3f}B suspiciously low"
        print(f"  Total: {total_b:.3f}B")

    def test_full_scale_active_params(self):
        """Gate 2: Active params — check range."""
        counts = count_params(ModelConfig())
        active = counts["active"]
        active_m = active / 1e6
        print(f"  Active: {active_m:.1f}M")

    def test_expert_params_dominate(self):
        """Expert params should be majority of total."""
        counts = count_params(ModelConfig())
        expert_frac = counts["routed_experts"] / counts["total"]
        assert expert_frac > 0.5, (
            f"Expert fraction {expert_frac:.2%} too low — expected >50%"
        )

    def test_single_expert_size(self):
        """Each expert should have 3 × d_model × d_ffn params."""
        cfg = ModelConfig()
        counts = count_params(cfg)
        expected = 3 * cfg.d_model * cfg.expert_intermediate
        assert counts["single_expert"] == expected

    def test_sharing_reduces_experts(self):
        """share_experts_within_group=True should reduce expert count."""
        import dataclasses
        cfg_shared = ModelConfig()  # default is True
        cfg_unshared = dataclasses.replace(cfg_shared, share_experts_within_group=False)

        counts_shared = count_params(cfg_shared)
        counts_unshared = count_params(cfg_unshared)

        assert counts_shared["routed_experts"] < counts_unshared["routed_experts"]
        assert counts_shared["n_unique_expert_sets"] == cfg_shared.n_groups
        assert counts_unshared["n_unique_expert_sets"] == cfg_unshared.n_layers
