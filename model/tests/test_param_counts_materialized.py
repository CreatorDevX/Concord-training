"""
Test Parameter Counts (Materialized) — Gate 3 validation.

Instantiates the full model and compares actual param counts
against the mathematically computed expected values.

At full scale:
    Total ∈ [2.8B, 3.2B] (with original config)
    Active ∈ [90M, 110M]
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.config import ModelConfig
from model.model import HybridMoE
from model.param_count import count_params


def get_small_config():
    """Small config for fast CPU testing."""
    return ModelConfig(
        d_model=128,
        embed_dim=64,
        n_layers=4,
        group_size=2,
        n_groups=2,
        n_experts=4,
        n_routed_delta=1,
        n_routed_attn=1,
        expert_intermediate=64,
        n_shared=1,
        delta_v_heads=2,
        delta_qk_heads=2,
        delta_head_dim=64,
        attn_q_heads=2,
        attn_kv_heads=1,
        attn_head_dim=64,
        rope_axis_dims={"x": 8, "y": 8, "u": 16, "w": 32},
        vocab_size=4096,
        mtp_steps=1,
        csa_top_k=8,
        csa_window=4,
        csa_compress=2,
        hca_compress=4,
        use_vision=False,
    )


def get_full_config():
    return ModelConfig(
        d_model=768,
        embed_dim=384,
        n_layers=16,
        group_size=4,
        n_groups=4,
        n_experts=36,
        n_routed_delta=2,
        n_routed_attn=2,
        expert_intermediate=256,
        n_shared=1,
        delta_v_heads=12,
        delta_qk_heads=12,
        delta_head_dim=64,
        attn_q_heads=6,
        attn_kv_heads=2,
        attn_head_dim=128,
        rope_axis_dims={"x": 16, "y": 16, "u": 32, "w": 64},
        use_vision=False,
    )


class TestParamCountsMaterialized:
    def test_materialized_matches_math_total_small(self):
        """Small config: materialized total params should match pure math count."""
        config = get_small_config()
        math_counts = count_params(config)

        torch.set_default_dtype(torch.float16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)

        seen_ids = set()
        materialized_total = 0
        for p in model.parameters():
            if id(p) not in seen_ids:
                seen_ids.add(id(p))
                materialized_total += p.numel()

        math_total = math_counts["total"]
        diff_pct = abs(materialized_total - math_total) / math_total * 100
        assert diff_pct < 0.1, \
            f"Materialized ({materialized_total:,}) vs math ({math_total:,}), diff={diff_pct:.4f}%"

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_materialized_matches_math_total_full(self):
        """Full-scale config: materialized total matches pure math count — GPU only."""
        if not torch.cuda.is_available():
            pytest.skip("Full-scale test requires GPU")
        config = get_full_config()
        math_counts = count_params(config)

        torch.set_default_dtype(torch.float16)
        device = torch.device("cuda")
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)

        seen_ids = set()
        materialized_total = 0
        for p in model.parameters():
            if id(p) not in seen_ids:
                seen_ids.add(id(p))
                materialized_total += p.numel()

        math_total = math_counts["total"]
        diff_pct = abs(materialized_total - math_total) / math_total * 100
        assert diff_pct < 0.1

        del model
        torch.cuda.empty_cache()

    def test_expert_params_dominate_materialized(self):
        """Expert parameters should be majority of total materialized count (small config)."""
        config = get_small_config()
        torch.set_default_dtype(torch.float16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)

        expert_params = 0
        total_params = 0
        seen_ids = set()

        for name, p in model.named_parameters():
            pid = id(p)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            total_params += p.numel()
            if "routed_experts" in name or "shared_experts" in name:
                expert_params += p.numel()

        expert_frac = expert_params / max(total_params, 1)
        assert expert_frac > 0.3, \
            f"Expert fraction {expert_frac:.2%} too low"

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_materialized_expert_count_matches(self):
        """Each expert should have exactly 3 * d_model * expert_intermediate params."""
        config = get_small_config()

        torch.set_default_dtype(torch.float16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)

        for name, p in model.named_parameters():
            if "routed_experts" in name and "w_gate" in name:
                assert p.shape == (config.expert_intermediate, config.d_model), \
                    f"{name}: expected ({config.expert_intermediate}, {config.d_model}), got {p.shape}"
            if "routed_experts" in name and "w_down" in name:
                assert p.shape == (config.d_model, config.expert_intermediate), \
                    f"{name}: expected ({config.d_model}, {config.expert_intermediate}), got {p.shape}"

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_router_output_dim(self):
        """Router output should match n_experts."""
        config = get_small_config()
        torch.set_default_dtype(torch.float16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)

        for name, p in model.named_parameters():
            if "router" in name and "w2" in name:
                assert p.shape[-1] == config.n_experts, \
                    f"Router {name} output dim {p.shape[-1]} != n_experts {config.n_experts}"

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_shared_vs_unshared_materialized(self):
        """Materialized counts should differ with share_experts_within_group."""
        import dataclasses
        cfg_shared = dataclasses.replace(get_small_config(), share_experts_within_group=True)
        cfg_unshared = dataclasses.replace(get_small_config(), share_experts_within_group=False)

        torch.set_default_dtype(torch.float16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        counts = []
        for cfg in [cfg_shared, cfg_unshared]:
            model = HybridMoE(cfg).to(device)
            seen = set()
            total = sum(p.numel() for p in model.parameters() if id(p) not in seen and not seen.add(id(p)))
            counts.append(total)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        assert counts[0] < counts[1], \
            f"Shared ({counts[0]:,}) should have fewer params than unshared ({counts[1]:,})"

    def test_lm_head_tied_in_materialized(self):
        """When tie_embeddings=True, lm_head should share storage with embed."""
        config = get_small_config()
        torch.set_default_dtype(torch.float16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)

        assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr(), \
            "Tied: lm_head.weight must share storage with embed.weight"

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_materialized_breakdown_small(self):
        """Small config: verify component parameter counts match math."""
        config = get_small_config()
        math_counts = count_params(config)

        torch.set_default_dtype(torch.float16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)

        math_expert = math_counts["routed_experts"]
        math_router = math_counts["router_total"]

        mat_expert = 0
        mat_router = 0
        seen = set()

        for name, p in model.named_parameters():
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            if "routed_experts" in name or "shared_experts" in name:
                mat_expert += p.numel()
            elif "router" in name:
                mat_router += p.numel()

        expert_diff = abs(mat_expert - math_expert) / max(math_expert, 1) * 100
        router_diff = abs(mat_router - math_router) / max(math_router, 1) * 100

        assert expert_diff < 0.1, f"Expert params: materialized={mat_expert:,}, math={math_expert:,}"
        assert router_diff < 0.1, f"Router params: materialized={mat_router:,}, math={math_router:,}"

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_materialized_breakdown_full(self):
        """Full-scale: verify component breakdown — GPU only."""
        if not torch.cuda.is_available():
            pytest.skip("Full-scale test requires GPU")
        config = get_full_config()
        math_counts = count_params(config)

        torch.set_default_dtype(torch.float16)
        device = torch.device("cuda")
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)

        math_expert = math_counts["routed_experts"]
        math_router = math_counts["router_total"]

        mat_expert = 0
        mat_router = 0
        seen = set()

        for name, p in model.named_parameters():
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            if "routed_experts" in name or "shared_experts" in name:
                mat_expert += p.numel()
            elif "router" in name:
                mat_router += p.numel()

        expert_diff = abs(mat_expert - math_expert) / math_expert * 100
        router_diff = abs(mat_router - math_router) / math_router * 100

        assert expert_diff < 0.1
        assert router_diff < 0.1

        del model
        torch.cuda.empty_cache()
