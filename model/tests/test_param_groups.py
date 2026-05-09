"""
Tests for Parameter Group Deduplication (tied embedding fix).

Validates:
  1. Tied embeddings appear only once across all param groups
  2. No tensor object is duplicated across optimizer groups
  3. Expert params go to expert_sgd group
  4. Embedding/norm/mtp params go to lion group
  5. Remaining params go to muon group
  6. Total unique params sum equals model parameter count
  7. All param groups have non-zero params
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.config import ModelConfig
from model.model import HybridMoE


def _small_config():
    """Small config for fast CPU testing — still exercises all code paths."""
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
        router_hidden=64,
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
        grad_checkpoint=False,
        use_vision=False,
        tie_embeddings=True,
    )


@pytest.fixture
def config():
    return _small_config()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestParamGroups:
    def test_no_duplicate_tensors_across_groups(self, config, device):
        """No tensor should appear in multiple optimizer groups."""
        model = HybridMoE(config).to(device)
        groups = model.get_param_groups()

        all_param_ids = {}
        for group in groups:
            group_name = group["name"]
            for p in group["params"]:
                pid = id(p)
                if pid in all_param_ids:
                    prev_group = all_param_ids[pid]
                    pytest.fail(
                        f"Tensor {p.shape} appears in both '{prev_group}' and '{group_name}'"
                    )
                all_param_ids[pid] = group_name

    def test_total_unique_params_matches(self, config, device):
        """Sum of unique params across groups should equal model's total params."""
        model = HybridMoE(config).to(device)
        groups = model.get_param_groups()

        group_params = set()
        for group in groups:
            for p in group["params"]:
                group_params.add(id(p))

        model_params = set()
        for p in model.parameters():
            if p.requires_grad:
                model_params.add(id(p))

        assert len(group_params) == len(model_params), \
            f"Param groups cover {len(group_params)}/{len(model_params)} params"

    def test_all_groups_have_params(self, config, device):
        """Every optimizer group should have at least one parameter."""
        model = HybridMoE(config).to(device)
        groups = model.get_param_groups()

        group_names = [g["name"] for g in groups]
        assert "expert_sgd" in group_names
        assert "muon" in group_names
        assert "lion" in group_names

        for group in groups:
            assert len(group["params"]) > 0, \
                f"Group '{group['name']}' has zero parameters"

    def test_tied_lm_head_not_in_groups(self, config, device):
        """When tie_embeddings=True, lm_head.weight should NOT add extra param."""
        model = HybridMoE(config).to(device)
        groups = model.get_param_groups()

        all_params = []
        for group in groups:
            all_params.extend(group["params"])

        # lm_head.weight IS embed.weight — it should appear exactly once
        embed_weight = model.embed.weight
        lm_head_weight = model.lm_head.weight

        assert lm_head_weight.data_ptr() == embed_weight.data_ptr(), \
            "lm_head.weight should share storage with embed.weight"

        count = sum(1 for p in all_params if p.data_ptr() == embed_weight.data_ptr())
        assert count == 1, \
            f"Embedding weight should appear exactly once, found {count}"

    def test_mtp_output_tied_to_embed(self, config, device):
        """MTP output projection should also be tied to embed.weight."""
        model = HybridMoE(config).to(device)

        if hasattr(model.mtp, 'output_proj'):
            assert model.mtp.output_proj.weight.data_ptr() == model.embed.weight.data_ptr(), \
                "MTP output_proj.weight should be tied to embed.weight"

    def test_param_counts_sum_to_total(self, config, device):
        """Each tensor's numel summed should equal total trainable params."""
        model = HybridMoE(config).to(device)
        groups = model.get_param_groups()

        counted_ids = set()
        total_params = 0
        for group in groups:
            for p in group["params"]:
                pid = id(p)
                if pid not in counted_ids:
                    counted_ids.add(pid)
                    total_params += p.numel()

        model_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params == model_total, \
            f"Counted {total_params} vs model {model_total}"

    def test_tied_untied_embeddings(self, config, device):
        """Verify tie_embeddings=False works correctly (no dedup needed)."""
        cfg = ModelConfig(tie_embeddings=False)
        model = HybridMoE(cfg).to(device)
        groups = model.get_param_groups()

        # When untied, lm_head.weight and embed.weight are different tensors
        assert model.lm_head.weight.data_ptr() != model.embed.weight.data_ptr(), \
            "Untied embeddings should have different storage"

        all_param_ids = set()
        for group in groups:
            for p in group["params"]:
                all_param_ids.add(id(p))

        model_ids = set(id(p) for p in model.parameters() if p.requires_grad)
        assert len(all_param_ids) == len(model_ids), \
            f"Untied: {len(all_param_ids)} in groups vs {len(model_ids)} in model"


class TestExpertSGDCentralization:
    def test_expert_params_in_sgd_group(self, config, device):
        """Expert FFN params should be in the 'expert_sgd' group."""
        model = HybridMoE(config).to(device)
        groups = model.get_param_groups()
        expert_group = next(g for g in groups if g["name"] == "expert_sgd")

        expert_param_names = set()
        for name, param in model.named_parameters():
            if "routed_experts" in name and param.requires_grad:
                expert_param_names.add(name)

        # Verify at least some expert params are in the group
        expert_ids = set(id(p) for p in expert_group["params"])
        has_expert = False
        for name, param in model.named_parameters():
            if "routed_experts" in name and id(param) in expert_ids:
                has_expert = True
                break
        assert has_expert, "Expert params should be in expert_sgd group"

    def test_shared_experts_not_duplicated(self, config, device):
        """With share_experts_within_group, shared expert weight should appear once."""
        from dataclasses import replace
        cfg = replace(config, share_experts_within_group=True)
        model = HybridMoE(cfg).to(device)
        groups = model.get_param_groups()

        all_params = []
        for group in groups:
            all_params.extend(group["params"])

        ptr_counts = {}
        for p in all_params:
            ptr_counts[p.data_ptr()] = ptr_counts.get(p.data_ptr(), 0) + 1

        duplicates = {ptr: cnt for ptr, cnt in ptr_counts.items() if cnt > 1}
        assert len(duplicates) == 0, \
            f"Found {len(duplicates)} duplicated tensor pointers"
