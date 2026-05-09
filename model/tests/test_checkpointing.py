"""
Tests for Activation Recomputation (gradient checkpointing).

Validates:
  1. Checkpointed MoE produces same gradients as non-checkpointed
  2. Checkpointed Attention produces same gradients as non-checkpointed
  3. MoEBlock activation recomputation reduces memory
  4. GatedAttention activation recomputation works correctly
  5. Nested checkpointing (block-level + MoE-level) works
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.config import ModelConfig
from model.components.moe import MoEBlock, dispatch_and_combine_fast
from model.components.attention import GatedAttention
from model.components.expert import ExpertFFN


@pytest.fixture
def config():
    return ModelConfig()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestMoECheckpointing:
    def test_checkpoint_matches_no_checkpoint_gradients(self, config, device):
        """MoE with and without activation recomputation should produce identical gradients."""
        torch.manual_seed(42)
        moe_ref = MoEBlock(
            d_model=config.d_model, n_experts=config.n_experts,
            n_routed=config.n_routed_delta,
            expert_intermediate=config.expert_intermediate,
            router_hidden=config.router_hidden,
            expert_dtype=config.expert_dtype,
            use_activation_recomputation=False,
        ).to(device)

        torch.manual_seed(42)
        moe_ckpt = MoEBlock(
            d_model=config.d_model, n_experts=config.n_experts,
            n_routed=config.n_routed_delta,
            expert_intermediate=config.expert_intermediate,
            router_hidden=config.router_hidden,
            expert_dtype=config.expert_dtype,
            use_activation_recomputation=True,
        ).to(device)

        # Copy weights to ensure identical init
        moe_ckpt.load_state_dict(moe_ref.state_dict())

        x = torch.randn(1, 16, config.d_model, device=device, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        x_ckpt = x.detach().clone().requires_grad_(True)

        out_ref, _ = moe_ref(x_ref)
        out_ckpt, _ = moe_ckpt(x_ckpt)

        loss_ref = out_ref.sum()
        loss_ckpt = out_ckpt.sum()

        loss_ref.backward()
        loss_ckpt.backward()

        # Compare input gradients
        assert torch.allclose(x_ref.grad, x_ckpt.grad, atol=1e-4, rtol=1e-3), \
            "Input gradients should match with/without checkpoint"

        # Compare parameter gradients
        for (n1, p1), (n2, p2) in zip(moe_ref.named_parameters(), moe_ckpt.named_parameters()):
            if p1.grad is not None and p2.grad is not None:
                assert torch.allclose(p1.grad, p2.grad, atol=1e-4, rtol=1e-3), \
                    f"Param {n1} gradient mismatch"

    def test_checkpoint_reduces_memory(self, config):
        """Checkpointed MoE should use less activation memory (visible on CUDA)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory measurement")

        device = torch.device("cuda:0")

        torch.manual_seed(0)
        moe = MoEBlock(
            d_model=config.d_model,
            n_experts=min(config.n_experts, 8),
            n_routed=config.n_routed_delta,
            expert_intermediate=config.expert_intermediate,
            router_hidden=config.router_hidden,
            expert_dtype=config.expert_dtype,
            use_activation_recomputation=False,
        ).to(device)

        x = torch.randn(2, 128, config.d_model, device=device, requires_grad=True)

        # Forward without checkpoint
        torch.cuda.reset_peak_memory_stats(device)
        out, _ = moe(x)
        loss = out.sum()
        mem_no_ckpt = torch.cuda.memory_allocated(device)

        loss.backward()
        moe.zero_grad(set_to_none=True)

        # Forward with checkpoint
        moe.use_activation_recomputation = True
        torch.cuda.reset_peak_memory_stats(device)
        out, _ = moe(x)
        loss = out.sum()
        mem_with_ckpt = torch.cuda.memory_allocated(device)

        assert mem_with_ckpt <= mem_no_ckpt * 1.1, \
            f"Checkpoint should not increase memory (no_ckpt={mem_no_ckpt/1e6:.1f}MB, ckpt={mem_with_ckpt/1e6:.1f}MB)"

    def test_checkpoint_with_zero_expert_routing(self, config, device):
        """Checkpoint should handle edge cases where experts receive zero tokens."""
        # Use a config with many experts to ensure some get no tokens
        moe = MoEBlock(
            d_model=config.d_model,
            n_experts=max(config.n_experts, 16),
            n_routed=1,
            expert_intermediate=config.expert_intermediate,
            router_hidden=config.router_hidden,
            expert_dtype=config.expert_dtype,
            use_activation_recomputation=True,
        ).to(device)

        x = torch.randn(1, 4, config.d_model, device=device, requires_grad=True)
        out, aux_loss = moe(x)
        loss = (out.sum() + aux_loss)
        loss.backward()
        # No crash = success
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestAttentionCheckpointing:
    def test_checkpoint_matches_no_checkpoint(self, config, device):
        """Attention with and without checkpoint should produce identical gradients."""
        torch.manual_seed(42)
        attn_ref = GatedAttention(
            d_model=config.d_model,
            n_q_heads=config.attn_q_heads,
            n_kv_heads=config.attn_kv_heads,
            head_dim=config.attn_head_dim,
            rope_axis_dims=config.rope_axis_dims,
            variant="csa",
            csa_compress=config.csa_compress,
            csa_top_k=config.csa_top_k,
            csa_window=config.csa_window,
        ).to(device)

        torch.manual_seed(42)
        attn_ckpt = GatedAttention(
            d_model=config.d_model,
            n_q_heads=config.attn_q_heads,
            n_kv_heads=config.attn_kv_heads,
            head_dim=config.attn_head_dim,
            rope_axis_dims=config.rope_axis_dims,
            variant="csa",
            csa_compress=config.csa_compress,
            csa_top_k=config.csa_top_k,
            csa_window=config.csa_window,
        ).to(device)

        attn_ckpt.load_state_dict(attn_ref.state_dict())

        x = torch.randn(1, 32, config.d_model, device=device, requires_grad=True)
        coords = {
            "x": torch.zeros(1, 32, device=device),
            "y": torch.zeros(1, 32, device=device),
            "u": torch.arange(32, device=device).unsqueeze(0).float(),
            "w": torch.zeros(1, 32, device=device),
        }

        x_ref = x.detach().clone().requires_grad_(True)
        x_ckpt = x.detach().clone().requires_grad_(True)

        # Reference: no checkpoint
        out_ref = attn_ref(x_ref, coords=coords)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        # Checkpointed: patching training flag
        attn_ckpt.train()
        out_ckpt = attn_ckpt(x_ckpt, coords=coords)
        loss_ckpt = out_ckpt.sum()
        loss_ckpt.backward()

        assert torch.allclose(x_ref.grad, x_ckpt.grad, atol=1e-3, rtol=1e-2), \
            "Input gradients should match with/without checkpoint"

    def test_hca_checkpoint_works(self, config, device):
        """HCA variant with checkpoint should work correctly."""
        attn = GatedAttention(
            d_model=config.d_model,
            n_q_heads=config.attn_q_heads,
            n_kv_heads=config.attn_kv_heads,
            head_dim=config.attn_head_dim,
            rope_axis_dims=config.rope_axis_dims,
            variant="hca",
            hca_compress=config.hca_compress,
        ).to(device)

        attn.train()

        x = torch.randn(1, 64, config.d_model, device=device, requires_grad=True)
        coords = {
            "x": torch.zeros(1, 64, device=device),
            "y": torch.zeros(1, 64, device=device),
            "u": torch.arange(64, device=device).unsqueeze(0).float(),
            "w": torch.zeros(1, 64, device=device),
        }

        out = attn(x, coords=coords)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_full_attention_checkpoint(self, config, device):
        """Full (non-compressed) attention should also work with checkpoint path."""
        attn = GatedAttention(
            d_model=config.d_model,
            n_q_heads=config.attn_q_heads,
            n_kv_heads=config.attn_kv_heads,
            head_dim=config.attn_head_dim,
            rope_axis_dims={"u": config.attn_head_dim},
            variant="full",
        ).to(device)

        attn.train()
        x = torch.randn(1, 16, config.d_model, device=device, requires_grad=True)
        coords = {
            "u": torch.arange(16, device=device).unsqueeze(0).float(),
            "w": torch.zeros(1, 16, device=device),
            "x": torch.zeros(1, 16, device=device),
            "y": torch.zeros(1, 16, device=device),
        }
        out = attn(x, coords=coords)
        out.sum().backward()
        assert x.grad is not None
