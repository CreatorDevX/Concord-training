"""
Test Shapes — Gate 1: Forward pass shape correctness.

Tests each component individually and the full model.
Runs at full config scale but with tiny batch/seq for speed.

Gate 1: input_ids (2,128) → model → logits (2,128,248320)
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.config import ModelConfig
from model.components.rms_norm import RMSNorm
from model.components.rope import PartialRoPE
from model.components.delta_net import GatedDeltaNet
from model.components.attention import GatedAttention
from model.components.attn_res import BlockAttnRes
from model.components.expert import ExpertFFN
from model.components.router import MLPRouter
from model.components.moe import MoEBlock
from model.components.mtp import MTPHead


@pytest.fixture
def config():
    return ModelConfig()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestRMSNorm:
    def test_shape(self, config, device):
        norm = RMSNorm(config.d_model).to(device)
        x = torch.randn(2, 16, config.d_model, device=device)
        out = norm(x)
        assert out.shape == (2, 16, config.d_model)

    def test_variance_approx_one(self, config, device):
        norm = RMSNorm(config.d_model).to(device)
        x = torch.randn(2, 16, config.d_model, device=device)
        out = norm(x)
        rms = out.float().pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestPartialRoPE:
    def test_shape(self, config, device):
        rope = PartialRoPE(
            head_dim=config.attn_head_dim, rope_dim=config.rope_dim, max_seq_len=512
        ).to(device)
        q = torch.randn(2, config.attn_q_heads, 16, config.attn_head_dim, device=device)
        k = torch.randn(2, config.attn_q_heads, 16, config.attn_head_dim, device=device)
        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_unrotated_dims_unchanged(self, config, device):
        rope = PartialRoPE(
            head_dim=config.attn_head_dim, rope_dim=config.rope_dim, max_seq_len=512
        ).to(device)
        q = torch.randn(2, config.attn_q_heads, 16, config.attn_head_dim, device=device)
        k = torch.randn(2, config.attn_q_heads, 16, config.attn_head_dim, device=device)
        q_out, k_out = rope(q, k)
        rd = config.rope_dim
        assert torch.allclose(q[..., rd:], q_out[..., rd:], atol=1e-5)
        assert torch.allclose(k[..., rd:], k_out[..., rd:], atol=1e-5)

    def test_rotated_dims_position_dependent(self, config, device):
        rope = PartialRoPE(
            head_dim=config.attn_head_dim, rope_dim=config.rope_dim, max_seq_len=512
        ).to(device)
        q = torch.ones(1, 1, 4, config.attn_head_dim, device=device)
        k = torch.ones(1, 1, 4, config.attn_head_dim, device=device)
        q_out, _ = rope(q, k)
        rd = config.rope_dim
        assert not torch.allclose(q_out[0, 0, 0, :rd], q_out[0, 0, 1, :rd])


class TestGatedDeltaNet:
    def test_shape(self, config, device):
        delta = GatedDeltaNet(
            d_model=config.d_model,
            n_v_heads=config.delta_v_heads,
            n_qk_heads=config.delta_qk_heads,
            head_dim=config.delta_head_dim,
            chunk_size=config.delta_chunk_size,
        ).to(device)
        x = torch.randn(1, 32, config.d_model, device=device)
        out, state = delta(x)
        assert out.shape == (1, 32, config.d_model)
        assert state.shape[0] == 1
        assert state.shape[1] == config.delta_qk_heads

    def test_causal(self, config, device):
        """Output at position t should not depend on future positions."""
        delta = GatedDeltaNet(
            d_model=config.d_model,
            n_v_heads=config.delta_v_heads,
            n_qk_heads=config.delta_qk_heads,
            head_dim=config.delta_head_dim,
            chunk_size=config.delta_chunk_size,
        ).to(device)
        delta.eval()
        x = torch.randn(1, 32, config.d_model, device=device)
        out_full, _ = delta(x)
        out_partial, _ = delta(x[:, :16, :])
        assert torch.allclose(out_full[:, :16, :], out_partial, atol=1e-4)


class TestGatedAttention:
    def test_csa_shape(self, config, device):
        attn = GatedAttention(
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
        x = torch.randn(1, 32, config.d_model, device=device)
        coords = {
            "x": torch.zeros(1, 32, device=device),
            "y": torch.zeros(1, 32, device=device),
            "u": torch.arange(32, device=device).unsqueeze(0).float(),
            "w": torch.zeros(1, 32, device=device),
        }
        out = attn(x, coords=coords)
        assert out.shape == (1, 32, config.d_model)

    def test_hca_shape(self, config, device):
        attn = GatedAttention(
            d_model=config.d_model,
            n_q_heads=config.attn_q_heads,
            n_kv_heads=config.attn_kv_heads,
            head_dim=config.attn_head_dim,
            rope_axis_dims=config.rope_axis_dims,
            variant="hca",
            hca_compress=config.hca_compress,
        ).to(device)
        x = torch.randn(1, 128, config.d_model, device=device)
        coords = {
            "x": torch.zeros(1, 128, device=device),
            "y": torch.zeros(1, 128, device=device),
            "u": torch.arange(128, device=device).unsqueeze(0).float(),
            "w": torch.zeros(1, 128, device=device),
        }
        out = attn(x, coords=coords)
        assert out.shape == (1, 128, config.d_model)


class TestBlockAttnRes:
    def test_shape(self, config, device):
        attn_res = BlockAttnRes(
            d_model=config.d_model, n_layers=config.n_layers, n_blocks=config.n_groups
        ).to(device)
        block_reprs = [torch.randn(1, 16, config.d_model, device=device) for _ in range(2)]
        partial = torch.randn(1, 16, config.d_model, device=device)
        out = attn_res(layer_idx=8, block_reprs=block_reprs, partial_residual=partial)
        assert out.shape == (1, 16, config.d_model)

    def test_zero_init_uniform(self, config, device):
        attn_res = BlockAttnRes(
            d_model=config.d_model, n_layers=config.n_layers, n_blocks=config.n_groups
        ).to(device)
        assert torch.allclose(attn_res.pseudo_queries, torch.zeros_like(attn_res.pseudo_queries))


class TestExpertFFN:
    def test_shape(self, config, device):
        expert = ExpertFFN(
            d_model=config.d_model, d_ffn=config.expert_intermediate
        ).to(device)
        x = torch.randn(32, config.d_model, device=device)
        out = expert(x)
        assert out.shape == (32, config.d_model)

    def test_params(self, config, device):
        expert = ExpertFFN(
            d_model=config.d_model, d_ffn=config.expert_intermediate
        ).to(device)
        params = sum(p.numel() for p in expert.parameters())
        expected = 3 * config.d_model * config.expert_intermediate
        assert params == expected


class TestMLPRouter:
    def test_shape(self, config, device):
        router = MLPRouter(
            d_model=config.d_model, n_experts=config.n_experts,
            n_routed=config.n_routed_delta, hidden_dim=config.router_hidden
        ).to(device)
        x = torch.randn(1, 16, config.d_model, device=device)
        indices, scores, aux_loss = router(x)
        assert indices.shape == (1, 16, config.n_routed_delta)
        assert scores.shape == (1, 16, config.n_routed_delta)
        assert aux_loss.dim() == 0

    def test_scores_sum_to_one(self, config, device):
        router = MLPRouter(
            d_model=config.d_model, n_experts=config.n_experts,
            n_routed=config.n_routed_delta, hidden_dim=config.router_hidden
        ).to(device)
        x = torch.randn(1, 16, config.d_model, device=device)
        _, scores, _ = router(x)
        sums = scores.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


class TestMoEBlock:
    def test_shape(self, config, device):
        moe = MoEBlock(
            d_model=config.d_model, n_experts=config.n_experts,
            n_routed=config.n_routed_delta,
            expert_intermediate=config.expert_intermediate,
            router_hidden=config.router_hidden,
            expert_dtype=config.expert_dtype,
        ).to(device)
        x = torch.randn(1, 8, config.d_model, device=device)
        out, aux_loss = moe(x)
        assert out.shape == (1, 8, config.d_model)
        assert aux_loss.dim() == 0


class TestMTPHead:
    def test_shape_and_loss(self, config, device):
        mtp = MTPHead(
            d_model=config.d_model, embed_dim=config.embed_dim,
            vocab_size=config.vocab_size, mtp_steps=config.mtp_steps
        ).to(device)
        hidden = torch.randn(1, 32, config.d_model, device=device)
        labels = torch.randint(0, config.vocab_size, (1, 32), device=device)
        embed_down = torch.nn.Linear(config.d_model, config.embed_dim, bias=False).to(device)
        loss = mtp(hidden, labels, embed_down=embed_down)
        assert loss.dim() == 0
        assert loss.item() > 0
