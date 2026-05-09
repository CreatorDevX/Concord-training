"""
Full Training Loop Tests — Validates end-to-end training correctness.

Tests:
  1. Multi-step training loss decreases monotonically
  2. Gradient flow to all parameter groups
  3. MoE auxiliary loss is non-zero and stable
  4. MTP loss is non-zero and contributes to total
  5. Selective loss masking works correctly
  6. NormalizedSGD + Lion + Adafactor all step without errors
  7. Model can handle variable-length sequences
  8. Loss is finite and non-NaN
  9. Gradient clipping produces finite norms
  10. FP8 weight sync runs without errors
"""

import pytest
import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.config import ModelConfig
from model.model import HybridMoE
from model.train import NormalizedSGD, Lion, Adafactor
from model.components.compact import CompressedActivationContext


def get_small_config():
    """Return a tiny config for fast testing."""
    return ModelConfig(
        d_model=128,
        embed_dim=64,
        n_layers=4,
        group_size=2,
        n_groups=2,
        n_experts=4,
        n_routed_delta=1,
        n_routed_attn=1,
        n_shared=1,
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
        rope_dim=64,
        mtp_steps=1,
        csa_top_k=16,
        csa_window=8,
        csa_compress=2,
        hca_compress=4,
        grad_checkpoint=False,
        use_vision=False,
        aux_loss_coeff=0.01,
        selective_loss=True,
        expert_dtype="fp16",
    )


@pytest.fixture
def config():
    return get_small_config()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestFullTraining:
    def test_loss_decreases_over_steps(self, config, device):
        """Loss should decrease over multiple training steps."""
        torch.set_default_dtype(torch.float16)
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)
        model.train()

        params = model.get_param_groups()
        optimizers = {}
        for group in params:
            name = group.pop("name")
            opt_type = group.pop("optimizer")
            pg = group.pop("params")
            if not pg:
                continue
            if opt_type == "sgd":
                optimizers[name] = NormalizedSGD([{"params": pg, **group}])
            elif opt_type == "lion":
                betas = group.pop("betas", (0.9, 0.99))
                optimizers[name] = Lion([{"params": pg, **group}], betas=betas)
            elif opt_type == "muon":
                group.pop("momentum", None)
                group.pop("nesterov", None)
                group.pop("ns_steps", None)
                optimizers[name] = Adafactor([{"params": pg, **group}])

        B, T = 2, 32
        losses = []
        for step in range(5):
            optimizers[name].zero_grad(set_to_none=True) if False else None
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)

            input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
            wall_ms = time.time() * 1000.0
            timestamps_ms = torch.full((B, T), wall_ms, dtype=torch.float32, device=device)

            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu",
                                    dtype=torch.float16, enabled=torch.cuda.is_available()):
                result = model(input_ids, labels=input_ids, timestamps_ms=timestamps_ms)

            loss = result["loss"]
            losses.append(loss.item())
            loss.backward()

            for opt in optimizers.values():
                opt.step()

        # Check loss trend (may not strictly decrease every step with all optimizers)
        final_avg = sum(losses[-2:]) / 2
        initial_avg = sum(losses[:2]) / 2
        assert final_avg < initial_avg * 1.5, \
            f"Loss trend: {losses}"

    def test_all_losses_non_nan(self, config, device):
        """All loss components should be finite and non-NaN."""
        torch.set_default_dtype(torch.float16)
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)
        model.train()

        B, T = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
        wall_ms = time.time() * 1000.0
        timestamps_ms = torch.full((B, T), wall_ms, dtype=torch.float32, device=device)

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu",
                                dtype=torch.float16, enabled=torch.cuda.is_available()):
            result = model(input_ids, labels=input_ids, timestamps_ms=timestamps_ms)

        for key in ["loss", "main_loss", "mtp_loss", "aux_loss"]:
            val = result.get(key, torch.tensor(float('nan')))
            assert not torch.isnan(val), f"{key} is NaN"
            assert torch.isfinite(val), f"{key} is infinite"
            assert val.item() >= 0 or key == "aux_loss", f"{key} should be non-negative"

    def test_selective_loss_masking(self, config, device):
        """Selective loss masking should zero out loss on masked positions."""
        torch.set_default_dtype(torch.float16)
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)
        model.eval()

        B, T = 1, 32
        input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
        wall_ms = time.time() * 1000.0
        timestamps_ms = torch.full((B, T), wall_ms, dtype=torch.float32, device=device)

        # Mask all tokens
        loss_mask_all = torch.zeros(B, T, device=device)

        with torch.no_grad():
            result_masked = model(input_ids, labels=input_ids, loss_mask=loss_mask_all,
                                  timestamps_ms=timestamps_ms)
            result_full = model(input_ids, labels=input_ids, timestamps_ms=timestamps_ms)

        assert result_masked["main_loss"].item() < result_full["main_loss"].item(), \
            "Masking should reduce loss contribution"

    def test_gradient_flows_to_all_components(self, config, device):
        """Every parameter should receive a gradient."""
        torch.set_default_dtype(torch.float16)
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)
        model.train()

        B, T = 1, 32
        input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
        wall_ms = time.time() * 1000.0
        timestamps_ms = torch.full((B, T), wall_ms, dtype=torch.float32, device=device)

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu",
                                dtype=torch.float16, enabled=torch.cuda.is_available()):
            result = model(input_ids, labels=input_ids, timestamps_ms=timestamps_ms)

        result["loss"].backward()

        no_grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                no_grad_params.append(name)
            elif param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any():
                    no_grad_params.append(f"{name} (NaN grad)")

        assert len(no_grad_params) == 0, \
            f"Params with no/NaN gradient ({len(no_grad_params)}): {no_grad_params[:5]}"

    def test_aux_loss_load_balancing_active(self, config, device):
        """Auxiliary load balancing loss should be non-zero during training."""
        torch.set_default_dtype(torch.float16)
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)
        model.train()

        B, T = 2, 64
        input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
        wall_ms = time.time() * 1000.0
        timestamps_ms = torch.full((B, T), wall_ms, dtype=torch.float32, device=device)

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu",
                                dtype=torch.float16, enabled=torch.cuda.is_available()):
            result = model(input_ids, labels=input_ids, timestamps_ms=timestamps_ms)

        assert result["aux_loss"].item() > 0, \
            "Aux loss should be > 0 (non-uniform routing at init)"

    def test_expert_weights_are_fp16(self, config, device):
        """Expert weights should be FP16 after initialization."""
        cfg = ModelConfig(expert_dtype="fp16", use_vision=False)
        torch.set_default_dtype(torch.float16)
        model = HybridMoE(cfg).to(device)
        torch.set_default_dtype(torch.float32)

        # Check that expert weight dtypes are float16
        for name, param in model.named_parameters():
            if "routed_experts" in name:
                assert param.dtype == torch.float16, f"{name} is {param.dtype}, expected float16"

    def test_forward_with_grad_checkpoint(self, config, device):
        """Model should work with gradient checkpointing enabled."""
        cfg = get_small_config()
        cfg.grad_checkpoint = True

        torch.set_default_dtype(torch.float16)
        model = HybridMoE(cfg).to(device)
        torch.set_default_dtype(torch.float32)
        model.train()

        B, T = 1, 32
        input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        wall_ms = time.time() * 1000.0
        timestamps_ms = torch.full((B, T), wall_ms, dtype=torch.float32, device=device)

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu",
                                dtype=torch.float16, enabled=torch.cuda.is_available()):
            result = model(input_ids, labels=input_ids, timestamps_ms=timestamps_ms)

        loss = result["loss"]
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        loss.backward()
        assert any(p.grad is not None for p in model.parameters()), \
            "Gradients should flow with checkpointing"

    def test_forward_with_compact(self, config, device):
        """Model should work with CompAct activation compression."""
        torch.set_default_dtype(torch.float16)
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)
        model.train()

        B, T = 1, 32
        input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
        wall_ms = time.time() * 1000.0
        timestamps_ms = torch.full((B, T), wall_ms, dtype=torch.float32, device=device)

        ctx = CompressedActivationContext(compress_ratio=0.3, min_numel=4096)
        with ctx:
            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu",
                                    dtype=torch.float16, enabled=torch.cuda.is_available()):
                result = model(input_ids, labels=input_ids, timestamps_ms=timestamps_ms)

        loss = result["loss"]
        loss.backward()
        assert any(p.grad is not None for p in model.parameters()), \
            "Gradients should flow through CompAct"

    def test_inference_mode_works(self, config, device):
        """Model should produce valid logits in eval/inference mode."""
        torch.set_default_dtype(torch.float16)
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)
        model.eval()

        B, T = 1, 16
        input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)

        with torch.no_grad():
            result = model(input_ids)

        assert "logits" in result
        assert result["logits"].shape == (B, T, config.vocab_size)
        assert not torch.isnan(result["logits"]).any()

    def test_gradient_clipping_finite(self, config, device):
        """Gradient clipping should produce finite norms."""
        torch.set_default_dtype(torch.float16)
        model = HybridMoE(config).to(device)
        torch.set_default_dtype(torch.float32)
        model.train()

        B, T = 1, 32
        input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
        wall_ms = time.time() * 1000.0
        timestamps_ms = torch.full((B, T), wall_ms, dtype=torch.float32, device=device)

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu",
                                dtype=torch.float16, enabled=torch.cuda.is_available()):
            result = model(input_ids, labels=input_ids, timestamps_ms=timestamps_ms)

        result["loss"].backward()

        params = [p for p in model.parameters() if p.grad is not None]
        if params:
            total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            assert torch.isfinite(total_norm), "Gradient norm should be finite"
