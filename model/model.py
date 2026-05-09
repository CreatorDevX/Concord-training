"""
HybridMoE — Full 3B Hybrid MoE Model.

Architecture:
    embed → 6 groups × (3 DeltaBlocks + 1 AttnBlock) → RMSNorm → lm_head → MTP

Features:
    - Tied embeddings (embed ↔ lm_head)
    - Block AttnRes for depth-wise residual selection
    - Expert weight sharing within groups (configurable)
    - Selective token loss masking
    - Jinja2 template for pretraining formatting
    - Gradient checkpointing support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, List, Tuple
import math

from model.config import ModelConfig
from model.components.rms_norm import RMSNorm
from model.components.attn_res import BlockAttnRes
from model.components.mtp import MTPHead
from model.components.moe import MoEBlock
from model.blocks.delta_block import DeltaBlock
from model.blocks.attn_block import AttnBlock


class HybridMoE(nn.Module):
    """
    Full 3B Hybrid MoE model.

    Architecture:
        - Embedding layer (vocab_size × d_model)
        - 24 layers in 6 groups of 4:
            - 3 DeltaNet blocks (linear recurrent attention)
            - 1 Attention block (CSA or HCA, alternating)
        - Block AttnRes (depth-wise softmax residuals)
        - RMSNorm final
        - LM head (tied to embeddings)
        - MTP head (multi-token prediction)

    When share_experts_within_group=True:
        All 4 layers in each group share the same expert pool.
        Each layer keeps its own router (so routing can differ per depth).
        Expert weights: 6 groups × 192 experts (instead of 24 layers × 192).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ── Embedding ───────────────────────────────────────────────────
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # ── Block Attention Residuals (shared across all layers) ────────
        self.attn_res = BlockAttnRes(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_blocks=config.n_groups,
        )

        # ── Shared MoE pools (when share_experts_within_group=True) ────
        self.group_moes = None
        if config.share_experts_within_group:
            # One MoE pool per group. Router is inside MoE but each block
            # also has its own routing via the shared MoE's router call.
            # Actually — we share expert weights, but each block gets its
            # own router. So we create shared expert sets, pass them into blocks.
            self.group_moes = nn.ModuleList()
            for g in range(config.n_groups):
                # Delta layers use n_routed_delta, attn layer uses n_routed_attn
                # For sharing, we create a "base" MoE with max routing width
                # and each block type uses its own top-k from the same experts.
                # Simplest correct approach: create the MoE with shared experts,
                # blocks just reference it.
                self.group_moes.append(MoEBlock(
                    d_model=config.d_model,
                    n_experts=config.n_experts,
                    n_routed=config.n_routed_delta,  # base routing width
                    n_shared=config.n_shared,
                    expert_intermediate=config.expert_intermediate,
                    router_hidden=config.router_hidden,
                    router_bias_update_interval=config.router_bias_update_interval,
                    expert_dtype=config.expert_dtype,
                ))

        # ── Blocks ──────────────────────────────────────────────────────
        self.blocks = nn.ModuleList()
        for g in range(config.n_groups):
            shared_moe = self.group_moes[g] if self.group_moes is not None else None

            # 3 DeltaBlocks per group
            for i in range(config.group_size - 1):
                layer_idx = g * config.group_size + i
                self.blocks.append(DeltaBlock(
                    config, layer_idx=layer_idx,
                    attn_res=self.attn_res,
                    shared_moe=shared_moe,
                ))
            # 1 AttnBlock per group
            attn_layer_idx = g * config.group_size + (config.group_size - 1)
            self.blocks.append(AttnBlock(
                config,
                layer_idx=attn_layer_idx,
                attn_idx=g,
                attn_res=self.attn_res,
                shared_moe=shared_moe,
            ))

        # ── Final norm ──────────────────────────────────────────────────
        self.norm = RMSNorm(config.d_model)

        # ── Vision encoders (M1/M2/M3 Stages) ──────────────────────────
        from model.components.vision import VisionConnector, TemporalVideoEncoder
        self.vision_connector = VisionConnector(config, d_vision=1024)
        self.video_encoder = TemporalVideoEncoder(config, d_vision=1024)

        # ── LM Head ────────────────────────────────────────────────────
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # ── MTP Head ───────────────────────────────────────────────────
        self.mtp = MTPHead(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            mtp_steps=config.mtp_steps,
            tie_output=config.mtp_tie_output,
            embed_weight=self.embed.weight if config.tie_embeddings else None,
        )

        # ── Jinja template ─────────────────────────────────────────────
        self._jinja_template = None
        if config.jinja_template_path:
            self._load_jinja_template(config.jinja_template_path)

        # ── Initialize weights ─────────────────────────────────────────
        self.apply(self._init_weights)

        if config.vision_weights_path:
            self.load_vision_weights(config.vision_weights_path)

    def load_vision_weights(self, path: str):
        from safetensors.torch import load_file
        state_dict = load_file(path) if path.endswith(".safetensors") else torch.load(path)
        # Assuming the weights are prefixed properly, we extract vision subset or let load_state_dict handle strict=False
        self.video_encoder.load_state_dict({k.replace('video_encoder.', ''): v for k, v in state_dict.items() if k.startswith('video_encoder.')}, strict=False)
        self.vision_connector.load_state_dict({k.replace('vision_connector.', ''): v for k, v in state_dict.items() if k.startswith('vision_connector.')}, strict=False)
        print(f"Loaded TIPSv2 vision weights from {path}")

    def save_vision_weights(self, path: str):
        from safetensors.torch import save_file
        vision_state = {}
        for k, v in self.video_encoder.state_dict().items(): vision_state[f'video_encoder.{k}'] = v
        for k, v in self.vision_connector.state_dict().items(): vision_state[f'vision_connector.{k}'] = v
        save_file(vision_state, path)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _load_jinja_template(self, path: str):
        """Load Jinja2 template for pretraining data formatting."""
        try:
            from jinja2 import Template
            with open(path, 'r') as f:
                self._jinja_template = Template(f.read())
        except ImportError:
            print("Warning: jinja2 not installed. Template formatting disabled.")
        except FileNotFoundError:
            print(f"Warning: Template file not found: {path}")

    def render_template(self, **kwargs) -> str:
        """Render the Jinja template with given variables."""
        if self._jinja_template is None:
            raise RuntimeError("No Jinja template loaded")
        return self._jinja_template.render(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        delta_states: Optional[List[torch.Tensor]] = None,
        image_patches: Optional[torch.Tensor] = None,
        video_patches: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (B, T) input token IDs
            labels: (B, T) target token IDs for loss computation
            loss_mask: (B, T) mask for selective loss (1=compute loss, 0=skip)
            position_ids: (B, T) position IDs for RoPE
            delta_states: optional list of DeltaNet recurrent states

        Returns:
            dict with keys:
                - logits: (B, T, vocab_size)
                - loss: scalar (if labels provided)
                - main_loss: scalar (if labels provided)
                - mtp_loss: scalar (if labels provided)
                - delta_states: list of updated DeltaNet states
        """
        B, T = input_ids.shape

        # ── Embed ──────────────────────────────────────────────────────
        x = self.embed(input_ids)  # (B, T, d_model)

        # ── Multimodal Prefix Injection (Stage M2/M3) ───────────────────
        visual_tokens = None
        if image_patches is not None:
            visual_tokens = self.vision_connector(image_patches)
        elif video_patches is not None:
            visual_tokens = self.video_encoder(video_patches)
            
        if visual_tokens is not None:
            # Prepend visual context features to text sequence
            x = torch.cat([visual_tokens, x], dim=1)
            visual_len = visual_tokens.shape[1]
            B_vis = x.shape[0]
            
            # Align labels and loss_mask by padding visual tokens with ignore/zero
            if labels is not None:
                visual_labels = torch.full((B_vis, visual_len), -100, device=labels.device, dtype=labels.dtype)
                labels = torch.cat([visual_labels, labels], dim=1)
            if loss_mask is not None:
                v_mask = torch.zeros((B_vis, visual_len), device=loss_mask.device, dtype=loss_mask.dtype)
                loss_mask = torch.cat([v_mask, loss_mask], dim=1)

        # ── Block processing with AttnRes ──────────────────────────────
        block_reprs: List[torch.Tensor] = []
        partial_residual = torch.zeros_like(x)
        new_delta_states = []
        delta_state_idx = 0

        for i, block in enumerate(self.blocks):
            if isinstance(block, DeltaBlock):
                ds = None
                if delta_states is not None and delta_state_idx < len(delta_states):
                    ds = delta_states[delta_state_idx]

                if self.config.grad_checkpoint and self.training:
                    def create_delta_forward(blk, blk_reprs, ds_val):
                        def custom_forward(x_in, partial_res):
                            return blk(x_in, blk_reprs, partial_res, ds_val)
                        return custom_forward
                    x, partial_residual, ds_out = checkpoint(
                        create_delta_forward(block, block_reprs, ds),
                        x, partial_residual,
                        use_reentrant=False,
                    )
                else:
                    x, partial_residual, ds_out = block(x, block_reprs, partial_residual, ds)

                new_delta_states.append(ds_out)
                delta_state_idx += 1

            elif isinstance(block, AttnBlock):
                if self.config.grad_checkpoint and self.training:
                    def create_attn_forward(blk, blk_reprs):
                        def custom_forward(x_in, partial_res):
                            return blk(x_in, blk_reprs, partial_res, position_ids)
                        return custom_forward
                    x, partial_residual = checkpoint(
                        create_attn_forward(block, block_reprs),
                        x, partial_residual,
                        use_reentrant=False,
                    )
                else:
                    x, partial_residual = block(x, block_reprs, partial_residual, position_ids)

            # End of group: save block representation
            if (i + 1) % self.config.group_size == 0:
                block_reprs.append(partial_residual.detach())
                partial_residual = torch.zeros_like(x)

        # ── Final norm + LM head ───────────────────────────────────────
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        result = {"logits": logits, "delta_states": new_delta_states}

        # ── Loss computation ───────────────────────────────────────────
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            B_logits, T_logits, _ = shift_logits.shape
            main_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                reduction="none",
            ).view(B_logits, T_logits)

            if loss_mask is not None and self.config.selective_loss:
                shift_mask = loss_mask[:, 1:].contiguous()
                main_loss = (main_loss * shift_mask).sum() / (shift_mask.sum() + 1e-10)
            else:
                main_loss = main_loss.mean()

            mtp_mask = loss_mask if (loss_mask is not None and self.config.selective_loss) else None
            mtp_loss = self.mtp(x, labels, self.embed.weight, mtp_mask)

            total_loss = main_loss + self.config.mtp_weight * mtp_loss

            result["loss"] = total_loss
            result["main_loss"] = main_loss
            result["mtp_loss"] = mtp_loss

        return result

    def get_param_groups(self) -> List[dict]:
        """
        Get parameter groups for mixed optimizer training.

        Returns three groups:
            1. Expert FFN weights → SGD
            2. Non-expert non-embedding weights → Muon
            3. Embeddings, norms, router, MTP, AttnRes → AdamW
        """
        expert_params = []
        muon_params = []
        adamw_params = []

        adamw_keywords = {"embed", "norm", "mtp", "router", "pseudo_queries",
                          "expert_bias", "bias"}
        expert_keywords = {"routed_experts"}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            is_expert = any(k in name for k in expert_keywords)
            is_adamw = any(k in name for k in adamw_keywords)

            if is_expert and not is_adamw:
                expert_params.append(param)
            elif is_adamw:
                adamw_params.append(param)
            else:
                muon_params.append(param)

        return [
            {
                "name": "expert_sgd",
                "params": expert_params,
                "optimizer": "sgd",
                "lr": 9e-4,
                "weight_decay": 0.1,
                "momentum": 0.0,
            },
            {
                "name": "muon",
                "params": muon_params,
                "optimizer": "muon",
                "lr": 3e-4,
                "weight_decay": 0.01,
            },
            {
                "name": "adamw",
                "params": adamw_params,
                "optimizer": "adamw",
                "lr": 1e-4,
                "betas": (0.9, 0.95),
                "weight_decay": 0.1,
            },
        ]

    def sync_fp8_weights(self):
        """Sync all FP8 expert weight shadows after optimizer step."""
        if self.group_moes is not None:
            for moe in self.group_moes:
                moe.sync_fp8_weights()
        else:
            for block in self.blocks:
                if hasattr(block, 'moe') and block._owns_moe:
                    block.moe.sync_fp8_weights()

    def maybe_update_router_biases(self):
        """Update router biases across all blocks if scheduled."""
        if self.group_moes is not None:
            for moe in self.group_moes:
                moe.maybe_update_router_bias()
        else:
            for block in self.blocks:
                if hasattr(block, 'moe') and block._owns_moe:
                    block.moe.maybe_update_router_bias()

    def stats(self) -> dict:
        """Aggregate stats from all components."""
        block_stats = []
        for i, block in enumerate(self.blocks):
            block_stats.append({
                "block_idx": i,
                "type": block.__class__.__name__,
                **block.stats(),
            })

        return {
            "attn_res": self.attn_res.stats(),
            "mtp": self.mtp.stats(),
            "final_norm": self.norm.stats(),
            "n_blocks": len(self.blocks),
            "share_experts": self.config.share_experts_within_group,
            "blocks": block_stats,
        }
