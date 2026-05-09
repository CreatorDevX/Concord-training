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
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.embed_up = nn.Sequential(
            nn.Linear(config.embed_dim, config.d_model, bias=False),
            nn.SiLU()
        )

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
                    use_activation_recomputation=not config.grad_checkpoint,
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
        if getattr(config, 'use_vision', True):
            from model.components.vision import VisionConnector, TemporalVideoEncoder
            self.vision_connector = VisionConnector(config, d_vision=1024)
            self.video_encoder = TemporalVideoEncoder(config, d_vision=1024)
        else:
            self.vision_connector = None
            self.video_encoder = None

        # ── Transpose projection and LM Head ────────────────────────────
        self.embed_down = nn.Linear(config.d_model, config.embed_dim, bias=False)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # ── AgentRoPE ───────────────────────────────────────────────────
        from model.components.rope import TemporalMultimodalRoPE
        self.agent_rope = TemporalMultimodalRoPE(
            axis_dims=config.rope_axis_dims,
            wallclock_scale_seconds=60.0,
            use_log_wallclock=True,
        )

        # ── MTP Head ───────────────────────────────────────────────────
        self.mtp = MTPHead(
            d_model=config.d_model,
            embed_dim=config.embed_dim,
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

        if getattr(config, 'use_vision', True) and config.vision_weights_path:
            self.load_vision_weights(config.vision_weights_path)

    def load_vision_weights(self, path: str):
        if not getattr(self.config, 'use_vision', True): return
        import os
        if not os.path.exists(path):
            print(f"Vision weights '{path}' not found locally. Attempting to download from HuggingFace Hub...")
            try:
                from huggingface_hub import hf_hub_download
                try:
                    path = hf_hub_download(repo_id=path, filename="model.safetensors")
                except Exception:
                    path = hf_hub_download(repo_id=path, filename="pytorch_model.bin")
            except ImportError:
                print("huggingface_hub is not installed! Run `pip install huggingface_hub`.")
                return
            except Exception as e:
                print(f"Failed to fetch {path} from HF: {e}")
                return

        from safetensors.torch import load_file
        state_dict = load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu", weights_only=False)
        # Assuming the weights are prefixed properly, we extract vision subset or let load_state_dict handle strict=False
        self.video_encoder.load_state_dict({k.replace('video_encoder.', ''): v for k, v in state_dict.items() if k.startswith('video_encoder.')}, strict=False)
        self.vision_connector.load_state_dict({k.replace('vision_connector.', ''): v for k, v in state_dict.items() if k.startswith('vision_connector.')}, strict=False)
        print(f"Loaded TIPSv2 vision weights from {path}")

    def save_vision_weights(self, path: str):
        if not getattr(self.config, 'use_vision', True): return
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
        timestamps_ms: Optional[torch.Tensor] = None,
        spatial_coords: Optional[Dict[str, torch.Tensor]] = None,
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
        x = self.embed(input_ids)  # (B, T, embed_dim)
        x = self.embed_up(x)       # (B, T, d_model)

        # ── Multimodal Prefix Injection (Stage M2/M3) ───────────────────
        visual_tokens = None
        if getattr(self.config, 'use_vision', True):
            if image_patches is not None and self.vision_connector is not None:
                visual_tokens = self.vision_connector(image_patches)
            elif video_patches is not None and self.video_encoder is not None:
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

        # ── Prepare AgentRoPE Coords ──────────────────────────────────
        coords = {}
        # 1. Local sequential 'u'
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        coords["u"] = position_ids

        # 2. Wallclock 'w'
        if timestamps_ms is not None:
            # Use the last token of the first batch element as reference if not provided
            ref_ms = timestamps_ms[:, -1:]
            coords["w"] = self.agent_rope.encode_wallclock(timestamps_ms, ref_ms)
        else:
            # Fallback to zeros if no timestamps provided
            coords["w"] = torch.zeros_like(position_ids).float()

        # 3. Spatial 'x', 'y'
        if spatial_coords is not None:
            coords["x"] = spatial_coords.get("x", torch.zeros_like(position_ids).float())
            coords["y"] = spatial_coords.get("y", torch.zeros_like(position_ids).float())
        else:
            coords["x"] = torch.zeros_like(position_ids).float()
            coords["y"] = torch.zeros_like(position_ids).float()

        # ── Block processing with AttnRes ──────────────────────────────
        block_reprs: List[torch.Tensor] = []
        partial_residual = torch.zeros_like(x)
        new_delta_states = []
        delta_state_idx = 0
        total_aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)

        for i, block in enumerate(self.blocks):
            if isinstance(block, DeltaBlock):
                ds = None
                if delta_states is not None and delta_state_idx < len(delta_states):
                    ds = delta_states[delta_state_idx]

                if self.config.grad_checkpoint and self.training:
                    def delta_custom_forward(x_in, partial_res, blk, blk_reprs, ds_val):
                        return blk(x_in, blk_reprs, partial_res, ds_val)
                    x, partial_residual, ds_out, aux_loss = checkpoint(
                        delta_custom_forward, x, partial_residual, block, list(block_reprs), ds,
                        use_reentrant=False,
                    )
                else:
                    x, partial_residual, ds_out, aux_loss = block(x, block_reprs, partial_residual, ds)

                new_delta_states.append(ds_out)
                delta_state_idx += 1
                total_aux_loss = total_aux_loss + aux_loss

            elif isinstance(block, AttnBlock):
                if self.config.grad_checkpoint and self.training:
                    def attn_custom_forward(x_in, partial_res, blk, blk_reprs, pos_ids, coords_dict):
                        return blk(x_in, blk_reprs, partial_res, pos_ids, coords_dict)
                    x, partial_residual, aux_loss = checkpoint(
                        attn_custom_forward, x, partial_residual, block, list(block_reprs), position_ids, coords,
                        use_reentrant=False,
                    )
                else:
                    x, partial_residual, aux_loss = block(x, block_reprs, partial_residual, position_ids, coords)
                
                total_aux_loss = total_aux_loss + aux_loss

            # End of group: save block representation
            if (i + 1) % self.config.group_size == 0:
                # Sanitize before storing: NaN in block_reprs poisons all subsequent _attn_res calls
                if partial_residual.dtype == torch.float16 or partial_residual.dtype == torch.bfloat16:
                    partial_residual = torch.where(
                        torch.isnan(partial_residual) | torch.isinf(partial_residual),
                        torch.zeros_like(partial_residual), partial_residual
                    )
                block_reprs.append(partial_residual.detach())
                partial_residual = torch.zeros_like(x)

        # ── Final norm ───────────────────────────────────────
        x = self.norm(x)

        # Sanitize hidden state: if any layer produced NaN (from fp16 overflow),
        # zero it here to prevent total_loss from becoming NaN.
        if self.training:
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x = torch.where(torch.isinf(x), torch.zeros_like(x), x)

        result = {"delta_states": new_delta_states}
        
        # ── Loss computation ───────────────────────────────────────────
        if labels is not None:
            # Find indices where we actually want to compute loss (t+1 must be in mask)
            if loss_mask is not None and self.config.selective_loss:
                active_mask = (loss_mask[:, 1:] == 1).reshape(-1)
            else:
                active_mask = torch.ones(x.shape[0] * (x.shape[1] - 1), dtype=torch.bool, device=x.device)

            # Slice hidden states and labels BEFORE massive vocab expansion
            x_shift = x[:, :-1, :].reshape(-1, x.shape[-1])[active_mask]
            labels_shift = labels[:, 1:].reshape(-1)[active_mask]
            
            if x_shift.shape[0] > 0:
                chunk_size = 128
                main_loss = 0.0
                total_tokens = x_shift.shape[0]
                for chunk_start in range(0, total_tokens, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_tokens)
                    x_chunk = x_shift[chunk_start:chunk_end]
                    y_chunk = labels_shift[chunk_start:chunk_end]
                    
                    # Checkpoint the logit expanding and cross entropy.
                    # This prevents Autograd from keeping 1GB of FP32 softmax probabilities
                    # in memory for *each chunk* concurrently before backward!
                    def compute_chunk_loss(xc, yc):
                        x_down_chunk = self.embed_down(xc)
                        logits_chunk = self.lm_head(x_down_chunk)
                        logits_chunk = logits_chunk.float()
                        # Clamp to fp16-safe range BEFORE softmax so gradients
                        # flow cleanly (nan_to_num would zero them at Inf points).
                        # nan_to_num stays as a last-resort guard for residual NaN.
                        logits_chunk = logits_chunk.clamp(min=-65504.0, max=65504.0)
                        logits_chunk = torch.nan_to_num(logits_chunk, nan=0.0)
                        return F.cross_entropy(logits_chunk, yc, reduction="sum", ignore_index=-100)
                    
                    if x_chunk.requires_grad:
                        loss_chunk = torch.utils.checkpoint.checkpoint(
                            compute_chunk_loss, x_chunk, y_chunk, use_reentrant=False
                        )
                    else:
                        loss_chunk = compute_chunk_loss(x_chunk, y_chunk)
                        
                    main_loss = main_loss + loss_chunk
                
                main_loss = main_loss / total_tokens
            else:
                main_loss = torch.tensor(0.0, device=x.device)

            mtp_mask = loss_mask if (loss_mask is not None and self.config.selective_loss) else None
            mtp_loss = self.mtp(x, labels, loss_mask=mtp_mask, embed_down=self.embed_down)
            
            # total_loss = main + mtp + aux
            # NOTE: aux_loss_coeff is already applied inside router.py — do NOT re-apply here
            total_loss = main_loss + self.config.mtp_weight * mtp_loss + total_aux_loss
            
            result["loss"] = total_loss
            result["main_loss"] = main_loss
            result["mtp_loss"] = mtp_loss
            result["aux_loss"] = total_aux_loss
        else:
            # Inference: compute all logits
            x_down = self.embed_down(x)
            logits = self.lm_head(x_down)
            result["logits"] = logits

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

        # Track seen tensor pointers to avoid duplicating tied embeddings
        seen_params = set()

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Skip duplicates of tied embeddings (lm_head.weight ties to embed.weight,
            # mtp.output_proj.weight ties to embed.weight). Both are the same tensor
            # object — adding them to separate optimizer groups causes DOUBLE UPDATES.
            if id(param) in seen_params:
                continue
            seen_params.add(id(param))

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
                "name": "lion",
                "params": adamw_params,
                "optimizer": "lion",
                "lr": 1e-4,
                "betas": (0.9, 0.99),
                "weight_decay": 0.1,
            },
        ]

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
