"""
ModelConfig — all hyperparameters for the 3B Hybrid MoE.

Constraints solved simultaneously:
    Total ≈ 3.0B  |  Active (MoE budget) ≈ 100M
    With n_routed_delta=2, n_routed_attn=4:
        active_invocations = 18×2 + 6×4 = 60 per forward pass
        84 × single_expert ≈ 88M → expert_intermediate = 352
        routed_total ≈ 2.07B → n_experts = 80
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # ── Dimensions ──────────────────────────────────────────────────────
    d_model: int = 1024
    vocab_size: int = 262189  # updated with the custom tokenizer

    # ── Layers ──────────────────────────────────────────────────────────
    n_layers: int = 24
    group_size: int = 4           # 3 DeltaNet + 1 Attention per group
    n_groups: int = 6             # 6 × 4 = 24 layers

    # ── DeltaNet ────────────────────────────────────────────────────────
    # d_qk = d_v = 16 × 64 = 1024 = d_model
    delta_v_heads: int = 16
    delta_qk_heads: int = 16
    delta_head_dim: int = 64

    # ── Full Attention ──────────────────────────────────────────────────
    # d_q = 8 × 128 = 1024 = d_model  |  d_kv = 2 × 128 = 256
    attn_q_heads: int = 8
    attn_kv_heads: int = 2
    attn_head_dim: int = 128
    rope_dim: int = 64            # partial RoPE on first 64 of 128 dims

    # ── Attention types ─────────────────────────────────────────────────
    csa_top_k: int = 1024
    csa_window: int = 128
    csa_compress: int = 4
    hca_compress: int = 128

    # ── MoE ─────────────────────────────────────────────────────────────
    # Solved: 80 experts × 24 layers × 1,081,344 per expert = 2.08B routed
    # Active: (18×2 + 6×4) = 60 invocations × 1.08M/expert = 64.9M
    #       + 24 shared × 1.08M = 25.9M  +  router/attnres ~12M  ≈ 103M
    n_experts: int = 96
    n_routed_delta: int = 2       # active routed experts per DeltaNet layer
    n_routed_attn: int = 4        # active routed experts per Attention layer
    n_shared: int = 1
    expert_intermediate: int = 384  # SwiGLU: 3 × 1024 × 352 = 1.08M per expert

    # ── Router ──────────────────────────────────────────────────────────
    router_hidden: int = 256
    router_bias_update_interval: int = 750

    # ── Block Attention Residuals ───────────────────────────────────────
    n_attnres_blocks: int = 6

    # ── MTP ─────────────────────────────────────────────────────────────
    mtp_steps: int = 2
    mtp_weight: float = 0.1
    mtp_tie_output: bool = True  # per-step output projections (untied)

    # ── Memory optimizations ────────────────────────────────────────────
    expert_dtype: str = "fp8"     # fp8 storage, bf16 compute
    use_expert_sgd: bool = True   # SGD for experts, Muon for rest
    grad_checkpoint: bool = True
    grad_offload_cpu: bool = True
    share_experts_within_group: bool = False

    # ── Training ────────────────────────────────────────────────────────
    tie_embeddings: bool = True
    train_seq_len_stage3: int = 4096
    max_seq_len: int = 8192

    # ── Selective loss & templating ─────────────────────────────────────
    selective_loss: bool = True
    jinja_template_path: Optional[str] = None
    corpus_tokens: int = 2048
    vision_weights_path: Optional[str] = None

    # ── DeltaNet chunk size for parallel scan ───────────────────────────
    delta_chunk_size: int = 128

    # ── Derived (computed post-init) ────────────────────────────────────
    def __post_init__(self):
        assert self.n_layers == self.n_groups * self.group_size, (
            f"n_layers ({self.n_layers}) must equal "
            f"n_groups ({self.n_groups}) × group_size ({self.group_size})"
        )
        self.delta_d_v = self.delta_v_heads * self.delta_head_dim
        self.delta_d_qk = self.delta_qk_heads * self.delta_head_dim
        self.attn_d_q = self.attn_q_heads * self.attn_head_dim
        self.attn_d_kv = self.attn_kv_heads * self.attn_head_dim

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        import dataclasses
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})
