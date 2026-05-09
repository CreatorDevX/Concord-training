"""
Memory Audit — Pure math, no model instantiation.

Estimates GPU memory for each component and validates <3.5GB budget.
Run as: python -m model.memory_audit (from parent dir)
"""

from model.config import ModelConfig


def memory_audit(config: ModelConfig = None, batch_size: int = 1) -> dict:
    """Estimate training memory budget using pure arithmetic. Zero GPU memory."""
    if config is None:
        config = ModelConfig()

    d = config.d_model
    V = config.vocab_size
    n_layers = config.n_layers
    n_groups = config.n_groups
    group_size = config.group_size
    n_delta_layers = n_groups * (group_size - 1)
    n_attn_layers = n_groups
    seq_len = 2080

    single_expert_params = 3 * d * config.expert_intermediate

    # ── Expert weight storage ──────────────────────────────────────────
    if config.share_experts_within_group:
        n_unique_expert_sets = n_groups
    else:
        n_unique_expert_sets = n_layers

    total_expert_params = config.n_experts * n_unique_expert_sets * single_expert_params

    expert_weight_bytes = total_expert_params * 2
    expert_master_bytes = 0

    # ── Non-expert weights ─────────────────────────────────────────────
    dqk = config.delta_d_qk
    dv = config.delta_d_v
    aq = config.attn_d_q
    akv = config.attn_d_kv

    # Group 1: "Muon" group -> Adafactor (factored O(m+n))
    # Attention and DeltaNet projections
    n_csa = (n_attn_layers + 1) // 2
    n_hca = n_attn_layers // 2
    compressor_params = (
        n_csa * config.attn_head_dim * config.csa_compress * config.attn_head_dim +
        n_hca * config.attn_head_dim * config.hca_compress * config.attn_head_dim
    )
    delta_params = (d*dqk + d*dqk + d*dv + dv*d + d*dv + d*config.delta_qk_heads) * n_delta_layers
    attn_params = (d*aq + d*akv + d*akv + aq*d + d*aq) * n_attn_layers
    muon_group_params = delta_params + attn_params + compressor_params

    # Group 2: "Lion" group -> Lion (1 moment, same dtype as param)
    # Embeddings, Norms, Router, MTP, AttnRes, Biases
    embed_params = V * config.embed_dim
    embed_proj_params = config.embed_dim * d + d * config.embed_dim
    router_params = (d * config.router_hidden + config.router_hidden * config.n_experts) * n_layers
    norm_params = (2 * d * n_layers + d) + (n_delta_layers * d + n_attn_layers * d) # layer norms + delta norms
    attn_res_params = n_layers * d + d * d + d
    mtp_params = config.mtp_steps * (2*d*d + d*d) + d*d # MLP + embed_proj
    # Biases are tiny, ignoring.
    lion_group_params = embed_params + embed_proj_params + router_params + norm_params + attn_res_params + mtp_params + (config.n_shared * n_layers * single_expert_params)

    non_expert_params = muon_group_params + lion_group_params
    non_expert_weight_bytes = non_expert_params * 2  # FP16

    # ── Optimizer states ───────────────────────────────────────────────
    expert_optim_bytes = 0  # SGD = zero optimizer state

    # Adafactor for muon_group (factored O(m+n))
    # Approximate: for each layer's 5-6 matrices, store ~2*d fp32 values
    n_ada_matrices = (5 * n_delta_layers + 5 * n_attn_layers + n_layers)
    adafactor_bytes = n_ada_matrices * (2 * d) * 4 # fp32 state

    # Lion for lion_group (1 moment, same dtype as param)
    lion_optim_bytes = lion_group_params * 2 # fp16 moment

    non_expert_optim_bytes = adafactor_bytes + lion_optim_bytes

    # ── Gradients (FP16, freed after optimizer step via set_to_none) ───
    grad_bytes = (total_expert_params + non_expert_params) * 2

    # ── Activations (grad checkpointing: only block boundaries) ────────
    # With grad_checkpoint: store input to each block + block_reprs at group boundaries
    if config.grad_checkpoint:
        # Input/output per block: B × T × d × 2 bytes, re-computed during backward
        # Only block boundary checkpoints + partial_residual kept
        activation_bytes = batch_size * seq_len * d * 2 * (n_groups + 2)
    else:
        activation_bytes = batch_size * seq_len * d * 2 * n_layers * 2

    # ── DeltaNet recurrent state (tiny — no KV cache!) ─────────────────
    head_dim_v = dv // config.delta_v_heads
    delta_state_bytes = (
        batch_size * config.delta_qk_heads *
        config.delta_head_dim * head_dim_v *
        n_delta_layers * 2
    )

    # ── KV cache: ONLY attention layers, compressed ────────────────────
    # DeltaNet has NO KV cache (linear recurrent). Only the 4 attention
    # layers have KV, and they use compressed KV (CSA=4×, HCA=128×).
    # These are transient during forward, not stored between steps.
    # At training time with grad_checkpoint, they're recomputed.
    kv_cache_bytes = 0  # Transient only, not persistent

    # ── CUDA allocator overhead ────────────────────────────────────────
    cuda_overhead_bytes = int(0.3 * 1024**3)  # ~300MB typical fragmentation

    # ── Total ──────────────────────────────────────────────────────────
    total_bytes = (
        expert_weight_bytes +
        non_expert_weight_bytes +
        expert_optim_bytes + non_expert_optim_bytes +
        grad_bytes + activation_bytes +
        delta_state_bytes + kv_cache_bytes +
        cuda_overhead_bytes
    )
    # Note: expert_master_bytes are on CPU, not counted in GPU budget
    total_gb = total_bytes / (1024 ** 3)
    budget = 3.5

    result = {
        "expert_weights_gb": expert_weight_bytes / (1024**3),
        "expert_masters_cpu_gb": expert_master_bytes / (1024**3),
        "non_expert_weights_gb": non_expert_weight_bytes / (1024**3),
        "expert_optim_gb": expert_optim_bytes / (1024**3),
        "non_expert_optim_gb": non_expert_optim_bytes / (1024**3),
        "gradients_gb": grad_bytes / (1024**3),
        "activations_gb": activation_bytes / (1024**3),
        "delta_state_gb": delta_state_bytes / (1024**3),
        "kv_cache_gb": kv_cache_bytes / (1024**3),
        "cuda_overhead_gb": cuda_overhead_bytes / (1024**3),
        "total_gb": total_gb,
        "within_budget": total_gb < budget,
    }

    print("=" * 60)
    print(f"  Memory Audit (pure math) -- batch={batch_size}, seq={seq_len}")
    print(f"  d_model={d}, n_layers={n_layers}, experts={config.n_experts}")
    print(f"  Expert dtype={config.expert_dtype}, grad_ckpt={config.grad_checkpoint}")
    print(f"  Optimizer: Adafactor (factored O(m+n) for 2D params)")
    print(f"  share_experts={config.share_experts_within_group}")
    print("=" * 60)
    print()
    for key, val in result.items():
        if key == "within_budget":
            continue
        label = key.replace("_gb", "").replace("_", " ").title()
        suffix = " (CPU)" if "cpu" in key else ""
        print(f"  {label:30s} {val:8.3f} GB{suffix}")
    print(f"  {'-' * 40}")
    print(f"  {'TOTAL (GPU)':30s} {total_gb:8.3f} GB")
    print()
    print(f"  Gate -- <{budget}GB budget:  {'PASS' if total_gb < budget else 'FAIL'}")
    print(f"  Headroom: {budget - total_gb:.3f} GB")
    print("=" * 60)

    return result

if __name__ == "__main__":
    memory_audit()
