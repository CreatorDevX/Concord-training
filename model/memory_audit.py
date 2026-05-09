"""
Memory Audit — Pure math, no model instantiation.

Estimates GPU memory for each component and validates <15GB budget.
Run as: python -m model.memory_audit (from parent dir)
"""

from model.config import ModelConfig


def memory_audit(config: ModelConfig = None, batch_size: int = 4) -> dict:
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
    seq_len = config.train_seq_len_stage1

    single_expert_params = 3 * d * config.expert_intermediate

    # ── Expert weight storage ──────────────────────────────────────────
    if config.share_experts_within_group:
        n_unique_expert_sets = n_groups
    else:
        n_unique_expert_sets = n_layers

    total_expert_params = config.n_experts * n_unique_expert_sets * single_expert_params

    if config.expert_dtype == "fp8":
        expert_weight_bytes = total_expert_params * 1   # FP8 = 1 byte
        expert_master_bytes = total_expert_params * 2   # BF16 master copies for grad
    else:
        expert_weight_bytes = total_expert_params * 2
        expert_master_bytes = 0

    # ── Non-expert weights ─────────────────────────────────────────────
    dqk = config.delta_d_qk
    dv = config.delta_d_v
    aq = config.attn_d_q
    akv = config.attn_d_kv

    delta_params = (d*dqk + d*dqk + d*dv + dv*d + d*dv + d*config.delta_qk_heads + d) * n_delta_layers
    attn_params = (d*aq + d*akv + d*akv + aq*d + d*aq + d) * n_attn_layers

    n_csa = (n_attn_layers + 1) // 2
    n_hca = n_attn_layers // 2
    compressor_params = (
        n_csa * config.attn_head_dim * config.csa_compress * config.attn_head_dim +
        n_hca * config.attn_head_dim * config.hca_compress * config.attn_head_dim
    )

    shared_expert_params = config.n_shared * n_layers * single_expert_params
    router_params = (d * config.router_hidden + config.router_hidden * config.n_experts + d) * n_layers
    norm_params = 2 * d * n_layers + d
    attn_res_params = n_layers * d + d * d + d
    embed_params = V * d
    mtp_params = config.mtp_steps * (2*d + 2*d*d + d + d*d) + d*d

    non_expert_params = (
        embed_params + delta_params + attn_params + compressor_params +
        shared_expert_params + router_params + norm_params +
        attn_res_params + mtp_params
    )
    non_expert_weight_bytes = non_expert_params * 2  # BF16

    # ── Optimizer states ───────────────────────────────────────────────
    expert_optim_bytes = 0  # SGD = zero optimizer state

    # Non-expert: ~60% Muon (1 FP32 moment), ~40% AdamW (2 FP32 moments)
    non_expert_optim_bytes = int(
        non_expert_params * 0.6 * 4 +   # Muon: 1 moment × 4 bytes
        non_expert_params * 0.4 * 8     # AdamW: 2 moments × 4 bytes
    )

    # ── Gradients ──────────────────────────────────────────────────────
    if config.grad_offload_cpu:
        active_expert_params = (
            (config.n_routed_delta * n_delta_layers + config.n_routed_attn * n_attn_layers) *
            single_expert_params
        )
        grad_bytes = (active_expert_params + non_expert_params) * 2 // 16
    else:
        grad_bytes = (total_expert_params + non_expert_params) * 2

    # ── Activations ────────────────────────────────────────────────────
    if config.grad_checkpoint:
        activation_bytes = batch_size * seq_len * d * 2 * 4
    else:
        activation_bytes = batch_size * seq_len * d * 2 * n_layers * 2

    # ── DeltaNet recurrent state ───────────────────────────────────────
    head_dim_v = dv // config.delta_v_heads
    delta_state_bytes = (
        batch_size * config.delta_qk_heads *
        config.delta_head_dim * head_dim_v *
        n_delta_layers * 2
    )

    # ── KV cache ───────────────────────────────────────────────────────
    kv_cache_bytes = (
        batch_size * n_attn_layers * 2 *
        config.attn_kv_heads * config.attn_head_dim *
        seq_len * 2
    )

    # ── Total ──────────────────────────────────────────────────────────
    total_bytes = (
        expert_weight_bytes + expert_master_bytes +
        non_expert_weight_bytes +
        expert_optim_bytes + non_expert_optim_bytes +
        grad_bytes + activation_bytes +
        delta_state_bytes + kv_cache_bytes
    )
    total_gb = total_bytes / (1024 ** 3)

    result = {
        "expert_weights_gb": expert_weight_bytes / (1024**3),
        "expert_masters_gb": expert_master_bytes / (1024**3),
        "non_expert_weights_gb": non_expert_weight_bytes / (1024**3),
        "expert_optim_gb": expert_optim_bytes / (1024**3),
        "non_expert_optim_gb": non_expert_optim_bytes / (1024**3),
        "gradients_gb": grad_bytes / (1024**3),
        "activations_gb": activation_bytes / (1024**3),
        "delta_state_gb": delta_state_bytes / (1024**3),
        "kv_cache_gb": kv_cache_bytes / (1024**3),
        "total_gb": total_gb,
        "within_budget": total_gb < 15.0,
    }

    print("=" * 60)
    print(f"  Memory Audit (pure math) — batch={batch_size}, seq={seq_len}")
    print(f"  d_model={d}, n_layers={n_layers}, experts={config.n_experts}")
    print(f"  Expert dtype={config.expert_dtype}, grad_ckpt={config.grad_checkpoint}")
    print(f"  CPU offload={config.grad_offload_cpu}, share_experts={config.share_experts_within_group}")
    print("=" * 60)
    print()
    for key, val in result.items():
        if key == "within_budget":
            continue
        label = key.replace("_gb", "").replace("_", " ").title()
        print(f"  {label:30s} {val:8.3f} GB")
    print(f"  {'─' * 40}")
    print(f"  {'TOTAL':30s} {total_gb:8.3f} GB")
    print()
    print(f"  Gate 6 — <15GB budget:  {'✓ PASS' if total_gb < 15.0 else '✗ FAIL'}")
    print(f"  Headroom: {15.0 - total_gb:.3f} GB")
    print("=" * 60)

    return result


if __name__ == "__main__":
    memory_audit()
