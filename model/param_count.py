"""
Parameter Count — Pure math, no model instantiation.

Two active metrics:
    active_total: standard MoE = total - inactive_expert_params (includes embed/MTP)
    active_moe:   MoE routing budget = active_routed + shared + router + attnres + norms
                  This is the ~100M target from the spec.

Run as: python -m model.param_count
"""

from model.config import ModelConfig


def count_params(config: ModelConfig = None) -> dict:
    """Count total and active parameters using pure arithmetic. Zero GPU memory."""
    if config is None:
        config = ModelConfig()

    d = config.d_model
    V = config.vocab_size
    n_layers = config.n_layers
    n_groups = config.n_groups
    group_size = config.group_size
    n_delta_layers = n_groups * (group_size - 1)  # 18
    n_attn_layers = n_groups                       # 6

    # ── Embedding ──────────────────────────────────────────────────────
    # Table is factorized: [V, embed_dim]
    embed_params = V * config.embed_dim
    lm_head_params = 0 if config.tie_embeddings else V * config.embed_dim
    
    # Projections
    embed_up_params = config.embed_dim * d  # Linear(512, 1024)
    embed_down_params = d * config.embed_dim # Linear(1024, 512)

    # ── DeltaNet ───────────────────────────────────────────────────────
    dqk = config.delta_d_qk
    dv = config.delta_d_v
    delta_per_layer = (
        d * dqk +       # q_proj
        d * dqk +       # k_proj
        d * dv +        # v_proj
        dv * d +        # o_proj
        d * dv +        # gate_proj
        d * config.delta_qk_heads +  # beta_proj
        config.delta_head_dim  # FPLayerNorm (per-head, not d_model)
    )
    delta_total = delta_per_layer * n_delta_layers

    # ── Attention ──────────────────────────────────────────────────────
    dq = config.attn_d_q
    dkv = config.attn_d_kv
    hd = config.attn_head_dim

    attn_proj_per_layer = (
        d * dq +        # q_proj
        d * dkv +       # k_proj
        d * dkv +       # v_proj
        dq * d +        # o_proj
        d * dq          # gate_proj
    )

    n_csa = (n_attn_layers + 1) // 2
    n_hca = n_attn_layers // 2
    csa_compressor = hd * config.csa_compress * hd
    hca_compressor = hd * config.hca_compress * hd

    attn_total = (
        attn_proj_per_layer * n_attn_layers +
        n_csa * csa_compressor +
        n_hca * hca_compressor
    )

    # ── Experts ────────────────────────────────────────────────────────
    single_expert = 3 * d * config.expert_intermediate

    if config.share_experts_within_group:
        n_expert_sets = n_groups
    else:
        n_expert_sets = n_layers

    routed_total = config.n_experts * n_expert_sets * single_expert
    shared_total = config.n_shared * n_layers * single_expert

    # ── Router ─────────────────────────────────────────────────────────
    n_routers = n_expert_sets if config.share_experts_within_group else n_layers
    router_per = (
        d +                                      # norm
        d * config.router_hidden +               # w1
        config.router_hidden * config.n_experts  # w2
    )
    router_total = router_per * n_routers

    # ── Norms (2 per block + final) ────────────────────────────────────
    norms_total = 2 * d * n_layers + d

    # ── AttnRes ────────────────────────────────────────────────────────
    attn_res_total = n_layers * d + d * d + d

    # ── MTP ────────────────────────────────────────────────────────────
    # prediction_heads: Linear(2d, d) + Linear(d, d)
    mtp_mlp_per_step = (2 * d * d) + (d * d)
    mtp_output_total = 0 # Tied to embed.weight logically
    mtp_embed_proj = d * d
    mtp_total = config.mtp_steps * mtp_mlp_per_step + mtp_output_total + mtp_embed_proj

    # ── Grand total ────────────────────────────────────────────────────
    total = (
        embed_params + lm_head_params +
        embed_up_params + embed_down_params +
        delta_total + attn_total +
        routed_total + shared_total +
        router_total + norms_total +
        attn_res_total + mtp_total
    )

    # ── Active (MoE routing budget) ────────────────────────────────────
    # What varies per token based on routing decisions + small overhead.
    # This is the spec's "~100M active" target.
    n_active_invocations = (
        n_delta_layers * config.n_routed_delta +
        n_attn_layers * config.n_routed_attn
    )
    active_routed = n_active_invocations * single_expert
    active_moe = active_routed + shared_total + router_total + attn_res_total + norms_total + embed_up_params + embed_down_params

    # Also compute standard MoE "activated params" (total - dormant experts)
    dormant_experts = (config.n_experts * n_expert_sets - n_active_invocations) * single_expert
    active_full = total - dormant_experts

    result = {
        "total": int(total),
        "active_moe": int(active_moe),
        "active_full": int(active_full),
        "embedding": int(embed_params),
        "lm_head_extra": int(lm_head_params),
        "embed_projs": int(embed_up_params + embed_down_params),
        "delta_total": int(delta_total),
        "attn_total": int(attn_total),
        "routed_experts": int(routed_total),
        "shared_experts": int(shared_total),
        "router_total": int(router_total),
        "norms": int(norms_total),
        "attn_res": int(attn_res_total),
        "mtp": int(mtp_total),
        "single_expert": int(single_expert),
        "n_expert_sets": int(n_expert_sets),
        "active_routed": int(active_routed),
    }

    # -- Print ----------------------------------------------------------
    print("=" * 62)
    print(f"  3B Hybrid MoE -- Parameter Count")
    print(f"  d_model={d}  n_layers={n_layers}  n_experts={config.n_experts}")
    print(f"  expert_intermediate={config.expert_intermediate}  "
          f"routing={config.n_routed_delta}/{config.n_routed_attn}")
    print(f"  share_experts={config.share_experts_within_group}")
    print("=" * 62)

    rows = [
        ("Embedding (Rank 512)", embed_params),
        ("Embed Up/Down Projs", embed_up_params + embed_down_params),
        ("LM Head (tied)", lm_head_params),
        ("DeltaNet (x18)", delta_total),
        ("Attention (x6)", attn_total),
        ("Routed experts", routed_total),
        ("Shared experts", shared_total),
        ("Routers", router_total),
        ("Norms", norms_total),
        ("AttnRes", attn_res_total),
        ("MTP head", mtp_total),
    ]

    print()
    for label, val in rows:
        if val == 0:
            continue
        suffix = f"({val/1e9:.2f}B)" if val > 1e9 else f"({val/1e6:.1f}M)" if val > 1e6 else f"({val/1e3:.1f}K)"
        print(f"  {label:28s} {val:>15,}  {suffix}")

    print(f"  {'-' * 50}")
    print(f"  {'TOTAL':28s} {total:>15,}  ({total/1e9:.3f}B)")
    print()
    print(f"  Per expert: {single_expert:,} params ({single_expert/1e6:.2f}M)")
    print(f"  Unique experts: {config.n_experts} × {n_expert_sets} = "
          f"{config.n_experts * n_expert_sets:,}")
    print(f"  Active invocations/fwd: {n_active_invocations} "
          f"(18x{config.n_routed_delta} + 6x{config.n_routed_attn})")
    print()
    print(f"  Active (MoE budget):  {active_moe/1e6:>8.1f}M  "
          f"<- routed({active_routed/1e6:.1f}M) + shared({shared_total/1e6:.1f}M) + overhead")
    print(f"  Active (full):        {active_full/1e6:>8.1f}M  "
          f"<- total - dormant experts")
    print()

    in_total = 4.8e8 <= total <= 5.2e8
    in_active = 28e6 <= active_moe <= 35e6
    print()
    print(f"  Gate 1 -- Total  [480M, 520M]:  {'PASS' if in_total else f'FAIL ({total/1e6:.1f}M)'}")
    print(f"  Gate 2 -- Active [28M, 35M]:    {'PASS' if in_active else f'FAIL ({active_moe/1e6:.1f}M)'}")
    print("=" * 62)

    return result


if __name__ == "__main__":
    count_params()
