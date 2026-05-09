"""
Param Count (Materialized) — Instantiates the model and counts parameters.

Usage:
    python -m model.param_count_materialized

Compared to param_count.py (pure math), this actually builds the model
and uses named_parameters() / named_buffers() for exact counts.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.model import HybridMoE


def count_materialized(config: ModelConfig = None) -> dict:
    if config is None:
        config = ModelConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)

    model = HybridMoE(config).to(device)
    torch.set_default_dtype(orig_dtype)

    # Per-component breakdown
    component_counts = {}
    total_params = 0
    total_buffers = 0
    total_trainable = 0

    # Named parameter breakdown by module path prefix
    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        if param.requires_grad:
            total_trainable += n
        prefix = name.split(".")[0]
        component_counts[prefix] = component_counts.get(prefix, 0) + n

    for name, buf in model.named_buffers():
        total_buffers += buf.numel()

    # Expert-specific breakdown
    expert_params = 0
    router_params = 0
    attn_params = 0
    delta_params = 0
    embed_params = 0
    norm_params = 0
    mtp_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        if "routed_experts" in name or "shared_experts" in name:
            expert_params += n
        elif "router" in name:
            router_params += n
        elif "attention" in name:
            attn_params += n
        elif "delta_net" in name:
            delta_params += n
        elif "embed" in name or "lm_head" in name or "embed_up" in name or "embed_down" in name:
            embed_params += n
        elif "norm" in name or "scale" in name:
            norm_params += n
        elif "mtp" in name:
            mtp_params += n

    # Active parameters (what's compute per token)
    active_routed = 0
    n_delta = config.n_groups * (config.group_size - 1)
    n_attn = config.n_groups
    single_expert = 3 * config.d_model * config.expert_intermediate
    active_routed = (n_delta * config.n_routed_delta + n_attn * config.n_routed_attn) * single_expert
    active_shared = config.n_shared * config.n_layers * single_expert

    result = {
        "total_params": total_params,
        "total_buffers": total_buffers,
        "trainable_params": total_trainable,
        "component_breakdown": component_counts,
        "expert_params": expert_params,
        "router_params": router_params,
        "attention_params": attn_params,
        "delta_params": delta_params,
        "embedding_params": embed_params,
        "norm_params": norm_params,
        "mtp_params": mtp_params,
        "active_routed": active_routed,
        "active_shared": active_shared,
        "active_total": active_routed + active_shared,
        "n_parameters_objects": len(list(model.parameters())),
        "n_buffers_objects": len(list(model.buffers())),
        "unique_expert_sets": config.n_groups if config.share_experts_within_group else config.n_layers,
    }

    print("=" * 62)
    print(f"  Param Count (Materialized) — {config.d_model=}, {config.n_layers=}")
    print(f"  {config.n_experts=}, {config.share_experts_within_group=}")
    print("=" * 62)
    print()
    rows = [
        ("Total parameters", total_params),
        ("Trainable parameters", total_trainable),
        ("Buffers (FP8 shadows, scales)", total_buffers),
        ("Expert FFN weights", expert_params),
        ("Router parameters", router_params),
        ("Attention parameters", attn_params),
        ("DeltaNet parameters", delta_params),
        ("Embedding & projections", embed_params),
        ("Norm parameters", norm_params),
        ("MTP head", mtp_params),
    ]
    for label, val in rows:
        suffix = f"({val/1e9:.2f}B)" if val > 1e9 else f"({val/1e6:.1f}M)" if val > 1e6 else f"({val/1e3:.1f}K)"
        print(f"  {label:28s} {val:>15,}  {suffix}")
    print(f"  {'-' * 50}")
    print(f"  Active routed:   {active_routed:>15,}  ({active_routed/1e6:.1f}M)")
    print(f"  Active shared:   {active_shared:>15,}  ({active_shared/1e6:.1f}M)")
    print(f"  Active total:    {active_routed+active_shared:>15,}  ({(active_routed+active_shared)/1e6:.1f}M)")
    print(f"  Parameter tensors: {result['n_parameters_objects']}")
    print(f"  Buffer tensors:    {result['n_buffers_objects']}")
    print()
    print(f"  Component breakdown (by prefix):")
    for prefix, count in sorted(component_counts.items()):
        suffix = f"({count/1e6:.1f}M)" if count > 1e6 else f"({count/1e3:.1f}K)"
        print(f"    {prefix:20s} {count:>12,}  {suffix}")
    print("=" * 62)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    count_materialized()
