"""
Initialize a fresh Concord model in mixed precision (bf16/fp16 with fp32 parts)
and save in HuggingFace-compatible format for direct inference.

Usage:
    python finetuning/HF/init_and_save.py [--output-dir ./finetuning/HF/saved]
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from finetuning.HF import ConcordConfig, ConcordForCausalLM


FP32_MODULES = {
    "norm",           # RMSNorm scales
    "router",         # MLPRouter (w1, w2)
    "attn_res",       # BlockAttnRes (pseudo_queries, k_proj)
    "mtp",            # MTPHead if present
    "wallclock_gain", # AgentRoPE
    "bias_embed",     # TemporalDecayBias
    "expert_bias",    # Router bias buffer
}


def is_fp32_param(name: str) -> bool:
    """Check if a parameter should stay in fp32 for numerical stability."""
    return any(k in name for k in FP32_MODULES)


def cast_to_mixed_precision(model: torch.nn.Module, expert_dtype: str = "fp16"):
    """
    Apply mixed-precision casting:
      - Expert FFN weights → fp16 (or bf16)
      - Most other weights  → bf16
      - Norms, router, attn_res → fp32
    """
    expert_dt = torch.float16 if expert_dtype == "fp16" else torch.bfloat16

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if is_fp32_param(name):
            param.data = param.data.float()
        elif "routed_experts" in name or "shared_experts" in name:
            param.data = param.data.to(expert_dt)
        else:
            param.data = param.data.bfloat16()

    # Buffers (non-parameter tensors like inv_freq, expert_bias)
    for name, buf in model.named_buffers():
        if is_fp32_param(name):
            buf.data = buf.data.float()
        elif "inv_freq" in name:
            buf.data = buf.data.float()

    return model


def main():
    parser = argparse.ArgumentParser("Concord-α HF Initialization")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "saved"),
                        help="Where to save the HF model")
    parser.add_argument("--bf16-experts", action="store_true",
                        help="Use bf16 for experts instead of fp16")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Building ConcordConfig...")
    config = ConcordConfig()

    print(f"  d_model={config.d_model}, n_layers={config.n_layers}, n_groups={config.n_groups}")
    print(f"  n_experts={config.n_experts}, expert_intermediate={config.expert_intermediate}")
    print(f"  vocab_size={config.vocab_size}")
    print(f"  expert_dtype={config.expert_dtype}")

    print("\nBuilding ConcordForCausalLM (random init)...")
    model = ConcordForCausalLM(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,} ({n_params / 1e9:.2f}B)")

    print("\nCasting to mixed precision:")
    expert_dtype = "bf16" if args.bf16_experts else "fp16"
    cast_to_mixed_precision(model, expert_dtype=expert_dtype)

    # Log dtype distribution
    fp32_count = 0
    bf16_count = 0
    fp16_count = 0
    for p in model.parameters():
        if p.dtype == torch.float32:
            fp32_count += p.numel()
        elif p.dtype == torch.bfloat16:
            bf16_count += p.numel()
        elif p.dtype == torch.float16:
            fp16_count += p.numel()

    total = fp32_count + bf16_count + fp16_count
    print(f"  fp32: {fp32_count:>12,} params ({100 * fp32_count / total:.1f}%) -> norms, routers, attn_res")
    print(f"  bf16: {bf16_count:>12,} params ({100 * bf16_count / total:.1f}%) -> transformer backbone")
    print(f"  fp16: {fp16_count:>12,} params ({100 * fp16_count / total:.1f}%) -> expert FFNs")

    model.eval()

    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"  [OK] config.json saved")
    print(f"  [OK] model.safetensors saved")

    print("\nVerifying: loading back with from_pretrained...")
    loaded = ConcordForCausalLM.from_pretrained(output_dir)
    loaded.eval()

    # Quick smoke test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = loaded.to(device)
    dummy = torch.randint(0, config.vocab_size - 1, (1, 8), device=device)

    with torch.no_grad():
        out = loaded(dummy)
        logits = out.logits if hasattr(out, "logits") else out["logits"]

    print(f"  [OK] Forward pass OK: input shape {tuple(dummy.shape)} -> logits shape {tuple(logits.shape)}")
    print(f"  [OK] First token logit mean: {logits[0, 0, :10].mean().item():.4f}")

    print(f"\nDone! Model saved to: {output_dir}".encode('ascii', 'replace').decode('ascii'))
    print(f"\nTo use for inference:")
    print(f"  from finetuning.HF import ConcordForCausalLM")
    print(f"  model = ConcordForCausalLM.from_pretrained(r'{output_dir}'.encode('ascii', 'replace').decode('ascii')).eval().cuda()")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained(r'{output_dir}'.encode('ascii', 'replace').decode('ascii'), trust_remote_code=True)")


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    main()
