"""
Autoregressive inference via HF-compatible ConcordForCausalLM.
Reports TTFT, tokens/sec, FLOP estimates per forward pass.

Usage:
    python finetuning/HF/inference.py --prompt "Hello, world!"
    python finetuning/HF/inference.py --prompt "|<STATE>|\n### INIT\n" --max-new-tokens 200
"""

import os
import sys
import time
import argparse

import torch
from transformers import AutoTokenizer


def estimate_model_flops(config, seq_len: int, batch_size: int = 1):
    d = config.d_model
    v = config.vocab_size
    e = config.embed_dim
    L = config.n_layers
    n_g = config.n_groups
    gs = config.group_size
    n_delta = gs - 1
    n_attn = 1
    n_exp = config.n_experts
    n_rt_d = config.n_routed_delta
    n_rt_a = config.n_routed_attn
    e_i = config.expert_intermediate
    r_h = config.router_hidden
    n_q = config.attn_q_heads
    n_kv = config.attn_kv_heads
    hd = config.attn_head_dim
    d_v = config.delta_v_heads
    d_hd = config.delta_head_dim

    flops = 0.0

    # Embed + embed_up: seq * embed_dim * d_model * 2
    flops += batch_size * seq_len * 2 * e * d

    # Per-layer compute
    for g in range(n_g):
        for i in range(gs):
            is_delta = i < (gs - 1)

            if is_delta:
                # DeltaNet projections: q, k, v, o, gate, beta = 6 * 2 * seq * d^2
                flops += batch_size * seq_len * 6 * 2 * d * d
                # DeltaNet recurrence: 4 * seq * n_qk * d_hd^2
                dh = config.delta_qk_heads
                flops += batch_size * seq_len * 4 * dh * d_hd * d_hd
                # MoE router: 2 * seq * (d * r_h + r_h * n_exp)
                flops += batch_size * seq_len * 2 * (d * r_h + r_h * n_exp)
                # Shared expert: 2 * seq * 3 * d * e_i
                flops += batch_size * seq_len * 2 * 3 * d * e_i
                # Routed experts (only n_routed_delta active): n_rt_d * 2 * seq * 3 * d * e_i
                flops += batch_size * seq_len * n_rt_d * 2 * 3 * d * e_i
            else:
                # Attention QKV + O + gate: 5 * 2 * seq * d^2
                flops += batch_size * seq_len * 5 * 2 * d * d
                # Attention scores: 2 * seq * n_q * hd * T_k (avg of CSA + HCA)
                csa_k = max(seq_len // config.csa_compress + min(seq_len, config.csa_window), 1)
                hca_k = max(seq_len // config.hca_compress, 1)
                avg_k = (csa_k + hca_k) / 2
                flops += batch_size * 2 * n_q * hd * seq_len * avg_k
                # MoE router
                flops += batch_size * seq_len * 2 * (d * r_h + r_h * n_exp)
                # Shared expert
                flops += batch_size * seq_len * 2 * 3 * d * e_i
                # Routed experts (n_routed_attn active)
                flops += batch_size * seq_len * n_rt_a * 2 * 3 * d * e_i

            # Two RMSNorm
            flops += batch_size * seq_len * 4 * d

    # Final norm
    flops += batch_size * seq_len * 2 * d

    # embed_down + lm_head: seq * (d * e + e * v) * 2
    flops += batch_size * seq_len * 2 * (d * e + e * v)

    return flops


def analyze_params(model, config):
    n_total = sum(p.numel() for p in model.parameters())
    n_embed = config.vocab_size * config.embed_dim
    n_dense = 0
    n_routed = 0
    for name, p in model.named_parameters():
        if "routed_experts" in name:
            n_routed += p.numel()
        elif "embed" in name or "lm_head" in name:
            pass
        else:
            n_dense += p.numel()
    return n_total, n_embed, n_dense, n_routed


@torch.no_grad()
def generate_with_stats(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, device):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    model.eval()
    starter = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    ender = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    # === Prefill (TTFT) ===
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    if starter:
        starter.record()

    out = model(input_ids)
    logits = out.logits[:, -1, :]

    if ender:
        ender.record()
    torch.cuda.synchronize() if device.type == "cuda" else None
    ttft = (time.perf_counter() - t0) * 1000
    if starter and ender:
        starter.synchronize()
        ender.synchronize()
        ttft = starter.elapsed_time(ender)

    # === Decode loop ===
    generated = input_ids.clone()
    token_times = []

    for step in range(max_new_tokens):
        if temperature > 0:
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            if top_p > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, stable=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        torch.cuda.synchronize() if device.type == "cuda" else None
        t_step = time.perf_counter()

        out = model(generated)
        logits = out.logits[:, -1, :]

        torch.cuda.synchronize() if device.type == "cuda" else None
        token_times.append((time.perf_counter() - t_step) * 1000)

    total_tokens = generated.shape[1] - prompt_len
    avg_ms_per_token = sum(token_times) / len(token_times) if token_times else 0
    tokens_per_sec = 1000 / avg_ms_per_token if avg_ms_per_token > 0 else 0

    return generated, ttft, avg_ms_per_token, tokens_per_sec, total_tokens


def main():
    parser = argparse.ArgumentParser("Concord HF Inference")
    parser.add_argument("--model-dir", type=str, default=os.path.join(os.path.dirname(__file__), "saved"),
                        help="Path to HF model directory")
    parser.add_argument("--tokenizer-dir", type=str, default=os.path.join(os.path.dirname(__file__), "tokenizer"),
                        help="Path to tokenizer directory")
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true", help="torch.compile model for faster inference")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs to average")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup iterations before measuring")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"Loading tokenizer from {args.tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

    print(f"Loading model from {args.model_dir}...")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from finetuning.HF import ConcordForCausalLM

    model = ConcordForCausalLM.from_pretrained(args.model_dir).to(device).eval()
    if args.compile:
        model = torch.compile(model)

    config = model.config
    n_total, n_embed, n_dense, n_routed = analyze_params(model, config)
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]

    prefill_flops = estimate_model_flops(config, prompt_len)
    decode_flops_per_token = estimate_model_flops(config, 1)

    print(f"  Vocabulary: {config.vocab_size:,}")
    print(f"  Parameters: {n_total:,} ({n_total/1e6:.1f}M)")
    print(f"    Embedding: {n_embed:,} ({100*n_embed/n_total:.1f}%)")
    print(f"    Dense:     {n_dense:,} ({100*n_dense/n_total:.1f}%)")
    print(f"    Routed:    {n_routed:,} ({100*n_routed/n_total:.1f}%)")
    print(f"  Prompt: {prompt_len} tokens")
    print(f"  FLOPs (prefill, S={prompt_len}): {prefill_flops/1e9:.1f}G FLOPs")
    print(f"  FLOPs (decode, per token):       {decode_flops_per_token/1e6:.1f}M FLOPs")

    for w in range(args.warmup):
        with torch.no_grad():
            model(prompt_ids)

    for run in range(1, args.num_runs + 1):
        print(f"\n--- Run {run}/{args.num_runs} ---")
        generated, ttft, avg_ms, tok_s, n_tokens = generate_with_stats(
            model, tokenizer, args.prompt, args.max_new_tokens,
            args.temperature, args.top_k, args.top_p, device,
        )
        text = tokenizer.decode(generated[0], skip_special_tokens=False)

        print(f"  TTFT:              {ttft:.1f} ms")
        print(f"  Generated tokens:  {n_tokens}")
        print(f"  Avg ms/token:      {avg_ms:.2f} ms")
        print(f"  Tokens/sec:        {tok_s:.1f}")
        if avg_ms > 0:
            est_gflops_prefill = prefill_flops / (ttft / 1000) / 1e9
            est_gflops_decode = decode_flops_per_token / (avg_ms / 1000) / 1e9
            print(f"  Est. TFLOPS (prefill): {est_gflops_prefill:.1f} TFLOPS")
            print(f"  Est. TFLOPS (decode):  {est_gflops_decode:.1f} TFLOPS")
        print(f"  Output:\n  {text}")


if __name__ == "__main__":
    main()
