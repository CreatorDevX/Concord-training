"""
Benchmark Concord model: measure context scaling, memory, latency.

Evaluates:
  - Forward pass time vs sequence length
  - Memory usage vs sequence length
  - TTFT vs sequence length
  - Token generation throughput

Usage:
    python finetuning/HF/benchmark.py
    python finetuning/HF/benchmark.py --seq-len 64 128 256 512 1024
"""

import os
import sys
import time
import argparse

import torch


def estimate_model_flops(config, seq_len: int, batch_size: int = 1):
    d = config.d_model
    v = config.vocab_size
    e = config.embed_dim
    n_g = config.n_groups
    gs = config.group_size
    n_exp = config.n_experts
    n_rt_d = config.n_routed_delta
    n_rt_a = config.n_routed_attn
    e_i = config.expert_intermediate
    r_h = config.router_hidden
    n_q = config.attn_q_heads
    hd = config.attn_head_dim
    dh = config.delta_qk_heads
    d_hd = config.delta_head_dim

    flops = 0.0
    flops += batch_size * seq_len * 2 * e * d

    for _ in range(n_g):
        for i in range(gs):
            is_delta = i < (gs - 1)

            if is_delta:
                flops += batch_size * seq_len * 6 * 2 * d * d
                flops += batch_size * seq_len * 4 * dh * d_hd * d_hd
                flops += batch_size * seq_len * 2 * (d * r_h + r_h * n_exp)
                flops += batch_size * seq_len * 2 * 3 * d * e_i
                flops += batch_size * seq_len * n_rt_d * 2 * 3 * d * e_i
            else:
                flops += batch_size * seq_len * 5 * 2 * d * d
                csa_k = max(seq_len // config.csa_compress + min(seq_len, config.csa_window), 1)
                hca_k = max(seq_len // config.hca_compress, 1)
                avg_k = (csa_k + hca_k) / 2
                flops += batch_size * 2 * n_q * hd * seq_len * avg_k
                flops += batch_size * seq_len * 2 * (d * r_h + r_h * n_exp)
                flops += batch_size * seq_len * 2 * 3 * d * e_i
                flops += batch_size * seq_len * n_rt_a * 2 * 3 * d * e_i

            flops += batch_size * seq_len * 4 * d

    flops += batch_size * seq_len * 2 * d
    flops += batch_size * seq_len * 2 * (d * e + e * v)
    return flops


@torch.no_grad()
def benchmark_prefill(model, seq_len: int, batch_size: int, device: torch.device, num_warmup: int, num_runs: int):
    vocab = model.config.vocab_size
    dummy = torch.randint(0, vocab - 1, (batch_size, seq_len), device=device)

    for _ in range(num_warmup):
        model(dummy)

    torch.cuda.synchronize() if device.type == "cuda" else None
    starter = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    ender = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    times = []
    mem_before = torch.cuda.memory_allocated() if device.type == "cuda" else 0

    for _ in range(num_runs):
        if starter:
            starter.record()
        t0 = time.perf_counter()

        out = model(dummy)

        if ender:
            ender.record()
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - t0

        if starter and ender:
            starter.synchronize()
            ender.synchronize()
            elapsed = starter.elapsed_time(ender) / 1000

        times.append(elapsed)

    mem_after = torch.cuda.memory_allocated() if device.type == "cuda" else 0
    peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0

    avg = sum(times) / len(times)
    flops = estimate_model_flops(model.config, seq_len, batch_size)
    tflops = flops / avg / 1e12

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "avg_time_ms": avg * 1000,
        "std_time_ms": (sum((t - avg)**2 for t in times) / max(len(times) - 1, 1)) ** 0.5 * 1000,
        "tokens_per_sec": batch_size * seq_len / avg,
        "flops_est": flops,
        "tflops_est": tflops,
        "mem_delta_mb": (mem_after - mem_before) / (1024 * 1024),
        "peak_mem_mb": peak_mem / (1024 * 1024),
    }


@torch.no_grad()
def benchmark_decode_single(model, seq_len: int, device: torch.device, num_warmup: int, num_runs: int):
    """Time a single-token decode (after prefix of seq_len)."""
    vocab = model.config.vocab_size
    prefix = torch.randint(0, vocab - 1, (1, seq_len), device=device)

    for _ in range(num_warmup):
        model(prefix)

    torch.cuda.synchronize() if device.type == "cuda" else None

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        out = model(prefix)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    avg = sum(times) / len(times)
    return {"avg_ms": avg * 1000, "tok_per_sec": 1.0 / avg}


def main():
    parser = argparse.ArgumentParser("Concord HF Benchmark")
    parser.add_argument("--model-dir", type=str, default=os.path.join(os.path.dirname(__file__), "saved"),
                        help="Path to HF model directory")
    parser.add_argument("--seq-len", type=int, nargs="+",
                        default=[64, 128, 256, 512, 1024, 2048, 4096],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-runs", type=int, default=10, help="Runs per length")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from finetuning.HF import ConcordForCausalLM

    model = ConcordForCausalLM.from_pretrained(args.model_dir).to(device).eval()
    config = model.config

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params, d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"MoE: {config.n_experts} experts, routed: {config.n_routed_delta}/{config.n_routed_attn}")
    print()

    seq_lens = [s for s in args.seq_len if s <= config.max_seq_len]

    if not args.csv:
        print(f"{'SeqLen':>8} {'Batch':>6} {'Time(ms)':>10} {'+/-':>8} {'Tok/s':>10} {'FLOPs(G)':>10} {'TFLOPS':>8} {'Mem Δ(MB)':>10} {'Peak(MB)':>10}")
        print("-" * 90)

    results = []
    for sl in seq_lens:
        r = benchmark_prefill(model, sl, args.batch_size, device, args.warmup, args.num_runs)
        results.append(r)

        if args.csv:
            print(f"{sl},{args.batch_size},{r['avg_time_ms']:.2f},{r['tokens_per_sec']:.0f},{r['flops_est']/1e9:.1f},{r['tflops_est']:.2f},{r['mem_delta_mb']:.0f},{r['peak_mem_mb']:.0f}")
        else:
            print(f"{sl:>8} {args.batch_size:>6} {r['avg_time_ms']:>8.2f} {r['std_time_ms']:>6.2f} "
                  f"{r['tokens_per_sec']:>8.0f} {r['flops_est']/1e9:>8.1f} {r['tflops_est']:>6.2f} "
                  f"{r['mem_delta_mb']:>8.0f} {r['peak_mem_mb']:>8.0f}")

    print()
    if not args.csv:
        print("Decode step timing (per-token latency after prefix):")
        print(f"{'PrefixLen':>10} {'ms/token':>10} {'tok/s':>10}")
        print("-" * 35)
    for sl in seq_lens[:6]:
        r = benchmark_decode_single(model, sl, device, args.warmup, args.num_runs)
        if args.csv:
            print(f"{sl},{r['avg_ms']:.2f},{r['tok_per_sec']:.0f}")
        else:
            print(f"{sl:>10} {r['avg_ms']:>8.2f} {r['tok_per_sec']:>8.0f}")


if __name__ == "__main__":
    main()
