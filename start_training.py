import os
import argparse

from model.config import ModelConfig


def main():
    parser = argparse.ArgumentParser("Concord-α 3B Pretraining Orchestrator")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory where data files live")
    parser.add_argument("--data-format", type=str, default="auto", choices=["auto", "jsonl", "parquet", "memmap", "huggingface"], help="Data format")
    parser.add_argument("--tokenizer", type=str, default="./model/custom_tokenizer", help="Local or remote tokenizer")
    parser.add_argument("--vision-weights", type=str, default="google/tipsv2-l14", help="Local or remote model for vision weights")
    parser.add_argument("--corpus-tokens", type=int, default=2048, help="Token length per sequence ignoring harness")
    parser.add_argument("--disable-vision", action="store_true", help="Disable multimodal data fetching and vision backbone execution.")
    
    parser.add_argument("--steps", type=int, default=100000, help="Total steps to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Micro-batch size")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt or .zip to resume from")
    
    parser.add_argument("--multi-gpu", action="store_true", help="Use DDP for multi GPU")
    parser.add_argument("--pipeline-parallel", action="store_true", help="Legacy 2xT4 PIPELINE parallel")
    parser.add_argument("--tpu", action="store_true", help="Enable Native xmp.spawn for TPU v5e-8 Scaling")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")

    parser.add_argument("--compress", type=str, default=None, help="Just compress a given checkpoint folder to .zip and exit")
    parser.add_argument("--export-safetensors", type=str, default="", help="If run finishes, export fully parsed safetensors")
    parser.add_argument("--wandb-key", type=str, default="", help="W&B API Key for fully automated backend logging")
    
    # Fast dev run: quick smoke test with minimal steps/params
    parser.add_argument("--fast-dev-run", action="store_true", help="Run 5 steps with tiny batch for quick validation")
    
    args = parser.parse_args()
    
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
        os.environ["WANDB_MODE"] = "online"

    if args.compress:
        import shutil
        print(f"Compressing checkpoint {args.compress} -> {args.compress}.zip...")
        shutil.make_archive(args.compress.replace('.zip', ''), 'zip', args.compress)
        print("Done!")
        return

    # ── Fast dev run overrides ──────────────────────────────────────────
    if args.fast_dev_run:
        print("\n[Fast Dev Run] Overriding: steps=5, batch_size=2, checkpoint_interval=50, log_interval=1")
        args.steps = 5
        args.batch_size = 2
        args.grad_accum = 1
    
    # Validate data directory early
    if not args.data_dir.startswith("hf:"):
        if not os.path.isdir(args.data_dir):
            print(f"Warning: data directory '{args.data_dir}' does not exist yet.")
    
    # Initiate Phase 1 Training Loop
    print("\n[Concord-α] Initializing Phase 1 Hybrid MoE Training Engine...")
    
    # Lazy imports for speed
    import torch
    from model.train import TrainConfig, train
    
    model_config = ModelConfig(
        corpus_tokens=args.corpus_tokens,
        use_vision=not args.disable_vision,
        vision_weights_path=args.vision_weights if args.vision_weights.lower() != "none" else None,
    )
    
    if not args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if not args.pipeline_parallel:
            print("[System] Multiple GPUs detected. Auto-enabling DDP Copied Parallelism.")
            args.multi_gpu = True

    train_config = TrainConfig(
        data_dir=args.data_dir,
        data_format=args.data_format,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        total_steps=args.steps,
        resume_from=args.resume,
        use_ddp=args.multi_gpu,
        use_tpu=args.tpu,
        use_pipeline_parallel=args.pipeline_parallel,
        checkpoint_interval=1 if args.fast_dev_run else 500,
        log_interval=1 if args.fast_dev_run else 10,
    )
    
    global_bs = args.batch_size * args.grad_accum * (torch.cuda.device_count() if args.multi_gpu else 1)
    print(f"[Config] Global Batch Size: {global_bs}")
    print(f"[Config] Precision: FP16 + Autocast (FP32 upcast for loss)")
    
    train(model_config=model_config, train_config=train_config)
    
if __name__ == "__main__":
    main()
