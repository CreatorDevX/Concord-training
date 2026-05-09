import os
import argparse
import glob
import zipfile
import shutil

from model.config import ModelConfig
from model.train import TrainConfig, train

def compress_checkpoint(source_dir: str, target_zip: str):
    print(f"Compressing checkpoint {source_dir} -> {target_zip}...")
    shutil.make_archive(target_zip.replace('.zip', ''), 'zip', source_dir)
    print("Done!")

def main():
    parser = argparse.ArgumentParser("Concord-α 3B Pretraining Orchestrator")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory where data files live")
    parser.add_argument("--data-format", type=str, default="auto", choices=["auto", "jsonl", "parquet", "memmap"], help="Data format")
    parser.add_argument("--tokenizer", type=str, default="./model/custom_tokenizer", help="Local or remote tokenizer")
    parser.add_argument("--vision-weights", type=str, default="google/tipsv2-l14", help="Local or remote model for vision weights")
    parser.add_argument("--corpus-tokens", type=int, default=2048, help="Token length per sequence ignoring harness")
    
    parser.add_argument("--steps", type=int, default=100000, help="Total steps to train")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt or .zip to resume from")
    
    parser.add_argument("--multi-gpu", action="store_true", help="Use DDP for multi GPU")
    parser.add_argument("--pipeline-parallel", action="store_true", help="Legacy 2xT4 PIPELINE parallel")

    parser.add_argument("--compress", type=str, default=None, help="Just compress a given checkpoint folder to .zip and exit")
    
    args = parser.parse_args()
    
    if args.compress:
        compress_checkpoint(args.compress, f"{args.compress}.zip")
        return
            
    # Initiate Phase 1 Training Loop
    print("\nStarting Phase 1 Training Engine...")
    model_config = ModelConfig(
        corpus_tokens=args.corpus_tokens,
        vision_weights_path=args.vision_weights if args.vision_weights.lower() != "none" else None,
    )
    
    train_config = TrainConfig(
        data_dir=args.data_dir,
        data_format=args.data_format,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        total_steps=args.steps,
        resume_from=args.resume,
        use_ddp=args.multi_gpu,
        use_pipeline_parallel=args.pipeline_parallel,
    )
    
    # Run the compiled train pipeline
    train(model_config=model_config, train_config=train_config)
    
if __name__ == "__main__":
    main()
