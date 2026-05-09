import os
import sys
import json
import time
import math
import argparse
import signal
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset

from model.config import ModelConfig
from model.model import HybridMoE
from model.dataset import HarnessFormatter, JsonlParquetDataset, MemmapPretrainingDataset, TokenizerWrapper, MultimodalDataset

# ═══════════════════════════════════════════════════════════════════════
#  Muon Optimizer (No changes)
# ═══════════════════════════════════════════════════════════════════════
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, momentum=0.95, weight_decay=0.01,
                 nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                       nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad.data
                if weight_decay != 0: p.data.mul_(1 - lr * weight_decay)
                state = self.state[p]
                if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(p.data)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                if nesterov: update = grad + momentum * buf
                else: update = buf
                if p.dim() >= 2: update = self._newton_schulz_orthogonalize(update, ns_steps)
                p.data.add_(update, alpha=-lr)
        return loss

    @staticmethod
    def _newton_schulz_orthogonalize(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
        if M.dim() < 2: return M
        original_shape = M.shape
        if M.dim() > 2: M = M.view(M.shape[0], -1)
        rows, cols = M.shape
        if rows > cols: M = M.T; transposed = True
        else: transposed = False
        M_norm = M.norm()
        if M_norm < 1e-8: return torch.zeros(original_shape, device=M.device, dtype=M.dtype)
        M = M / M_norm
        a, b, c = 3.4445, -4.7750, 2.0315
        X = M
        for _ in range(steps):
            A = X @ X.T
            X = a * X + b * (A @ X) + c * (A @ A @ X)
        if transposed: X = X.T
        return (X * M_norm).view(original_shape)

# ═══════════════════════════════════════════════════════════════════════
#  CPU Gradient Offload (No changes)
# ═══════════════════════════════════════════════════════════════════════
class CPUGradientOffloader:
    def __init__(self, model: nn.Module):
        self.cpu_grads: Dict[str, torch.Tensor] = {}
        self.model = model
    def accumulate(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name not in self.cpu_grads: self.cpu_grads[name] = param.grad.data.cpu().clone()
                else: self.cpu_grads[name].add_(param.grad.data.cpu())
                param.grad = None
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.cpu_grads: param.grad = self.cpu_grads[name].to(param.device)
        self.cpu_grads.clear()

# ═══════════════════════════════════════════════════════════════════════
#  Pipeline Parallelism
# ═══════════════════════════════════════════════════════════════════════
class PipelineParallel:
    def __init__(self, model: HybridMoE, device0: torch.device, device1: torch.device):
        self.model = model
        self.device0 = device0
        self.device1 = device1
        self.split_point = model.config.n_layers // 2
        model.embed.to(device0)
        model.attn_res.to(device0)
        for i, block in enumerate(model.blocks):
            if i < self.split_point: block.to(device0)
            else: block.to(device1)
        model.norm.to(device1)
        model.lm_head.to(device1)
        model.mtp.to(device1)
    
    def forward(self, input_ids, labels=None, loss_mask=None, image_patches=None, video_patches=None):
        # Modified to handle extra args silently by just using input_ids for now, as vision pipeline needs proper cross-device handling
        x = self.model.embed(input_ids.to(self.device0))
        block_reprs = []
        partial_residual = torch.zeros_like(x)
        for i in range(self.split_point):
            block = self.model.blocks[i]
            if hasattr(block, 'delta_net'): x, partial_residual, _ = block(x, block_reprs, partial_residual)
            else: x, partial_residual = block(x, block_reprs, partial_residual)
            if (i + 1) % self.model.config.group_size == 0:
                block_reprs.append(partial_residual.detach())
                partial_residual = torch.zeros_like(x)
        x = x.to(self.device1)
        partial_residual = partial_residual.to(self.device1)
        block_reprs = [br.to(self.device1) for br in block_reprs]
        for i in range(self.split_point, len(self.model.blocks)):
            block = self.model.blocks[i]
            if hasattr(block, 'delta_net'): x, partial_residual, _ = block(x, block_reprs, partial_residual)
            else: x, partial_residual = block(x, block_reprs, partial_residual)
            if (i + 1) % self.model.config.group_size == 0:
                block_reprs.append(partial_residual.detach())
                partial_residual = torch.zeros_like(x)
        x = self.model.norm(x)
        logits = self.model.lm_head(x)
        result = {"logits": logits}
        if labels is not None:
            labels = labels.to(self.device1)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            B_logits, T_logits, _ = shift_logits.shape
            main_loss = F.cross_entropy(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1), reduction="none").view(B_logits, T_logits)
            if loss_mask is not None and self.model.config.selective_loss:
                shift_mask = loss_mask[:, 1:].to(self.device1).contiguous()
                main_loss = (main_loss * shift_mask).sum() / (shift_mask.sum() + 1e-10)
            else:
                main_loss = main_loss.mean()
            mtp_mask = loss_mask.to(self.device1) if loss_mask is not None else None
            mtp_loss = self.model.mtp(x, labels, self.model.embed.weight.to(self.device1), mtp_mask)
            result["loss"] = main_loss + self.model.config.mtp_weight * mtp_loss
            result["main_loss"] = main_loss
            result["mtp_loss"] = mtp_loss
        return result

# ═══════════════════════════════════════════════════════════════════════
#  Progress Tracking
# ═══════════════════════════════════════════════════════════════════════
class ProgressTracker:
    def __init__(self, total_target: int = 70_000_000_000):
        self.tokens_seen = 0
        self.total_target = total_target

    def add_tokens(self, count: int):
        self.tokens_seen += count

    def get_percentage(self) -> float:
        return (self.tokens_seen / self.total_target) * 100.0

    def state_dict(self):
        return {
            "tokens_seen": self.tokens_seen,
            "total_target": self.total_target,
            "percentage": self.get_percentage(),
            "timestamp": time.time()
        }

    def load_state_dict(self, state):
        self.tokens_seen = state.get("tokens_seen", 0)
        self.total_target = state.get("total_target", self.total_target)


# ═══════════════════════════════════════════════════════════════════════
#  Checkpointing
# ═══════════════════════════════════════════════════════════════════════
def save_checkpoint(
    model: nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    step: int,
    loss: float,
    config: ModelConfig,
    progress: ProgressTracker,
    checkpoint_dir: str,
    zip_checkpoint: bool = True
):
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    os.makedirs(checkpoint_dir, exist_ok=True)
    base_name = f"step_{step:06d}"
    tmp_dir = os.path.join(checkpoint_dir, f"tmp_{base_name}")
    os.makedirs(tmp_dir, exist_ok=True)
    
    model_to_save = model.module if isinstance(model, DDP) else model
    state = {
        "step": step,
        "loss": loss,
        "config": config.to_dict(),
        "model_state_dict": model_to_save.state_dict(),
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
    }
    torch.save(state, os.path.join(tmp_dir, "model.pt"))
    
    with open(os.path.join(tmp_dir, "progress.json"), "w") as f:
        json.dump(progress.state_dict(), f, indent=2)

    if zip_checkpoint:
        target_zip = os.path.join(checkpoint_dir, f"{base_name}.zip")
        shutil.make_archive(target_zip.replace('.zip', ''), 'zip', tmp_dir)
        shutil.rmtree(tmp_dir)
        print(f"  Checkpoint saved: step={step}, loss={loss:.4f} → {target_zip}")
    else:
        # Just rename tmp_dir
        final_dir = os.path.join(checkpoint_dir, base_name)
        if os.path.exists(final_dir): shutil.rmtree(final_dir)
        os.rename(tmp_dir, final_dir)
        print(f"  Checkpoint saved: step={step}, loss={loss:.4f} → {final_dir}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    progress: Optional[ProgressTracker] = None
) -> int:
    is_zip = path.endswith('.zip')
    if is_zip:
        tmp_extract = path + "_tmp_extract"
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(tmp_extract)
        pt_path = os.path.join(tmp_extract, "model.pt")
        prog_path = os.path.join(tmp_extract, "progress.json")
    else:
        pt_path = path if path.endswith('.pt') else os.path.join(path, "model.pt")
        prog_path = os.path.join(os.path.dirname(pt_path), "progress.json")

    state = torch.load(pt_path, map_location="cpu", weights_only=False)
    model_to_load = model.module if isinstance(model, DDP) else model
    model_to_load.load_state_dict(state["model_state_dict"])
    
    if optimizers is not None:
        for name, opt in optimizers.items():
            if name in state["optimizers"]:
                opt.load_state_dict(state["optimizers"][name])
                
    if progress is not None and os.path.exists(prog_path):
        with open(prog_path, "r") as f:
            progress.load_state_dict(json.load(f))

    if is_zip:
        shutil.rmtree(tmp_extract)

    print(f"  Checkpoint loaded: step={state['step']}, loss={state['loss']:.4f}")
    return state["step"]


def cosine_decay_with_warmup(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps: return step / warmup_steps
    elif step >= total_steps: return min_lr_ratio
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

# ═══════════════════════════════════════════════════════════════════════
#  Signal Handler
# ═══════════════════════════════════════════════════════════════════════
class GracefulExiter:
    def __init__(self):
        self.state = False
        signal.signal(signal.SIGINT, self.change_state)
        signal.signal(signal.SIGTERM, self.change_state)
    def change_state(self, signum, frame):
        print("\nSignal received! Will checkpoint and exit after current step...")
        self.state = True
    def should_exit(self):
        return self.state

# ═══════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class TrainConfig:
    data_dir: str = "./data"
    data_format: str = "auto"
    batch_size: int = 4
    grad_accumulation: int = 16
    total_steps: int = 100000
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 500
    log_interval: int = 10
    use_pipeline_parallel: bool = False
    use_ddp: bool = False
    use_amp: bool = True
    use_wandb: bool = True
    wandb_project: str = "Concord-3b"
    wandb_run_name: str = ""
    resume_from: Optional[str] = None
    tokenizer_name: str = "./model/custom_tokenizer"

def build_optimizers(model: nn.Module) -> Dict[str, torch.optim.Optimizer]:
    raw_model = model.module if isinstance(model, DDP) else model
    param_groups = raw_model.get_param_groups()
    optimizers = {}
    for group in param_groups:
        name = group.pop("name")
        opt_type = group.pop("optimizer")
        params = group.pop("params")
        if not params: continue
        if opt_type == "sgd": optimizers[name] = torch.optim.SGD([{"params": params, **group}])
        elif opt_type == "muon": optimizers[name] = Muon([{"params": params, **group}])
        elif opt_type == "adamw":
            betas = group.pop("betas", (0.9, 0.95))
            optimizers[name] = torch.optim.AdamW([{"params": params, **group}], betas=betas)
        else: raise ValueError(f"Unknown optimizer: {opt_type}")
    return optimizers


def train_worker(local_rank: int, world_size: int, model_config: ModelConfig, train_config: TrainConfig):
    dist_initialized = False
    if train_config.use_ddp:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist_initialized = True
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    if rank == 0: print(f"  Device: {device}")

    if rank == 0: print("  Building model...")
    model = HybridMoE(model_config)

    tokenizer = TokenizerWrapper(model_name_or_path=train_config.tokenizer_name)
    formatter = HarnessFormatter(tokenizer, corpus_tokens=model_config.corpus_tokens)
    computed_seq_len = formatter.get_seq_len()
    if rank == 0: print(f"  Computed dynamic seq_len: {computed_seq_len} (overhead: {computed_seq_len - model_config.corpus_tokens})")

    pipeline = None
    if train_config.use_pipeline_parallel and torch.cuda.device_count() >= 2 and not train_config.use_ddp:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
        pipeline = PipelineParallel(model, device0, device1)
        if rank == 0: print(f"  Pipeline parallel: GPU0 ← layers 0-11, GPU1 ← layers 12-23")
    else:
        model = model.to(device)
        if train_config.use_ddp:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizers = build_optimizers(model)
    grad_offloader = CPUGradientOffloader(model) if model_config.grad_offload_cpu else None
    progress_tracker = ProgressTracker(total_target=70_000_000_000)
    start_step = 0

    if train_config.resume_from and os.path.exists(train_config.resume_from):
        start_step = load_checkpoint(train_config.resume_from, model, optimizers, progress_tracker)

    dfmt = train_config.data_format
    if dfmt == "auto":
        # Check if HF format via prefix
        if train_config.data_dir.startswith("hf:"):
            dfmt = "huggingface"
        else:
            has_bin = any(f.endswith('.bin') for f in os.listdir(train_config.data_dir)) if os.path.exists(train_config.data_dir) else False
            dfmt = "memmap" if has_bin else "parquet_jsonl"

    start_samples = start_step * train_config.grad_accumulation * train_config.batch_size * (dist.get_world_size() if dist_initialized else 1)
    
    # NEW DYNAMIC HF LOADER OVERRIDE
    from model.dataset import HuggingFaceDataset
    if train_config.data_dir.startswith("hf:") or dfmt == "huggingface":
        hf_path = train_config.data_dir.replace("hf:", "")
        dataset = HuggingFaceDataset(dataset_path=hf_path, formatter=formatter, progress_tracker=progress_tracker)
    elif dfmt == "memmap":
        dataset = MemmapPretrainingDataset(data_dir=train_config.data_dir, seq_len=computed_seq_len, start_sample_idx=start_samples)
    else:
        dataset = JsonlParquetDataset(data_dir=train_config.data_dir, formatter=formatter, progress_tracker=progress_tracker)

    dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    if train_config.use_wandb and rank == 0:
        try:
            import wandb
            wandb.init(project=train_config.wandb_project, name=train_config.wandb_run_name or f"run-{int(time.time())}", 
                       config={**model_config.to_dict(), "batch_size": train_config.batch_size, "seq_len": computed_seq_len})
        except ImportError:
            train_config.use_wandb = False

    use_amp = torch.cuda.is_available() and train_config.use_amp
    if use_amp and rank == 0: print("  AMP enabled: Autocast + GradScaler (float16)")
    # Using modern GraduationScaler
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    step = start_step
    accum_loss, accum_main_loss, accum_mtp_loss, accum_count = 0.0, 0.0, 0.0, 0
    t_start = time.time()
    data_iter = iter(dataloader)
    
    # Capture initial LR before scheduler modifies it
    for opt in optimizers.values():
        for pg in opt.param_groups:
            if "initial_lr" not in pg:
                pg["initial_lr"] = pg["lr"]

    exiter = GracefulExiter()

    while step < train_config.total_steps:
        if exiter.should_exit(): break
        
        # ── LR schedule ───────────────────────────────────────────────
        lr_scale = cosine_decay_with_warmup(step, train_config.total_steps, train_config.warmup_steps, train_config.min_lr_ratio)
        for opt in optimizers.values():
            for pg in opt.param_groups:
                pg["lr"] = pg["initial_lr"] * lr_scale

        for micro_step in range(train_config.grad_accumulation):
            try: batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            loss_mask = batch.get("loss_mask")
            if loss_mask is not None: loss_mask = loss_mask.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                if pipeline is not None:
                    result = pipeline.forward(input_ids, labels=input_ids, loss_mask=loss_mask)
                else:
                    result = model(input_ids, labels=input_ids, loss_mask=loss_mask)

            loss = result["loss"] / train_config.grad_accumulation
            scaler.scale(loss).backward()

            if grad_offloader is not None and micro_step < train_config.grad_accumulation - 1:
                grad_offloader.accumulate()

            accum_loss += result.get("loss", torch.tensor(0.0)).item()
            accum_main_loss += result.get("main_loss", torch.tensor(0.0)).item()
            accum_mtp_loss += result.get("mtp_loss", torch.tensor(0.0)).item()
            accum_count += 1

        if grad_offloader is not None: grad_offloader.restore()

        non_expert_params = []
        for name, opt in optimizers.items():
            if name != "expert_sgd":
                for pg in opt.param_groups: non_expert_params.extend(pg["params"])
        
        grad_norm = 0.0
        if non_expert_params:
            for opt in optimizers.values(): scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(non_expert_params, max_norm=1.0).item()

        for opt in optimizers.values(): scaler.step(opt)
        scaler.update()

        for opt in optimizers.values(): opt.zero_grad(set_to_none=True)

        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.sync_fp8_weights()
        raw_model.maybe_update_router_biases()

        step += 1

        if step % train_config.log_interval == 0 and rank == 0:
            avg_loss = accum_loss / max(accum_count, 1)
            avg_main = accum_main_loss / max(accum_count, 1)
            avg_mtp = accum_mtp_loss / max(accum_count, 1)
            elapsed = time.time() - t_start
            tokens_per_sec = (accum_count * train_config.batch_size * computed_seq_len / elapsed) * (dist.get_world_size() if dist_initialized else 1)
            pct_val = progress_tracker.get_percentage()
            print(f"  step {step:>6d} | loss {avg_loss:.4f} | main {avg_main:.4f} | mtp {avg_mtp:.4f} | lr {lr_scale:.4f} | pct {pct_val:.4f}% | tok/s {tokens_per_sec:.0f}")
            if train_config.use_wandb:
                import wandb
                wandb.log({"loss": avg_loss, "main_loss": avg_main, "mtp_loss": avg_mtp, "lr_scale": lr_scale, "tokens_per_sec": tokens_per_sec, "step": step, "progress_pct": pct_val})
            accum_loss, accum_main_loss, accum_mtp_loss, accum_count = 0.0, 0.0, 0.0, 0
            t_start = time.time()

        if step % train_config.checkpoint_interval == 0:
            avg_loss = accum_loss / max(accum_count, 1)
            save_checkpoint(model, optimizers, step, avg_loss, model_config, progress_tracker, train_config.checkpoint_dir)

    if rank == 0:
        print(f"\n  Training complete/interrupted. Final step: {step}")
        final_path = os.path.join(train_config.checkpoint_dir, "final")
        save_checkpoint(model, optimizers, step, 0.0, model_config, progress_tracker, train_config.checkpoint_dir, zip_checkpoint=True)

    if dist_initialized:
        dist.destroy_process_group()

    return model

def train(model_config: ModelConfig = None, train_config: TrainConfig = None):
    if model_config is None: model_config = ModelConfig()
    if train_config is None: train_config = TrainConfig()

    try:
        if train_config.use_ddp:
            world_size = torch.cuda.device_count()
            if world_size < 2:
                print("DDP requested but less than 2 GPUs mapped. Falling back to single.")
                train_config.use_ddp = False
                train_worker(0, 1, model_config, train_config)
            else:
                print(f"Native mp.spawn scaling across {world_size} target devices...")
                import torch.multiprocessing as mp
                mp.spawn(train_worker, args=(world_size, model_config, train_config), nprocs=world_size, join=True)
        else:
            train_worker(0, 1, model_config, train_config)
    except KeyboardInterrupt:
        print("\n[Main Orchestrator] KeyboardInterrupt caught. Handled gracefully by workers saving ZIPs.")
