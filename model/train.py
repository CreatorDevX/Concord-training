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
from datetime import datetime
from contextlib import nullcontext

# Setup PyTorch allocator BEFORE importing torch to ensure limits are respected natively
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

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
#  Adafactor Optimizer — Sublinear Memory via Factored Second Moments
#  For 2D params: stores O(m+n) state instead of O(m*n).
#  Replaces Muon + Lion to eliminate ~1.6GB optimizer overhead.
# ═══════════════════════════════════════════════════════════════════════
class Adafactor(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, eps=(1e-30, 1e-3), clip_threshold=1.0,
                 decay_rate=-0.8, weight_decay=0.0):
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                       decay_rate=decay_rate, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.float().norm(2) / max(tensor.numel() ** 0.5, 1.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps_sq, eps_root = group["eps"]
            clip = group["clip_threshold"]
            decay_rate = group["decay_rate"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                shape = grad.shape
                factored = len(shape) >= 2
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if factored:
                        state["v_row"] = torch.zeros(shape[:-1], dtype=torch.float32, device=p.device)
                        state["v_col"] = torch.zeros(shape[:-2] + shape[-1:], dtype=torch.float32, device=p.device)
                    else:
                        state["v"] = torch.zeros(shape, dtype=torch.float32, device=p.device)
                state["step"] += 1
                rho = min(1.0 - state["step"] ** decay_rate, 0.999)

                # Compute row/col means of squared gradient without full grad_sq temp
                if factored:
                    row_sq_mean = (grad * grad).mean(dim=-1)
                    col_sq_mean = (grad * grad).mean(dim=-2)
                    v_row = state["v_row"]
                    v_col = state["v_col"]
                    v_row.mul_(rho).add_(row_sq_mean, alpha=1.0 - rho)
                    v_row.add_(eps_sq * (1.0 - rho))
                    v_col.mul_(rho).add_(col_sq_mean, alpha=1.0 - rho)
                    v_col.add_(eps_sq * (1.0 - rho))

                    rms = v_row.mean(dim=-1, keepdim=True).clamp(min=eps_sq).sqrt()
                    update = grad.clone()
                    update.mul_(rms.unsqueeze(-1))
                    update.div_(v_row.unsqueeze(-1).sqrt())
                    update.div_(v_col.unsqueeze(-2).sqrt())
                else:
                    v = state["v"]
                    grad_sq = grad * grad
                    v.mul_(rho).add_(grad_sq, alpha=1.0 - rho)
                    v.add_(eps_sq * (1.0 - rho))
                    update = grad / (v.sqrt().add_(eps_root))

                update_rms = self._rms(update)
                if update_rms > clip:
                    update.mul_(clip / update_rms)
                if wd > 0:
                    p.data.add_(p.data, alpha=-wd * lr)
                p.data.add_(update.to(p.data.dtype), alpha=-lr)
        return loss

# ═══════════════════════════════════════════════════════════════════════
#  Pipeline Parallelism
# ═══════════════════════════════════════════════════════════════════════
class PipelineParallel:
    def __init__(self, model: HybridMoE, device0: torch.device, device1: torch.device):
        self.model = model
        self.device0 = device0
        self.device1 = device1
        # Split at group boundary: even split across GPUs
        self.split_point = model.config.group_size * (model.config.n_groups // 2)
        
        import copy
        self.attn_res_1 = copy.deepcopy(model.attn_res).to(device1)
        
        model.attn_res.to(device0)
        model.embed_up.to(device0)
        model.embed.to(device0)
        
        for i, block in enumerate(model.blocks):
            if i < self.split_point:
                block.to(device0)
                object.__setattr__(block, '_attn_res', model.attn_res)
            else:
                block.to(device1)
                object.__setattr__(block, '_attn_res', self.attn_res_1)
        
        model.norm.to(device1)
        model.lm_head.to(device1)
        model.embed_down.to(device1)
        model.mtp.to(device1)
        
        self.use_ckpt = model.config.grad_checkpoint
        if self.use_ckpt:
            print(f"  Pipeline: Gradient checkpointing enabled (saves ~60% activation memory)")
        print(f"  Pipeline: GPU0 ← blocks 0-{self.split_point-1}, GPU1 ← blocks {self.split_point}-{len(model.blocks)-1}")
        
        # Track delta states for DeltaBlocks across pipeline forward calls
        self._delta_states = {}
    
    def forward(self, input_ids, labels=None, loss_mask=None, image_patches=None, video_patches=None, 
                timestamps_ms=None, spatial_coords=None, position_ids=None):
        embed_device = self.model.embed.weight.device
        x = self.model.embed(input_ids.to(embed_device)).to(self.device0)
        x = self.model.embed_up(x).to(self.device0)

        # ── Multimodal Prefix Injection ──────────────────────────────────
        visual_tokens = None
        if getattr(self.model.config, 'use_vision', True):
            if image_patches is not None and self.model.vision_connector is not None:
                visual_tokens = self.model.vision_connector(image_patches)
            elif video_patches is not None and self.model.video_encoder is not None:
                visual_tokens = self.model.video_encoder(video_patches)

        if visual_tokens is not None:
            visual_tokens = visual_tokens.to(self.device0)
            x = torch.cat([visual_tokens, x], dim=1)
            visual_len = visual_tokens.shape[1]
            B_vis = x.shape[0]
            if labels is not None:
                visual_labels = torch.full((B_vis, visual_len), -100, device=labels.device, dtype=labels.dtype)
                labels = torch.cat([visual_labels, labels.to(self.device0)], dim=1)
            if loss_mask is not None:
                v_mask = torch.zeros((B_vis, visual_len), device=loss_mask.device, dtype=loss_mask.dtype)
                loss_mask = torch.cat([v_mask, loss_mask.to(self.device0)], dim=1)

        # ── Prepare AgentRoPE Coords ──────────────────────────────────
        coords = {}
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        coords["u"] = position_ids

        if timestamps_ms is not None:
            ref_ms = timestamps_ms[:, -1:]
            coords["w"] = self.model.agent_rope.encode_wallclock(timestamps_ms.to(self.device0), ref_ms.to(self.device0))
        else:
            coords["w"] = torch.zeros_like(position_ids).float().to(self.device0)

        if spatial_coords is not None:
            coords["x"] = spatial_coords.get("x", torch.zeros_like(position_ids).float()).to(self.device0)
            coords["y"] = spatial_coords.get("y", torch.zeros_like(position_ids).float()).to(self.device0)
        else:
            coords["x"] = torch.zeros_like(position_ids).float().to(self.device0)
            coords["y"] = torch.zeros_like(position_ids).float().to(self.device0)
        
        block_reprs = []
        partial_residual = torch.zeros_like(x)
        total_aux_loss = torch.zeros((), device=self.device0, dtype=torch.float32)
        # Reset delta states for new batch (each batch is independent)
        self._delta_states = {}
        
        # --- First half: blocks on device0, attn_res on device0 ---
        delta_idx = 0
        for i in range(self.split_point):
            block = self.model.blocks[i]
            if hasattr(block, 'delta_net'):
                ds = self._delta_states.get(delta_idx, None)
                if self.use_ckpt:
                    def delta_ckpt_fn(b, x_in, br, pr, ds_in):
                        return b(x_in, br, pr, ds_in)
                    x, partial_residual, ds_out, aux_loss = torch.utils.checkpoint.checkpoint(
                        delta_ckpt_fn, block, x, list(block_reprs), partial_residual, ds,
                        use_reentrant=False
                    )
                else:
                    x, partial_residual, ds_out, aux_loss = block(x, block_reprs, partial_residual, ds)
                self._delta_states[delta_idx] = ds_out.detach() if ds_out is not None else None
                delta_idx += 1
            else:
                if self.use_ckpt:
                    def attn_ckpt_fn(b, x_in, br, pr, pid, c):
                        return b(x_in, br, pr, pid, c)
                    x, partial_residual, aux_loss = torch.utils.checkpoint.checkpoint(
                        attn_ckpt_fn, block, x, list(block_reprs), partial_residual, position_ids, coords,
                        use_reentrant=False
                    )
                else:
                    x, partial_residual, aux_loss = block(x, block_reprs, partial_residual, position_ids, coords)
            
            total_aux_loss = total_aux_loss + aux_loss
            if (i + 1) % self.model.config.group_size == 0:
                block_reprs.append(partial_residual.detach())
                partial_residual = torch.zeros_like(x)
        
        # --- Cross-device transfer at split boundary ---
        x = x.to(self.device1)
        partial_residual = partial_residual.to(self.device1)
        block_reprs = [br.to(self.device1) for br in block_reprs]
        position_ids = position_ids.to(self.device1)
        coords = {k: v.to(self.device1) for k, v in coords.items()}
        total_aux_loss = total_aux_loss.to(self.device1)
        
        # Transfer delta states to device1
        delta_states_dev1 = {}
        for k, v in self._delta_states.items():
            if v is not None:
                delta_states_dev1[k] = v.to(self.device1)
        self._delta_states = delta_states_dev1
        
        # --- Second half: blocks on device1, attn_res on device1 ---
        for i in range(self.split_point, len(self.model.blocks)):
            block = self.model.blocks[i]
            if hasattr(block, 'delta_net'):
                ds = self._delta_states.get(delta_idx, None)
                if self.use_ckpt:
                    def delta_ckpt_fn(b, x_in, br, pr, ds_in):
                        return b(x_in, br, pr, ds_in)
                    x, partial_residual, ds_out, aux_loss = torch.utils.checkpoint.checkpoint(
                        delta_ckpt_fn, block, x, list(block_reprs), partial_residual, ds,
                        use_reentrant=False
                    )
                else:
                    x, partial_residual, ds_out, aux_loss = block(x, block_reprs, partial_residual, ds)
                self._delta_states[delta_idx] = ds_out.detach() if ds_out is not None else None
                delta_idx += 1
            else:
                if self.use_ckpt:
                    def attn_ckpt_fn(b, x_in, br, pr, pid, c):
                        return b(x_in, br, pr, pid, c)
                    x, partial_residual, aux_loss = torch.utils.checkpoint.checkpoint(
                        attn_ckpt_fn, block, x, list(block_reprs), partial_residual, position_ids, coords,
                        use_reentrant=False
                    )
                else:
                    x, partial_residual, aux_loss = block(x, block_reprs, partial_residual, position_ids, coords)
            
            total_aux_loss = total_aux_loss + aux_loss
            if (i + 1) % self.model.config.group_size == 0:
                block_reprs.append(partial_residual.detach())
                partial_residual = torch.zeros_like(x)
        
        x = self.model.norm(x)
        result = {}
        
        if labels is not None:
            labels = labels.to(self.device1)
            # Find indices where we actually want to compute loss (t+1 must be in mask)
            # If loss_mask[t+1] == 1, then logits[t] is "active"
            if loss_mask is not None and self.model.config.selective_loss:
                mask = loss_mask.to(self.device1)
                active_mask = (mask[:, 1:] == 1).reshape(-1)
            else:
                active_mask = torch.ones(x.shape[0] * (x.shape[1] - 1), dtype=torch.bool, device=x.device)

            # Slice hidden states and labels BEFORE massive vocab expansion
            x_shift = x[:, :-1, :].reshape(-1, x.shape[-1])[active_mask]
            labels_shift = labels[:, 1:].reshape(-1)[active_mask]
            
            if x_shift.shape[0] > 0:
                # Predict ONLY over masked tokens! This saves massive activation memory.
                # Chunked cross entropy to perfectly restrain peak memory (256 * 262k * dtype ~ 134MB)
                chunk_size = 128  # 128 × 262k vocab × 4B ≈ 134MB per chunk, peak-controlled
                main_loss = 0.0
                total_tokens = x_shift.shape[0]
                for chunk_start in range(0, total_tokens, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_tokens)
                    x_chunk = x_shift[chunk_start:chunk_end]
                    y_chunk = labels_shift[chunk_start:chunk_end]
                    def compute_chunk_loss(xc, yc):
                        logits_chunk = self.model.lm_head(self.model.embed_down(xc))
                        logits_chunk = logits_chunk.float()
                        logits_chunk = torch.nan_to_num(logits_chunk, nan=0.0, posinf=65504.0, neginf=-65504.0)
                        return F.cross_entropy(logits_chunk, yc, reduction="sum", ignore_index=-100)
                    
                    if x_chunk.requires_grad:
                        loss_chunk = torch.utils.checkpoint.checkpoint(
                            compute_chunk_loss, x_chunk, y_chunk, use_reentrant=False
                        )
                    else:
                        loss_chunk = compute_chunk_loss(x_chunk, y_chunk)
                        
                    main_loss = main_loss + loss_chunk
                
                main_loss = main_loss / total_tokens
            else:
                main_loss = torch.tensor(0.0, device=x.device)

            mtp_mask = loss_mask.to(self.device1) if loss_mask is not None else None
            mtp_loss = self.model.mtp(x, labels, loss_mask=mtp_mask, embed_down=self.model.embed_down)
            
            # total_loss = main + mtp + aux
            # NOTE: aux_loss_coeff is already applied inside router.py — do NOT re-apply here
            total_loss = main_loss + self.model.config.mtp_weight * mtp_loss + total_aux_loss
            
            result["loss"] = total_loss
            result["main_loss"] = main_loss
            result["mtp_loss"] = mtp_loss
            result["aux_loss"] = total_aux_loss
        else:
            # Inference or no-labels path
            logits = self.model.lm_head(self.model.embed_down(x))
            result["logits"] = logits
            
        return result

    def sync_gradients(self):
        for p0, p1 in zip(self.model.attn_res.parameters(), self.attn_res_1.parameters()):
            if p0.grad is not None and p1.grad is not None:
                p0.grad.add_(p1.grad.to(p0.device))
            elif p1.grad is not None:
                p0.grad = p1.grad.to(p0.device)
                
    def sync_weights(self):
        for p0, p1 in zip(self.model.attn_res.parameters(), self.attn_res_1.parameters()):
            p1.data.copy_(p0.data.to(p1.device))

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
    zip_checkpoint: bool = True,
    is_tpu: bool = False
):
    rank = 0
    if is_tpu:
        import torch_xla.core.xla_model as xm
        rank = xm.get_ordinal()
    elif dist.is_initialized():
        rank = dist.get_rank()
        
    if rank != 0:
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
    if step < warmup_steps: return step / max(warmup_steps, 1)
    elif step >= total_steps: return min_lr_ratio
    denom = max(total_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / denom
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
    total_steps: int = 100000
    warmup_steps: int = 2000
    grad_accum_steps: int = 1
    min_lr_ratio: float = 0.1
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 500
    log_interval: int = 10
    use_pipeline_parallel: bool = False
    use_ddp: bool = False
    use_tpu: bool = False
    use_amp: bool = True
    use_wandb: bool = True
    wandb_project: str = "Concord-3b"
    wandb_run_name: str = ""
    resume_from: Optional[str] = None
    tokenizer_name: str = "./model/custom_tokenizer"

class NormalizedSGD(torch.optim.Optimizer):
    """
    Memory-efficient SGD improvement for expert weights with NO optimizer state.

    Features (all zero memory overhead):
      1. Gradient centralization: subtracts mean from gradient channels
         (reduces internal covariate shift, stabilizes training).
      2. Per-parameter gradient normalization: scales gradient by its RMS
         so all experts learn at a consistent rate regardless of token frequency.
      3. Weight decay (standard L2).

    The key insight: MoE experts receive wildly different gradient magnitudes
    depending on how many tokens were routed to each expert. NormalizedSGD
    ensures each expert's update step is uniformly scaled.

    Args:
        params: iterable of parameters to optimize
        lr: learning rate
        weight_decay: weight decay coefficient
        eps: small constant for numerical stability
        centralize: if True, apply gradient centralization
        normalize: if True, apply per-parameter gradient normalization
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.0, eps=1e-8,
                 centralize=True, normalize=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, eps=eps,
                        centralize=centralize, normalize=normalize)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            eps = group['eps']
            centralize = group['centralize']
            normalize = group['normalize']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # Gradient centralization: zero-mean along non-batch dims
                if centralize and grad.dim() > 1:
                    grad = grad - grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)

                # Gradient normalization: scale by RMS for consistent step size
                if normalize:
                    rms = grad.norm() / max(grad.numel() ** 0.5, 1.0)
                    grad = grad / (rms + eps)

                # Weight decay
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                # Update
                p.data.add_(grad, alpha=-lr)
        return loss


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # In-place math to avoid massive temporary tensor copies
                update = exp_avg.clone()
                update.mul_(beta1).add_(grad, alpha=1 - beta1)
                update.sign_()
                
                p.data.add_(update, alpha=-group['lr'])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                
                del update
        return loss

def build_optimizers(model: nn.Module) -> Dict[str, torch.optim.Optimizer]:
    raw_model = model.module if isinstance(model, DDP) else model
    param_groups = raw_model.get_param_groups()
    optimizers = {}
    for group in param_groups:
        name = group.pop("name")
        opt_type = group.pop("optimizer")
        params = group.pop("params")
        if not params: continue
        
        if opt_type == "sgd":
            group.pop("momentum", None)
            group.pop("nesterov", None)
            optimizers[name] = NormalizedSGD([{"params": params, **group}])
        elif opt_type == "lion":
            betas = group.pop("betas", (0.9, 0.99))
            optimizers[name] = Lion([{"params": params, **group}], betas=betas)
        elif opt_type in ("muon", "adafactor"):
            # Replace Muon with Adafactor as requested
            group.pop("momentum", None)
            group.pop("nesterov", None)
            group.pop("ns_steps", None)
            optimizers[name] = Adafactor([{"params": params, **group}])
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
    return optimizers


def train_worker(local_rank: int, world_size: int, model_config: ModelConfig, train_config: TrainConfig):
    dist_initialized = False
    is_tpu = train_config.use_tpu
    
    if is_tpu:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        if rank == 0: print("Initializing PyTorch XLA for TPU scaling...")
    elif train_config.use_ddp:
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
    # Build model in fp32 for numerical stability. autocast fp16 during forward.
    model = HybridMoE(model_config)

    # After fp32 build, cast expert FFN weights to fp16 to save memory
    # (experts are the largest weight component and fp16 is sufficient for them)
    if model_config.expert_dtype == "fp16":
        for name, param in model.named_parameters():
            if "routed_experts" in name or "shared_experts" in name:
                param.data = param.data.half()

    tokenizer = TokenizerWrapper(model_name_or_path=train_config.tokenizer_name)
    formatter = HarnessFormatter(tokenizer, corpus_tokens=model_config.corpus_tokens)
    computed_seq_len = formatter.get_seq_len()
    if rank == 0: print(f"  Computed dynamic seq_len: {computed_seq_len} (overhead: {computed_seq_len - model_config.corpus_tokens})")

    pipeline = None
    if train_config.use_pipeline_parallel and torch.cuda.device_count() >= 2 and not train_config.use_ddp:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
        pipeline = PipelineParallel(model, device0, device1)
        if rank == 0: print(f"  Pipeline parallel: GPU0 ← layers 0-{pipeline.split_point-1}, GPU1 ← layers {pipeline.split_point}-{len(model.blocks)-1}")
    else:
        # ── Single-GPU memory optimization ──────────────────────
        model = model.to(device)

    # DDP wrapping (must happen before expert offload since DDP wraps the model)
    if train_config.use_ddp and not is_tpu and pipeline is None:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizers = build_optimizers(model)
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

    start_samples = start_step * train_config.batch_size * (dist.get_world_size() if dist_initialized else 1)
    
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

    use_amp = torch.cuda.is_available() and train_config.use_amp and not is_tpu
    if use_amp and rank == 0:
        print("  AMP enabled: Autocast FP16 + FP32 loss upcast")

    from model.components.activation_recompute import ActivationRecomputer

    model.train()
    step = start_step
    accum_loss, accum_main_loss, accum_mtp_loss, accum_aux_loss = 0.0, 0.0, 0.0, 0.0
    t_start = time.time()
    data_iter = iter(dataloader)
    consecutive_nan_steps = 0
    
    # Capture initial LR before scheduler modifies it
    for opt in optimizers.values():
        for pg in opt.param_groups:
            if "initial_lr" not in pg:
                pg["initial_lr"] = pg["lr"]

    exiter = GracefulExiter()

    def audit_mem(point_name):
        if torch.cuda.is_available() and rank == 0:
            alloc = torch.cuda.memory_allocated() / (1024**3)
            res = torch.cuda.memory_reserved() / (1024**3)
            ts = datetime.now().strftime("%H:%M:%S")
            # Clear previous audit lines to avoid flooding, but keep them in background logs if redirected
            print(f"    [MEM] {ts} | {point_name:<30} | Alloc: {alloc:.2f}GB | Res: {res:.2f}GB", flush=True)

    audit_mem("Before Training Loop Begins")

    while step < train_config.total_steps:
        if exiter.should_exit(): break
        
        # ── LR schedule ───────────────────────────────────────────────
        lr_scale = cosine_decay_with_warmup(step, train_config.total_steps, train_config.warmup_steps, train_config.min_lr_ratio)
        for opt in optimizers.values():
            for pg in opt.param_groups:
                pg["lr"] = pg["initial_lr"] * lr_scale

        for opt in optimizers.values(): opt.zero_grad(set_to_none=True)
        
        accum_loss, accum_main_loss, accum_mtp_loss, accum_aux_loss = 0.0, 0.0, 0.0, 0.0
        accum_micro_steps = 0
        
        for micro_step in range(train_config.grad_accum_steps):
            try: batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            input_ids = batch["input_ids"].to(device)
            loss_mask = batch.get("loss_mask")
            if loss_mask is not None: loss_mask = loss_mask.to(device)

            # ── Wallclock injection ─────────────────────────────────────
            # Every forward pass gets the real Python wallclock so RoPE's
            # temporal axis encodes actual time. During inference, callers
            # can pass synthetic timestamps for injected context / user turns.
            wall_now_ms = time.time() * 1000.0
            timestamps_ms = torch.full(
                input_ids.shape, wall_now_ms, dtype=torch.float32, device=device
            )

            if is_tpu:
                if pipeline is not None:
                    result = pipeline.forward(input_ids, labels=input_ids, loss_mask=loss_mask, timestamps_ms=timestamps_ms)
                else:
                    result = model(input_ids, labels=input_ids, loss_mask=loss_mask, timestamps_ms=timestamps_ms)
                loss = result["loss"] / train_config.grad_accum_steps
            else:
                # NOTE: ActivationRecomputer (saved_tensors_hooks) and
                # torch.checkpoint (use_reentrant=False, also uses
                # saved_tensors_hooks) are mutually incompatible when nested.
                # Skip ActivationRecomputer when grad_checkpoint is ON
                # since block-level checkpoint already saves activation memory.
                if model_config.grad_checkpoint:
                    activate_recomputer = nullcontext()
                else:
                    activate_recomputer = ActivationRecomputer(
                        offload_threshold=512_000, compress_threshold=4_000_000,
                        min_numel=4096, compress_ratio=0.3)
                with activate_recomputer:
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        if pipeline is not None:
                            result = pipeline.forward(input_ids, labels=input_ids, loss_mask=loss_mask, timestamps_ms=timestamps_ms)
                        else:
                            result = model(input_ids, labels=input_ids, loss_mask=loss_mask, timestamps_ms=timestamps_ms)
                        loss = result["loss"] / train_config.grad_accum_steps
                
            # NaN detection: skip micro-batches with NaN/Inf loss
            micro_loss = result.get("loss", torch.tensor(0.0))
            if torch.isnan(micro_loss).any() or torch.isinf(micro_loss).any():
                if rank == 0:
                    print(f"  [WARN] NaN/Inf loss at step {step}, micro_step {micro_step}. Skipping.")
                continue
            
            accum_loss += micro_loss.item()
            accum_main_loss += result.get("main_loss", torch.tensor(0.0)).item()
            accum_mtp_loss += result.get("mtp_loss", torch.tensor(0.0)).item()
            accum_aux_loss += result.get("aux_loss", torch.tensor(0.0)).item()
            accum_micro_steps += 1

            loss.backward()

        # Step optimizers after accumulation
        if pipeline is not None:
            pipeline.sync_gradients()

        # Protect against NaN death spiral: if ALL micro-batches were NaN,
        # skip optimizer step (no grads to apply) and track consecutive NaN steps.
        if accum_micro_steps == 0:
            consecutive_nan_steps += 1
            if consecutive_nan_steps >= 100:
                if rank == 0:
                    print(f"\n  [FATAL] {consecutive_nan_steps} consecutive NaN steps. "
                          f"Aborting training. Check model initialization and data.")
                break
            if rank == 0:
                print(f"  [WARN] All micro-batches NaN at step {step} "
                      f"(consecutive: {consecutive_nan_steps}). Skipping step.")
            step += 1
            # Clear any partial gradients that may have been set
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            t_start = time.time()
            continue
        else:
            consecutive_nan_steps = 0

        # Detect and sanitize NaN/Inf gradients before optimizer step.
        # nan_to_num on grads is critical: a single NaN/Inf grad poisons
        # the entire parameter — and without this guard, clip_grad_norm_
        # of an Inf-normed tensor produces NaN for ALL params in the group.
        has_nan_grad = False
        all_params = []
        for pg in optimizers.values():
            for p in pg.param_groups[0]["params"]:
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        has_nan_grad = True
                        p.grad.zero_()
                    else:
                        all_params.append(p)

        if has_nan_grad and rank == 0:
            print(f"  [WARN] NaN/Inf grads detected at step {step}. Zeroing affected params.")

        if is_tpu:
            import torch_xla.core.xla_model as xm
            if all_params:
                raw_model = model.module if isinstance(model, DDP) else model
                xm.reduce_gradients(raw_model)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            for opt in optimizers.values(): 
                xm.optimizer_step(opt)
        else:
            if all_params:
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            for opt in optimizers.values():
                opt.step()

        if pipeline is not None:
            pipeline.sync_weights()

        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.maybe_update_router_biases()

        step += 1

        if step % train_config.log_interval == 0 and rank == 0:
            accum_div = max(accum_micro_steps, 1)
            avg_loss = accum_loss / accum_div
            avg_main = accum_main_loss / accum_div
            avg_mtp = accum_mtp_loss / accum_div
            avg_aux = accum_aux_loss / accum_div
            elapsed = time.time() - t_start
            tokens_per_sec = (train_config.batch_size * train_config.grad_accum_steps * computed_seq_len / elapsed) * (dist.get_world_size() if dist_initialized else 1)
            pct_val = progress_tracker.get_percentage()
            ts = datetime.now().strftime("%H:%M:%S")
            mem_str = ""
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024**3)
                mem_str = f" | mem {alloc:.2f}GB"
            eta_str = ""
            if tokens_per_sec > 0:
                remaining_steps = train_config.total_steps - step
                remaining_tokens = remaining_steps * train_config.batch_size * train_config.grad_accum_steps * computed_seq_len
                eta_secs = remaining_tokens / tokens_per_sec
                eta_h, eta_m = int(eta_secs // 3600), int((eta_secs % 3600) // 60)
                eta_str = f" | ETA {eta_h}h{eta_m:02d}m"
            print(f"  [{ts}] step {step:>6d}/{train_config.total_steps} | loss {avg_loss:.4f} (main {avg_main:.4f} + mtp {avg_mtp:.4f} + aux {avg_aux:.4f}) | lr {lr_scale:.4f} | {tokens_per_sec:.0f} tok/s{mem_str}{eta_str}")
            if train_config.use_wandb:
                import wandb
                wandb.log({"loss": avg_loss, "main_loss": avg_main, "mtp_loss": avg_mtp, "aux_loss": avg_aux, "lr_scale": lr_scale, "tokens_per_sec": tokens_per_sec, "step": step, "progress_pct": pct_val})
            t_start = time.time()

        if step % train_config.checkpoint_interval == 0:
            avg_loss = accum_loss / max(accum_micro_steps, 1)
            save_checkpoint(model, optimizers, step, avg_loss, model_config, progress_tracker, train_config.checkpoint_dir, is_tpu=is_tpu)
            
            if rank == 0:
                print("\n  [Sample Inference]")
                raw_model.eval()
                with torch.no_grad():
                    # Check whether TokenizerWrapper exposes encode
                    prompt_str = "The origin of life is"
                    if hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "encode"):
                        samp_ids = tokenizer._tokenizer.encode(prompt_str, return_tensors="pt").to(device)
                    else:
                        # Fallback for dynamic tokenizer wrappers directly calling instance encode
                        samp_ids = torch.tensor([tokenizer.encode(prompt_str)], dtype=torch.long, device=device)
                    
                    for _ in range(20):
                        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                            if pipeline is not None:
                                out = pipeline.forward(samp_ids)
                            else:
                                out = raw_model(samp_ids)
                        next_token = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
                        samp_ids = torch.cat([samp_ids, next_token], dim=1)
                        
                    if hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "decode"):
                        print("  Output: " + tokenizer._tokenizer.decode(samp_ids[0].cpu().tolist()))
                    else:
                        print("  Output: " + tokenizer.decode(samp_ids[0].cpu().tolist()))
                raw_model.train()

    if rank == 0:
        print(f"\n  Training complete/interrupted. Final step: {step}")
        save_checkpoint(model, optimizers, step, 0.0, model_config, progress_tracker, train_config.checkpoint_dir, zip_checkpoint=True, is_tpu=is_tpu)

    if dist_initialized:
        dist.destroy_process_group()

    return model

def train(model_config: ModelConfig = None, train_config: TrainConfig = None):
    if model_config is None: model_config = ModelConfig()
    if train_config is None: train_config = TrainConfig()

    try:
        if train_config.use_tpu:
            print("Native PyTorch XLA requested. Scaling environment across TPU endpoints...")
            try:
                import torch_xla.distributed.xla_multiprocessing as xmp
                xmp.spawn(train_worker, args=(8, model_config, train_config), nprocs=8, start_method='fork')
            except ImportError:
                print("Error: torch_xla not installed. Run `pip install torch_xla` to utilize TPU scaling.")
        elif train_config.use_ddp:
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
