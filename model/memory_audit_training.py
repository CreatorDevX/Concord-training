"""
Memory Audit (Training) — Tracks GPU memory during a training step.

Usage:
    python -m model.memory_audit_training

This instantiates the full model, runs a forward + backward pass,
and reports detailed GPU memory allocation at each phase.
"""

import torch
import time
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.model import HybridMoE
from model.components.compact import CompressedActivationContext


class TrainingMemoryAuditor:
    def __init__(self, config: ModelConfig = None, batch_size: int = 1, seq_len: int = 128):
        self.config = config or ModelConfig()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.phase_memories = {}

    def _mem_gb(self):
        if not torch.cuda.is_available():
            return 0.0, 0.0
        return (
            torch.cuda.memory_allocated(self.device) / (1024 ** 3),
            torch.cuda.memory_reserved(self.device) / (1024 ** 3),
        )

    def _snapshot(self, label: str):
        alloc, reserved = self._mem_gb()
        self.phase_memories[label] = {"alloc_gb": alloc, "reserved_gb": reserved}
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {label:40s}  alloc={alloc:.3f}GB  reserved={reserved:.3f}GB")
        return alloc, reserved

    def run(self):
        if not torch.cuda.is_available():
            print("  CUDA not available. Running on CPU — memory tracking disabled.")
            return {}

        print("=" * 62)
        print(f"  Training Memory Audit")
        print(f"  batch_size={self.batch_size}, seq_len={self.seq_len}")
        print(f"  d_model={self.config.d_model}, n_layers={self.config.n_layers}")
        print(f"  n_experts={self.config.n_experts}, expert_dtype={self.config.expert_dtype}")
        print(f"  grad_checkpoint={self.config.grad_checkpoint}")
        print("=" * 62)
        print()

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
        self._snapshot("Initial (empty cache)")

        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)

        self._snapshot("Before model creation")
        model = HybridMoE(self.config).to(self.device)
        torch.set_default_dtype(orig_dtype)
        self._snapshot("After model.to(device)")

        model.train()

        input_ids = torch.randint(
            0, self.config.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device,
        )
        wall_ms = time.time() * 1000.0
        timestamps_ms = torch.full(
            (self.batch_size, self.seq_len), wall_ms,
            dtype=torch.float32, device=self.device,
        )
        self._snapshot("After input creation")

        # ── Forward (no grad checkpoint, no CompAct) ──
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            result = model(input_ids, labels=input_ids, timestamps_ms=timestamps_ms)
        loss = result["loss"]
        self._snapshot("After forward (no ckpt)")

        loss.backward()
        self._snapshot("After backward (no ckpt)")

        # Zero gradients and reset
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        self._snapshot("After grad zero + cache empty")

        # ── Forward (with grad checkpoint) ──
        self.config.grad_checkpoint = True
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            result = model(input_ids, labels=input_ids, timestamps_ms=timestamps_ms)
        loss = result["loss"]
        self._snapshot("After forward (with ckpt)")

        loss.backward()
        self._snapshot("After backward (with ckpt)")

        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        self._snapshot("After grad zero (ckpt)")

        # ── Forward (with ckpt + CompAct) ──
        with CompressedActivationContext(compress_ratio=0.3, min_numel=4096):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                result = model(input_ids, labels=input_ids, timestamps_ms=timestamps_ms)
            loss = result["loss"]
        self._snapshot("After forward (ckpt + CompAct)")

        loss.backward()
        self._snapshot("After backward (ckpt + CompAct)")

        # ── Summary ──
        peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        print()
        print(f"  Peak memory: {peak:.3f} GB")
        print(f"  Model weights: {self.phase_memories.get('After model.to(device)', {}).get('alloc_gb', 0):.3f} GB")

        model_size = self.phase_memories.get("After model.to(device)", {}).get("alloc_gb", 0) - \
                     self.phase_memories.get("Before model creation", {}).get("alloc_gb", 0)
        no_ckpt_forward = self.phase_memories.get("After forward (no ckpt)", {}).get("alloc_gb", 0) - \
                          self.phase_memories.get("After input creation", {}).get("alloc_gb", 0)
        ckpt_forward = self.phase_memories.get("After forward (with ckpt)", {}).get("alloc_gb", 0) - \
                       self.phase_memories.get("After grad zero + cache empty", {}).get("alloc_gb", 0)
        compact_forward = self.phase_memories.get("After forward (ckpt + CompAct)", {}).get("alloc_gb", 0) - \
                          self.phase_memories.get("After grad zero (ckpt)", {}).get("alloc_gb", 0)

        print()
        diff_phases = [
            ("Model weights estimate", model_size),
            ("Activation memory (no ckpt)", no_ckpt_forward),
            ("Activation memory (with ckpt)", ckpt_forward),
            ("Activation memory (ckpt + CompAct)", compact_forward),
        ]
        for label, val in diff_phases:
            print(f"  {label:35s}  {val:.3f} GB")
        print()
        print(f"  Peak GPU memory:  {peak:.3f} GB")
        print("=" * 62)

        del model, input_ids, timestamps_ms
        torch.cuda.empty_cache()
        return self.phase_memories


def memory_audit_training(config: ModelConfig = None, batch_size: int = 1, seq_len: int = 128):
    auditor = TrainingMemoryAuditor(config, batch_size, seq_len)
    return auditor.run()


if __name__ == "__main__":
    memory_audit_training()
