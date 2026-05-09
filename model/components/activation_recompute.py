"""
Activation Recomputation — Multi-strategy activation memory management.

Three strategies applied per-tensor based on size:
  1. Small tensors (< offload_threshold): Keep on GPU (no overhead)
  2. Medium tensors (< compress_threshold): Offload to CPU pinned memory
  3. Large tensors (>= compress_threshold): Compress via random projection

Works additively with gradient checkpointing:
  - Checkpointing eliminates block-level activation storage (macro)
  - Activation recomputation handles fine-grained intra-block tensors (micro)

Usage:
    model = HybridMoE(config)
    recompute = ActivationRecomputer(
        offload_threshold=1_000_000,   # tensors with >1M elems offloaded to CPU
        compress_threshold=10_000_000,  # tensors with >10M elems compressed
    )
    with recompute:
        loss = model(input_ids, labels=input_ids)
    loss.backward()
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class ActivationRecomputer:
    """
    Context manager for multi-strategy activation recomputation.

    Uses PyTorch's saved_tensors_hooks to intercept all tensors saved
    by autograd for the backward pass, then applies the optimal strategy.

    IMPORTANT: Seeds are derived deterministically from tensor metadata
    (shape + dtype + device) so that forward and checkpoint recomputation
    produce IDENTICAL random projections. This prevents gradient corruption
    and CheckpointError when used inside torch.checkpoint regions.

    Args:
        offload_threshold: Tensors with more elements than this are offloaded
                           to CPU pinned memory (saves GPU memory, costs PCIe BW).
        compress_threshold: Tensors with more elements than this are compressed
                            via random projection (saves GPU memory, adds noise).
        min_numel: Minimum elements to apply any strategy (< this → keep on GPU).
        compress_ratio: Compression ratio for random projection (0.3 = 30%).
    """

    def __init__(
        self,
        offload_threshold: int = 256_000,
        compress_threshold: int = 1_000_000,
        min_numel: int = 8192,
        compress_ratio: float = 0.3,
    ):
        self.offload_threshold = offload_threshold
        self.compress_threshold = compress_threshold
        self.min_numel = min_numel
        self.compress_ratio = compress_ratio

    def _tensor_key(self, tensor: torch.Tensor) -> int:
        return hash((tensor.shape, tensor.dtype, tensor.device)) & 0x7FFFFFFF

    def _offload_to_cpu(self, tensor: torch.Tensor) -> tuple:
        """Offload tensor to CPU pinned memory."""
        cpu_tensor = tensor.detach().to("cpu", non_blocking=tensor.is_cuda)
        if tensor.is_cuda:
            cpu_tensor = cpu_tensor.pin_memory()
        return (cpu_tensor, tensor.shape, tensor.dtype, tensor.device, "cpu_offload")

    def _load_from_cpu(self, data: tuple) -> torch.Tensor:
        """Load tensor back from CPU to original device."""
        cpu_tensor, shape, dtype, device, _ = data
        result = cpu_tensor.to(device, non_blocking=device.type == "cuda")
        if result.dtype != dtype:
            result = result.to(dtype)
        return result

    def _compress_projection(self, tensor: torch.Tensor) -> tuple:
        """Compress via random Gaussian projection.

        Runs projection in fp32 regardless of tensor dtype to avoid
        fp16 overflow. Results are cast back for storage.
        """
        *batch_dims, D = tensor.shape
        proj_dim = max(int(D * self.compress_ratio), 1)

        seed = self._tensor_key(tensor)
        gen = torch.Generator(device=tensor.device)
        gen.manual_seed(seed)
        proj = torch.randn(D, proj_dim, generator=gen, device=tensor.device, dtype=torch.float32)
        proj.mul_(1.0 / math.sqrt(proj_dim))

        flat = tensor.reshape(-1, D).float()
        compressed = flat @ proj
        compressed = compressed.to(tensor.dtype)

        return (compressed, seed, tensor.shape, tensor.dtype, "proj")

    def _decompress_projection(self, data: tuple) -> torch.Tensor:
        """Decompress from random projection (fp32 math for stability)."""
        compressed, seed, shape, dtype, _ = data
        *batch_dims, D = shape
        proj_dim = compressed.shape[-1]

        gen = torch.Generator(device=compressed.device)
        gen.manual_seed(seed)
        proj = torch.randn(D, proj_dim, generator=gen, device=compressed.device, dtype=torch.float32)
        proj.mul_(1.0 / math.sqrt(proj_dim))

        flat = compressed.float() @ proj.T
        result = flat.reshape(shape).to(dtype)
        return result

    def pack(self, tensor: torch.Tensor):
        """Pack hook: decide strategy and transform."""
        if (
            tensor.numel() < self.min_numel
            or tensor.dim() < 2
            or not tensor.is_floating_point()
        ):
            return tensor

        if tensor.numel() >= self.compress_threshold:
            return self._compress_projection(tensor)
        elif tensor.numel() >= self.offload_threshold and tensor.is_cuda:
            return self._offload_to_cpu(tensor)
        else:
            return tensor

    def unpack(self, data):
        """Unpack hook: restore original tensor."""
        if isinstance(data, torch.Tensor):
            return data

        if isinstance(data, tuple) and len(data) >= 4:
            tag = data[-1]
            if tag == "cpu_offload":
                return self._load_from_cpu(data)
            elif tag == "proj":
                return self._decompress_projection(data)

        return data

    def __enter__(self):
        self._hook = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        self._hook.__enter__()
        return self

    def __exit__(self, *args):
        self._hook.__exit__(*args)
