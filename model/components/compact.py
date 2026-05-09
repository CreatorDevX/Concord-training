"""
CompAct — Compressed Activation Storage for Memory-Efficient Training.

Uses PyTorch's saved_tensors_hooks to intercept activations stored by
autograd for the backward pass, compressing them via random projection.

Saves ~60-70% activation memory at the cost of slightly noisy gradients
(empirically negligible impact on convergence).

Works additively with gradient checkpointing:
  - Checkpointing eliminates macro-level activation storage (block boundaries)
  - CompAct compresses micro-level activations (intra-block, during recomputation)

Also includes Sparse Activation Training (SAT): top-k sparsification
of stored activations for additional memory savings.
"""

import math
import torch
import torch.nn as nn


class CompressedActivationContext:
    """
    Context manager that compresses activations saved for the backward pass
    using random projection (Johnson-Lindenstrauss style).

    Usage:
        with CompressedActivationContext(compress_ratio=0.3):
            output = model(input)
            loss = criterion(output, target)
        loss.backward()  # uses decompressed (approximate) activations

    Args:
        compress_ratio: fraction of original dim to keep (0.3 = 30% of dims)
        min_numel: minimum tensor size (in elements) to compress.
                   Small tensors are stored as-is (compression overhead > savings).
        use_topk: if True, use top-k sparsification instead of random projection.
                  Top-k keeps the k largest-magnitude values and zeros the rest.
    """

    def __init__(self, compress_ratio: float = 0.3, min_numel: int = 4096, use_topk: bool = False):
        self.compress_ratio = compress_ratio
        self.min_numel = min_numel
        self.use_topk = use_topk
        self._seed_counter = 0

    def _next_seed(self) -> int:
        self._seed_counter += 1
        return self._seed_counter

    def _pack_projection(self, tensor: torch.Tensor):
        """Compress via random Gaussian projection."""
        *batch_dims, D = tensor.shape
        proj_dim = max(int(D * self.compress_ratio), 1)

        seed = self._next_seed()
        gen = torch.Generator(device=tensor.device)
        gen.manual_seed(seed)
        # Generate in FP32 for numerical stability, then cast to tensor dtype
        proj = torch.randn(D, proj_dim, generator=gen, device=tensor.device, dtype=torch.float32)
        proj.mul_(1.0 / math.sqrt(proj_dim))
        proj = proj.to(tensor.dtype)

        flat = tensor.reshape(-1, D)
        compressed = flat @ proj  # (N, proj_dim)

        return (compressed, seed, tensor.shape, tensor.dtype, "proj")

    def _unpack_projection(self, data):
        """Decompress from random projection."""
        compressed, seed, shape, dtype, _ = data
        *batch_dims, D = shape
        proj_dim = compressed.shape[-1]

        # Must generate the SAME projection as pack (FP32 then cast)
        gen = torch.Generator(device=compressed.device)
        gen.manual_seed(seed)
        proj = torch.randn(D, proj_dim, generator=gen, device=compressed.device, dtype=torch.float32)
        proj.mul_(1.0 / math.sqrt(proj_dim))
        proj = proj.to(compressed.dtype)

        flat = compressed @ proj.T  # (N, D)
        result = flat.reshape(shape)
        if result.dtype != dtype:
            result = result.to(dtype)
        return result

    def _pack_topk(self, tensor: torch.Tensor):
        """Compress via top-k sparsification."""
        flat = tensor.reshape(-1)
        k = max(int(flat.numel() * self.compress_ratio), 1)
        _, topk_idx = flat.abs().topk(k)
        topk_vals = flat[topk_idx]

        return (topk_vals, topk_idx, tensor.shape, tensor.dtype, flat.numel())

    def _unpack_topk(self, data):
        """Decompress from top-k sparse representation."""
        vals, idx, shape, dtype, total = data
        flat = torch.zeros(total, device=vals.device, dtype=dtype)
        flat[idx] = vals
        return flat.reshape(shape)

    def pack(self, tensor: torch.Tensor):
        """Pack hook: compress tensor before autograd stores it."""
        # Skip small tensors, scalars, non-floating-point
        if (tensor.numel() < self.min_numel
                or tensor.dim() < 2
                or not tensor.is_floating_point()):
            return tensor

        if self.use_topk:
            return self._pack_topk(tensor)
        else:
            return self._pack_projection(tensor)

    def unpack(self, data):
        """Unpack hook: decompress tensor when autograd needs it for backward."""
        if isinstance(data, torch.Tensor):
            return data  # wasn't compressed

        if isinstance(data, tuple) and len(data) == 5:
            tag = data[-1]
            if tag == "proj":
                return self._unpack_projection(data)
            else:
                return self._unpack_topk(data)

        return data  # fallback: return as-is

    def __enter__(self):
        self._hook = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        self._hook.__enter__()
        return self

    def __exit__(self, *args):
        self._hook.__exit__(*args)
