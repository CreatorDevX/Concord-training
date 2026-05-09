"""
Gated DeltaNet -- Linear Recurrent Attention.

State update (delta rule):
    S_t = S_{t-1} - beta * k_t (k_t^T S_{t-1}) + beta * k_t v_t^T
    y_t = S_t^T q_t
    out = gate_t * y_t

Fixes over original:
    - EMA state rescaling (not hard clipping) to prevent unbounded drift
    - Learned V-expand projection instead of repeat_interleave
    - State-aware gating (input + retrieval strength signal)
    - Minimized per-token allocations in chunk recurrence
    - No state.clone() -- operate on state directly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class FPLayerNorm(nn.Module):
    """LayerNorm that always runs in fp32 for numerical stability."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return self.norm(x.float()).to(dtype)


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet linear recurrent attention.

    Args:
        d_model: model dimension
        n_v_heads: number of value heads
        n_qk_heads: number of query/key heads
        head_dim: dimension per head
        chunk_size: chunk size for chunked recurrence
        state_target_norm: target norm for EMA state rescaling
    """

    def __init__(
        self,
        d_model: int,
        n_v_heads: int,
        n_qk_heads: int,
        head_dim: int,
        chunk_size: int = 64,
        state_target_norm: float = 10.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_v_heads = n_v_heads
        self.n_qk_heads = n_qk_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.state_target_norm = state_target_norm
        self.use_checkpoint = use_checkpoint

        # Total dims
        self.d_v = n_v_heads * head_dim
        self.d_qk = n_qk_heads * head_dim

        # Input projections
        self.q_proj = nn.Linear(d_model, self.d_qk, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_qk, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_v, bias=False)

        # Learned V-expand: project from n_v_heads to n_qk_heads
        # instead of repeat_interleave which destroys value subspace independence
        if n_qk_heads != n_v_heads:
            self.v_expand = nn.Linear(self.d_v, self.d_qk, bias=False)
        else:
            self.v_expand = None

        # Output projection
        self.o_proj = nn.Linear(self.d_qk if n_qk_heads != n_v_heads else self.d_v,
                                d_model, bias=False)

        # State-aware gating: input projection + state norm signal
        # gate = sigmoid(gate_proj(x) + state_gate_proj(state_summary))
        self.gate_proj = nn.Linear(d_model, self.d_qk if n_qk_heads != n_v_heads else self.d_v, bias=False)
        self.state_gate_scale = nn.Parameter(torch.zeros(1))  # learned scalar for state signal

        # Beta (per-head learning rate for delta rule update)
        self.beta_proj = nn.Linear(d_model, n_qk_heads, bias=False)

        # Per-head layer norm for state (applied after each chunk) — fp32 for stability
        self.state_norm = FPLayerNorm(head_dim)

        # Runtime stats
        self._last_state_norm = 0.0
        self._last_gate_mean = 0.0

    def _chunk_recurrence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single chunk with the delta rule recurrence.
        Minimized per-token allocations -- no state.clone(), pre-shaped tensors.

        Args:
            q: (B, H, C, D) queries
            k: (B, H, C, D) keys (normalized)
            v: (B, H, C, D_v) values
            beta: (B, H, C) learning rates
            state: (B, H, D, D_v) recurrent state

        Returns:
            output: (B, H, C, D_v)
            state: (B, H, D, D_v) updated state
        """
        B, H, C, D = q.shape
        D_v = v.shape[-1]
        dtype = state.dtype

        state_fp32 = state.float()

        output = torch.zeros(B, H, C, D_v, dtype=dtype, device=q.device)

        for t in range(C):
            q_t = q[:, :, t, :].float()
            k_t = k[:, :, t, :].float()
            v_t = v[:, :, t, :].float()
            b_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1).float()

            k_t_row = k_t.unsqueeze(-2)
            k_t_col = k_t.unsqueeze(-1)

            kS = torch.matmul(k_t_row, state_fp32)
            erase = torch.matmul(k_t_col, kS)
            write = torch.matmul(k_t_col, v_t.unsqueeze(-2))

            state_fp32.sub_(b_t * erase)
            state_fp32.add_(b_t * write)

            # Clamp state periodically within chunk to prevent runaway growth
            if (t + 1) % 16 == 0:
                state_norm = state_fp32.norm(dim=(-2, -1), keepdim=True)
                scale = (self.state_target_norm / (state_norm + 1e-10)).clamp(max=1.0)
                state_fp32.mul_(scale)

            y_t = torch.matmul(state_fp32.transpose(-2, -1), q_t.unsqueeze(-1)).squeeze(-1)
            output[:, :, t, :] = y_t.to(dtype)

        # Sanitize output: fp16 overflow in the state-fp32-to-target-dtype cast
        # or matmul can produce NaN. Zero it to prevent cascading NaN.
        if dtype == torch.float16 or dtype == torch.bfloat16:
            output = torch.where(
                torch.isnan(output) | torch.isinf(output),
                torch.zeros_like(output),
                output
            )
        return output, state_fp32.to(dtype)

    def _rescale_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        EMA-style state rescaling: normalize per-head state to prevent
        unbounded drift while preserving directional information.

        Uses per-head layer norm on the value dimension of each key slot,
        which distinguishes signal from noise better than hard clipping.

        Args:
            state: (B, H, D, D_v)
        Returns:
            rescaled state: (B, H, D, D_v)
        """
        # Apply layer norm along the last dim (D_v) for each key slot
        # This normalizes each row of the key->value mapping independently
        state = self.state_norm(state)
        # Clamp to fp16 range to prevent overflow when casting fp32→fp16
        if state.dtype == torch.float16 or state.dtype == torch.bfloat16:
            state = state.clamp(-65504.0, 65504.0)
        return state

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model)
            state: optional (B, n_qk_heads, head_dim, head_dim_v) recurrent state

        Returns:
            output: (B, T, d_model)
            new_state: (B, n_qk_heads, head_dim, head_dim_v)
        """
        B, T, _ = x.shape

        # Projections
        q = self.q_proj(x)  # (B, T, d_qk)
        k = self.k_proj(x)  # (B, T, d_qk)
        v = self.v_proj(x)  # (B, T, d_v)

        # Reshape to heads
        head_dim_qk = self.head_dim

        q = q.view(B, T, self.n_qk_heads, head_dim_qk).transpose(1, 2)  # (B, H_qk, T, D)
        k = k.view(B, T, self.n_qk_heads, head_dim_qk).transpose(1, 2)

        # Normalize keys for stability
        k = F.normalize(k, dim=-1)

        # V-head projection: learned expand instead of repeat_interleave
        if self.v_expand is not None:
            v = self.v_expand(v)  # (B, T, d_qk) -- now matched to n_qk_heads
            head_dim_v = head_dim_qk
            n_heads_v = self.n_qk_heads
        else:
            head_dim_v = self.head_dim
            n_heads_v = self.n_v_heads

        v = v.view(B, T, n_heads_v, head_dim_v).transpose(1, 2)  # (B, H, T, D_v)

        # Beta (per-head learning rate)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)  # (B, H, T)

        # Initialize state
        if state is None:
            state = torch.zeros(
                B, self.n_qk_heads, head_dim_qk, head_dim_v,
                device=x.device, dtype=x.dtype,
            )

        # State-aware gating: compute gate from input + state retrieval signal
        gate_input = torch.sigmoid(self.gate_proj(x))  # (B, T, d_out)
        # State signal: per-head norm of state, broadcast to gate shape
        with torch.no_grad():
            state_norms = state.norm(dim=(-2, -1))  # (B, H)
            # Normalize to [0, 1] range via sigmoid-like transform
            state_signal = torch.tanh(state_norms / self.state_target_norm)  # (B, H)

        # Modulate gate: scale gate down when state is saturated
        # state_gate_scale is learned -- model decides how much to trust state signal
        gate_modulation = 1.0 + self.state_gate_scale * state_signal.unsqueeze(-1)  # (B, H, 1)
        gate_input = gate_input.view(B, T, n_heads_v, head_dim_v).transpose(1, 2)  # (B, H, T, D_v)
        gate = gate_input * gate_modulation.unsqueeze(2)  # (B, H, T, D_v) * (B, H, 1, 1)

        # Chunked recurrence
        n_chunks = (T + self.chunk_size - 1) // self.chunk_size
        chunk_outputs = []

        for c in range(n_chunks):
            start = c * self.chunk_size
            end = min(start + self.chunk_size, T)

            q_chunk = q[:, :, start:end, :]
            k_chunk = k[:, :, start:end, :]
            v_chunk = v[:, :, start:end, :]
            beta_chunk = beta[:, :, start:end]

            if q.requires_grad and self.use_checkpoint:
                def run_chunk(q_c, k_c, v_c, b_c, s_c):
                    return self._chunk_recurrence(q_c, k_c, v_c, b_c, s_c)
                chunk_out, state = torch.utils.checkpoint.checkpoint(
                    run_chunk, q_chunk, k_chunk, v_chunk, beta_chunk, state,
                    use_reentrant=False
                )
            else:
                chunk_out, state = self._chunk_recurrence(
                    q_chunk, k_chunk, v_chunk, beta_chunk, state
                )
            chunk_outputs.append(chunk_out)

            # EMA state rescaling after each chunk -- prevents unbounded drift
            state = self._rescale_state(state)

        # Concatenate chunks
        output = torch.cat(chunk_outputs, dim=2)  # (B, H, T, D_v)

        # Apply state-aware gate
        output = gate * output

        # Merge heads and project
        output = output.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_out)
        output = self.o_proj(output)

        # Sanitize output: delta rule recurrence + fp16 matmuls can produce NaN
        if self.training and (output.dtype == torch.float16 or output.dtype == torch.bfloat16):
            output = torch.where(
                torch.isnan(output) | torch.isinf(output),
                torch.zeros_like(output),
                output
            )
            state = torch.where(
                torch.isnan(state) | torch.isinf(state),
                torch.zeros_like(state),
                state
            )

        # Stats -- only during eval
        if not self.training:
            self._last_state_norm = state.detach().float().norm().item()
            self._last_gate_mean = gate.detach().float().mean().item()

        return output, state

    def stats(self) -> dict:
        return {
            "state_norm": self._last_state_norm,
            "gate_mean": self._last_gate_mean,
            "n_qk_heads": self.n_qk_heads,
            "n_v_heads": self.n_v_heads,
            "head_dim": self.head_dim,
            "chunk_size": self.chunk_size,
            "params": sum(p.numel() for p in self.parameters()),
        }
