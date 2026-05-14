import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin


class ConcordConfig(PretrainedConfig):
    model_type = "concord"

    def __init__(
        self,
        d_model: int = 768,
        vocab_size: int = 262189,
        embed_dim: int = 384,
        n_layers: int = 16,
        group_size: int = 4,
        n_groups: int = 4,
        delta_v_heads: int = 12,
        delta_qk_heads: int = 12,
        delta_head_dim: int = 64,
        attn_q_heads: int = 6,
        attn_kv_heads: int = 2,
        attn_head_dim: int = 128,
        rope_axis_dims: dict = None,
        csa_top_k: int = 1024,
        csa_window: int = 128,
        csa_compress: int = 4,
        hca_compress: int = 128,
        n_experts: int = 36,
        n_routed_delta: int = 2,
        n_routed_attn: int = 2,
        n_shared: int = 1,
        expert_intermediate: int = 256,
        aux_loss_coeff: float = 0.01,
        router_hidden: int = 256,
        router_bias_update_interval: int = 750,
        mtp_steps: int = 2,
        mtp_weight: float = 0.1,
        mtp_tie_output: bool = True,
        expert_dtype: str = "fp16",
        grad_checkpoint: bool = True,
        share_experts_within_group: bool = False,
        tie_embeddings: bool = True,
        max_seq_len: int = 8192,
        selective_loss: bool = True,
        use_vision: bool = False,
        delta_chunk_size: int = 128,
        **kwargs,
    ):
        kwargs["tie_word_embeddings"] = tie_embeddings
        super().__init__(**kwargs)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.group_size = group_size
        self.n_groups = n_groups
        self.delta_v_heads = delta_v_heads
        self.delta_qk_heads = delta_qk_heads
        self.delta_head_dim = delta_head_dim
        self.attn_q_heads = attn_q_heads
        self.attn_kv_heads = attn_kv_heads
        self.attn_head_dim = attn_head_dim
        self.rope_axis_dims = rope_axis_dims or {"x": 16, "y": 16, "u": 32, "w": 64}
        self.csa_top_k = csa_top_k
        self.csa_window = csa_window
        self.csa_compress = csa_compress
        self.hca_compress = hca_compress
        self.n_experts = n_experts
        self.n_routed_delta = n_routed_delta
        self.n_routed_attn = n_routed_attn
        self.n_shared = n_shared
        self.expert_intermediate = expert_intermediate
        self.aux_loss_coeff = aux_loss_coeff
        self.router_hidden = router_hidden
        self.router_bias_update_interval = router_bias_update_interval
        self.mtp_steps = mtp_steps
        self.mtp_weight = mtp_weight
        self.mtp_tie_output = mtp_tie_output
        self.expert_dtype = expert_dtype
        self.grad_checkpoint = grad_checkpoint
        self.share_experts_within_group = share_experts_within_group
        self.tie_embeddings = tie_embeddings
        self.max_seq_len = max_seq_len
        self.selective_loss = selective_loss
        self.use_vision = use_vision
        self.delta_chunk_size = delta_chunk_size

        self.delta_d_v = delta_v_heads * delta_head_dim
        self.delta_d_qk = delta_qk_heads * delta_head_dim
        self.attn_d_q = attn_q_heads * attn_head_dim
        self.attn_d_kv = attn_kv_heads * attn_head_dim

        self.num_hidden_layers = n_layers
        self.hidden_size = d_model
        self.num_attention_heads = attn_q_heads
        self.num_key_value_heads = attn_kv_heads


class FPLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.float()).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f / rms * self.scale).to(x.dtype)


class RotaryAxis(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        angles = coord[:, None, :, None].float() * self.inv_freq[None, None, None, :].float()
        cos, sin = angles.cos(), angles.sin()
        x_even, x_odd = x[..., ::2].float(), x[..., 1::2].float()
        out = torch.empty_like(x)
        out[..., ::2] = (x_even * cos - x_odd * sin).to(x.dtype)
        out[..., 1::2] = (x_even * sin + x_odd * cos).to(x.dtype)
        return out


class TemporalMultimodalRoPE(nn.Module):
    def __init__(self, axis_dims: Dict[str, int], wallclock_scale_seconds: float = 60.0, use_log_wallclock: bool = True):
        super().__init__()
        self.axis_names = list(axis_dims.keys())
        self.axis_dims = axis_dims
        self.wallclock_scale = wallclock_scale_seconds
        self.use_log_wallclock = use_log_wallclock
        if "w" in axis_dims:
            self.wallclock_gain = nn.Parameter(torch.tensor(0.1))
        else:
            self.wallclock_gain = None
        axis_bases = {k: 10000.0 for k in axis_dims}
        self.axes = nn.ModuleDict({k: RotaryAxis(axis_dims[k], base=axis_bases[k]) for k in axis_dims})

    def encode_wallclock(self, t_ms: torch.Tensor, ref_ms: torch.Tensor) -> torch.Tensor:
        dt = (t_ms.float() - ref_ms.float()) / 1000.0
        if self.use_log_wallclock:
            dt = torch.sign(dt) * torch.log1p(dt.abs() / self.wallclock_scale)
        else:
            dt = dt / self.wallclock_scale
        return dt

    def forward(self, q: torch.Tensor, k: torch.Tensor, coords: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        q_parts = torch.split(q, [self.axis_dims[n] for n in self.axis_names], dim=-1)
        k_parts = torch.split(k, [self.axis_dims[n] for n in self.axis_names], dim=-1)
        q_out, k_out = [], []
        for name, qp, kp in zip(self.axis_names, q_parts, k_parts):
            coord = coords[name].to(dtype=q.dtype)
            if name == "w" and self.wallclock_gain is not None:
                coord = coord * self.wallclock_gain
            q_out.append(self.axes[name](qp, coord))
            k_out.append(self.axes[name](kp, coord))
        return torch.cat(q_out, dim=-1), torch.cat(k_out, dim=-1)


class TemporalDecayBias(nn.Module):
    def __init__(self, n_heads: int, n_buckets: int = 32, max_log_gap: float = 10.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.max_log_gap = max_log_gap
        self.bias_embed = nn.Embedding(n_buckets, n_heads)
        nn.init.zeros_(self.bias_embed.weight)

    def _bucketize(self, delta_w: torch.Tensor) -> torch.Tensor:
        log_gap = torch.sign(delta_w) * torch.log1p(delta_w.abs())
        normalized = (log_gap + self.max_log_gap) / (2 * self.max_log_gap)
        return (normalized * (self.n_buckets - 1)).clamp(0, self.n_buckets - 1).long()

    def forward(self, w_q: torch.Tensor, w_k: torch.Tensor) -> torch.Tensor:
        delta_w = w_q.unsqueeze(-1) - w_k.unsqueeze(-2)
        bucket_ids = self._bucketize(delta_w)
        bias = self.bias_embed(bucket_ids)
        return bias.permute(0, 3, 1, 2)


class KVCompressor(nn.Module):
    def __init__(self, compress_ratio: int, d_kv: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.compress_proj = nn.Linear(d_kv * compress_ratio, d_kv, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        r = self.compress_ratio
        n_complete = T // r
        if n_complete == 0:
            return x.new_zeros(B, H, 0, D)
        usable_len = n_complete * r
        x_usable = x[:, :, :usable_len, :]
        x_grouped = x_usable.reshape(B, H, n_complete, r * D)
        return self.compress_proj(x_grouped)


class GatedAttention(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, head_dim: int,
                 rope_axis_dims: Dict[str, int], variant: str = "csa",
                 csa_top_k: int = 1024, csa_window: int = 128,
                 csa_compress: int = 4, hca_compress: int = 128):
        super().__init__()
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.variant = variant
        self.csa_top_k = csa_top_k
        self.csa_window = csa_window
        assert n_q_heads % n_kv_heads == 0
        self.n_rep = n_q_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q_heads * head_dim, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)

        self.rope = TemporalMultimodalRoPE(axis_dims=rope_axis_dims)
        self.temporal_bias = TemporalDecayBias(n_q_heads, n_buckets=32)

        d_kv = head_dim
        if variant == "csa":
            self.kv_compressor = KVCompressor(csa_compress, d_kv)
            self.compress_ratio = csa_compress
        elif variant == "hca":
            self.kv_compressor = KVCompressor(hca_compress, d_kv)
            self.compress_ratio = hca_compress
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.scale = 1.0 / math.sqrt(head_dim)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, H, T, D = x.shape
        x = x[:, :, None, :, :].expand(B, H, self.n_rep, T, D)
        return x.reshape(B, H * self.n_rep, T, D)

    def _build_causal_compress_mask(self, T_q: int, T_comp: int, compress_ratio: int,
                                     device: torch.device, q_start: int = 0) -> torch.Tensor:
        block_end_positions = (torch.arange(T_comp, device=device) + 1) * compress_ratio - 1
        query_positions = torch.arange(T_q, device=device) + q_start
        return query_positions.unsqueeze(1) >= block_end_positions.unsqueeze(0)

    def _csa_forward(self, q, k, v, k_compressed, v_compressed, w_q=None, w_k=None, q_start=0):
        B, H, T, D = q.shape
        T_comp = k_compressed.shape[2]

        if T_comp == 0:
            k_local = self._repeat_kv(k)
            v_local = self._repeat_kv(v)
            return F.scaled_dot_product_attention(q, k_local, v_local, is_causal=True)

        if T_comp <= self.csa_top_k:
            k_exp = self._repeat_kv(k_compressed)
            v_exp = self._repeat_kv(v_compressed)
            k_local = self._repeat_kv(k)
            v_local = self._repeat_kv(v)
            k_cat = torch.cat([k_exp, k_local], dim=2)
            v_cat = torch.cat([v_exp, v_local], dim=2)
            T_full = k.shape[2]
            comp_mask = self._build_causal_compress_mask(T, T_comp, self.compress_ratio, q.device, q_start=q_start)
            i_idx = (torch.arange(T, device=q.device) + q_start).unsqueeze(1)
            j_idx = torch.arange(T_full, device=q.device).unsqueeze(0)
            local_mask = (i_idx >= j_idx) & (i_idx - j_idx < self.csa_window)
            combined_mask = torch.cat([comp_mask, local_mask], dim=1)
            attn_mask = combined_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
            float_mask = torch.where(attn_mask, torch.tensor(0.0, device=q.device, dtype=q.dtype),
                                     torch.tensor(float('-inf'), device=q.device, dtype=q.dtype))
            if w_q is not None and w_k is not None:
                w_k_comp = w_k[:, self.compress_ratio - 1::self.compress_ratio]
                if w_k_comp.shape[1] < T_comp:
                    pad_len = T_comp - w_k_comp.shape[1]
                    w_k_comp = torch.cat([w_k_comp, w_k_comp[:, -1:].expand(-1, pad_len)], dim=1)
                w_k_cat = torch.cat([w_k_comp, w_k], dim=1)
                float_mask = float_mask + self.temporal_bias(w_q, w_k_cat)
            return F.scaled_dot_product_attention(q, k_cat, v_cat, attn_mask=float_mask)

        k_exp = self._repeat_kv(k_compressed)
        v_exp = self._repeat_kv(v_compressed)
        scores_comp = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
        comp_causal_mask = self._build_causal_compress_mask(T, T_comp, self.compress_ratio, q.device, q_start=q_start)
        scores_comp.masked_fill_(~comp_causal_mask.unsqueeze(0).unsqueeze(0), -float('inf'))

        top_k = min(self.csa_top_k, T_comp)
        if top_k < T_comp:
            sanitized = scores_comp.masked_fill(scores_comp == float('-inf'), torch.finfo(scores_comp.dtype).min)
            threshold = sanitized.topk(top_k, dim=-1, largest=True).values[..., -1:]
            scores_comp = scores_comp.masked_fill(scores_comp < threshold, -float('inf'))

        T_full = k.shape[2]
        k_local = self._repeat_kv(k)
        v_local = self._repeat_kv(v)
        scores_local = torch.matmul(q, k_local.transpose(-2, -1)) * self.scale
        i_idx = (torch.arange(T, device=q.device) + q_start).unsqueeze(1)
        j_idx = torch.arange(T_full, device=q.device).unsqueeze(0)
        mask_local = (i_idx >= j_idx) & (i_idx - j_idx < self.csa_window)
        scores_local.masked_fill_(~mask_local, -float('inf'))

        if w_q is not None and w_k is not None:
            w_k_comp = w_k[:, self.compress_ratio - 1::self.compress_ratio]
            if w_k_comp.shape[1] < T_comp:
                pad_len = T_comp - w_k_comp.shape[1]
                w_k_comp = torch.cat([w_k_comp, w_k_comp[:, -1:].expand(-1, pad_len)], dim=1)
            scores_comp = scores_comp + self.temporal_bias(w_q, w_k_comp)
            scores_local = scores_local + self.temporal_bias(w_q, w_k)

        scores_all = torch.cat([scores_comp, scores_local], dim=-1)
        attn_weights = F.softmax(scores_all, dim=-1)
        attn_weights_comp = attn_weights[..., :T_comp]
        attn_weights_local = attn_weights[..., T_comp:]
        return torch.matmul(attn_weights_comp, v_exp) + torch.matmul(attn_weights_local, v_local)

    def _hca_forward(self, q, k_compressed, v_compressed, w_q=None, w_k=None, q_start=0):
        B, H, T, D = q.shape
        T_comp = k_compressed.shape[2]
        if T_comp == 0:
            return torch.zeros_like(q)
        k_exp = self._repeat_kv(k_compressed)
        v_exp = self._repeat_kv(v_compressed)
        comp_causal_mask = self._build_causal_compress_mask(T, T_comp, self.compress_ratio, q.device, q_start=q_start)
        attn_mask = comp_causal_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        float_mask = torch.where(attn_mask, torch.tensor(0.0, device=q.device, dtype=q.dtype),
                                 torch.tensor(float('-inf'), device=q.device, dtype=q.dtype))
        if w_q is not None and w_k is not None:
            float_mask = float_mask + self.temporal_bias(w_q, w_k)
        return F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=float_mask)

    def forward(self, x: torch.Tensor, coords: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(x))
        gate = gate.view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)

        q_rope, k_rope = self.rope(q, k, coords or {"u": torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1),
                                                      "w": torch.zeros(B, T, device=x.device),
                                                      "x": torch.zeros(B, T, device=x.device),
                                                      "y": torch.zeros(B, T, device=x.device)})

        k_compressed = self.kv_compressor(k_rope)
        v_compressed = self.kv_compressor(v)
        w_q = coords.get("w") if coords is not None else None
        w_k = coords.get("w") if coords is not None else None

        q_chunk_size = 256
        output_chunks = []
        for start in range(0, T, q_chunk_size):
            end = min(start + q_chunk_size, T)
            q_chunk = q_rope[:, :, start:end, :]
            gate_chunk = gate[:, :, start:end, :]
            w_q_chunk = w_q[:, start:end] if w_q is not None else None
            if self.variant == "csa":
                out_chunk = self._csa_forward(q_chunk, k_rope, v, k_compressed, v_compressed,
                                              w_q=w_q_chunk, w_k=w_k, q_start=start)
            else:
                out_chunk = self._hca_forward(q_chunk, k_compressed, v_compressed,
                                              w_q=w_q_chunk, w_k=w_k, q_start=start)
            out_chunk = gate_chunk * out_chunk
            out_chunk = out_chunk.transpose(1, 2).contiguous().view(B, end - start, -1)
            output_chunks.append(out_chunk)

        output = torch.cat(output_chunks, dim=1)
        return self.o_proj(output)


class GatedDeltaNet(nn.Module):
    def __init__(self, d_model: int, n_v_heads: int, n_qk_heads: int, head_dim: int,
                 chunk_size: int = 64, state_target_norm: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.n_v_heads = n_v_heads
        self.n_qk_heads = n_qk_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.state_target_norm = state_target_norm
        self.d_v = n_v_heads * head_dim
        self.d_qk = n_qk_heads * head_dim

        self.q_proj = nn.Linear(d_model, self.d_qk, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_qk, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_v, bias=False)

        if n_qk_heads != n_v_heads:
            self.v_expand = nn.Linear(self.d_v, self.d_qk, bias=False)
        else:
            self.v_expand = None

        self.o_proj = nn.Linear(self.d_qk if n_qk_heads != n_v_heads else self.d_v, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, self.d_qk if n_qk_heads != n_v_heads else self.d_v, bias=False)
        self.state_gate_scale = nn.Parameter(torch.zeros(1))
        self.beta_proj = nn.Linear(d_model, n_qk_heads, bias=False)
        self.state_norm = FPLayerNorm(head_dim)

    def _chunk_recurrence(self, q, k, v, beta, state):
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

            kS = torch.matmul(k_t.unsqueeze(-2), state_fp32)
            erase = torch.matmul(k_t.unsqueeze(-1), kS)
            write = torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(-2))

            state_fp32.sub_(b_t * erase)
            state_fp32.add_(b_t * write)

            if (t + 1) % 16 == 0:
                state_norm = state_fp32.norm(dim=(-2, -1), keepdim=True)
                scale = (self.state_target_norm / (state_norm + 1e-10)).clamp(max=1.0)
                state_fp32.mul_(scale)

            y_t = torch.matmul(state_fp32.transpose(-2, -1), q_t.unsqueeze(-1)).squeeze(-1)
            output[:, :, t, :] = y_t.to(dtype)

        if dtype == torch.float16 or dtype == torch.bfloat16:
            output = torch.where(torch.isnan(output) | torch.isinf(output), torch.zeros_like(output), output)
        return output, state_fp32.to(dtype)

    def _rescale_state(self, state: torch.Tensor) -> torch.Tensor:
        state = self.state_norm(state)
        if state.dtype == torch.float16 or state.dtype == torch.bfloat16:
            state = state.clamp(-65504.0, 65504.0)
        return state

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_qk_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_qk_heads, self.head_dim).transpose(1, 2)
        k = F.normalize(k, dim=-1)
        v = self.v_proj(x)

        if self.v_expand is not None:
            v = self.v_expand(v)
            head_dim_v = self.head_dim
            n_heads_v = self.n_qk_heads
        else:
            head_dim_v = self.head_dim
            n_heads_v = self.n_v_heads

        v = v.view(B, T, n_heads_v, head_dim_v).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)

        if state is None:
            state = torch.zeros(B, self.n_qk_heads, self.head_dim, head_dim_v,
                                 device=x.device, dtype=x.dtype)

        gate_input = torch.sigmoid(self.gate_proj(x))
        with torch.no_grad():
            state_norms = state.norm(dim=(-2, -1))
            state_signal = torch.tanh(state_norms / self.state_target_norm)
        gate_modulation = 1.0 + self.state_gate_scale * state_signal.unsqueeze(-1)
        gate = gate_input.view(B, T, n_heads_v, head_dim_v).transpose(1, 2) * gate_modulation.unsqueeze(2)

        n_chunks = (T + self.chunk_size - 1) // self.chunk_size
        chunk_outputs = []
        for c in range(n_chunks):
            start = c * self.chunk_size
            end = min(start + self.chunk_size, T)
            q_chunk = q[:, :, start:end, :]
            k_chunk = k[:, :, start:end, :]
            v_chunk = v[:, :, start:end, :]
            beta_chunk = beta[:, :, start:end]
            chunk_out, state = self._chunk_recurrence(q_chunk, k_chunk, v_chunk, beta_chunk, state)
            chunk_outputs.append(chunk_out)
            state = self._rescale_state(state)

        output = torch.cat(chunk_outputs, dim=2)
        output = gate * output
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.o_proj(output)
        return output, state


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Parameter(torch.empty(d_ffn, d_model))
        self.w_up = nn.Parameter(torch.empty(d_ffn, d_model))
        self.w_down = nn.Parameter(torch.empty(d_model, d_ffn))
        nn.init.kaiming_uniform_(self.w_gate)
        nn.init.kaiming_uniform_(self.w_up)
        nn.init.kaiming_uniform_(self.w_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.w_gate.dtype)
        gate = F.silu(F.linear(x, self.w_gate))
        up = F.linear(x, self.w_up)
        return F.linear(gate * up, self.w_down)


class MLPRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, n_routed: int, hidden_dim: int = 384,
                 bias_update_interval: int = 1000, aux_loss_coeff: float = 0.01):
        super().__init__()
        self.n_experts = n_experts
        self.n_routed = n_routed
        self.aux_loss_coeff = aux_loss_coeff
        self.norm = RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(n_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.silu(self.w1(self.norm(x)))
        logits = self.w2(h) + self.expert_bias
        scores = F.softmax(logits, dim=-1)
        top_scores, top_indices = torch.topk(scores, self.n_routed, dim=-1)
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-10)

        aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(top_indices, self.n_experts).float()
                tokens_per_expert = one_hot.sum(dim=(0, 1, 2))
                total_tokens = x.shape[0] * x.shape[1] * self.n_routed
                f_e = tokens_per_expert / total_tokens
            P_e = scores.mean(dim=(0, 1))
            aux_loss = self.aux_loss_coeff * self.n_experts * (f_e * P_e).sum()
            if torch.isnan(aux_loss) or torch.isinf(aux_loss):
                aux_loss = torch.zeros_like(aux_loss)

        return top_indices, top_scores, aux_loss


def dispatch_and_combine(x, indices, scores, experts):
    B, T, d = x.shape
    n_routed = indices.shape[-1]
    n_experts = len(experts)
    x_flat = x.view(B * T, d)
    indices_flat = indices.view(B * T, n_routed)
    scores_flat = scores.view(B * T, n_routed)
    output = torch.zeros_like(x_flat)

    for slot in range(n_routed):
        slot_indices = indices_flat[:, slot]
        slot_scores = scores_flat[:, slot]
        for expert_idx in range(n_experts):
            mask = (slot_indices == expert_idx)
            if not mask.any():
                continue
            expert_input = x_flat[mask]
            expert_output = experts[expert_idx](expert_input)
            if expert_output.dtype == torch.float16 or expert_output.dtype == torch.bfloat16:
                expert_output = torch.where(torch.isnan(expert_output) | torch.isinf(expert_output),
                                            torch.zeros_like(expert_output), expert_output)
            output[mask] += slot_scores[mask].unsqueeze(-1) * expert_output

    return output.view(B, T, d)


class MoEBlock(nn.Module):
    def __init__(self, d_model: int, n_experts: int, n_routed: int, n_shared: int = 1,
                 expert_intermediate: int = 192, router_hidden: int = 384,
                 router_bias_update_interval: int = 1000, expert_dtype: str = "fp16"):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_routed = n_routed
        self.router = MLPRouter(d_model=d_model, n_experts=n_experts, n_routed=n_routed,
                                hidden_dim=router_hidden, bias_update_interval=router_bias_update_interval)
        self.shared_experts = nn.ModuleList([ExpertFFN(d_model, expert_intermediate) for _ in range(n_shared)])
        self.routed_experts = nn.ModuleList([ExpertFFN(d_model, expert_intermediate) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_out = sum(e(x.view(-1, self.d_model)).view(x.shape) for e in self.shared_experts)
        indices, scores, aux_loss = self.router(x)
        routed_out = dispatch_and_combine(x, indices, scores, self.routed_experts)
        return shared_out + routed_out, aux_loss


class BlockAttnRes(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_blocks: int):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.pseudo_queries = nn.Parameter(torch.zeros(n_layers, d_model))
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, layer_idx: int, block_reprs: List[torch.Tensor], partial_residual: torch.Tensor) -> torch.Tensor:
        sources = block_reprs + [partial_residual]
        stacked = torch.stack(sources, dim=2)
        stacked_norm = self.norm(stacked)
        keys = self.k_proj(stacked_norm)
        q = self.pseudo_queries[layer_idx].view(1, 1, 1, -1)
        attn_logits = (q * keys).sum(dim=-1) * self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        return (attn_weights.unsqueeze(-1) * stacked).sum(dim=2)


class DeltaBlock(nn.Module):
    def __init__(self, config: ConcordConfig, layer_idx: int, attn_res: BlockAttnRes, shared_moe: Optional[MoEBlock] = None):
        super().__init__()
        self.layer_idx = layer_idx
        object.__setattr__(self, '_attn_res', attn_res)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.delta_net = GatedDeltaNet(d_model=config.d_model, n_v_heads=config.delta_v_heads,
                                       n_qk_heads=config.delta_qk_heads, head_dim=config.delta_head_dim,
                                       chunk_size=config.delta_chunk_size)
        if shared_moe is not None:
            self.moe = shared_moe
        else:
            self.moe = MoEBlock(d_model=config.d_model, n_experts=config.n_experts,
                                n_routed=config.n_routed_delta, n_shared=config.n_shared,
                                expert_intermediate=config.expert_intermediate,
                                router_hidden=config.router_hidden,
                                router_bias_update_interval=config.router_bias_update_interval,
                                expert_dtype=config.expert_dtype)

    def forward(self, x, block_reprs, partial_residual, delta_state=None):
        x_res = self._attn_res(self.layer_idx, block_reprs, partial_residual)
        delta_out, delta_state = self.delta_net(self.norm1(x_res), delta_state)
        x = x + x_res + delta_out
        partial_residual = partial_residual + x
        if x.dtype in (torch.float16, torch.bfloat16):
            partial_residual = torch.where(torch.isnan(partial_residual) | torch.isinf(partial_residual),
                                           torch.zeros_like(partial_residual), partial_residual)
        x_res2 = self._attn_res(self.layer_idx, block_reprs, partial_residual)
        moe_out, aux_loss = self.moe(self.norm2(x_res2))
        x = x + x_res2 + moe_out
        partial_residual = partial_residual + x
        return x, partial_residual, delta_state, aux_loss


class AttnBlock(nn.Module):
    def __init__(self, config: ConcordConfig, layer_idx: int, attn_idx: int, attn_res: BlockAttnRes,
                 shared_moe: Optional[MoEBlock] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_idx = attn_idx
        object.__setattr__(self, '_attn_res', attn_res)
        variant = "csa" if attn_idx % 2 == 0 else "hca"
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.attention = GatedAttention(d_model=config.d_model, n_q_heads=config.attn_q_heads,
                                        n_kv_heads=config.attn_kv_heads, head_dim=config.attn_head_dim,
                                        rope_axis_dims=config.rope_axis_dims, variant=variant,
                                        csa_top_k=config.csa_top_k, csa_window=config.csa_window,
                                        csa_compress=config.csa_compress, hca_compress=config.hca_compress)
        if shared_moe is not None:
            self.moe = shared_moe
        else:
            self.moe = MoEBlock(d_model=config.d_model, n_experts=config.n_experts,
                                n_routed=config.n_routed_attn, n_shared=config.n_shared,
                                expert_intermediate=config.expert_intermediate,
                                router_hidden=config.router_hidden,
                                router_bias_update_interval=config.router_bias_update_interval,
                                expert_dtype=config.expert_dtype)

    def forward(self, x, block_reprs, partial_residual, coords=None):
        x_res = self._attn_res(self.layer_idx, block_reprs, partial_residual)
        attn_out = self.attention(self.norm1(x_res), coords=coords)
        x = x + x_res + attn_out
        partial_residual = partial_residual + x
        if x.dtype in (torch.float16, torch.bfloat16):
            partial_residual = torch.where(torch.isnan(partial_residual) | torch.isinf(partial_residual),
                                           torch.zeros_like(partial_residual), partial_residual)
        x_res2 = self._attn_res(self.layer_idx, block_reprs, partial_residual)
        moe_out, aux_loss = self.moe(self.norm2(x_res2))
        x = x + x_res2 + moe_out
        partial_residual = partial_residual + x
        return x, partial_residual, aux_loss


class ConcordPreTrainedModel(PreTrainedModel):
    config_class = ConcordConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class ConcordModel(ConcordPreTrainedModel):
    def __init__(self, config: ConcordConfig):
        super().__init__(config)
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.embed_up = nn.Sequential(nn.Linear(config.embed_dim, config.d_model, bias=False), nn.SiLU())

        self.attn_res = BlockAttnRes(d_model=config.d_model, n_layers=config.n_layers, n_blocks=config.n_groups)

        self.group_moes = None
        if config.share_experts_within_group:
            self.group_moes = nn.ModuleList()
            for g in range(config.n_groups):
                self.group_moes.append(MoEBlock(d_model=config.d_model, n_experts=config.n_experts,
                                                n_routed=config.n_routed_delta, n_shared=config.n_shared,
                                                expert_intermediate=config.expert_intermediate,
                                                router_hidden=config.router_hidden,
                                                router_bias_update_interval=config.router_bias_update_interval,
                                                expert_dtype=config.expert_dtype))

        self.blocks = nn.ModuleList()
        for g in range(config.n_groups):
            shared_moe = self.group_moes[g] if self.group_moes is not None else None
            for i in range(config.group_size - 1):
                layer_idx = g * config.group_size + i
                self.blocks.append(DeltaBlock(config, layer_idx=layer_idx, attn_res=self.attn_res, shared_moe=shared_moe))
            attn_layer_idx = g * config.group_size + (config.group_size - 1)
            self.blocks.append(AttnBlock(config, layer_idx=attn_layer_idx, attn_idx=g, attn_res=self.attn_res, shared_moe=shared_moe))

        self.norm = RMSNorm(config.d_model)

        self.embed_down = nn.Linear(config.d_model, config.embed_dim, bias=False)

        self.agent_rope = TemporalMultimodalRoPE(axis_dims=config.rope_axis_dims, wallclock_scale_seconds=60.0, use_log_wallclock=True)

        self.post_init()

    def forward(self, input_ids: torch.Tensor, delta_states: Optional[List[torch.Tensor]] = None,
                coords: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        x = self.embed(input_ids)
        x = self.embed_up(x)

        if coords is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
            coords = {"u": position_ids, "w": torch.zeros_like(position_ids).float(),
                      "x": torch.zeros_like(position_ids).float(), "y": torch.zeros_like(position_ids).float()}

        block_reprs = []
        partial_residual = torch.zeros_like(x)
        new_delta_states = []
        delta_state_idx = 0

        for block in self.blocks:
            if isinstance(block, DeltaBlock):
                ds = delta_states[delta_state_idx] if delta_states is not None and delta_state_idx < len(delta_states) else None
                x, partial_residual, ds_out, _ = block(x, block_reprs, partial_residual, ds)
                new_delta_states.append(ds_out)
                delta_state_idx += 1
            elif isinstance(block, AttnBlock):
                x, partial_residual, _ = block(x, block_reprs, partial_residual, coords)

            if (block.layer_idx + 1) % self.config.group_size == 0:
                if partial_residual.dtype in (torch.float16, torch.bfloat16):
                    partial_residual = torch.where(torch.isnan(partial_residual) | torch.isinf(partial_residual),
                                                   torch.zeros_like(partial_residual), partial_residual)
                block_reprs.append(partial_residual.detach())
                partial_residual = torch.zeros_like(x)

        x = self.norm(x)
        return {"last_hidden": x, "delta_states": new_delta_states}


class ConcordForCausalLM(ConcordPreTrainedModel, GenerationMixin):
    # Provide _tied_weights_keys as a dictionary to avoid the "AttributeError: 'list' object has no attribute 'keys'" 
    # bug in specific HF transformers versions, while still successfully registering the tied weights.
    _tied_weights_keys = {"lm_head.weight": "model.embed.weight"}
    all_tied_weights_keys = {"lm_head.weight": "model.embed.weight"}

    def __init__(self, config: ConcordConfig):
        # Ensure HF's native tie_word_embeddings is set correctly
        config.tie_word_embeddings = config.tie_embeddings
        super().__init__(config)
        self.model = ConcordModel(config)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.model.embed.weight

    def get_input_embeddings(self):
        return self.model.embed

    def set_input_embeddings(self, value):
        self.model.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, *args, **kwargs):
        if hasattr(self, "config") and self.config.tie_embeddings:
            self.lm_head.weight = self.model.embed.weight
        super().tie_weights(*args, **kwargs)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                delta_states: Optional[List[torch.Tensor]] = None, return_dict: bool = True,
                **kwargs) -> CausalLMOutputWithPast:
        out = self.model(input_ids, delta_states=delta_states)
        x = out["last_hidden"]
        x_down = self.model.embed_down(x)
        logits = self.lm_head(x_down).float()

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        if return_dict:
            return CausalLMOutputWithPast(loss=loss, logits=logits)
        return (loss, logits)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        return {"input_ids": input_ids}


ConcordConfig.register_for_auto_class()
ConcordForCausalLM.register_for_auto_class("AutoModelForCausalLM")
