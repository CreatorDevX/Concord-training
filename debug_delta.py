import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.components.delta_net import GatedDeltaNet
from model.config import ModelConfig

def main():
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.set_default_dtype(torch.float16)
    delta = GatedDeltaNet(
        d_model=config.d_model,
        n_v_heads=config.delta_v_heads,
        n_qk_heads=config.delta_qk_heads,
        head_dim=config.delta_head_dim,
        chunk_size=config.delta_chunk_size,
    ).to(device)
    torch.set_default_dtype(torch.float32)

    x = torch.randn(1, 128, config.d_model, device=device, dtype=torch.float16)
    
    with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu",
                            dtype=torch.float16, enabled=torch.cuda.is_available()):
        B, T, _ = x.shape
        q = delta.q_proj(x)
        k = delta.k_proj(x)
        v = delta.v_proj(x)
        
        print("q nan:", torch.isnan(q).any().item())
        print("k nan:", torch.isnan(k).any().item())
        print("v nan:", torch.isnan(v).any().item())

        head_dim_qk = delta.head_dim
        q = q.view(B, T, delta.n_qk_heads, head_dim_qk).transpose(1, 2)
        k = k.view(B, T, delta.n_qk_heads, head_dim_qk).transpose(1, 2)
        
        k_norm = torch.nn.functional.normalize(k, dim=-1)
        print("k_norm nan:", torch.isnan(k_norm).any().item())

        if delta.v_expand is not None:
            v = delta.v_expand(v)
            head_dim_v = head_dim_qk
        else:
            head_dim_v = delta.head_dim
        v = v.view(B, T, delta.n_qk_heads, head_dim_v).transpose(1, 2)
        
        beta = torch.nn.functional.softplus(delta.beta_proj(x))
        print("beta nan:", torch.isnan(beta).any().item())
        beta = beta.view(B, T, delta.n_qk_heads).transpose(1, 2)
        
        chunk_start = 0
        chunk_end = 64
        q_chunk = q[:, :, chunk_start:chunk_end, :]
        k_chunk = k_norm[:, :, chunk_start:chunk_end, :]
        v_chunk = v[:, :, chunk_start:chunk_end, :]
        beta_chunk = beta[:, :, chunk_start:chunk_end]
        
        kv = torch.einsum('b h n d, b h n e -> b h d e', k_chunk, v_chunk)
        print("kv nan:", torch.isnan(kv).any().item())
        print("kv max:", kv.abs().max().item())
        
        state = beta_chunk.unsqueeze(-1) * kv
        print("state nan:", torch.isnan(state).any().item())
        
        state_normed = delta.state_norm(state)
        print("state_normed nan:", torch.isnan(state_normed).any().item())

if __name__ == '__main__':
    main()
