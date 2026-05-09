import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.config import ModelConfig
from model.model import HybridMoE

def main():
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.set_default_dtype(torch.float16)
    model = HybridMoE(config).to(device)
    torch.set_default_dtype(torch.float32)

    B, T = 1, 64
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    wall_ms = time.time() * 1000.0
    timestamps_ms = torch.full((B, T), wall_ms, dtype=torch.float32, device=device)

    # Let's hook into the model to find exactly where NaN first appears
    nan_found = False
    
    def get_nan_hook(name):
        def hook(module, inp, output):
            nonlocal nan_found
            if nan_found: return
            
            tensors_to_check = []
            if isinstance(output, torch.Tensor):
                tensors_to_check.append(output)
            elif isinstance(output, tuple):
                tensors_to_check.extend([t for t in output if isinstance(t, torch.Tensor)])
                
            for i, t in enumerate(tensors_to_check):
                if torch.isnan(t).any():
                    print(f"!!! NaN found in output of {name} (index {i}) !!!")
                    print(f"Shape: {t.shape}")
                    print(f"Input sum: {inp[0].sum().item() if isinstance(inp[0], torch.Tensor) else 'N/A'}")
                    nan_found = True
                    return
        return hook

    for name, module in model.named_modules():
        module.register_forward_hook(get_nan_hook(name))

    print("Running forward pass...")
    with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu",
                            dtype=torch.float16, enabled=torch.cuda.is_available()):
        result = model(input_ids, labels=input_ids, timestamps_ms=timestamps_ms)
        
    print(f"Loss: {result['loss'].item()}")
    print(f"Main Loss: {result['main_loss'].item()}")
    print(f"MTP Loss: {result['mtp_loss'].item()}")
    print(f"Aux Loss: {result['aux_loss'].item()}")

if __name__ == '__main__':
    main()
