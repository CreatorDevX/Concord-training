import os
import argparse
import torch
import zipfile
import shutil

from model.config import ModelConfig
from model.model import HybridMoE
from model.dataset import TokenizerWrapper

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=50):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    print("\n" + "=" * 40)
    print("Generating:")
    print("=" * 40)
    print(prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        logits = model(input_tensor)["logits"]
        # Take the logits for the last token in sequence
        next_token_logits = logits[0, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Apply top_k sampling
        v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
        next_token_logits[next_token_logits < v[[-1]]] = -float('Inf')
        
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)
        
        word = tokenizer.decode([next_token])
        print(word, end="", flush=True)
        
        if next_token == tokenizer.tokenizer.eos_token_id:
            break
            
    print("\n\n" + "=" * 40)

def main():
    parser = argparse.ArgumentParser("Concord-α 3B Autoregressive Inference Generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to step_XXXXXX.zip or model.pt")
    parser.add_argument("--tokenizer", type=str, default="./model/custom_tokenizer")
    parser.add_argument("--prompt", type=str, default="|<STATE>|\\n### INIT\\nYou are in pretraining mode.\\n### END\\n100.00 %\\nPhase 1/2 of pretraining.\\nAll blocks implemented, context Nominal.\\n|</STATE>|\\n\\n# GO.\\n\\n|<INTERRUPT>|\\n## TRAINING INTERRUPT\\n")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    path = args.checkpoint
    is_zip = path.endswith('.zip')
    
    if is_zip:
        tmp_extract = path + "_tmp_extract"
        print(f"Extracting zip to {tmp_extract}...")
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(tmp_extract)
        pt_path = os.path.join(tmp_extract, "model.pt")
    else:
        pt_path = path if path.endswith('.pt') else os.path.join(path, "model.pt")
        tmp_extract = None

    print(f"Loading weights from {pt_path}...")
    state = torch.load(pt_path, map_location="cpu", weights_only=False)
    
    # Reload model dynamically using identical saved hyperparameters dict
    config_dict = state.get("config", {})
    model_config = ModelConfig.from_dict(config_dict)
    
    print("Building model architecture...")
    model = HybridMoE(model_config)
    
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.to(device)
    print("Model initialized successfully!")
    
    if tmp_extract:
        shutil.rmtree(tmp_extract)
        
    print(f"Loading tokenizer from {args.tokenizer}...")
    t = TokenizerWrapper(args.tokenizer)
    
    prompt = args.prompt.replace("\\n", "\n")
    generate(model, t, prompt, args.max_new_tokens, args.temperature)

if __name__ == "__main__":
    main()
