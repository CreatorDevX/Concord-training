import sys
import os

# Change to workspace directory if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.dataset import TokenizerWrapper

def run():
    path = "./model/custom_tokenizer"
    print(f"Loading custom tokenizer from {path}...")
    t = TokenizerWrapper(path)
    print(f"Vocab size: {t.vocab_size}")
    print("Loaded successfully!\n")
    
    sample_text = """### INIT
You are in pretraining mode.
### END
50.00 %
Phase 1/2 of pretraining.
All blocks implemented, context Nominal.
# GO.

## TRAINING INTERRUPT
# GO."""

    ids = t.encode(sample_text)
    
    print("=== Token by Token ===")
    for token_id in ids:
        # Decode single token
        decoded = t.decode([token_id])
        print(f"ID: {token_id:>6d} | String: {repr(decoded)}")
        
    print("\n=== Reconstructed with -|- separator ===")
    tokens = [t.decode([token_id]) for token_id in ids]
    print("-|-".join(tokens))

if __name__ == "__main__":
    run()
