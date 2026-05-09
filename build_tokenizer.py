import os
import json
import argparse
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

def main():
    base_model = "google/gemma-4-E2B"
    local_dir = "./tmp_gemma_tokenizer"
    
    print(f"Downloading {base_model} locally to patch config...")
    # Download the tokenizer files
    snapshot_download(repo_id=base_model, local_dir=local_dir, 
                      allow_patterns=["*token*", "*config.json"])
                      
    # Patch tokenizer_config.json which seems to have a malformed `extra_special_tokens` list
    config_path = os.path.join(local_dir, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            t_config = json.load(f)
            
        modified = False
        # The error was caused by special_tokens expecting a dict with .keys() but getting a list
        for key in ["extra_special_tokens", "special_tokens"]:
            if key in t_config and isinstance(t_config[key], list):
                print(f"Patching malformed {key} in tokenizer_config.json (was list, converting/removing)")
                del t_config[key]
                modified = True
                
        if modified:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(t_config, f, indent=2)
                
    print(f"Loading patched tokenizer from {local_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)

    new_tokens = [
        "### INIT",
        "### END",
        "# GO.",
        "## TRAINING INTERRUPT",
        "## FUNCTION INTERRUPT",
        "## REALTIME INTERRUPT",
        "Friday",
        "Vincere",
        "|<speak>|",
        "|</speak>|",
        "System",
        "()}",
        "Agentic",
        "Cognitive",
        "Architecture",
        "Transformers",
        "Concord-α",
        "Concord",
        "Initialised",
        "### World",
        "### Emotional Affect",
        "### DOMAIN",
        "### PC",
        "### Home",
        "State@Home",
        "### Memory",
        "### Goals",
        "All blocks implemented, context Nominal. Go.",
        "{back(PC)}",
        "{back()}",
        "{back(Home)}",
        "{sleep(180)}",
        "{sleep(seconds)}",
        "## Domain View",
        "{status()}",
        "###",
        "##",
        "#",
        "State@PC",
        "{home(Home)}",
        "{home(PC)}",
        "{home()}",
        "Navigate to:",
        "||TIME_ELAPSED:",
        "||",
        "|<STATE>|",
        "|</STATE>|",
        "|<INTERRUPT>|",
        "|</INTERRUPT>|",
        "|<SELF>|",
        "|</SELF>|",
        "Phase 1/2 of pretraining.",
        "You are in pretraining mode."
    ]

    print(f"Original vocab size: {len(tokenizer)}")
    
    # Add tokens
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added} new tokens.")
    
    new_vocab_size = len(tokenizer)
    print(f"New vocab size: {new_vocab_size}")

    save_path = "./custom_tokenizer"
    os.makedirs(save_path, exist_ok=True)
    
    tokenizer.save_pretrained(save_path)
    print(f"Saved custom tokenizer to {save_path}")
    print(f"!!! Remember to update config.vocab_size to {new_vocab_size} !!!")

if __name__ == "__main__":
    main()
