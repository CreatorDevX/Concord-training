import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.dataset import TokenizerWrapper, HarnessFormatter

@pytest.fixture
def tokenizer():
    # Use standard GPT2 for fast test or the custom one if it exists
    path = "./model/custom_tokenizer"
    if not os.path.exists(path):
        from transformers import AutoTokenizer
        return TokenizerWrapper() # This will crash if no model_id is used. Let's patch.
    return TokenizerWrapper(path)

def test_harness_formatter(tokenizer):
    formatter = HarnessFormatter(tokenizer, corpus_tokens=2048, pad_percentage_tokens=3)
    corpus_text = "This is a simple test sequence for the formatting."
    
    input_ids, loss_mask = formatter.format(corpus_text, 50.0)
    
    assert len(input_ids) == formatter.get_seq_len()
    assert len(loss_mask) == formatter.get_seq_len()
    
    # 0 for prefix
    p1_len = len(formatter.p1_ids) + formatter.pct_max_token_len + len(formatter.p2_base_ids)
    
    assert (loss_mask[:p1_len] == 0).all()
    # 1 for actual corpus text. Usually ~10-12 tokens here.
    corpus_actual = len(tokenizer.encode(corpus_text))
    assert (loss_mask[p1_len:p1_len+corpus_actual] == 1).all()
    
    # 0 for suffix and padding
    assert (loss_mask[p1_len+corpus_actual:] == 0).all()
