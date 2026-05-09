import pytest
import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.dataset import JsonlParquetDataset, TokenizerWrapper, HarnessFormatter
from model.train import ProgressTracker

def test_jsonl_dataset(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    file1 = data_dir / "1.jsonl"
    with open(file1, "w") as f:
        f.write(json.dumps({"text": "Hello world part 1"}) + "\n")
        f.write(json.dumps({"text": "Hello world part 2"}) + "\n")
        f.write(json.dumps({"text": "Hello world part 3"}) + "\n")

    tokenizer = TokenizerWrapper("./model/custom_tokenizer")
    formatter = HarnessFormatter(tokenizer, corpus_tokens=20)
    tracker = ProgressTracker()
    
    ds = JsonlParquetDataset(str(data_dir), formatter, tracker, text_key="text")
    items = list(ds)
    
    assert len(items) == 3
    assert tracker.tokens_seen > 0
    assert "input_ids" in items[0]
    assert "loss_mask" in items[0]
    
    assert items[0]["input_ids"].shape == items[0]["loss_mask"].shape
