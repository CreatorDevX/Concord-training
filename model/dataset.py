import os
import json
import numpy as np
from typing import List, Generator, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


class TokenizerWrapper:
    """Wrapper to handle loading custom tokenizer."""
    def __init__(self, model_name_or_path: str = "./model/custom_tokenizer"):
        if AutoTokenizer is None:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            # Gemma/Custom tokenizers might not have a pad_token
            # we can add it or just use eos
            if "<pad>" in self.tokenizer.get_vocab():
                self.tokenizer.pad_token_id = self.tokenizer.get_vocab()["<pad>"]
            else:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.vocab_size = len(self.tokenizer)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)
        
    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)


class HarnessFormatter:
    """
    Formats corpus text into the strict agentic harness template.
    Template:
    ### INIT
    You are in pretraining mode.
    ### END
    [pct] %
    Phase 1/2 of pretraining.
    All blocks implemented, context Nominal.
    # GO.

    ## TRAINING INTERRUPT
    [corpus text]
    # GO.
    """
    def __init__(self, tokenizer: TokenizerWrapper, corpus_tokens: int = 2048, pad_percentage_tokens: int = 3):
        self.tokenizer = tokenizer
        self.corpus_tokens = corpus_tokens
        self.pad_percentage_tokens = pad_percentage_tokens
        
        self.prefix_1 = (
            "|<STATE>|\n"
            "### INIT\n"
            "You are in pretraining mode.\n"
            "### END\n"
        )
        self.prefix_2_template = (
            " %\n"
            "Phase 1/2 of pretraining.\n"
            "All blocks implemented, context Nominal.\n"
            "# GO.\n\n"
            "|</STATE>|\n\n"
            "|<INTERRUPT>|\n"
            "## TRAINING INTERRUPT\n"
        )
        self.suffix = "\n# GO.\n|</INTERRUPT>|"
        
        self.p1_ids = self.tokenizer.encode(self.prefix_1)
        self.p2_base_ids = self.tokenizer.encode(self.prefix_2_template)
        self.suffix_ids = self.tokenizer.encode(self.suffix)
        self.eos_id = [self.tokenizer.tokenizer.eos_token_id]
        self.pad_id = self.tokenizer.tokenizer.pad_token_id
        
        # Calculate exactly the extra token size
        # We need to reserve some tokens for the percentage. e.g. "  0.00", "100.00"
        # We will pad it to always be exactly `pct_token_len` tokens long using pad_id.
        # "100.00" is typically 2-3 tokens depending on the tokenizer.
        # So we allocate a fixed number of tokens. The user requested to pad out 2-3 tokens.
        pct_sample_ids = self.tokenizer.encode("100.00")
        self.pct_max_token_len = len(pct_sample_ids) + self.pad_percentage_tokens
        
        self.template_overhead = len(self.p1_ids) + self.pct_max_token_len + len(self.p2_base_ids) + len(self.suffix_ids) + len(self.eos_id)
        
        self.seq_len = self.corpus_tokens + self.template_overhead
        
    def get_seq_len(self):
        return self.seq_len
        
    def format(self, corpus_text: str, progress_pct: float) -> Tuple[torch.Tensor, torch.Tensor]:
        pct_str = f"{progress_pct:.2f}"
        pct_ids = self.tokenizer.encode(pct_str)
        
        if len(pct_ids) < self.pct_max_token_len:
            # Pad with pad_id to keep length constant
            pct_ids = pct_ids + [self.pad_id] * (self.pct_max_token_len - len(pct_ids))
        elif len(pct_ids) > self.pct_max_token_len:
            pct_ids = pct_ids[:self.pct_max_token_len]
            
        prefix_ids = self.p1_ids + pct_ids + self.p2_base_ids
        
        corpus_ids = self.tokenizer.encode(corpus_text)
        if len(corpus_ids) > self.corpus_tokens:
            corpus_ids = corpus_ids[:self.corpus_tokens]
            
        full_ids = prefix_ids + corpus_ids + self.suffix_ids + self.eos_id
        
        if len(full_ids) < self.seq_len:
            full_ids = full_ids + [self.pad_id] * (self.seq_len - len(full_ids))
        elif len(full_ids) > self.seq_len:
            full_ids = full_ids[:self.seq_len]
            
        mask = torch.zeros(self.seq_len, dtype=torch.float)
        corpus_start = len(prefix_ids)
        corpus_actual_len = len(corpus_ids)
        if corpus_start + corpus_actual_len <= self.seq_len:
            mask[corpus_start:corpus_start + corpus_actual_len] = 1.0
            
        return torch.tensor(full_ids, dtype=torch.long), mask


class MemmapPretrainingDataset(IterableDataset):
    """
    Streams sequence chunks from a directory of .bin memmap files.
    """
    def __init__(self, data_dir: str, seq_len: int, dtype: np.dtype = np.uint32, start_sample_idx: int = 0):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.dtype = dtype
        self.start_sample_idx = start_sample_idx
        
        self.bin_files = []
        if os.path.exists(data_dir):
            for f in sorted(os.listdir(data_dir)): 
                if f.endswith('.bin') and not f.endswith('.mask.bin'):
                    self.bin_files.append(os.path.join(data_dir, f))
        
    def __iter__(self):
        worker_info = get_worker_info()
        tokens_to_skip = self.start_sample_idx * self.seq_len
        
        for i, bin_file in enumerate(self.bin_files):
            if worker_info is not None and i % worker_info.num_workers != worker_info.id:
                continue

            mask_file = bin_file.replace('.bin', '.mask.bin')
            has_mask = os.path.exists(mask_file)

            mmap_array = np.memmap(bin_file, dtype=self.dtype, mode='r')
            total_len = len(mmap_array)
            
            mask_array = None
            if has_mask:
                mask_array = np.memmap(mask_file, dtype=np.uint8, mode='r')
                if len(mask_array) != total_len:
                    has_mask = False
            
            if tokens_to_skip >= total_len:
                tokens_to_skip -= total_len
                continue
            
            idx = tokens_to_skip
            tokens_to_skip = 0 
            
            while idx + self.seq_len <= total_len:
                chunk = mmap_array[idx : idx + self.seq_len]
                input_ids = torch.from_numpy(chunk.astype(np.int64))
                
                if has_mask:
                    mask_chunk = mask_array[idx : idx + self.seq_len]
                    loss_mask = torch.from_numpy(mask_chunk).float()
                else:
                    loss_mask = torch.ones(self.seq_len, dtype=torch.float)
                
                yield {
                    "input_ids": input_ids.long(),
                    "loss_mask": loss_mask,
                }
                idx += self.seq_len

class JsonlParquetDataset(IterableDataset):
    """
    Streams from .jsonl or .parquet files and formats them on-the-fly.
    """
    def __init__(self, data_dir: str, formatter: HarnessFormatter, progress_tracker: Any, text_key: str = "text"):
        self.data_dir = data_dir
        self.formatter = formatter
        self.progress_tracker = progress_tracker
        self.text_key = text_key
        
        self.files = []
        if os.path.exists(data_dir):
            for f in sorted(os.listdir(data_dir)):
                if f.endswith('.jsonl') or f.endswith('.parquet'):
                    self.files.append(os.path.join(data_dir, f))

    def _read_parquet(self, filepath: str):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")
        parquet_file = pq.ParquetFile(filepath)
        for batch in parquet_file.iter_batches():
            for row in batch.to_pylist():
                text = row.get(self.text_key, "")
                if text: yield text

    def _read_jsonl(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    text = data.get(self.text_key, "")
                    if text: yield text
                except json.JSONDecodeError:
                    continue

    def __iter__(self):
        worker_info = get_worker_info()
        for i, filepath in enumerate(self.files):
            if worker_info is not None and i % worker_info.num_workers != worker_info.id:
                continue
                
            stream = self._read_parquet(filepath) if filepath.endswith('.parquet') else self._read_jsonl(filepath)
            for text in stream:
                pct = self.progress_tracker.get_percentage()
                input_ids, loss_mask = self.formatter.format(text, pct)
                
                self.progress_tracker.add_tokens(len(input_ids))
                
                yield {
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                }


class HuggingFaceDataset(IterableDataset):
    """
    Streams seamlessly from an external HuggingFace dataset mapped repository.
    """
    def __init__(self, dataset_path: str, formatter: HarnessFormatter, progress_tracker: Any, text_key: str = "text", split: str = "train"):
        self.dataset_path = dataset_path
        self.formatter = formatter
        self.progress_tracker = progress_tracker
        self.text_key = text_key
        self.split = split
        
    def __iter__(self):
        if load_dataset is None:
            raise ImportError("Please install datasets library via `pip install datasets` to use HF datasets.")
            
        import torch.distributed as dist
        
        # We load dataset via stream architecture
        dataset = load_dataset(self.dataset_path, split=self.split, streaming=True)
        
        try:
            import torch_xla.core.xla_model as xm
            in_tpu = True
        except ImportError:
            in_tpu = False
            
        if in_tpu and xm.xrt_world_size() > 1:
            rank = xm.get_ordinal()
            world_size = xm.xrt_world_size()
        else:
            rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers
        
        
        import time
        import logging
        from itertools import islice

        def resilient_stream(dataset, start_idx=0, max_retries=100):
            """Wraps the dataset iterator with robust reconnection logic to prevent hanging."""
            current_idx = start_idx
            retries = 0
            
            while retries < max_retries:
                try:
                    # Skip to the current index using itertools.islice
                    iterator = iter(dataset)
                    if current_idx > 0:
                        iterator = islice(iterator, current_idx, None)
                        
                    for item in iterator:
                        yield current_idx, item
                        current_idx += 1
                        retries = 0  # Reset retries on successful yield
                        
                    break  # If we exhausted the dataset normally, we're done
                    
                except Exception as e:
                    retries += 1
                    err_msg = str(e).lower()
                    if "timeout" in err_msg or "connection" in err_msg or "ssl" in err_msg or "http" in err_msg:
                        logging.warning(f"HF Dataset network error at index {current_idx}: {e}. Retrying {retries}/{max_retries} in 15 seconds...")
                        time.sleep(15)
                    else:
                        logging.error(f"HF Dataset unexpected error at index {current_idx}: {e}.")
                        # For format/parsing errors, we want to skip the bad item and continue
                        current_idx += 1 
                        time.sleep(2)
        
        # Use the resilient stream to iterate
        for i, example in resilient_stream(dataset):
            # Deterministic modulo splitting ensures zero duplicate drops when distributed over multi-GPU environments!
            if i % global_num_workers != global_worker_id:
                continue
                
            text = example.get(self.text_key, "")
            if not text: continue
            
            try:
                pct = self.progress_tracker.get_percentage()
                input_ids, loss_mask = self.formatter.format(text, pct)
                self.progress_tracker.add_tokens(len(input_ids))
                
                yield {
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                }
            except Exception as e:
                # Catch any formatting errors to prevent the entire dataloader from dying
                logging.warning(f"Error formatting example {i}: {e}. Skipping.")
                continue


class MultimodalDataset(IterableDataset):
    """
    Multimodal pretraining dataset yielding text tokens, loss masks, and raw image/video patches.
    """
    def __init__(self, data_dir: str, formatter: HarnessFormatter, progress_tracker: Any, text_key: str = "text", image_key: str = "image"):
        self.data_dir = data_dir
        self.formatter = formatter
        self.progress_tracker = progress_tracker
        self.text_key = text_key
        self.image_key = image_key
        
        self.files = []
        if os.path.exists(data_dir):
            for f in sorted(os.listdir(data_dir)):
                if f.endswith('.jsonl') or f.endswith('.parquet'):
                    self.files.append(os.path.join(data_dir, f))

    def _read_stream(self, filepath: str):
        if filepath.endswith('.parquet'):
            if pq is None: raise ImportError("pyarrow required")
            table = pq.read_table(filepath)
            for batch in table.to_batches():
                for row in batch.to_pylist():
                    yield row
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass

    def __iter__(self):
        # Dummy PIL/Torchvision preprocessing stub - assumes real vision loader is external
        import torchvision.transforms as T
        from PIL import Image
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        
        worker_info = get_worker_info()
        for i, filepath in enumerate(self.files):
            if worker_info is not None and i % worker_info.num_workers != worker_info.id:
                continue
                
            for row in self._read_stream(filepath):
                text = row.get(self.text_key, "")
                img_path = row.get(self.image_key, None)
                if not text: continue
                
                pct = self.progress_tracker.get_percentage()
                input_ids, loss_mask = self.formatter.format(text, pct)
                
                self.progress_tracker.add_tokens(len(input_ids))
                
                item = {
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                }
                
                # Load dummy image patch representations for now (real loading via models like CLIP/TIPS needs image embeddings or pixels)
                # Assume vision wrapper will generate `image_patches` during forward or dataset transforms pixels:
                if img_path and os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        item["image_pixels"] = transform(img)
                    except Exception:
                        pass
                
                yield item
