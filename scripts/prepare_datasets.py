"""
Dataset Preparation Script for Mamba-Killer ResNet-BK

Downloads and preprocesses datasets for:
1. Phase 3: Logic/Physics (Wiki, Math, Code, ArXiv)
2. Phase 4: LOGOS (Sarcasm, Emotion, Dialogue)
3. Phase 5: Factuality (Knowledge extraction)

Supports English and Japanese.
Generates .pt (PyTorch) and .bin (Mmap-ready) formats.
Uses streaming and batched writes for memory efficiency.

Usage:
    python scripts/prepare_datasets.py --all --output_dir ./data
"""

import argparse
import os
import json
import struct
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterator

try:
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install datasets transformers torch numpy")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinaryWriter:
    """Handles append-only writing to .bin (tokens) and .idx (offsets)."""

    def __init__(self, output_prefix: Path):
        self.bin_path = output_prefix.with_suffix('.bin')
        self.idx_path = output_prefix.with_suffix('.idx')

        # Initialize files (overwrite)
        with open(self.bin_path, 'wb') as f:
            pass

        self.idx_file = open(self.idx_path, 'wb')
        # Write Header: Magic(4), Version(4)
        self.idx_file.write(b'MUSE')
        self.idx_file.write(struct.pack('<I', 1))

        self.current_offset = 0
        self.doc_count = 0

    def append(self, token_lists: List[List[int]]):
        """Append a batch of token sequences."""
        if not token_lists:
            return

        # Prepare binary data
        flat_tokens = []
        offsets = []

        for seq in token_lists:
            length = len(seq)
            flat_tokens.extend(seq)
            # Index format: offset (uint64), length (uint32)
            offsets.append((self.current_offset, length))
            self.current_offset += length
            self.doc_count += 1

        # Write tokens
        arr = np.array(flat_tokens, dtype=np.uint32)
        with open(self.bin_path, 'ab') as f:
            f.write(arr.tobytes())

        # Write offsets
        # We write (offset, length) pairs. offset is uint64, length is uint64 for alignment/safety
        offset_arr = np.array(offsets, dtype=np.uint64)
        self.idx_file.write(offset_arr.tobytes())

    def close(self):
        self.idx_file.close()
        logger.info(f"Closed {self.bin_path} ({self.current_offset} tokens, {self.doc_count} docs)")
        return str(self.bin_path)


class DatasetPreparator:
    """Handles downloading and preprocessing of benchmark datasets."""
    
    def __init__(self, output_dir: str = "./data", tokenizer_name: str = "gpt2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configuration
        self.dataset_configs = {
            # --- Teppei's "High-Density" Curriculum (OSS Safe) ---
            
            # 1. The "Brain" (Textbook Quality Logic)
            'cosmopedia': {
                'hf_name': 'HuggingFaceTB/cosmopedia',
                'hf_config': 'web_samples_v2',
                'splits': ['train'],
                'streaming': True,
                'text_col': 'text' # Contains synthesized textbook content
            },

            # 2. The "Hands" (Expert Code)
            'evol_instruct_code': {
                'hf_name': 'nickrosh/Evol-Instruct-Code-80k-v1',
                'splits': ['train'],
                'streaming': True,
                'text_col': 'special_evol' # Instruction + Output
            },

            # 3. The "Soul" (Japanese Culture & Politeness)
            'wiki_ja': {
                'hf_name': 'izumi-lab/wikipedia-ja-20230720',
                'splits': ['train'],
                'text_col': 'text'
            },
            'japanese_instruct': {
                'hf_name': 'kunishou/databricks-dolly-15k-ja',
                'splits': ['train'],
                'text_col': 'special_dolly'
            },

            # --- Legacy / Supplementary ---
            'wikitext103': {
                'hf_name': 'wikitext',
                'hf_config': 'wikitext-103-raw-v1',
                'splits': ['train', 'validation', 'test'],
                'text_col': 'text'
            },
             # --- Physics & Math ---
            'arxiv': {
                'hf_name': 'gfissore/arxiv-abstracts-2021',
                'splits': ['train'],
                'text_col': 'abstract'
            },
             # --- Factuality (Triples) ---
            'factuality_source': {
                'hf_name': 'wikimedia/wikipedia',
                'hf_config': '20231101.en',
                'splits': ['train'],
                'streaming': True,
                'text_col': 'text'
            }
        }

    def _process_dataset(self, key: str, max_samples: Optional[int] = None, val_ratio: float = 0.0) -> Optional[Dict[str, str]]:
        """Generic processing logic with streaming and batching."""
        config = self.dataset_configs[key]
        dataset_name = config['hf_name']
        logger.info(f"Processing {key} ({dataset_name})...")
        
        output_dir = self.output_dir / key
        output_dir.mkdir(exist_ok=True)
        output_paths = {}
        
        try:
            # Check if we need to artificially create a validation split
            splits_to_process = config['splits']
            artificial_val = False

            if 'validation' not in splits_to_process and val_ratio > 0:
                # We will split 'train' into train/val
                artificial_val = True
                # Ensure we only process 'train' from the list if it's there
                if 'train' not in splits_to_process:
                     logger.warning(f"Cannot create artificial validation split for {key} because 'train' split is missing.")
                     artificial_val = False

            for split in splits_to_process:
                logger.info(f"  Split: {split}")
                
                # Load Dataset
                kwargs = {'split': split}
                if config.get('streaming'):
                    kwargs['streaming'] = True
                
                try:
                    if config.get('hf_config'):
                        ds = load_dataset(dataset_name, config['hf_config'], **kwargs)
                    elif config.get('subset_lang'):
                         ds = load_dataset(dataset_name, **kwargs)
                    else:
                        ds = load_dataset(dataset_name, **kwargs)
                except Exception as e:
                    if config.get('gated') and ("401" in str(e) or "403" in str(e)):
                        logger.warning(f"Skipping {key}: Gated dataset authentication failed. Please login via `huggingface-cli login`.")
                        return None
                    logger.warning(f"Failed to load {key}: {e}. Skipping.")
                    return None

                # Setup Writers
                writer = BinaryWriter(output_dir / split)
                val_writer = None

                if split == 'train' and artificial_val:
                     val_writer = BinaryWriter(output_dir / 'validation')
                     logger.info(f"    (Splitting 'train' into 'train' and 'validation' with ratio {val_ratio})")

                # Processing Loop
                batch_tokens = []
                val_batch_tokens = []
                batch_size = 1000
                count = 0
                val_count = 0
                
                iterator = ds
                if not config.get('streaming'):
                    iterator = ds # list/arrow
                
                import random
                rng = random.Random(42)

                for item in iterator:
                    if max_samples and count >= max_samples:
                        break

                    # Filter for Stack V2 if needed
                    if config.get('subset_lang') and key.startswith('the_stack'):
                        lang = item.get('language', item.get('lang', None))
                        target = config['subset_lang']
                        if lang and target.lower() not in lang.lower():
                            continue
                    
                    # Text Extraction
                    text = ""
                    text_col = config['text_col']

                    if text_col == 'special_dolly':
                        instruction = item.get('instruction', '')
                        input_text = item.get('input', item.get('context', ''))
                        output_text = item.get('output', item.get('response', ''))
                        text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"

                    elif text_col == 'special_evol':
                        # Evol-Instruct: instruction + output
                        instruction = item.get('instruction', '')
                        output = item.get('output', '')
                        text = f"Instruction: {instruction}\nOutput: {output}"

                    elif text_col == 'special_magicoder':
                        # Magicoder: instruction + response
                        instruction = item.get('instruction', '')
                        response = item.get('response', '')
                        text = f"Instruction: {instruction}\nResponse: {response}"

                    else:
                        text = item.get(text_col, "")

                    if text and text.strip():
                        # Tokenize
                        ids = self.tokenizer(text, truncation=False, padding=False, return_attention_mask=False)['input_ids']

                        # Decide split
                        if val_writer and rng.random() < val_ratio:
                            val_batch_tokens.append(ids)
                            val_count += 1
                        else:
                            batch_tokens.append(ids)
                            count += 1

                    # Flush Batch (Train)
                    if len(batch_tokens) >= batch_size:
                        writer.append(batch_tokens)
                        batch_tokens = []
                        if count % 10000 == 0:
                            logger.info(f"    Processed {count} training samples...")

                    # Flush Batch (Val)
                    if val_writer and len(val_batch_tokens) >= batch_size:
                        val_writer.append(val_batch_tokens)
                        val_batch_tokens = []

                # Flush remaining
                if batch_tokens:
                    writer.append(batch_tokens)
                if val_batch_tokens and val_writer:
                    val_writer.append(val_batch_tokens)

                output_paths[f"{split}_bin"] = writer.close()
                if val_writer:
                    output_paths["validation_bin"] = val_writer.close()
                    logger.info(f"    Created validation split with {val_count} samples.")
                
        except Exception as e:
            logger.error(f"Failed to process {key}: {e}")
            return None

        # Save Metadata
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump({
                'dataset': key,
                'hf_name': dataset_name,
                'paths': output_paths,
                'doc_count': count
            }, f, indent=2)

        return output_paths

    def prepare_all(self, max_samples: int = 10000, val_ratio: float = 0.05):
        """Prepare all datasets."""
        results = {}
        for key in self.dataset_configs:
            results[key] = self._process_dataset(key, max_samples, val_ratio)
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("All datasets prepared.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./data')
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--val_ratio', type=float, default=0.05, help='Ratio of training data to use for validation if no split exists')
    parser.add_argument('--tokenizer', default='gpt2')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to prepare')
    args = parser.parse_args()
    
    prep = DatasetPreparator(args.output_dir, args.tokenizer)
    
    if args.datasets:
        for ds in args.datasets:
            if ds in prep.dataset_configs:
                prep._process_dataset(ds, args.max_samples, args.val_ratio)
            else:
                logger.warning(f"Dataset {ds} not found in config.")
    else:
        prep.prepare_all(args.max_samples, args.val_ratio)

if __name__ == '__main__':
    main()
