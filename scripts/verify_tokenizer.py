#!/usr/bin/env python3
"""Verify training data tokenization by decoding actual training samples."""

import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

# Load actual training data
data_path = Path("data/dolly_ja/train.bin")
if not data_path.exists():
    data_path = Path("data/wiki_ja/train.bin")
if not data_path.exists():
    data_path = Path("data/japanese_instruct/train.bin")

if not data_path.exists():
    print("No training data found!")
    exit(1)

print(f"Loading training data from: {data_path}")
tokens = np.memmap(data_path, dtype=np.uint32, mode='r')

print(f"Total tokens: {len(tokens):,}")
print(f"Token ID range: {tokens.min()} - {tokens.max()}")
print()

# Try decoding with GPT-2
print("=== Decoding with GPT-2 tokenizer ===")
tokenizer_gpt2 = AutoTokenizer.from_pretrained('gpt2')
sample_tokens = tokens[0:200].tolist()
decoded_gpt2 = tokenizer_gpt2.decode(sample_tokens)
print(f"Sample tokens (first 20): {sample_tokens[:20]}")
print(f"Decoded text: {decoded_gpt2[:500]}")
print()

# Check if it looks like Japanese
has_japanese = any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in decoded_gpt2)
print(f"Contains Japanese characters: {has_japanese}")
