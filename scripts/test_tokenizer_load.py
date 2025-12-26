#!/usr/bin/env python3
"""Test rinna tokenizer loading with HuggingFace token."""

import sys

try:
    from transformers import AutoTokenizer
    
    token = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Loading rinna tokenizer with token: {token[:10]}..." if token else "No token provided")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "rinna/japanese-gpt-neox-3.6b",
        token=token,
        use_fast=False  # rinna tokenizer may need this
    )
    
    print(f"SUCCESS! Tokenizer loaded: {type(tokenizer)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Test encoding
    test_text = "こんにちは、世界！"
    tokens = tokenizer.encode(test_text)
    print(f"Test encoding: '{test_text}' -> {tokens}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
