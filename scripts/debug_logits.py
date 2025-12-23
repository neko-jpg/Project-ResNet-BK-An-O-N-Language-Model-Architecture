#!/usr/bin/env python3
"""Debug model logits to understand tokenizer mismatch."""

import torch
import sys
sys.path.insert(0, '.')

from transformers import AutoTokenizer
from src.models.phase8.integrated_model import Phase8IntegratedModel
from src.models.phase8.config import Phase8Config

# Load checkpoint
print("Loading checkpoint...")
ckpt = torch.load('checkpoints/phase8_300m_japanese_chat/step_16000.pt', map_location='cpu', weights_only=False)
cfg = ckpt.get('config', {})

print(f"Config: d_model={cfg.get('d_model')}, vocab_size={cfg.get('vocab_size')}")

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model config
phase8_cfg = Phase8Config(
    d_model=cfg.get('d_model', 1024),
    n_layers=cfg.get('n_layers', 24),
    vocab_size=cfg.get('vocab_size', 50256),
    n_seq=cfg.get('n_seq', 512),
    num_heads=cfg.get('num_heads', 16),
    use_resonant_htt=cfg.get('use_resonant_htt', True),
    resonant_num_cores=cfg.get('resonant_num_cores', 4),
)
model = Phase8IntegratedModel(phase8_cfg).to(device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
print("Model loaded!")

# Test with tokenizer - use training format
tokenizer = AutoTokenizer.from_pretrained('gpt2')
eos_id = tokenizer.eos_token_id or 50256
print(f"EOS ID: {eos_id}, Token: |{repr(tokenizer.decode([eos_id]))}|")

prompt = '### 指示:\nこんにちはと挨拶して\n\n### 回答:\n'
input_ids_orig = tokenizer.encode(prompt, return_tensors='pt')
prompt_len = input_ids_orig.shape[1]
print(f'\nInput: {prompt}')
print(f'Token IDs: {input_ids_orig[0].tolist()}')

# --- Test 1: Right-Padding (Prompt at Position 0) ---
print("\n=== Test 1: Right-Padding (Prompt at Position 0) ===")
n_seq = cfg.get('n_seq', 512)
pad_token_id = 0 # Try 0 first (gpt2 0 is '!')
x_right = torch.full((1, n_seq), pad_token_id, dtype=torch.long)
x_right[0, :prompt_len] = input_ids_orig[0]
x_right = x_right.to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = model(x_right)
        if isinstance(logits, tuple): logits = logits[0]
        logits = logits.float()

print(f"Logits at last prompt token (index {prompt_len-1}):")
probs = torch.softmax(logits[0, prompt_len-1], dim=0)
top5 = torch.topk(probs, 5)
for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
    token = tokenizer.decode([idx.item()])
    print(f'  {i+1}. ID={idx.item():5d}, prob={prob.item():.4f}, token=|{repr(token)}|')

# --- Test 2: Left-Padding (Same as Chat Inference) ---
print("\n=== Test 2: Left-Padding (Prompt at the End) ===")
x_left = torch.full((1, n_seq), pad_token_id, dtype=torch.long)
x_left[0, -prompt_len:] = input_ids_orig[0]
x_left = x_left.to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = model(x_left)
        if isinstance(logits, tuple): logits = logits[0]
        logits = logits.float()

print(f"Logits at last prompt token (index 511):")
probs = torch.softmax(logits[0, -1], dim=0)
top5 = torch.topk(probs, 5)
for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
    token = tokenizer.decode([idx.item()])
    print(f'  {i+1}. ID={idx.item():5d}, prob={prob.item():.4f}, token=|{repr(token)}|')

# --- Test 3: No Padding (Short sequence) ---
# Check if model allows short sequences
try:
    print("\n=== Test 3: Short sequence (No Padding) ===")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(input_ids_orig.to(device))
            if isinstance(logits, tuple): logits = logits[0]
            logits = logits.float()
    print("Model allows short sequences!")
    probs = torch.softmax(logits[0, -1], dim=0)
    top5 = torch.topk(probs, 5)
    for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
        token = tokenizer.decode([idx.item()])
        print(f'  {i+1}. ID={idx.item():5d}, prob={prob.item():.4f}, token=|{repr(token)}|')
except Exception as e:
    print(f"Model failed short sequence: {e}")
