#!/usr/bin/env python3
"""Debug inference to find why model outputs garbage."""
import torch
import sys
sys.path.insert(0, '.')

from transformers import AutoTokenizer
from src.models.phase8.integrated_model import Phase8IntegratedModel
from src.models.phase8.config import Phase8Config

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load checkpoint
ckpt = torch.load("checkpoints/phase8_300m_scaling/step_4000.pt", map_location=device)
config = ckpt["config"]
print(f"Model config: vocab_size={config.get('vocab_size')}, d_model={config.get('d_model')}")

# Create model
phase8_config = Phase8Config(
    d_model=config.get("d_model", 1024),
    n_layers=config.get("n_layers", 24),
    n_seq=config.get("n_seq", 512),
    vocab_size=config.get("vocab_size", 32768),
    num_heads=config.get("num_heads", 16),
)
model = Phase8IntegratedModel(phase8_config).to(device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

# Load tokenizer - use GPT-2 for debugging (may not match training data perfectly)
try:
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b", trust_remote_code=True)
    print(f"Using rinna tokenizer")
except Exception as e:
    print(f"Failed to load rinna tokenizer: {e}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"Using GPT-2 tokenizer (fallback)")
    
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

# Test
prompt = "Hello?"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
print(f"Input IDs: {input_ids}")
print(f"Max input ID: {input_ids.max().item()}")

# Need to pad to n_seq for the model
n_seq = config.get("n_seq", 512)
if input_ids.shape[1] < n_seq:
    pad_token_id = tokenizer.pad_token_id or 0
    padding = torch.full((1, n_seq - input_ids.shape[1]), pad_token_id, dtype=torch.long, device=device)
    input_padded = torch.cat([padding, input_ids], dim=1)
    print(f"Padded input shape: {input_padded.shape}")
else:
    input_padded = input_ids[:, -n_seq:]

# Forward pass
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=device=="cuda"):
        output = model(input_padded)
        
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output
    print(f"Logits shape: {logits.shape}")
    
    # Check logits distribution
    logits_float = logits.float()
    print(f"Logits min: {logits_float.min().item():.4f}, max: {logits_float.max().item():.4f}")
    print(f"Logits mean: {logits_float.mean().item():.4f}, std: {logits_float.std().item():.4f}")
    
    # Check for NaN/Inf
    if torch.isnan(logits_float).any():
        print("WARNING: NaN in logits!")
    if torch.isinf(logits_float).any():
        print("WARNING: Inf in logits!")
    
    # Top 5 predictions at last position
    probs = torch.softmax(logits_float[0, -1, :] / 0.8, dim=-1)
    top5 = torch.topk(probs, 10)
    print(f"\nTop 10 predictions:")
    print(f"Token IDs: {top5.indices.tolist()}")
    print(f"Probs: {[f'{p:.4f}' for p in top5.values.tolist()]}")
    for idx, prob in zip(top5.indices.tolist(), top5.values.tolist()):
        if idx < tokenizer.vocab_size:
            decoded = tokenizer.decode([idx])
            print(f"  {idx}: '{decoded}' (prob={prob:.4f})")
        else:
            print(f"  {idx}: [OOV - larger than tokenizer vocab] (prob={prob:.4f})")

print("\n=== Check Generation ===")
# Generate a few tokens
generated = input_ids.clone()
for step in range(10):
    current_len = generated.shape[1]
    if current_len < n_seq:
        padding = torch.full((1, n_seq - current_len), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
        context = torch.cat([padding, generated], dim=1)
    else:
        context = generated[:, -n_seq:]
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device=="cuda"):
            output = model(context)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        next_logits = logits[:, -1, :].float() / 0.8
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        
        token_text = tokenizer.decode([next_token.item()]) if next_token.item() < tokenizer.vocab_size else "[OOV]"
        print(f"Step {step}: token={next_token.item()}, text='{token_text}'")

print(f"\nFull generated: {tokenizer.decode(generated[0].tolist())}")
