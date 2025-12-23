#!/usr/bin/env python3
"""Test if model actually learned by computing loss on real data."""
import torch
import sys
sys.path.insert(0, '.')

from transformers import AutoTokenizer
from src.models.phase8.integrated_model import Phase8IntegratedModel
from src.models.phase8.config import Phase8Config
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load checkpoint
ckpt = torch.load("checkpoints/phase8_300m_scaling/step_4000.pt", map_location=device)
config = ckpt["config"]

# Create model
phase8_config = Phase8Config(
    d_model=config.get("d_model", 1024),
    n_layers=config.get("n_layers", 24),
    n_seq=config.get("n_seq", 512),
    vocab_size=config.get("vocab_size", 50256),
    num_heads=config.get("num_heads", 16),
    htt_rank=config.get("htt_rank", 16),
    use_resonant_htt=config.get("use_resonant_htt", True),
    resonant_num_cores=config.get("resonant_num_cores", 4),
    use_zeta_init=config.get("use_zeta_init", True),
    low_rank_ffn=config.get("low_rank_ffn", True),
    low_rank_attention=config.get("low_rank_attention", True),
    low_rank_rank=config.get("low_rank_rank", 32),
    use_bk_hyperbolic=config.get("use_bk_hyperbolic", True),
)

model = Phase8IntegratedModel(phase8_config).to(device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

# Use GPT-2 tokenizer (same as training data)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Test sentences - compute loss like during training
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Hello, how are you today?",
    "The weather is nice today.",
]

print("\n=== Computing Loss on Test Texts ===")
for text in test_texts:
    try:
        tokens = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        
        # Need n_seq length for model
        n_seq = config.get("n_seq", 512)
        if tokens.shape[1] < n_seq + 1:
            # Pad
            pad_id = tokenizer.pad_token_id or 0
            padding = torch.full((1, n_seq + 1 - tokens.shape[1]), pad_id, dtype=torch.long, device=device)
            tokens = torch.cat([padding, tokens], dim=1)
        else:
            tokens = tokens[:, :n_seq + 1]
        
        input_ids = tokens[:, :-1]  # Input
        targets = tokens[:, 1:]     # Target (shifted by 1)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=device=="cuda"):
                output = model(input_ids)
            
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # Compute loss
            logits_flat = logits.view(-1, logits.size(-1)).float()
            targets_flat = targets.view(-1)
            
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
            ppl = torch.exp(loss).item()
            
            print(f"\nText: '{text[:50]}...' ({tokens.shape[1]} tokens)")
            print(f"  Loss: {loss.item():.4f}, PPL: {ppl:.2f}")
    except Exception as e:
        print(f"\nText: '{text[:50]}...' - ERROR: {e}")

print("\n=== Random Baseline ===")
# Random logits should give loss = ln(vocab_size)
random_loss = torch.log(torch.tensor(float(config.get("vocab_size", 50256)))).item()
random_ppl = config.get("vocab_size", 50256)
print(f"Random baseline: Loss = {random_loss:.4f}, PPL = {random_ppl}")

print("\n=== Checkpoint Stats ===")
print(f"Checkpoint step: {ckpt.get('step', 'unknown')}")
print(f"Checkpoint loss: {ckpt.get('loss', 'unknown')}")

