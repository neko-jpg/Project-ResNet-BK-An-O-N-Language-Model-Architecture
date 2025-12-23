#!/usr/bin/env python3
"""Deep debug to check model forward pass layer by layer."""
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

print(f"\n=== Config ===")
print(f"vocab_size: {config.get('vocab_size')}")
print(f"d_model: {config.get('d_model')}")
print(f"n_layers: {config.get('n_layers')}")

# Create model with SAME config as training
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

# Load weights
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
print(f"\n=== Weight Loading ===")
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")
if missing:
    print(f"First 5 missing: {missing[:5]}")
if unexpected:
    print(f"First 5 unexpected: {unexpected[:5]}")

model.eval()

# Use GPT-2 tokenizer (same as training data!)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"\nUsing GPT-2 tokenizer (vocab_size: {tokenizer.vocab_size})")

# Test input
prompt = "Hello? how are you?"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
print(f"\nInput: '{prompt}'")
print(f"Token IDs: {input_ids.tolist()}")

# Pad to n_seq
n_seq = config.get("n_seq", 512)
if input_ids.shape[1] < n_seq:
    pad_id = tokenizer.pad_token_id or 0
    padding = torch.full((1, n_seq - input_ids.shape[1]), pad_id, dtype=torch.long, device=device)
    input_padded = torch.cat([padding, input_ids], dim=1)
else:
    input_padded = input_ids[:, -n_seq:]

print(f"Padded shape: {input_padded.shape}")

# Forward with hooks to check intermediate values
activations = {}

def hook_fn(name):
    def fn(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        if isinstance(out, torch.Tensor):
            activations[name] = {
                'mean': out.float().mean().item(),
                'std': out.float().std().item(),
                'min': out.float().min().item(),
                'max': out.float().max().item(),
            }
    return fn

# Register hooks on key layers
hooks = []
# hooks.append(model.phase7_model.model.token_embedding.register_forward_hook(hook_fn('token_embedding')))
# hooks.append(model.phase7_model.model.blocks[0].register_forward_hook(hook_fn('block_0')))
# hooks.append(model.phase7_model.model.blocks[-1].register_forward_hook(hook_fn('block_last')))

# Forward pass
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=device=="cuda"):
        output = model(input_padded)

    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    logits_float = logits.float()
    
print(f"\n=== Output Analysis ===")
print(f"Logits shape: {logits.shape}")
print(f"Logits dtype: {logits.dtype}")
print(f"Logits mean: {logits_float.mean().item():.6f}")
print(f"Logits std: {logits_float.std().item():.6f}")
print(f"Logits min: {logits_float.min().item():.6f}")
print(f"Logits max: {logits_float.max().item():.6f}")

# Check if logits are all similar (near uniform)
probs = torch.softmax(logits_float[0, -1, :], dim=-1)
print(f"\nProbability stats:")
print(f"Max prob: {probs.max().item():.6f}")
print(f"Min prob: {probs.min().item():.6f}")
print(f"Entropy (higher=more random): {(-probs * probs.log()).sum().item():.2f}")

# Expected entropy for uniform distribution
uniform_entropy = torch.log(torch.tensor(float(config.get('vocab_size', 50256)))).item()
print(f"Max entropy (uniform): {uniform_entropy:.2f}")

# Clean up hooks
for h in hooks:
    h.remove()

print("\n=== Check Embedding Layer ===")
# Manually check embedding
try:
    # Get embedding output
    with torch.no_grad():
        # Access inner model
        inner = model.phase7_model.model
        
        # Token embedding
        token_emb = inner.token_embedding(input_padded)
        print(f"Token embedding shape: {token_emb.shape}")
        print(f"Token embedding mean: {token_emb.float().mean().item():.6f}")
        print(f"Token embedding std: {token_emb.float().std().item():.6f}")
        
        # After first block
        x = token_emb
        if hasattr(inner, 'position_embedding'):
            pos_emb = inner.position_embedding.weight[:n_seq, :]
            x = x + pos_emb
            print(f"\nAfter position: mean={x.float().mean().item():.6f}, std={x.float().std().item():.6f}")
        
        # First block
        if hasattr(inner, 'blocks') and len(inner.blocks) > 0:
            block_out = inner.blocks[0](x)
            if isinstance(block_out, tuple):
                block_out = block_out[0]
            print(f"After block 0: mean={block_out.float().mean().item():.6f}, std={block_out.float().std().item():.6f}")
            
except Exception as e:
    print(f"Error accessing internals: {e}")
    import traceback
    traceback.print_exc()
