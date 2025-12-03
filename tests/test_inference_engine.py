
import torch
import torch.nn as nn
import pytest
from src.inference.inference_engine import HyperbolicKVCache, InferenceEngine, patch_model_with_kv_cache
from src.models.phase7.hyperbolic_attention import HyperbolicMultiHeadAttention

class MockModel(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        self.attn = HyperbolicMultiHeadAttention(d_model, num_heads, use_triton_kernel=False)
        self.fc = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        out, _ = self.attn(x)
        return self.fc(out)

def test_hyperbolic_kv_cache_update():
    B, H, D = 1, 4, 16
    cache = HyperbolicKVCache(max_size=10, prune_ratio=0.2)
    
    # Update 1
    k1 = torch.randn(B, H, 1, D)
    v1 = torch.randn(B, H, 1, D)
    c = torch.tensor(1.0)
    cache.update(k1, v1, c)
    
    k_c, v_c = cache.get_cache()
    assert k_c.shape == (B, H, 1, D)
    
    # Update 2
    k2 = torch.randn(B, H, 1, D)
    v2 = torch.randn(B, H, 1, D)
    cache.update(k2, v2, c)
    
    k_c, v_c = cache.get_cache()
    assert k_c.shape == (B, H, 2, D)

def test_hyperbolic_kv_cache_pruning():
    B, H, D = 1, 1, 4
    cache = HyperbolicKVCache(max_size=5, prune_ratio=0.2)
    c = torch.tensor(1.0)
    
    # Add 5 tokens
    for i in range(5):
        k = torch.randn(B, H, 1, D)
        v = torch.randn(B, H, 1, D)
        cache.update(k, v, c)
        
    assert cache.k_cache.shape[2] == 5
    
    # Add 6th token -> Trigger Prune
    # Keep len = 5 * (1-0.2) = 4
    k = torch.randn(B, H, 1, D)
    v = torch.randn(B, H, 1, D)
    cache.update(k, v, c)
    
    # After update, size is 6. Then prune to 4.
    # Wait, update appends then prunes.
    # If size > max_size (5).
    # 6 > 5. Prune.
    # keep_len = 6 * 0.8 = 4.8 -> 4.
    
    assert cache.k_cache.shape[2] == 4

def test_inference_engine_integration():
    model = MockModel()
    engine = InferenceEngine(model, max_cache_size=10)
    patch_model_with_kv_cache(model, engine)
    
    input_ids = torch.randn(1, 5, 64) # (B, L, D) - Mock input embeddings
    
    # Forward pass (Prefill)
    out = model(input_ids)
    assert out.shape == (1, 5, 64)
    
    # Check cache populated
    assert len(engine.kv_caches) == 1
    cache = list(engine.kv_caches.values())[0]
    assert cache.k_cache.shape[2] == 5
    
    # Forward pass (Step)
    step_input = torch.randn(1, 1, 64)
    out_step = model(step_input)
    assert out_step.shape == (1, 1, 64)
    
    assert cache.k_cache.shape[2] == 6
