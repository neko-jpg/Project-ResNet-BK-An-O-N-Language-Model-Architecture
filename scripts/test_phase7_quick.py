#!/usr/bin/env python3
"""Quick test for Phase 7 model."""
import torch
from src.models.phase7 import Phase7IntegratedModel, Phase7Config

config = Phase7Config(
    vocab_size=1000,
    d_model=64,
    n_layers=2,
    n_seq=32,
    num_heads=4,
    htt_rank=8,
    hyperbolic_window_size=16,
    use_triton_kernel=False,
)

model = Phase7IntegratedModel(config)
print(f'Model created! Params: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M')
print(f'HTT params: {model.get_embedding_parameter_count() / 1e6:.6f}M')

x = torch.randint(0, 1000, (2, 32))
output = model(x)
print(f'Input: {x.shape}, Output: {output.shape}')
print('Forward pass successful!')
