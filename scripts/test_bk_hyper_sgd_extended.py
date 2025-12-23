#!/usr/bin/env python3
"""Extended stability test for BK-HyperSGD optimizer."""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')
from src.optimizers.bk_hyper_sgd import BKHyperSGD

print('='*60)
print('BK-HyperSGD Extended Stability Test (100 steps)')
print('='*60)

# Test model with various param types
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_proj = nn.Linear(64, 64)  # Unitary
        self.hyperbolic_layer = nn.Linear(64, 64)  # 'hyperbolic' in name
        self.ffn = nn.Linear(64, 64)  # Euclidean
    
    def forward(self, x):
        return self.ffn(self.hyperbolic_layer(self.v_proj(x)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

model = TestModel().to(device)

# Name params for optimizer
for name, param in model.named_parameters():
    param._param_name = name

optimizer = BKHyperSGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer.set_param_names(model)

# Print initial stats
stats = optimizer.get_statistics()
print(f'Parameter types: {stats["param_type_counts"]}')

# Run 100 steps
losses = []
for i in range(100):
    optimizer.zero_grad()
    x = torch.randn(8, 64, device=device)
    y = model(x)
    loss = y.mean()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (i+1) % 25 == 0:
        print(f'  Step {i+1}: loss={loss.item():.4f}')

# Final stats
stats = optimizer.get_statistics()
print(f'\nAfter 100 steps:')
print(f'  Cayley fallbacks: {stats["cayley_fallback_count"]}')
print(f'  Cayley normalized: {stats["cayley_normalized_count"]}')
print(f'  Poincare projections: {stats["poincare_projection_count"]}')

# Verify loss is finite
all_finite = all(not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in losses)
print(f'  All losses finite: {all_finite}')

# Check momentum buffer dtype
for p in model.parameters():
    if p in optimizer.state:
        buf_dtype = optimizer.state[p]['momentum_buffer'].dtype
        print(f'  Momentum buffer dtype: {buf_dtype} (expected: float32)')
        break

if all_finite:
    print('\n✅ Extended stability test PASSED!')
else:
    print('\n❌ Test FAILED - NaN/Inf detected')
    sys.exit(1)
