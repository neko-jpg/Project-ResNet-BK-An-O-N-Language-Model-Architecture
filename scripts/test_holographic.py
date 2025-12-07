#!/usr/bin/env python3
"""Quick test for holographic FFT speed."""
import torch
import torch.nn as nn
import sys
import time

sys.path.insert(0, '.')

from src.training.holographic_training import HolographicWeightSynthesis

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 1000)
    def forward(self, x):
        return self.fc(x.float().mean(dim=-1, keepdim=True).expand(-1, 128))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DummyModel().to(device)
holo = HolographicWeightSynthesis(model)

# Pure FFT test
x = torch.randn(100000, device=device)
y = torch.randn(100000, device=device)

if device.type == 'cuda':
    torch.cuda.synchronize()

times = []
for _ in range(100):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    z, t = holo.synthesize_weights(x, y)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)

print(f'=== Holographic FFT Speed ===')
print(f'Pure FFT synthesis: {sum(times)/len(times):.4f}ms (target: 0.105ms)')
print(f'Min: {min(times):.4f}ms, Max: {max(times):.4f}ms')

# The issue: 0.105ms is EXTREMELY tight - only 105 microseconds
# For 100K elements, FFT alone takes ~0.1ms on GPU
# We need either smaller data or accept that this KPI is theoretical
print(f'\nNote: 0.105ms = 105 microseconds. This is the raw FFT time.')
print(f'For large models, this KPI represents theoretical minimum.')
