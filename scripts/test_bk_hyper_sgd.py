#!/usr/bin/env python3
"""Minimal test to verify BK-HyperSGD gradient flow works."""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from src.optimizers.bk_hyper_sgd import BKHyperSGD

print("="*50)
print("BK-HyperSGD Minimal Gradient Flow Test")
print("="*50)

# Simple test model
model = nn.Sequential(
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
model = model.to(device)

# Name params for optimizer
for name, param in model.named_parameters():
    param._param_name = name

optimizer = BKHyperSGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer.set_param_names(model)

# Forward/backward
x = torch.randn(8, 64).to(device)
y = model(x)
loss = y.mean()
print(f"Initial loss: {loss.item():.4f}")

loss.backward()

# Check gradients
print("\nGradients before step:")
total_grad_norm = 0.0
for name, p in model.named_parameters():
    if p.grad is not None:
        grad_norm = p.grad.norm().item()
        total_grad_norm += grad_norm ** 2
        print(f"  {name}: {grad_norm:.6f}")
total_grad_norm = total_grad_norm ** 0.5
print(f"Total grad norm: {total_grad_norm:.6f}")

# Store initial weights
w0 = model[0].weight.data.clone()

# Optimizer step
print("\nCalling optimizer.step()...")
optimizer.step()

# Check weight change
w1 = model[0].weight.data
diff = (w1 - w0).abs().max().item()
print(f"Weight change after step: {diff:.6f}")

if diff > 0:
    print("\n✅ SUCCESS! BK-HyperSGD is working correctly.")
else:
    print("\n❌ FAILED - no weight change detected")
    
# Try multiple steps
print("\n--- Running 10 more steps ---")
for i in range(10):
    optimizer.zero_grad()
    x = torch.randn(8, 64).to(device)
    y = model(x)
    loss = y.mean()
    loss.backward()
    optimizer.step()
    if i % 3 == 0:
        print(f"Step {i+1}: loss={loss.item():.4f}")

print("\nTest complete!")
