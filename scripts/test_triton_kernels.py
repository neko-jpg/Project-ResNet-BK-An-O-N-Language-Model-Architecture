#!/usr/bin/env python3
"""Test Triton kernels"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

device = torch.device('cuda')
print(f"Testing on: {torch.cuda.get_device_name(0)}")

# Test 1: Hyperbolic Distance
print("\n1. Hyperbolic Distance Triton Kernel")
try:
    from src.kernels.hyperbolic_distance_batch import BatchedHyperbolicDistance
    dist = BatchedHyperbolicDistance(curvature=1.0, use_triton=True)
    x = torch.randn(2, 16, 64, device=device) * 0.4
    result = dist(x)
    print(f"   OK: input {x.shape} -> output {result.shape}")
    print(f"   Values: min={result.min():.4f}, max={result.max():.4f}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Mobius Addition
print("\n2. Möbius Addition Triton Kernel")
try:
    from src.kernels.hyperbolic_mobius_chain import mobius_add_fused
    x = torch.randn(2, 16, 64, device=device) * 0.3
    a = torch.randn(2, 16, 64, device=device) * 0.3
    result = mobius_add_fused(x, a, 1.0)
    print(f"   OK: input {x.shape} -> output {result.shape}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Scattering Gate
print("\n3. Scattering Gate Fused Kernel")
try:
    from src.kernels.scattering_gate_fused import FusedScatteringGate
    gate = FusedScatteringGate(d_model=64).to(device)
    G_ii = torch.complex(
        torch.randn(2, 16, device=device),
        torch.randn(2, 16, device=device) * 0.1
    )
    attn = torch.softmax(torch.randn(2, 4, 16, 16, device=device), dim=-1)
    result, diag = gate(G_ii, attn)
    print(f"   OK: attn {attn.shape} -> gated {result.shape}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
