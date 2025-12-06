#!/usr/bin/env python3
"""
Moonshot Optimizations Benchmark

Tests the new Phase 8 moonshot optimizations:
- #7 Scattering-Aware Attention Pruning
- Comparison with standard attention
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def bench(fn, warmup=3, iters=20, name=""):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name}: {ms:.3f} ms")
    return ms


print("\n=== Previous Phase 8 Optimizations ===")

# Run the existing benchmark
B, L, D = 4, 512, 256
x = torch.randn(B, L, D, device=device)
a = torch.randn(B, L, D, device=device)
b = torch.randn(B, L, D, device=device)
scale = torch.randn(B, L, D, device=device)

def mobius_add_pt(x, a, c=1.0):
    x2 = (x*x).sum(-1, keepdim=True)
    a2 = (a*a).sum(-1, keepdim=True)
    xa = (x*a).sum(-1, keepdim=True)
    num = (1+2*c*xa+c*a2)*x + (1-c*x2)*a
    denom = 1+2*c*xa+c*c*x2*a2
    return num/(denom+1e-7)

def chain_pt():
    y = mobius_add_pt(x, a)
    y = mobius_add_pt(y, b)
    return y * scale

t_pt = bench(chain_pt, name="Möbius PyTorch")

try:
    from src.kernels.hyperbolic_mobius_chain import mobius_chain_fused
    def chain_tr():
        return mobius_chain_fused(x, a, b, scale, 1.0)
    t_tr = bench(chain_tr, name="Möbius Triton")
    print(f"  Speedup: {t_pt/t_tr:.2f}x")
except Exception as e:
    print(f"  Triton: {e}")


print("\n=== NEW: Scattering-Aware Attention Pruning (#7) ===")

try:
    from src.kernels.scattering_attention_pruning import ScatteringAwareAttention
    import torch.nn.functional as F
    
    # Create attention modules
    d_model, num_heads = 256, 8
    std_attn = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True).to(device)
    scatter_attn = ScatteringAwareAttention(d_model, num_heads, threshold=0.1).to(device)
    
    # Test input
    B, L = 4, 256
    x = torch.randn(B, L, d_model, device=device)
    
    # Create G_ii with varying scattering energy (some blocks low, some high)
    G_ii = torch.randn(B, L, device=device)
    G_ii[:, :64] *= 0.01  # Low scattering - should be skipped
    G_ii[:, 64:] *= 1.0   # High scattering - should be computed
    
    # Benchmark standard attention
    def run_std():
        return std_attn(x, x, x)[0]
    t_std = bench(run_std, name="Standard Attention")
    
    # Benchmark scattering-aware attention
    scatter_attn.reset_stats()
    def run_scatter():
        return scatter_attn(x, G_ii)[0]
    t_scatter = bench(run_scatter, name="Scattering Attention")
    
    print(f"  Speedup: {t_std/t_scatter:.2f}x")
    print(f"  Blocks skipped: {scatter_attn.blocks_skipped}")
    print(f"  Blocks computed: {scatter_attn.blocks_computed}")
    print(f"  Skip ratio: {scatter_attn.blocks_skipped / max(1, scatter_attn.blocks_skipped + scatter_attn.blocks_computed):.1%}")

except Exception as e:
    import traceback
    print(f"  Error: {e}")
    traceback.print_exc()


print("\n=== SSM Scan (Phase 8) ===")
d_state = 64
A = torch.randn(d_state, d_state, device=device) * 0.1
B_proj = torch.randn(D, d_state, device=device) * 0.1
C_proj = torch.randn(d_state, D, device=device) * 0.1
x_ssm = torch.randn(4, 512, 256, device=device)

def ssm_seq():
    u = x_ssm @ B_proj
    h = torch.zeros(4, d_state, device=device)
    out = []
    for t in range(512):
        h = h @ A.T + u[:, t]
        out.append(h @ C_proj)
    return torch.stack(out, 1)

t_seq = bench(ssm_seq, name="Sequential SSM")

try:
    from src.kernels.low_rank_ssm_scan import LowRankSSMScan
    ssm_par = LowRankSSMScan(d_model=256, d_state=d_state, rank=16).to(device)
    def ssm_parallel():
        return ssm_par(x_ssm)
    t_par = bench(ssm_parallel, name="Parallel SSM Scan")
    print(f"  Speedup: {t_seq/t_par:.2f}x")
except Exception as e:
    print(f"  Parallel: {e}")


print("\n=== Summary ===")
print("Phase 8 + Moonshot Optimizations implemented:")
print("  ✅ #7 Scattering-Aware Attention Pruning")
print("  ✅ #6 Resonance-Locked Training (in train_phase8.py)")
print("  ✅ #10 Time-Reversed Training (in train_phase8.py)")
print("\nRun 'python scripts/train_phase8.py --dry-run --compile' to test full pipeline")
