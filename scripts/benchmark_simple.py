#!/usr/bin/env python3
"""
Simple Phase 8 Kernel Benchmark - Standalone tests
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


print("\n=== 1. Möbius Operations ===")
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

t_pt = bench(chain_pt, name="PyTorch")

try:
    from src.kernels.hyperbolic_mobius_chain import mobius_chain_fused
    def chain_tr():
        return mobius_chain_fused(x, a, b, scale, 1.0)
    t_tr = bench(chain_tr, name="Triton Fused")
    print(f"  Speedup: {t_pt/t_tr:.2f}x")
except Exception as e:
    print(f"  Triton: {e}")


print("\n=== 2. SSM Scan ===")
d_state = 64
A = torch.randn(d_state, d_state, device=device) * 0.1
B_proj = torch.randn(D, d_state, device=device) * 0.1
C_proj = torch.randn(d_state, D, device=device) * 0.1

def ssm_seq():
    u = x @ B_proj
    h = torch.zeros(B, d_state, device=device)
    out = []
    for t in range(L):
        h = h @ A.T + u[:, t]
        out.append(h @ C_proj)
    return torch.stack(out, 1)

t_seq = bench(ssm_seq, name="Sequential")

try:
    from src.kernels.low_rank_ssm_scan import LowRankSSMScan
    ssm_par = LowRankSSMScan(d_model=D, d_state=d_state, rank=16).to(device)
    def ssm_parallel():
        return ssm_par(x)
    t_par = bench(ssm_parallel, name="Parallel Scan")
    print(f"  Speedup: {t_seq/t_par:.2f}x")
except Exception as e:
    print(f"  Parallel: {e}")


print("\n=== 3. Scattering Gate ===")
H = 8
G_ii = torch.randn(B, L, device=device) + 1j*torch.randn(B, L, device=device)
attn = torch.softmax(torch.randn(B, H, L, L, device=device), -1)

def scatter_naive():
    e = G_ii.abs()
    g = torch.softmax(e, -1).unsqueeze(1).unsqueeze(-1)
    gated = attn * g
    return gated / (gated.sum(-1, keepdim=True)+1e-7)

t_naive = bench(scatter_naive, name="Naive")

try:
    from src.kernels.scattering_gate_fused import FusedScatteringGate
    sg = FusedScatteringGate(d_model=D).to(device)
    def scatter_fused():
        return sg(G_ii, attn)
    t_fused = bench(scatter_fused, name="Fused")
    print(f"  Speedup: {t_naive/t_fused:.2f}x")
except Exception as e:
    print(f"  Fused: {e}")


print("\n=== 4. Hyperbolic Distance ===")
x_norm = x / (x.norm(dim=-1, keepdim=True) + 0.1)

def dist_naive():
    n = x_norm.norm(dim=-1).clamp(max=0.99)
    return 2 * torch.atanh(n)

t_naive = bench(dist_naive, name="Naive")

try:
    from src.kernels.hyperbolic_distance_batch import BatchedHyperbolicDistance
    hd = BatchedHyperbolicDistance(curvature=1.0).to(device)
    def dist_batch():
        return hd(x_norm)
    t_batch = bench(dist_batch, name="Batched")
    print(f"  Speedup: {t_naive/t_batch:.2f}x")
except Exception as e:
    print(f"  Batched: {e}")


print("\n=== Done ===")
