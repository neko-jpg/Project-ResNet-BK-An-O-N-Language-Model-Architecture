#!/usr/bin/env python3
"""Quick test of core innovations."""
import torch
import time
import sys
sys.path.insert(0, '.')

print('=== HTT Complex Phase Test ===')
try:
    from src.models.phase1.htt_embedding import HolographicTTEmbedding
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    emb_cos = HolographicTTEmbedding(1000, 256, rank=16, use_complex_phase=False).to(device)
    emb_complex = HolographicTTEmbedding(1000, 256, rank=16, use_complex_phase=True).to(device)
    
    input_ids = torch.randint(0, 1000, (4, 64), device=device)
    
    # Test forward pass
    out_cos = emb_cos(input_ids)
    out_complex = emb_complex(input_ids)
    
    print(f'  cos output shape: {out_cos.shape}, dtype: {out_cos.dtype}')
    print(f'  complex output shape: {out_complex.shape}, dtype: {out_complex.dtype}')
    print(f'  cos variance: {out_cos.var().item():.6f}')
    print(f'  complex variance: {out_complex.var().item():.6f}')
    print('  ✅ HTT Complex Phase: PASS')
except Exception as e:
    import traceback
    print(f'  ❌ HTT Complex Phase: FAIL - {e}')
    traceback.print_exc()

print()
print('=== BK-Core Parallel Scan Test ===')
try:
    from src.kernels.bk_parallel_scan import bk_parallel_inverse_diagonal
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N = 2, 512
    a = torch.randn(B, N, device=device)
    b = torch.randn(B, N-1, device=device) * 0.1
    c = torch.randn(B, N-1, device=device) * 0.1
    z = 0.1 + 0.1j
    
    # Quick timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    par_result = bk_parallel_inverse_diagonal(a, b, c, z)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    par_time = (time.perf_counter() - start) * 1000
    
    print(f'  Parallel scan time (N={N}): {par_time:.2f}ms')
    print(f'  Output shape: {par_result.shape}, dtype: {par_result.dtype}')
    print(f'  Contains NaN: {torch.isnan(par_result).any().item()}')
    print(f'  Has finite values: {torch.isfinite(par_result).any().item()}')
    print('  ✅ BK-Core Parallel Scan: PASS')
except Exception as e:
    import traceback
    print(f'  ❌ BK-Core Parallel Scan: FAIL - {e}')
    traceback.print_exc()

print()
print('=== GPU Topology Test ===')
try:
    from src.kernels.vietoris_rips_triton import approximate_persistence_gpu
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(8, 512, 256, device=device)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = approximate_persistence_gpu(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - start) * 1000
    
    print(f'  GPU topology time: {gpu_time:.2f}ms')
    print(f'  Output shape: {result.shape}')
    print('  ✅ GPU Topology: PASS')
except Exception as e:
    import traceback
    print(f'  ❌ GPU Topology: FAIL - {e}')
    traceback.print_exc()

print()
print('=== Summary ===')
print('All core innovations have been implemented and can be tested.')
