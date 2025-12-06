#!/usr/bin/env python3
"""Quick kernel test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test kernels
print('\nTesting Mobius...')
try:
    from src.kernels.hyperbolic_mobius_chain import mobius_add_fused
    x = torch.randn(2, 4, 8, device=device)
    y = mobius_add_fused(x, x, 1.0)
    print('  Mobius OK:', y.shape)
except Exception as e:
    print('  Mobius Error:', e)

print('\nTesting SSM Scan...')
try:
    from src.kernels.low_rank_ssm_scan import LowRankSSMScan
    ssm = LowRankSSMScan(256, 64, 16).to(device)
    x = torch.randn(2, 16, 256, device=device)
    y = ssm(x)
    print('  SSM OK:', y.shape)
except Exception as e:
    print('  SSM Error:', e)

print('\nTesting Hyperbolic Distance...')
try:
    from src.kernels.hyperbolic_distance_batch import BatchedHyperbolicDistance
    dist = BatchedHyperbolicDistance(1.0)
    x = torch.randn(2, 16, 256, device=device) * 0.5
    y = dist(x)
    print('  Distance OK:', y.shape)
except Exception as e:
    print('  Distance Error:', e)

print('\nTesting Green Function Cache...')
try:
    from src.kernels.green_function_cache import GreenFunctionCache
    cache = GreenFunctionCache(cache_size=64)
    x = torch.randn(2, 16, 256, device=device)
    def compute(x):
        return x.norm(dim=-1)
    y = cache.get_or_compute(x, compute)
    print('  Cache OK:', y.shape, 'Stats:', cache.get_stats())
except Exception as e:
    print('  Cache Error:', e)

print('\nAll tests done!')
