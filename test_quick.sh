#!/bin/bash
# Quick test to see import timing details
cd /mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture
source venv_ubuntu/bin/activate
export PYTHONPATH=.

echo "=== Quick Import Test ==="
python -c "
import time
print('Step 1: Import torch...')
t0 = time.time()
import torch
print(f'  torch: {(time.time()-t0)*1000:.0f}ms')

print()
print('Step 2: Import gradient_feeder (check C++ ext first)...')
t0 = time.time()

import sys, os
_kernels_path = 'src/kernels'
if _kernels_path not in sys.path:
    sys.path.insert(0, _kernels_path)

try:
    import gradient_feeder_cpp
    print(f'  ✅ C++ extension loaded: {(time.time()-t0)*1000:.0f}ms')
    print(f'  Functions: {[x for x in dir(gradient_feeder_cpp) if not x.startswith(\"_\")]}')
except ImportError as e:
    print(f'  ❌ C++ extension failed: {e}')

print()
print('Step 3: Full GradientFeederV2 import...')
t0 = time.time()
from src.training.gradient_feeder import GradientFeederV2, _CPP_FEEDER_AVAILABLE
print(f'  Import: {(time.time()-t0)*1000:.0f}ms')
print(f'  C++ extension used: {_CPP_FEEDER_AVAILABLE}')
"
