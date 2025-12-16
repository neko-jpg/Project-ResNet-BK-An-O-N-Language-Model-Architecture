#!/bin/bash
# Precise timing to show gradient_feeder C++ vs JIT performance
cd /mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture
source venv_ubuntu/bin/activate
export PYTHONPATH=.

echo "=== Gradient Feeder C++ Extension Performance Test ==="
echo ""
python -c "
import time

# First import torch (this always takes a while)
t0 = time.time()
import torch
torch_time = time.time() - t0
print(f'1. PyTorch import: {torch_time*1000:.0f}ms')

# Now import ONLY the gradient_feeder_cpp extension directly
import sys
sys.path.insert(0, 'src/kernels')

t0 = time.time()
import gradient_feeder_cpp
cpp_time = time.time() - t0
print(f'2. C++ extension import (gradient_feeder_cpp): {cpp_time*1000:.0f}ms')

# Test the C++ extension
state = gradient_feeder_cpp.FeederState()
threshold, scale, action = gradient_feeder_cpp.feed(state, 0.5)
print(f'   - feed() works: threshold={threshold:.1f}, scale={scale:.2f}')

print()
print('=== Result ===')
if cpp_time < 0.5:  # Less than 500ms
    print('✅ SUCCESS: C++ extension loads in <500ms (no JIT delay)')
    print(f'   Before fix: ~2-3 seconds (JIT compilation)')
    print(f'   After fix:  {cpp_time*1000:.0f}ms (pre-built C++ extension)')
else:
    print(f'⚠️ C++ extension took {cpp_time*1000:.0f}ms (expected <500ms)')
"
