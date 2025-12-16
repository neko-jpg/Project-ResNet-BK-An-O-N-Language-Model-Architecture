#!/bin/bash
# Build and test GradientFeeder CUDA extension

cd /mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture
source ./venv_ubuntu/bin/activate

echo "=== Environment Check ==="
echo "Python: $(which python)"
echo "CUDA: $(nvcc --version 2>/dev/null | head -1 || echo 'nvcc not found')"
echo "Ninja: $(which ninja)"
echo ""

echo "=== Testing JIT Compilation ==="
python -c "
import os
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_extensions'

from src.kernels.gradient_feeder_jit import CUDA_AVAILABLE
print(f'CUDA JIT compiled: {CUDA_AVAILABLE}')

if CUDA_AVAILABLE:
    print('✅ SUCCESS: CUDA extension is ready!')
else:
    print('❌ FAILED: Check errors above')
"
