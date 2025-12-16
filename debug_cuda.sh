#!/bin/bash
# Check CUDA compilation errors

cd /mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture
source ./venv_ubuntu/bin/activate

echo "=== Checking CUDA Compiler ==="
which nvcc
nvcc --version

echo ""
echo "=== Testing Simple CUDA Compile ==="
python -c "
import torch
from torch.utils.cpp_extension import load
import tempfile
import os

# Simple test kernel
test_code = '''
#include <torch/extension.h>
void test_fn() {}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(\"test_fn\", &test_fn, \"Test function\");
}
'''

with tempfile.TemporaryDirectory() as tmpdir:
    src_path = os.path.join(tmpdir, 'test.cpp')
    with open(src_path, 'w') as f:
        f.write(test_code)
    try:
        ext = load(
            name='test_ext',
            sources=[src_path],
            verbose=True,
        )
        print('✅ Simple C++ extension compile works!')
    except Exception as e:
        print(f'❌ C++ compile failed: {e}')
"
