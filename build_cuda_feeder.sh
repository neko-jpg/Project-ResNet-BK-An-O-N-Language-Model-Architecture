#!/bin/bash
# Build script for C++ gradient_feeder extension (no CUDA compiler required)
# This avoids CUDA version mismatch issues while still providing C++ speedup
set -e

cd /mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture
source venv_ubuntu/bin/activate

echo "=== Environment Check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"

echo ""
echo "=== Building gradient_feeder_cpp extension (C++ only, no CUDA compiler needed) ==="

# Create setup.py for the C++ extension
cat > src/kernels/setup_gradient_feeder.py << 'EOF'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

setup(
    name='gradient_feeder_cpp',
    ext_modules=[
        CppExtension(
            'gradient_feeder_cpp',
            sources=['gradient_feeder_cpp.cpp'],
            extra_compile_args=['-O3', '-ffast-math']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
EOF

cd src/kernels
python setup_gradient_feeder.py build_ext --inplace

echo ""
echo "=== Verifying Build ==="
python -c "import gradient_feeder_cpp; print('SUCCESS: gradient_feeder_cpp extension loaded!')"

echo ""
echo "=== Build Complete ==="
