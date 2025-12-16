#!/bin/bash
cd /mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture
./venv_ubuntu/bin/python -c "from src.kernels.gradient_feeder_jit import CUDA_AVAILABLE; print(f'CUDA JIT: {CUDA_AVAILABLE}')"
