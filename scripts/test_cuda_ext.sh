#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.6
cd /mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture
source .venv_wsl/bin/activate
export PYTHONPATH=$PYTHONPATH:/mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture/src/cuda
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH
python -c "import holographic_cuda; print('CUDA extension loaded!')"
