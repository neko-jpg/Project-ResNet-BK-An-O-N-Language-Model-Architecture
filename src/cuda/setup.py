"""
Revolutionary Training Algorithms - PyTorch Extension Package

This package provides CUDA-accelerated implementations of revolutionary
training algorithms for extreme speedup of language model training.

Install:
    pip install -e src/cuda/

Usage:
    from revolutionary_cuda import holographic_bind
    output, time_ms = holographic_bind(gradients, inputs, lr=0.01)
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

here = os.path.dirname(os.path.abspath(__file__))

setup(
    name='revolutionary_cuda',
    version='1.0.0',
    description='CUDA kernels for revolutionary training algorithms',
    author='Project MUSE Team',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='holographic_cuda',
            sources=[
                os.path.join(here, 'holographic_kernel.cu'),
                os.path.join(here, 'holographic_binding.cpp'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-use_fast_math',
                    '--expt-relaxed-constexpr',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ],
            },
            libraries=['cufft'],
        ),
        CUDAExtension(
            name='hyperbolic_cuda',
            sources=[
                os.path.join(here, 'hyperbolic_ops.cu'),
                os.path.join(here, 'hyperbolic_binding.cpp'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-use_fast_math',
                    '--expt-relaxed-constexpr',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=2.0.0',
    ],
    python_requires='>=3.8',
)
