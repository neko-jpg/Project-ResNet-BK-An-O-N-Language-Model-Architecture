"""
Setup script for GradientFeeder CUDA extension

Build:
    python setup_gradient_feeder.py install

Or for development:
    python setup_gradient_feeder.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gradient_feeder_cuda',
    ext_modules=[
        CUDAExtension(
            'gradient_feeder_cuda',
            [
                'src/kernels/gradient_feeder_cuda.cu',  # Relative path
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math'],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
