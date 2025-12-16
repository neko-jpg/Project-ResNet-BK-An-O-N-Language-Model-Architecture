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
