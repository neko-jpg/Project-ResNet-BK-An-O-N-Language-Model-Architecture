"""
GradientFeeder CUDA Extension - JIT Compilation

This module uses torch.utils.cpp_extension.load() for JIT compilation,
avoiding the need for setup.py which has deprecation issues.

Usage:
    from src.kernels.gradient_feeder_jit import scale_all_gradients_cuda, feed_cpp
"""

import torch
import os

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available - cannot use gradient_feeder_cuda")

# JIT compile the CUDA extension
from torch.utils.cpp_extension import load

# Get the path to the CUDA source file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_source = os.path.join(_current_dir, 'gradient_feeder_cuda.cu')

# JIT compile with ninja (faster) if available
try:
    _cuda_ext = load(
        name='gradient_feeder_cuda',
        sources=[_cuda_source],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        extra_cflags=['-O3'],
        verbose=False,
    )
    print("✅ GradientFeeder CUDA extension JIT-compiled successfully!")
except Exception as e:
    print(f"⚠️ CUDA JIT compilation failed: {e}")
    _cuda_ext = None

# Export functions
def scale_all_gradients_cuda(grads, scale):
    """Scale all gradients using CUDA kernel."""
    if _cuda_ext is None:
        raise RuntimeError("CUDA extension not compiled")
    return _cuda_ext.scale_all_gradients(grads, scale)

def get_feeder_state():
    """Get a new FeederState object."""
    if _cuda_ext is None:
        raise RuntimeError("CUDA extension not compiled")
    return _cuda_ext.FeederState()

def feed_cpp(state, grad_norm):
    """C++ implementation of feed()."""
    if _cuda_ext is None:
        raise RuntimeError("CUDA extension not compiled")
    return _cuda_ext.feed(state, grad_norm)

# Check if extension is available
CUDA_AVAILABLE = _cuda_ext is not None
