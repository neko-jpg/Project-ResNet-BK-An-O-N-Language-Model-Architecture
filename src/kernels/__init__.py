"""
Custom GPU kernels for Project MUSE.

This module contains Triton-based custom kernels for efficient
GPU operations that are not well-optimized in standard PyTorch.
"""

from .associative_scan import fused_associative_scan
from .lns_kernel import lns_matmul

__all__ = ['fused_associative_scan', 'lns_matmul']
