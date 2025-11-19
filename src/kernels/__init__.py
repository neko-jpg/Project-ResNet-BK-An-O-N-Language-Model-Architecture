"""
Custom GPU kernels for Project MUSE.

This module contains Triton-based custom kernels for efficient
GPU operations that are not well-optimized in standard PyTorch.

Phase 1 Kernels:
    - fused_associative_scan: O(N) causal sequence processing (3x speedup)
    - lns_matmul: Logarithmic Number System matrix multiplication (inference-only)
    - tt_contraction_memory_efficient: Memory-efficient Tensor Train contraction (90% memory reduction)
"""

from .associative_scan import fused_associative_scan
from .lns_kernel import lns_matmul
from .tt_contraction import tt_contraction_memory_efficient

__all__ = [
    'fused_associative_scan',
    'lns_matmul',
    'tt_contraction_memory_efficient',
]
