"""
Custom GPU kernels for Project MUSE.

This module contains Triton-based custom kernels for efficient
GPU operations that are not well-optimized in standard PyTorch.

Phase 1 Kernels:
    - fused_associative_scan: O(N) causal sequence processing (3x speedup)
    - lns_matmul: Logarithmic Number System matrix multiplication (inference-only)
    - tt_contraction_memory_efficient: Memory-efficient Tensor Train contraction (90% memory reduction)

Phase 2 Kernels:
    - bk_scan_triton: Triton-accelerated BK-Core computation (3x+ speedup)
    - bk_scan_triton_forward: Forward scan for Theta recursion
    - bk_scan_triton_backward: Backward scan for Phi recursion
"""

from .associative_scan import fused_associative_scan
from .lns_kernel import lns_matmul
from .tt_contraction import tt_contraction_memory_efficient

# Phase 2: BK-Core Triton kernels
try:
    from .bk_scan import (
        bk_scan_triton,
        bk_scan_triton_forward,
        bk_scan_triton_backward,
        is_triton_available,
    )
    _BK_SCAN_AVAILABLE = True
except ImportError:
    _BK_SCAN_AVAILABLE = False
    bk_scan_triton = None
    bk_scan_triton_forward = None
    bk_scan_triton_backward = None
    is_triton_available = None

# Phase 8: Flash Hyperbolic Attention
try:
    from .flash_hyperbolic_triton import (
        flash_hyperbolic_attention,
        FlashHyperbolicAttentionModule,
        FlashHyperbolicConfig,
    )
    _FLASH_HYPERBOLIC_AVAILABLE = True
except ImportError:
    _FLASH_HYPERBOLIC_AVAILABLE = False
    flash_hyperbolic_attention = None
    FlashHyperbolicAttentionModule = None
    FlashHyperbolicConfig = None

# Phase 8: Safe Numerical Operations (Rubber Wall)
try:
    from .safe_ops_triton import (
        safe_exp,
        safe_log,
        safe_acosh,
        safe_atanh,
        safe_softmax_exp,
        safe_poincare_distance,
        safe_exp_pytorch,
        safe_log_pytorch,
        safe_acosh_pytorch,
        safe_atanh_pytorch,
        K_THRESHOLD,
    )
    _SAFE_OPS_AVAILABLE = True
except ImportError:
    _SAFE_OPS_AVAILABLE = False
    safe_exp = None
    safe_log = None
    safe_acosh = None
    safe_atanh = None
    safe_softmax_exp = None
    safe_poincare_distance = None
    safe_exp_pytorch = None
    safe_log_pytorch = None
    safe_acosh_pytorch = None
    safe_atanh_pytorch = None
    K_THRESHOLD = 88.0

__all__ = [
    'fused_associative_scan',
    'lns_matmul',
    'tt_contraction_memory_efficient',
]

if _BK_SCAN_AVAILABLE:
    __all__.extend([
        'bk_scan_triton',
        'bk_scan_triton_forward',
        'bk_scan_triton_backward',
        'is_triton_available',
    ])

if _FLASH_HYPERBOLIC_AVAILABLE:
    __all__.extend([
        'flash_hyperbolic_attention',
        'FlashHyperbolicAttentionModule',
        'FlashHyperbolicConfig',
    ])

if _SAFE_OPS_AVAILABLE:
    __all__.extend([
        'safe_exp',
        'safe_log',
        'safe_acosh',
        'safe_atanh',
        'safe_softmax_exp',
        'safe_poincare_distance',
        'safe_exp_pytorch',
        'safe_log_pytorch',
        'safe_acosh_pytorch',
        'safe_atanh_pytorch',
        'K_THRESHOLD',
    ])

# =============================================================================
# Phase 8: Custom Optimization Kernels
# =============================================================================

# Hyperbolic Möbius Chain Fusion
try:
    from .hyperbolic_mobius_chain import (
        FusedMobiusOperations,
        mobius_add_fused,
        mobius_chain_fused,
        exp_map_fused,
    )
    _MOBIUS_CHAIN_AVAILABLE = True
except ImportError:
    _MOBIUS_CHAIN_AVAILABLE = False
    FusedMobiusOperations = None
    mobius_add_fused = None
    mobius_chain_fused = None
    exp_map_fused = None

# Green Function Cache
try:
    from .green_function_cache import (
        GreenFunctionCache,
        AdaptiveGreenFunctionCache,
        CachedBKCoreWrapper,
    )
    _GREEN_CACHE_AVAILABLE = True
except ImportError:
    _GREEN_CACHE_AVAILABLE = False
    GreenFunctionCache = None
    AdaptiveGreenFunctionCache = None
    CachedBKCoreWrapper = None

# Low-Rank SSM Parallel Scan
try:
    from .low_rank_ssm_scan import (
        LowRankSSMScan,
        AdaptiveLowRankSSM,
        parallel_prefix_scan,
    )
    _SSM_SCAN_AVAILABLE = True
except ImportError:
    _SSM_SCAN_AVAILABLE = False
    LowRankSSMScan = None
    AdaptiveLowRankSSM = None
    parallel_prefix_scan = None

# Scattering Gate Fused
try:
    from .scattering_gate_fused import (
        FusedScatteringGate,
        FusedScatteringAttention,
    )
    _SCATTERING_FUSED_AVAILABLE = True
except ImportError:
    _SCATTERING_FUSED_AVAILABLE = False
    FusedScatteringGate = None
    FusedScatteringAttention = None

# Hyperbolic Distance Batch
try:
    from .hyperbolic_distance_batch import (
        BatchedHyperbolicDistance,
        HyperbolicRankGatingOptimized,
        poincare_distance_from_origin,
        poincare_distance,
    )
    _HYPER_DIST_AVAILABLE = True
except ImportError:
    _HYPER_DIST_AVAILABLE = False
    BatchedHyperbolicDistance = None
    HyperbolicRankGatingOptimized = None
    poincare_distance_from_origin = None
    poincare_distance = None

# Resonance Adaptive Curvature
try:
    from .resonance_adaptive_curvature import (
        ResonanceAdaptiveCurvature,
        StabilityMonitor,
    )
    _RESONANCE_AVAILABLE = True
except ImportError:
    _RESONANCE_AVAILABLE = False
    ResonanceAdaptiveCurvature = None
    StabilityMonitor = None

# Ternary Möbius MatMul
try:
    from .ternary_mobius_matmul import (
        TernaryMobiusLinear,
        TernaryMobiusMLP,
        ternary_matmul,
    )
    _TERNARY_MOBIUS_AVAILABLE = True
except ImportError:
    _TERNARY_MOBIUS_AVAILABLE = False
    TernaryMobiusLinear = None
    TernaryMobiusMLP = None
    ternary_matmul = None

# Add to __all__
if _MOBIUS_CHAIN_AVAILABLE:
    __all__.extend(['FusedMobiusOperations', 'mobius_add_fused', 'mobius_chain_fused', 'exp_map_fused'])

if _GREEN_CACHE_AVAILABLE:
    __all__.extend(['GreenFunctionCache', 'AdaptiveGreenFunctionCache', 'CachedBKCoreWrapper'])

if _SSM_SCAN_AVAILABLE:
    __all__.extend(['LowRankSSMScan', 'AdaptiveLowRankSSM', 'parallel_prefix_scan'])

if _SCATTERING_FUSED_AVAILABLE:
    __all__.extend(['FusedScatteringGate', 'FusedScatteringAttention'])

if _HYPER_DIST_AVAILABLE:
    __all__.extend(['BatchedHyperbolicDistance', 'HyperbolicRankGatingOptimized', 'poincare_distance_from_origin', 'poincare_distance'])

if _RESONANCE_AVAILABLE:
    __all__.extend(['ResonanceAdaptiveCurvature', 'StabilityMonitor'])

if _TERNARY_MOBIUS_AVAILABLE:
    __all__.extend(['TernaryMobiusLinear', 'TernaryMobiusMLP', 'ternary_matmul'])
