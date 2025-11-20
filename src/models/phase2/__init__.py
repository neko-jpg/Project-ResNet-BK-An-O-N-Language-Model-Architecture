"""
Phase 2: Breath of Life - Dynamic Memory and Forgetting Mechanisms

This module implements Phase 2 of Project MUSE, introducing:
- Non-Hermitian Forgetting (散逸的忘却)
- Dissipative Hebbian Dynamics (散逸的Hebbian動力学)
- Memory Resonance and Selection (記憶共鳴と選択)
"""

from .non_hermitian import (
    NonHermitianPotential,
    DissipativeBKLayer,
)
from .gradient_safety import (
    GradientSafetyModule,
    safe_complex_backward,
    clip_grad_norm_safe,
)
from .dissipative_hebbian import (
    DissipativeHebbianLayer,
    LyapunovStabilityMonitor,
)
from .memory_selection import (
    SNRMemoryFilter,
    MemoryImportanceEstimator,
)
from .memory_resonance import (
    MemoryResonanceLayer,
    ZetaBasisTransform,
)
from .zeta_init import (
    ZetaInitializer,
    ZetaEmbedding,
    apply_zeta_initialization,
    get_zeta_statistics,
)
from .integrated_model import (
    Phase2Block,
    Phase2IntegratedModel,
)
from .factory import (
    Phase2Config,
    create_phase2_model,
    convert_phase1_to_phase2,
    get_phase2_preset,
)

__all__ = [
    "NonHermitianPotential",
    "DissipativeBKLayer",
    "GradientSafetyModule",
    "safe_complex_backward",
    "clip_grad_norm_safe",
    "DissipativeHebbianLayer",
    "LyapunovStabilityMonitor",
    "SNRMemoryFilter",
    "MemoryImportanceEstimator",
    "MemoryResonanceLayer",
    "ZetaBasisTransform",
    "ZetaInitializer",
    "ZetaEmbedding",
    "apply_zeta_initialization",
    "get_zeta_statistics",
    "Phase2Block",
    "Phase2IntegratedModel",
    "Phase2Config",
    "create_phase2_model",
    "convert_phase1_to_phase2",
    "get_phase2_preset",
]
