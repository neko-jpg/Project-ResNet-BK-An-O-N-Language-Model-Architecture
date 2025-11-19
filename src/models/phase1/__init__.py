"""
Phase 1 Efficiency Engine

Phase 1は、Project MUSEの物理的制約（VRAM 8-10GB、計算量O(N)）を完全に満たしながら、
100B級モデルを家庭用GPUで動作可能にする3つの革新的アルゴリズムを実装します。

Core Components:
    1. AR-SSM (Adaptive Rank Semiseparable Matrix): 
       入力の複雑性に応じてランクを動的に調整する半可分行列層
    
    2. HTT (Holographic Tensor Train): 
       Tensor Train分解に位相回転を組み込んだEmbedding圧縮技術
    
    3. LNS (Logarithmic Number System): 
       乗算を加算に変換することで計算コストを削減する数値表現システム

Requirements:
    - 4.1: 既存インフラとの統合
    - 4.2: テストファイルの作成
    - 4.3: AGENTS.md準拠
    - 12.3: ハイパーパラメータドキュメント

Usage:
    >>> from src.models.phase1 import Phase1Config, Phase1Diagnostics, Phase1TrainingState
    >>> 
    >>> # Create configuration for 8GB VRAM
    >>> config = Phase1Config.for_hardware(vram_gb=8.0)
    >>> config.validate()
    >>> 
    >>> # Initialize diagnostics
    >>> diagnostics = Phase1Diagnostics()
    >>> 
    >>> # Initialize training state
    >>> training_state = Phase1TrainingState(rank_warmup_steps=1000)
    >>> training_state.update_rank_schedule(config)

Author: Project MUSE Team
License: See LICENSE file
"""

from .config import (
    Phase1Config,
    Phase1Diagnostics,
    Phase1TrainingState,
)

from .errors import (
    VRAMExhaustedError,
    NumericalInstabilityError,
    InvalidConfigError,
    HardwareCompatibilityError,
    raise_vram_exhausted,
    raise_numerical_instability,
    check_cuda_available,
)

from .recovery import (
    Phase1ErrorRecovery,
    RecoveryAction,
    create_recovery_context_manager,
)

from .ar_ssm_layer import AdaptiveRankSemiseparableLayer

from .htt_embedding import (
    HolographicTTEmbedding,
    create_htt_embedding,
    replace_embedding_with_htt,
    verify_compression_ratio,
    verify_gradient_flow,
    calculate_htt_memory_savings,
)

from .lns_linear import (
    LNSLinear,
    convert_linear_to_lns,
)

from .stability_monitor import (
    BKStabilityMonitor,
    StabilityThresholds,
    StabilityMetrics,
)

from .gradient_monitor import (
    GradientMonitor,
    GradientStatistics,
    create_gradient_monitor_from_config,
    check_gradient_health,
)

from .factory import (
    Phase1IntegratedModel,
    create_phase1_model,
    add_ar_ssm_to_model,
    convert_model_to_phase1,
)

from .presets import (
    get_preset_8gb,
    get_preset_10gb,
    get_preset_24gb,
    get_preset_inference,
    get_preset_maximum_quality,
    get_preset_maximum_efficiency,
    get_preset,
    list_presets,
    print_preset_comparison,
    PRESET_REGISTRY,
)

from .conversion import (
    initialize_htt_from_embedding,
    initialize_ar_ssm_from_semiseparable,
    convert_embedding_to_htt,
    convert_all_embeddings_to_htt,
    add_ar_ssm_layers,
    verify_conversion,
    get_conversion_summary,
    print_conversion_summary,
)

from .complex_utils import (
    is_complex_tensor,
    real_to_complex,
    complex_to_real,
    ensure_complex,
    ensure_real,
    complex_phase_rotation,
    check_dtype_compatibility,
    safe_complex_operation,
    ComplexLinear,
    document_complex_support,
    get_complex_conversion_guide,
)

from .memory_optimizer import (
    LowRankFFN,
    MemoryEfficientTransformerBlock,
    MemoryOptimizedModel,
    create_memory_optimized_model,
    replace_model_with_memory_optimized,
)

from .ultra_optimizer import (
    UltraLowRankFFN,
    UltraMemoryEfficientBlock,
    UltraMemoryOptimizedModel,
    create_ultra_memory_optimized_model,
)

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "Phase1Config",
    "Phase1Diagnostics",
    "Phase1TrainingState",
    
    # Error Handling
    "VRAMExhaustedError",
    "NumericalInstabilityError",
    "InvalidConfigError",
    "HardwareCompatibilityError",
    "raise_vram_exhausted",
    "raise_numerical_instability",
    "check_cuda_available",
    
    # Error Recovery
    "Phase1ErrorRecovery",
    "RecoveryAction",
    "create_recovery_context_manager",
    
    # AR-SSM Layer (Task 3)
    "AdaptiveRankSemiseparableLayer",
    
    # HTT Embedding (Task 4)
    "HolographicTTEmbedding",
    "create_htt_embedding",
    "replace_embedding_with_htt",
    "verify_compression_ratio",
    "verify_gradient_flow",
    "calculate_htt_memory_savings",
    
    # LNS Linear (Task 6)
    "LNSLinear",
    "convert_linear_to_lns",
    
    # Stability Monitor (Task 7)
    "BKStabilityMonitor",
    "StabilityThresholds",
    "StabilityMetrics",
    
    # Gradient Monitor (Task 8.4)
    "GradientMonitor",
    "GradientStatistics",
    "create_gradient_monitor_from_config",
    "check_gradient_health",
    
    # Phase 1 Model Factory (Task 9.1)
    "Phase1IntegratedModel",
    "create_phase1_model",
    "add_ar_ssm_to_model",
    "convert_model_to_phase1",
    
    # Configuration Presets (Task 9.2)
    "get_preset_8gb",
    "get_preset_10gb",
    "get_preset_24gb",
    "get_preset_inference",
    "get_preset_maximum_quality",
    "get_preset_maximum_efficiency",
    "get_preset",
    "list_presets",
    "print_preset_comparison",
    "PRESET_REGISTRY",
    
    # Model Conversion Utilities (Task 9.3)
    "initialize_htt_from_embedding",
    "initialize_ar_ssm_from_semiseparable",
    "convert_embedding_to_htt",
    "convert_all_embeddings_to_htt",
    "add_ar_ssm_layers",
    "verify_conversion",
    "get_conversion_summary",
    "print_conversion_summary",
    
    # Complex Number Support (Task 11.1 - Phase 2 Preparation)
    "is_complex_tensor",
    "real_to_complex",
    "complex_to_real",
    "ensure_complex",
    "ensure_real",
    "complex_phase_rotation",
    "check_dtype_compatibility",
    "safe_complex_operation",
    "ComplexLinear",
    "document_complex_support",
    "get_complex_conversion_guide",
    
    # Memory Optimizer (82-85% VRAM Reduction)
    "LowRankFFN",
    "MemoryEfficientTransformerBlock",
    "MemoryOptimizedModel",
    "create_memory_optimized_model",
    "replace_model_with_memory_optimized",
    
    # Ultra Memory Optimizer (84.8% VRAM Reduction)
    "UltraLowRankFFN",
    "UltraMemoryEfficientBlock",
    "UltraMemoryOptimizedModel",
    "create_ultra_memory_optimized_model",
    
    # Version
    "__version__",
]

# Fused Associative Scan (Task 5 - Completed)
# Note: Imported from src.kernels, not src.models.phase1
# Usage: from src.kernels.associative_scan import fused_associative_scan

# LNS Kernel (Task 6 - Completed)
# Note: Kernel imported from src.kernels, layer from src.models.phase1
# Usage: from src.kernels.lns_kernel import lns_matmul
#        from src.models.phase1 import LNSLinear

# Stability Monitor (Task 7 - Completed)
# Usage: from src.models.phase1 import BKStabilityMonitor, StabilityThresholds
