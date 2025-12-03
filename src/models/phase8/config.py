from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Import Phase7Config from the correct module
from src.models.phase7.integrated_model import Phase7Config

@dataclass
class Phase8Config(Phase7Config):
    """
    Configuration for Phase 8: Hyperbolic Transcendence.
    
    Inherits from Phase7Config to maintain compatibility with ResNetBK architecture.
    Phase 8 extends Phase 7 with:
    - BK-Core Hyperbolic Integration (using G_ii from ResNetBK)
    - AR-SSM Hyperbolic Fusion
    - Entailment Cones for logical reasoning
    - Persistent Homology for topological analysis
    - Sheaf Attention for multi-head consistency
    - Advanced optimization techniques
    
    All Phase 7 parameters (HTT embedding, hybrid attention, etc.) are inherited.
    """

    # ========== Phase 8 Component Enable/Disable Flags ==========
    # BK-Core Integration (Core Feature)
    use_bk_hyperbolic: bool = True  # Use BK-Core G_ii for hyperbolic gating
    
    # AR-SSM Integration (Core Feature)
    use_ar_ssm_fusion: bool = True  # Fuse AR-SSM with hyperbolic space
    
    # Optional Advanced Features
    enable_entailment_cones: bool = False  # Logical entailment checking
    enable_persistent_homology: bool = False  # Topological analysis
    enable_sheaf_attention: bool = False  # Multi-head consistency
    enable_adaptive_computation: bool = False  # Dynamic layer depth
    enable_koopman_bridge: bool = False  # Koopman-hyperbolic bridge
    enable_sparse_attention: bool = False  # Sparse attention patterns
    enable_kv_compression: bool = False  # KV cache compression
    enable_numerical_guards: bool = True  # Numerical safety (always recommended)
    enable_curvature_adaptation: bool = False  # Dynamic curvature adjustment

    # ========== BK-Core Hyperbolic Integration Settings ==========
    bk_hyperbolic_gate_scale: float = 1.0  # Scaling factor for G_ii gating
    bk_hyperbolic_resonance_threshold: float = 0.5  # Resonance detection threshold
    
    # ========== AR-SSM Hyperbolic Fusion Settings ==========
    ar_ssm_max_rank: int = 32  # Maximum AR-SSM rank
    ar_ssm_min_rank: int = 4   # Minimum AR-SSM rank
    ar_ssm_hyperbolic_rank_threshold: float = 0.7  # Rank-based gating threshold
    ar_ssm_curvature_adaptation_rate: float = 0.1  # Curvature adjustment rate

    # ========== Entailment Cone Settings ==========
    entailment_aperture: float = 0.5  # Initial aperture angle
    entailment_margin: float = 0.1    # Margin for order loss

    # ========== Topology Settings ==========
    topology_persistence_threshold: float = 0.1
    topology_max_dimension: int = 1
    topology_betti_threshold: float = 0.5  # β₁ threshold for circular reasoning

    # ========== Sheaf Attention Settings ==========
    sheaf_agreement_threshold: float = 0.7  # Agreement threshold for consensus
    sheaf_num_sections: int = 4  # Number of sheaf sections

    # ========== Adaptive Computation Settings ==========
    adaptive_exit_threshold: float = 0.8  # Exit if confidence > threshold
    adaptive_min_layers: int = 2          # Minimum layers to execute

    # ========== Sparse Attention Settings ==========
    sparse_top_k: int = 32
    sparse_block_size: int = 64

    # ========== KV Compression Settings ==========
    kv_cache_dim: int = 4
    kv_eviction_threshold: float = 2.0   # Evict if distance > threshold

    # ========== Curvature Settings ==========
    curvature_initial: float = 1.0
    curvature_min: float = 0.1
    curvature_max: float = 5.0

    # ========== Numerical Safety ==========
    max_norm: float = 0.99  # Clamp norms to this value
    grad_clip: float = 1.0  # Hyperbolic gradient clipping
    
    # ========== Phase 8 Optimization Settings ==========
    # Override Phase 7 defaults for Phase 8
    use_gradient_checkpointing: bool = True  # Always use for memory efficiency
    use_mixed_precision: bool = True  # Always use for speed
    use_triton_kernel: bool = True  # Use optimized Triton kernels
    triton_kernel_version: str = 'fast'  # Use fast kernel by default
    
    # Low-Rank Compression Settings
    low_rank_embedding: bool = True  # Use low-rank compression for embeddings
    low_rank_ffn: bool = True  # Use low-rank compression for FFN layers
    
    # ========== Speed Optimizations (NEW) ==========
    # torch.compile settings
    use_torch_compile: bool = False  # Enable torch.compile (experimental)
    compile_mode: str = "default"  # Options: "default", "reduce-overhead", "max-autotune"
    compile_fullgraph: bool = False  # Try to compile entire model as one graph
    
    # Flash Attention 2
    use_flash_attention_2: bool = False  # Use Flash Attention 2 if available
    
    # Data loading optimizations
    dataloader_num_workers: int = 8  # Number of dataloader workers
    dataloader_pin_memory: bool = True  # Pin memory for faster H2D transfer
    dataloader_prefetch_factor: int = 4  # Prefetch factor
    dataloader_persistent_workers: bool = True  # Keep workers alive
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients

@dataclass
class Phase8Diagnostics:
    """
    Diagnostic metrics for Phase 8 components.
    
    Extends Phase 7 diagnostics with Phase 8-specific metrics.
    """

    # ========== BK-Core Hyperbolic Integration Metrics ==========
    bk_hyperbolic_gate_mean: float = 0.0  # Average G_ii gating value
    bk_hyperbolic_gate_std: float = 0.0   # Std of G_ii gating
    bk_resonance_detected: bool = False   # Whether resonance was detected
    bk_resonance_strength: float = 0.0    # Strength of resonance

    # ========== AR-SSM Hyperbolic Fusion Metrics ==========
    ar_ssm_rank_mean: float = 0.0  # Average AR-SSM rank
    ar_ssm_hyperbolic_distance_mean: float = 0.0  # Average hyperbolic distance
    ar_ssm_curvature_adjusted: bool = False  # Whether curvature was adjusted

    # ========== Entailment Metrics ==========
    entailment_violation_rate: float = 0.0
    avg_aperture: float = 0.0
    entailment_checks_performed: int = 0

    # ========== Topology Metrics ==========
    betti_numbers: List[int] = field(default_factory=list)
    persistent_entropy: float = 0.0
    circular_reasoning_detected: bool = False
    topology_curvature_adjustment_suggested: bool = False
    topology_curvature_suggestion: Optional[float] = None

    # ========== Sheaf Attention Metrics ==========
    sheaf_agreement_mean: float = 0.0  # Average agreement score
    sheaf_consensus_rate: float = 0.0  # Rate of consensus achievement
    sheaf_cohomology_obstruction: bool = False  # Global obstruction detected

    # ========== Adaptive Computation Metrics ==========
    avg_layers_executed: float = 0.0
    early_exit_rate: float = 0.0
    adaptive_compute_savings: float = 0.0  # Percentage of compute saved

    # ========== Hyperbolic Metrics ==========
    avg_hyperbolic_norm: float = 0.0
    curvature_value: float = 0.0
    boundary_collapse_warnings: int = 0
    hyperbolic_distance_mean: float = 0.0
    hyperbolic_distance_std: float = 0.0

    # ========== Memory and Performance Metrics ==========
    peak_memory_mb: float = 0.0
    forward_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    
    # ========== Numerical Safety Metrics ==========
    gradient_overflow_count: int = 0
    nan_detected: bool = False
    inf_detected: bool = False
    precision_upcast_count: int = 0
