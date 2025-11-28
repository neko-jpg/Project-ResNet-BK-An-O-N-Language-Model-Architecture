from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class Phase8Config:
    """Configuration for Phase 8: Hyperbolic Transcendence."""

    # Enable/Disable Flags for Major Components
    enable_entailment_cones: bool = True
    enable_topological_norm: bool = True
    enable_adaptive_computation: bool = True
    enable_koopman_bridge: bool = True
    enable_sparse_attention: bool = True
    enable_kv_compression: bool = True
    enable_numerical_guards: bool = True
    enable_curvature_adaptation: bool = True

    # Entailment Cone Settings
    entailment_aperture: float = 0.5  # Initial aperture angle
    entailment_margin: float = 0.1    # Margin for order loss

    # Topology Settings
    topology_persistence_threshold: float = 0.1
    topology_max_dimension: int = 1

    # Adaptive Computation Settings
    adaptive_exit_threshold: float = 0.8  # Exit if confidence > threshold
    adaptive_min_layers: int = 2          # Minimum layers to execute

    # Sparse Attention Settings
    sparse_top_k: int = 32
    sparse_block_size: int = 64

    # KV Compression Settings
    kv_cache_dim: int = 4
    kv_eviction_threshold: float = 2.0   # Evict if distance > threshold

    # Curvature Settings
    curvature_initial: float = 1.0
    curvature_min: float = 0.1
    curvature_max: float = 5.0

    # Numerical Safety
    max_norm: float = 0.99  # Clamp norms to this value
    grad_clip: float = 1.0  # Hyperbolic gradient clipping

@dataclass
class Phase8Diagnostics:
    """Diagnostic metrics for Phase 8 components."""

    # Entailment Metrics
    entailment_violation_rate: float = 0.0
    avg_aperture: float = 0.0

    # Topology Metrics
    betti_numbers: List[int] = field(default_factory=list)
    persistent_entropy: float = 0.0

    # Adaptive Computation Metrics
    avg_layers_executed: float = 0.0
    early_exit_rate: float = 0.0

    # Hyperbolic Metrics
    avg_hyperbolic_norm: float = 0.0
    curvature_value: float = 0.0
    boundary_collapse_warnings: int = 0
