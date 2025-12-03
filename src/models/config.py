"""
Configuration classes for the models.
"""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ResNetBKConfig:
    """
    Configuration for the ResNet-BK Language Model.
    """
    vocab_size: int = 50257 # Default to GPT-2 vocab size
    d_model: int = 64
    n_layers: int = 4
    n_seq: int = 128
    num_experts: int = 4
    top_k: int = 1
    dropout_p: float = 0.1
    use_scattering_router: bool = False
    scattering_scale: float = 0.1
    scattering_scale_warmup_steps: int = 0
    use_hyperbolic_router: bool = False
    hyperbolic_router_curvature: float = 1.0
    hyperbolic_router_boundary: float = 0.85
    hyperbolic_router_update_proto: bool = False
    hyperbolic_router_proto_decay: float = 0.9
    prime_bump_init: bool = False
    prime_bump_scale: float = 0.02
    use_birman_schwinger: bool = False
    epsilon: float = 1.0
    use_mourre: bool = True
    use_lap: bool = True
    schatten_threshold: float = 100.0
    precision_upgrade_threshold: float = 1e6
    k_max: int = 3
    use_bitnet: bool = False
    use_symplectic: bool = False
    symplectic_dt: float = 0.1
    symplectic_mode: str = 'verlet'
    use_gradient_checkpointing: bool = False
    use_hybrid_attention: bool = False
    hyperbolic_window_size: int = 64
    num_heads: int = 4
    use_fused_moe_kernel: bool = False
    use_triton_kernel: bool = True # For Hyperbolic Attention
    triton_kernel_version: str = 'fast'  # 'fast', 'v2', 'v1'
    
    # Low-Rank Compression
    low_rank_embedding: bool = False
    low_rank_ffn: bool = False
    low_rank_attention: bool = False
    low_rank_rank: int = 64

    # AR-SSM specific parameters
    ar_ssm_max_rank: int = 32
    ar_ssm_min_rank: int = 4
    
    # Mixed Precision Training
    use_mixed_precision: bool = False

    # HTT / Quantization (Phase 1, 8)
    use_htt_embedding: bool = False
    htt_rank: int = 16
    quantized_htt: bool = False
