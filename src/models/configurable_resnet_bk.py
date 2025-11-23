"""
Configurable ResNet-BK Model
Supports all optimization flags for ablation studies and progressive optimization.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from .resnet_bk import LanguageModel


@dataclass
class ResNetBKConfig:
    """
    Configuration for ResNet-BK model with all optimization flags.
    
    Model Architecture:
        vocab_size: vocabulary size
        d_model: hidden dimension
        n_layers: number of ResNet-BK blocks
        n_seq: sequence length
        num_experts: number of MoE experts
        top_k: number of experts to route to (1=sparse, num_experts=dense)
        dropout_p: dropout probability
    
    Step 2 - Learning Algorithm:
        use_analytic_gradient: enable analytic gradient (vs autograd)
        grad_blend: blend factor for hybrid gradient (0.0=theoretical, 1.0=hypothesis-7)
        use_koopman: enable Koopman operator learning
        koopman_dim: Koopman space dimension
        use_physics_informed: enable physics-informed learning
    
    Step 4 - Compression:
        use_quantization: enable quantization-aware training
        quantization_bits: 8 for INT8, 4 for INT4
        use_pruning: enable structured pruning
        prune_threshold: usage threshold for expert pruning
        use_distillation: enable knowledge distillation
    
    Step 5 - Hardware:
        use_mixed_precision: enable automatic mixed precision
        use_custom_kernels: enable custom CUDA kernels
        use_gradient_checkpointing: enable gradient checkpointing
    
    Step 6 - Algorithms:
        use_adaptive_computation: enable ACT (adaptive computation time)
        use_multi_scale: enable multi-scale processing
        use_learned_sparsity: enable learned sparsity in BK-Core
    
    Step 7 - System:
        use_curriculum_learning: enable curriculum learning
        use_active_learning: enable active learning
        use_gradient_caching: enable gradient caching
    
    Phase 4 - New Physics & Optimization:
        use_bitnet: Enable 1.58-bit quantization
        use_symplectic: Enable Symplectic Integrator (Velocity Verlet)
        symplectic_dt: Time step for symplectic integration
        use_non_hermitian: Enable non-Hermitian decay (gamma learning)
        model_type: "resnet_bk" or "koopman"

    Numerical Stability:
        v_max: potential clipping range
        feature_clamp: BK feature clipping range
        grad_clip: gradient clipping threshold
    """
    
    # Model Architecture
    vocab_size: int = 30000
    d_model: int = 64
    n_layers: int = 4
    n_seq: int = 128
    num_experts: int = 4
    top_k: int = 1
    dropout_p: float = 0.1
    
    # Step 2 - Learning Algorithm
    use_analytic_gradient: bool = True
    grad_blend: float = 0.5
    use_koopman: bool = False
    koopman_dim: int = 256
    use_physics_informed: bool = False
    
    # Step 4 - Compression
    use_quantization: bool = False
    quantization_bits: int = 8
    use_pruning: bool = False
    prune_threshold: float = 0.05
    use_distillation: bool = False
    
    # Step 5 - Hardware
    use_mixed_precision: bool = False
    use_custom_kernels: bool = False
    use_gradient_checkpointing: bool = False
    
    # Step 6 - Algorithms
    use_adaptive_computation: bool = False
    use_multi_scale: bool = False
    use_learned_sparsity: bool = False
    
    # Step 7 - System
    use_curriculum_learning: bool = False
    use_active_learning: bool = False
    use_gradient_caching: bool = False
    
    # Phase 4 - New Physics
    use_bitnet: bool = False
    use_symplectic: bool = False
    symplectic_dt: float = 0.1
    use_non_hermitian: bool = False
    model_type: str = "resnet_bk" # or "koopman"

    # Numerical Stability
    v_max: float = 3.0
    feature_clamp: float = 10.0
    grad_clip: float = 0.5

    # Initialization
    prime_bump_init: bool = False
    prime_bump_scale: float = 0.02

    # Routing (Phase 2)
    use_scattering_router: bool = False
    scattering_scale: float = 0.1
    scattering_scale_warmup_steps: int = 0
    
    # Birman-Schwinger (Phase 2/3)
    use_birman_schwinger: bool = False
    epsilon: float = 1.0
    use_mourre: bool = True
    use_lap: bool = True
    schatten_threshold: float = 100.0
    precision_upgrade_threshold: float = 1e6

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.n_seq > 0, "n_seq must be positive"
        assert self.num_experts > 0, "num_experts must be positive"
        assert 0 <= self.grad_blend <= 1, "grad_blend must be in [0, 1]"
        assert self.quantization_bits in [4, 8, 16, 32], "quantization_bits must be 4, 8, 16, or 32"


# Predefined configuration presets
BASELINE_CONFIG = ResNetBKConfig(
    # Baseline: standard ResNet-BK with analytic gradient
    use_analytic_gradient=True,
    grad_blend=0.5,
    use_koopman=False,
    use_physics_informed=False,
    use_quantization=False,
    use_pruning=False,
    use_distillation=False,
    use_mixed_precision=False,
    use_custom_kernels=False,
    use_gradient_checkpointing=False,
    use_adaptive_computation=False,
    use_multi_scale=False,
    use_learned_sparsity=False,
    use_curriculum_learning=False,
    use_active_learning=False,
    use_gradient_caching=False,
)

STEP2_CONFIG = ResNetBKConfig(
    # Step 2: Optimized learning algorithm
    use_analytic_gradient=True,
    grad_blend=0.5,  # Will be tuned via grid search
    use_koopman=True,
    koopman_dim=256,
    use_physics_informed=True,
    # All other optimizations disabled
    use_quantization=False,
    use_pruning=False,
    use_distillation=False,
    use_mixed_precision=False,
    use_custom_kernels=False,
    use_gradient_checkpointing=False,
    use_adaptive_computation=False,
    use_multi_scale=False,
    use_learned_sparsity=False,
    use_curriculum_learning=False,
    use_active_learning=False,
    use_gradient_caching=False,
)

STEP4_CONFIG = ResNetBKConfig(
    # Step 4: Advanced compression
    use_analytic_gradient=True,
    grad_blend=0.5,
    use_koopman=True,
    koopman_dim=256,
    use_physics_informed=True,
    use_quantization=True,
    quantization_bits=8,
    use_pruning=True,
    prune_threshold=0.05,
    use_distillation=True,
    # Hardware and algorithmic optimizations disabled
    use_mixed_precision=False,
    use_custom_kernels=False,
    use_gradient_checkpointing=False,
    use_adaptive_computation=False,
    use_multi_scale=False,
    use_learned_sparsity=False,
    use_curriculum_learning=False,
    use_active_learning=False,
    use_gradient_caching=False,
)

STEP5_CONFIG = ResNetBKConfig(
    # Step 5: Hardware co-design
    use_analytic_gradient=True,
    grad_blend=0.5,
    use_koopman=True,
    koopman_dim=256,
    use_physics_informed=True,
    use_quantization=True,
    quantization_bits=8,
    use_pruning=True,
    prune_threshold=0.05,
    use_distillation=True,
    use_mixed_precision=True,
    use_custom_kernels=True,
    use_gradient_checkpointing=True,
    # Algorithmic and system optimizations disabled
    use_adaptive_computation=False,
    use_multi_scale=False,
    use_learned_sparsity=False,
    use_curriculum_learning=False,
    use_active_learning=False,
    use_gradient_caching=False,
)

STEP6_CONFIG = ResNetBKConfig(
    # Step 6: Algorithmic innovations
    use_analytic_gradient=True,
    grad_blend=0.5,
    use_koopman=True,
    koopman_dim=256,
    use_physics_informed=True,
    use_quantization=True,
    quantization_bits=8,
    use_pruning=True,
    prune_threshold=0.05,
    use_distillation=True,
    use_mixed_precision=True,
    use_custom_kernels=True,
    use_gradient_checkpointing=True,
    use_adaptive_computation=True,
    use_multi_scale=True,
    use_learned_sparsity=True,
    # System optimizations disabled
    use_curriculum_learning=False,
    use_active_learning=False,
    use_gradient_caching=False,
)

FULL_CONFIG = ResNetBKConfig(
    # Full: All optimizations enabled
    use_analytic_gradient=True,
    grad_blend=0.5,
    use_koopman=True,
    koopman_dim=256,
    use_physics_informed=True,
    use_quantization=True,
    quantization_bits=8,
    use_pruning=True,
    prune_threshold=0.05,
    use_distillation=True,
    use_mixed_precision=True,
    use_custom_kernels=True,
    use_gradient_checkpointing=True,
    use_adaptive_computation=True,
    use_multi_scale=True,
    use_learned_sparsity=True,
    use_curriculum_learning=True,
    use_active_learning=True,
    use_gradient_caching=True,
)

PHASE4_STRONGEST_CONFIG = ResNetBKConfig(
    # Strongest Phase 4 Model
    use_analytic_gradient=True,
    grad_blend=0.5,
    use_koopman=True,
    koopman_dim=100000,
    use_physics_informed=True,
    use_quantization=False, # BitNet handles quantization internally
    use_mixed_precision=True,
    use_custom_kernels=True,
    use_gradient_checkpointing=True,
    use_adaptive_computation=True,
    use_multi_scale=True,
    use_learned_sparsity=True,
    use_curriculum_learning=True,
    use_active_learning=True,
    use_gradient_caching=True,

    # Phase 4 Specifics
    use_bitnet=True,
    use_symplectic=True,
    symplectic_dt=0.1,
    use_non_hermitian=True,

    use_birman_schwinger=True,
    epsilon=1.0,
    use_mourre=True,
    use_lap=True,
)


class ConfigurableResNetBK(nn.Module):
    """
    Configurable ResNet-BK model supporting all optimization flags.
    
    This wrapper allows easy ablation studies and progressive optimization
    by enabling/disabling features through configuration.
    """
    
    def __init__(self, config: ResNetBKConfig):
        super().__init__()
        self.config = config
        
        # Logic to switch between Standard ResNet and Koopman Model
        if config.model_type == "koopman":
            from src.models.koopman.model import KoopmanBKModel
            from src.models.koopman.config import KoopmanConfig

            k_config = KoopmanConfig(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_seq=config.n_seq,
                num_experts=config.num_experts,
                top_k=config.top_k,
                dropout_p=config.dropout_p,
                use_scattering_router=config.use_scattering_router,
                scattering_scale=config.scattering_scale,
                scattering_scale_warmup_steps=config.scattering_scale_warmup_steps,
                use_birman_schwinger=config.use_birman_schwinger,
                epsilon=config.epsilon,
                use_mourre=config.use_mourre,
                use_lap=config.use_lap,
                schatten_threshold=config.schatten_threshold,
                precision_upgrade_threshold=config.precision_upgrade_threshold,
                use_bitnet=config.use_bitnet,
                use_symplectic=config.use_symplectic,
                symplectic_dt=config.symplectic_dt,
            )
            self.model = KoopmanBKModel(k_config)
        else:
            # Create base model (LanguageModel)
            self.model = LanguageModel(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_seq=config.n_seq,
                num_experts=config.num_experts,
                top_k=config.top_k,
                dropout_p=config.dropout_p,
                prime_bump_init=config.prime_bump_init,
                prime_bump_scale=config.prime_bump_scale,
                use_scattering_router=config.use_scattering_router,
                scattering_scale=config.scattering_scale,
                scattering_scale_warmup_steps=config.scattering_scale_warmup_steps,
                use_birman_schwinger=config.use_birman_schwinger,
                epsilon=config.epsilon,
                use_mourre=config.use_mourre,
                use_lap=config.use_lap,
                schatten_threshold=config.schatten_threshold,
                precision_upgrade_threshold=config.precision_upgrade_threshold,
                use_bitnet=config.use_bitnet,
                use_symplectic=config.use_symplectic,
                symplectic_dt=config.symplectic_dt,
            )
        
        # Apply configuration to model components
        self._apply_config()
    
    def _apply_config(self):
        """Apply configuration settings to model components."""
        from .bk_core import BKCoreFunction
        from .moe import SparseMoELayer
        
        # Step 2: Learning algorithm
        if self.config.use_analytic_gradient:
            BKCoreFunction.GRAD_BLEND = self.config.grad_blend
        
        # Iterate over blocks to set runtime flags
        if hasattr(self.model, 'blocks'):
            for block in self.model.blocks:
                # Handle Symplectic vs Standard blocks
                if hasattr(block, 'force_field'):
                    layer = block.force_field
                else:
                    layer = block.bk_layer

                # Routing proxy settings
                if isinstance(layer.moe_ffn, SparseMoELayer):
                    layer.moe_ffn.use_scattering_router = self.config.use_scattering_router
                    layer.moe_ffn.scattering_scale = self.config.scattering_scale

                # Apply numerical stability settings
                layer.v_max = self.config.v_max
                layer.feature_clamp = self.config.feature_clamp

                # Apply Non-Hermitian setting (if not handled in init)
                # Actually, gamma logic is internal to layer, but we can verify or init here
                pass
    
    def forward(self, x):
        """Forward pass through configured model."""
        return self.model(x)
    
    def get_num_parameters(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config_summary(self):
        """Get human-readable configuration summary."""
        enabled_features = []
        
        if self.config.use_analytic_gradient:
            enabled_features.append(f"Analytic Gradient (blend={self.config.grad_blend})")
        if self.config.use_koopman:
            enabled_features.append(f"Koopman Learning (dim={self.config.koopman_dim})")
        if self.config.use_physics_informed:
            enabled_features.append("Physics-Informed Learning")
        if self.config.use_quantization:
            enabled_features.append(f"Quantization (INT{self.config.quantization_bits})")
        if self.config.use_pruning:
            enabled_features.append(f"Pruning (threshold={self.config.prune_threshold})")
        if self.config.use_distillation:
            enabled_features.append("Knowledge Distillation")
        if self.config.use_mixed_precision:
            enabled_features.append("Mixed Precision")
        if self.config.use_custom_kernels:
            enabled_features.append("Custom CUDA Kernels")
        if self.config.use_gradient_checkpointing:
            enabled_features.append("Gradient Checkpointing")
        if self.config.use_adaptive_computation:
            enabled_features.append("Adaptive Computation Time")
        if self.config.use_multi_scale:
            enabled_features.append("Multi-Scale Processing")
        if self.config.use_learned_sparsity:
            enabled_features.append("Learned Sparsity")
        if self.config.use_curriculum_learning:
            enabled_features.append("Curriculum Learning")
        if self.config.use_active_learning:
            enabled_features.append("Active Learning")
        if self.config.use_gradient_caching:
            enabled_features.append("Gradient Caching")
        if self.config.prime_bump_init:
            enabled_features.append(f"Prime-bump Init (scale={self.config.prime_bump_scale})")
        if self.config.use_bitnet:
            enabled_features.append("BitNet b1.58 (1.58-bit Weights)")
        if self.config.use_symplectic:
            enabled_features.append(f"Symplectic Integrator (dt={self.config.symplectic_dt})")
        if self.config.use_non_hermitian:
            enabled_features.append("Non-Hermitian Physics (Decay Gamma)")
        if self.config.model_type == "koopman":
            enabled_features.append("KOOPMAN MODEL ARCHITECTURE")
        
        summary = {
            "Model Type": self.config.model_type,
            "Specs": f"d={self.config.d_model}, L={self.config.n_layers}, N={self.config.n_seq}",
            "Parameters": f"{self.get_num_parameters()/1e6:.2f}M",
            "MoE": f"{self.config.num_experts} experts, top-{self.config.top_k}",
            "Enabled Features": enabled_features if enabled_features else ["Baseline only"],
        }
        
        return summary
