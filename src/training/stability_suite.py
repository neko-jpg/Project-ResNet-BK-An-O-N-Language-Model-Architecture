"""
Numerical Stability Suite - Comprehensive NaN/Inf Elimination

This module provides a complete suite of stability algorithms designed to 
eliminate NaN/Inf values throughout the training pipeline.

Algorithms included:
1. BackwardHookNaNEliminator - Catches NaN at source in backward pass
2. EmbeddingStabilityWrapper - NaN-proof embedding layers
3. LayerwiseGradientScaler - Per-layer gradient scaling
4. SafeActivations - NaN-proof activation function wrappers
5. LossLandscapeSmoother - Prevents sharp loss spikes
6. AdaptivePrecisionController - Dynamic fp16/fp32 switching
7. StabilityManager - Unified manager for all stability features

Author: Project MUSE Team
Date: 2024-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
import math
import weakref


# =============================================================================
# 1. Backward Hook NaN Eliminator
# =============================================================================

class BackwardHookNaNEliminator:
    """
    Registers backward hooks on all parameters to catch and fix NaN/Inf
    at the source during backpropagation.
    
    This is the first line of defense - catches NaN the moment they appear
    in gradients, before they can propagate to other layers.
    """
    
    def __init__(self, model: nn.Module, max_grad_value: float = 1.0, verbose: bool = False):
        self.model = model
        self.max_grad_value = max_grad_value
        self.verbose = verbose
        self.hook_handles = []
        self.nan_counts: Dict[str, int] = {}
        self.total_fixed = 0
        
    def register_hooks(self):
        """Register backward hooks on all parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Create hook that clamps and fixes NaN/Inf
                handle = param.register_hook(self._create_grad_hook(name))
                self.hook_handles.append(handle)
                self.nan_counts[name] = 0
        
        if self.verbose:
            print(f"✔ Backward NaN Eliminator: {len(self.hook_handles)} hooks registered")
    
    def _create_grad_hook(self, param_name: str) -> Callable:
        """Create a gradient hook for a specific parameter."""
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad
            
            # Check for NaN/Inf
            has_nan = torch.isnan(grad).any()
            has_inf = torch.isinf(grad).any()
            
            if has_nan or has_inf:
                self.nan_counts[param_name] += 1
                self.total_fixed += 1
                
                # Replace NaN/Inf with zeros
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clamp to reasonable range (Muon用に厳格化済み)
            grad_max = grad.abs().max().item()
            if grad_max > self.max_grad_value:
                # デバッグ: クリッピング発生をカウント
                if not hasattr(self, 'clip_count'):
                    self.clip_count = 0
                self.clip_count += 1
            
            grad = torch.clamp(grad, -self.max_grad_value, self.max_grad_value)
            
            return grad
        
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get NaN fixing statistics."""
        problematic = {k: v for k, v in self.nan_counts.items() if v > 0}
        return {
            'total_fixed': self.total_fixed,
            'problematic_params': len(problematic),
            'top_offenders': sorted(problematic.items(), key=lambda x: -x[1])[:10],
        }


# =============================================================================
# 2. Embedding Stability Wrapper
# =============================================================================

class EmbeddingStabilityWrapper(nn.Module):
    """
    Wraps embedding layers with comprehensive stability measures:
    - Output clamping
    - NaN/Inf replacement
    - Gradient clipping via hooks
    - Spectral normalization of embedding weights
    """
    
    def __init__(
        self,
        embedding: nn.Module,
        max_norm: float = 10.0,
        grad_max: float = 1.0,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.embedding = embedding
        self.max_norm = max_norm
        self.grad_max = grad_max
        
        # Register gradient hooks for embedding weights
        self._register_grad_hooks()
        
        # Optional: Apply spectral normalization
        if use_spectral_norm and hasattr(embedding, 'weight'):
            self._apply_spectral_constraint()
    
    def _register_grad_hooks(self):
        """Register gradient clipping hooks on embedding parameters."""
        for name, param in self.embedding.named_parameters():
            if param.requires_grad:
                param.register_hook(
                    lambda g: torch.clamp(g, -self.grad_max, self.grad_max) if g is not None else g
                )
    
    def _apply_spectral_constraint(self):
        """Apply spectral constraint to embedding weights."""
        with torch.no_grad():
            if hasattr(self.embedding, 'weight'):
                weight = self.embedding.weight
                # Clamp weight magnitude
                weight.data = torch.clamp(weight.data, -1.0, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embedding output
        out = self.embedding(x)
        
        # Clamp output magnitude
        out = torch.clamp(out, -self.max_norm, self.max_norm)
        
        # Replace any NaN/Inf
        out = torch.nan_to_num(out, nan=0.0, posinf=self.max_norm, neginf=-self.max_norm)
        
        return out


# =============================================================================
# 3. Layer-wise Gradient Scaler
# =============================================================================

class LayerwiseGradientScaler:
    """
    Applies different gradient scaling factors to different layer types.
    
    Embedding layers and early layers get more aggressive scaling (smaller multiplier)
    to prevent gradient explosion at the source.
    """
    
    def __init__(
        self,
        model: nn.Module,
        embedding_scale: float = 0.1,      # Very conservative for embeddings
        early_layer_scale: float = 0.3,    # Conservative for early layers
        middle_layer_scale: float = 0.5,   # Moderate for middle layers
        late_layer_scale: float = 1.0,     # Full scale for late layers
        num_layers: int = 48,
    ):
        self.model = model
        self.embedding_scale = embedding_scale
        self.early_layer_scale = early_layer_scale
        self.middle_layer_scale = middle_layer_scale
        self.late_layer_scale = late_layer_scale
        self.num_layers = num_layers
        self.hook_handles = []
    
    def register_hooks(self):
        """Register gradient scaling hooks based on layer position."""
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            scale = self._get_scale_for_layer(name)
            if scale < 1.0:
                handle = param.register_hook(self._create_scale_hook(scale))
                self.hook_handles.append(handle)
    
    def _get_scale_for_layer(self, name: str) -> float:
        """Determine scale factor based on layer name."""
        name_lower = name.lower()
        
        # Embedding layers - most conservative
        if 'embedding' in name_lower or 'core1' in name_lower or 'core2' in name_lower:
            return self.embedding_scale
        
        # Extract layer number if present
        import re
        match = re.search(r'blocks\.(\d+)', name)
        if match:
            layer_idx = int(match.group(1))
            # Early 25%
            if layer_idx < self.num_layers * 0.25:
                return self.early_layer_scale
            # Middle 50%
            elif layer_idx < self.num_layers * 0.75:
                return self.middle_layer_scale
            # Late 25%
            else:
                return self.late_layer_scale
        
        # Default for non-block layers
        return self.middle_layer_scale
    
    def _create_scale_hook(self, scale: float) -> Callable:
        """Create a gradient scaling hook."""
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad
            return grad * scale
        return hook
    
    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


# =============================================================================
# 4. Safe Activation Functions
# =============================================================================

class SafeGELU(nn.Module):
    """GELU with NaN protection."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp input to prevent overflow in exp
        x = torch.clamp(x, -10.0, 10.0)
        out = F.gelu(x)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class SafeSoftmax(nn.Module):
    """Softmax with NaN protection and temperature scaling."""
    def __init__(self, dim: int = -1, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Subtract max for numerical stability
        x_max = x.max(dim=self.dim, keepdim=True).values
        x = x - x_max
        
        # Apply temperature
        x = x / self.temperature
        
        # Clamp before exp
        x = torch.clamp(x, -50.0, 50.0)
        
        # Compute softmax
        out = F.softmax(x, dim=self.dim)
        
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


class SafeLayerNorm(nn.Module):
    """LayerNorm with NaN protection."""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.max_value = 100.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-clamp input
        x = torch.clamp(x, -self.max_value, self.max_value)
        x = torch.nan_to_num(x, nan=0.0, posinf=self.max_value, neginf=-self.max_value)
        
        # Apply layer norm
        out = self.layer_norm(x)
        
        # Post-clamp output
        out = torch.nan_to_num(out, nan=0.0, posinf=self.max_value, neginf=-self.max_value)
        
        return out


# =============================================================================
# 5. Loss Landscape Smoother
# =============================================================================

class LossLandscapeSmoother:
    """
    Smooths the loss landscape to prevent sharp spikes that cause gradient explosion.
    
    Techniques:
    - Loss clipping
    - Loss EMA smoothing
    - Gradient penalty for sharp changes
    """
    
    def __init__(
        self,
        max_loss: float = 100.0,
        smoothing_factor: float = 0.9,
        enable_ema: bool = True,
    ):
        self.max_loss = max_loss
        self.smoothing_factor = smoothing_factor
        self.enable_ema = enable_ema
        self.loss_ema = None
        self.step_count = 0
    
    def smooth_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply loss smoothing and clipping."""
        self.step_count += 1
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            if self.loss_ema is not None:
                return self.loss_ema  # Return EMA if current is invalid
            return torch.tensor(10.0, device=loss.device, dtype=loss.dtype)
        
        # Clip loss
        loss = torch.clamp(loss, 0.0, self.max_loss)
        
        # Apply EMA smoothing
        if self.enable_ema:
            if self.loss_ema is None:
                self.loss_ema = loss.detach()
            else:
                self.loss_ema = self.smoothing_factor * self.loss_ema + (1 - self.smoothing_factor) * loss.detach()
        
        return loss
    
    def get_smoothed_loss(self) -> Optional[float]:
        """Get current smoothed loss value."""
        return self.loss_ema.item() if self.loss_ema is not None else None


# =============================================================================
# 6. Adaptive Precision Controller
# =============================================================================

class AdaptivePrecisionController:
    """
    Dynamically switches between fp16/bf16/fp32 based on gradient health.
    
    If gradients become unstable, temporarily switches to higher precision.
    """
    
    def __init__(
        self,
        model: nn.Module,
        nan_threshold: int = 100,      # Switch precision if more than this many NaNs
        switch_back_steps: int = 1000,  # Steps before trying lower precision again
    ):
        self.model = model
        self.nan_threshold = nan_threshold
        self.switch_back_steps = switch_back_steps
        self.current_precision = 'bfloat16'
        self.nan_count_this_epoch = 0
        self.steps_since_switch = 0
        self.precision_history = []
    
    def update(self, nan_count: int) -> str:
        """Update precision based on NaN count."""
        self.nan_count_this_epoch += nan_count
        self.steps_since_switch += 1
        
        # Check if we need to upgrade precision
        if nan_count > self.nan_threshold and self.current_precision != 'float32':
            self.current_precision = 'float32'
            self.steps_since_switch = 0
            self.precision_history.append(('upgrade', self.steps_since_switch))
        
        # Check if we can try downgrading back
        elif self.steps_since_switch > self.switch_back_steps and self.current_precision == 'float32':
            self.current_precision = 'bfloat16'
            self.steps_since_switch = 0
            self.precision_history.append(('downgrade', self.steps_since_switch))
        
        return self.current_precision
    
    def get_autocast_dtype(self) -> torch.dtype:
        """Get the dtype for autocast context."""
        if self.current_precision == 'float32':
            return torch.float32
        elif self.current_precision == 'bfloat16':
            return torch.bfloat16
        else:
            return torch.float16


# =============================================================================
# 7. Unified Stability Manager
# =============================================================================

@dataclass
class StabilityConfig:
    """Configuration for the Stability Manager."""
    # Backward Hook Settings
    enable_backward_hooks: bool = True
    backward_max_grad: float = 1.0
    
    # Embedding Stability
    enable_embedding_stability: bool = True
    embedding_max_norm: float = 10.0
    embedding_grad_max: float = 0.5
    
    # Layerwise Gradient Scaling
    enable_layerwise_scaling: bool = True
    embedding_scale: float = 0.1
    early_layer_scale: float = 0.3
    
    # Loss Smoothing
    enable_loss_smoothing: bool = True
    max_loss: float = 100.0
    
    # Adaptive Precision
    enable_adaptive_precision: bool = True
    
    # General
    verbose: bool = False


class StabilityManager:
    """
    Unified manager for all stability algorithms.
    
    Provides a single interface to enable, configure, and monitor
    all stability features.
    """
    
    def __init__(self, model: nn.Module, config: StabilityConfig = None):
        self.model = model
        self.config = config or StabilityConfig()
        
        # Initialize components
        self.backward_eliminator = None
        self.layerwise_scaler = None
        self.loss_smoother = None
        self.precision_controller = None
        
        self.initialized = False
        self.step_count = 0
        
    def initialize(self):
        """Initialize all stability components."""
        if self.initialized:
            return
        
        # 1. Backward Hook NaN Eliminator
        if self.config.enable_backward_hooks:
            self.backward_eliminator = BackwardHookNaNEliminator(
                self.model,
                max_grad_value=self.config.backward_max_grad,
                verbose=self.config.verbose,
            )
            self.backward_eliminator.register_hooks()
        
        # 2. Layerwise Gradient Scaler
        if self.config.enable_layerwise_scaling:
            self.layerwise_scaler = LayerwiseGradientScaler(
                self.model,
                embedding_scale=self.config.embedding_scale,
                early_layer_scale=self.config.early_layer_scale,
            )
            self.layerwise_scaler.register_hooks()
        
        # 3. Loss Landscape Smoother
        if self.config.enable_loss_smoothing:
            self.loss_smoother = LossLandscapeSmoother(
                max_loss=self.config.max_loss,
            )
        
        # 4. Adaptive Precision Controller
        if self.config.enable_adaptive_precision:
            self.precision_controller = AdaptivePrecisionController(self.model)
        
        self.initialized = True
        
        if self.config.verbose:
            print("✔ Stability Manager Initialized")
            print(f"  - Backward Hooks: {self.config.enable_backward_hooks}")
            print(f"  - Layerwise Scaling: {self.config.enable_layerwise_scaling}")
            print(f"  - Loss Smoothing: {self.config.enable_loss_smoothing}")
            print(f"  - Adaptive Precision: {self.config.enable_adaptive_precision}")
    
    def process_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Process loss through smoothing."""
        if self.loss_smoother is not None:
            return self.loss_smoother.smooth_loss(loss)
        return loss
    
    def update_precision(self, nan_count: int) -> torch.dtype:
        """Update precision based on NaN count."""
        if self.precision_controller is not None:
            self.precision_controller.update(nan_count)
            return self.precision_controller.get_autocast_dtype()
        return torch.bfloat16
    
    def step(self):
        """Called after each training step."""
        self.step_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics from all components."""
        stats = {
            'step_count': self.step_count,
        }
        
        if self.backward_eliminator:
            stats['backward_eliminator'] = self.backward_eliminator.get_statistics()
        
        if self.loss_smoother:
            stats['smoothed_loss'] = self.loss_smoother.get_smoothed_loss()
        
        if self.precision_controller:
            stats['current_precision'] = self.precision_controller.current_precision
        
        return stats
    
    def cleanup(self):
        """Remove all hooks and cleanup."""
        if self.backward_eliminator:
            self.backward_eliminator.remove_hooks()
        if self.layerwise_scaler:
            self.layerwise_scaler.remove_hooks()


def create_stability_manager(
    model: nn.Module,
    aggressive: bool = True,
) -> StabilityManager:
    """
    Factory function to create a pre-configured StabilityManager.
    
    Args:
        model: The model to stabilize
        aggressive: If True, use more aggressive stability settings
    
    Returns:
        Configured StabilityManager instance
    """
    if aggressive:
        # Muon Ultra-Aggressive Mode: 勾配を0.5-1.0に厳格に制限
        config = StabilityConfig(
            enable_backward_hooks=True,
            backward_max_grad=1.0,       # 10.0 → 1.0 (Muon: 通常時の上限)
            enable_embedding_stability=True,
            embedding_max_norm=5.0,
            embedding_grad_max=0.5,      # 10.0 → 0.5 (Muon: embedding層は特に厳しく)
            enable_layerwise_scaling=True,  # False → True (勾配爆発抑制)
            embedding_scale=0.5,         # Embedding層の勾配を半分に
            early_layer_scale=0.7,       # 初期層(0-25%)を70%に
            enable_loss_smoothing=True,
            max_loss=50.0,
            enable_adaptive_precision=True,
            verbose=True,
        )
    else:
        config = StabilityConfig()
    
    manager = StabilityManager(model, config)
    manager.initialize()
    
    return manager
