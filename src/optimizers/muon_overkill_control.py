"""
Muon Overkill Gradient Control - Maximum Stability

This module contains AGGRESSIVE algorithms to force gradient norms
into safe range (0.3 - 2.0). These are "overkill" level controls.

Algorithms:
9.  Per-Layer Gradient Clipping
10. Global Gradient Rescaling
11. Exponential Gradient Decay
12. Gradient Magnitude Normalizer
13. Ultimate Gradient Limiter
"""

import torch
import math
from typing import Dict, List, Tuple, Optional


class PerLayerGradientClipper:
    """
    Algorithm 9: Per-Layer Gradient Clipping
    
    Clips each parameter's gradient independently to max_norm.
    This ensures no single layer can dominate the update.
    """
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
        self.clip_count = 0
        
    def clip(self, grad: torch.Tensor) -> torch.Tensor:
        grad_norm = grad.norm().item()
        if grad_norm > self.max_norm:
            grad = grad * (self.max_norm / (grad_norm + 1e-8))
            self.clip_count += 1
        return grad


class GlobalGradientRescaler:
    """
    Algorithm 10: Global Gradient Rescaling
    
    Tracks the maximum gradient norm seen and rescales all
    gradients to ensure they're within a target range.
    """
    
    def __init__(self, target_norm: float = 0.5, ema_decay: float = 0.99):
        self.target_norm = target_norm
        self.ema_decay = ema_decay
        self.ema_max_norm = 1.0
        
    def rescale(self, grad: torch.Tensor) -> torch.Tensor:
        grad_norm = grad.norm().item()
        
        # Update EMA of max norm
        self.ema_max_norm = max(
            self.ema_decay * self.ema_max_norm,
            grad_norm
        )
        
        # Rescale to target
        if self.ema_max_norm > self.target_norm:
            scale = self.target_norm / self.ema_max_norm
            grad = grad * scale
            
        return grad


class ExponentialGradientDecay:
    """
    Algorithm 11: Exponential Gradient Decay
    
    Applies exponential decay to gradients based on magnitude.
    Larger gradients get more aggressive decay.
    
    decay_factor = exp(-alpha * (grad_norm - threshold))
    """
    
    def __init__(self, threshold: float = 0.5, alpha: float = 2.0):
        self.threshold = threshold
        self.alpha = alpha
        
    def decay(self, grad: torch.Tensor) -> torch.Tensor:
        grad_norm = grad.norm().item()
        
        if grad_norm > self.threshold:
            excess = grad_norm - self.threshold
            decay_factor = math.exp(-self.alpha * excess)
            grad = grad * decay_factor
            
        return grad


class GradientMagnitudeNormalizer:
    """
    Algorithm 12: Gradient Magnitude Normalizer
    
    Normalizes gradient to have unit norm, then scales to target magnitude.
    This completely removes the original magnitude and replaces it.
    """
    
    def __init__(self, target_magnitude: float = 0.1):
        self.target_magnitude = target_magnitude
        
    def normalize(self, grad: torch.Tensor) -> torch.Tensor:
        grad_norm = grad.norm()
        
        if grad_norm > 1e-8:
            # Normalize to unit norm
            unit_grad = grad / grad_norm
            # Scale to target magnitude
            grad = unit_grad * self.target_magnitude
            
        return grad


class UltimateGradientLimiter:
    """
    Algorithm 13: Ultimate Gradient Limiter
    
    The final safety net. Applies multiple techniques in sequence:
    1. Hard clip to absolute maximum
    2. Soft clip with tanh
    3. Per-element clipping
    """
    
    def __init__(
        self,
        hard_max: float = 1.0,
        soft_max: float = 0.5,
        per_element_max: float = 0.01,
    ):
        self.hard_max = hard_max
        self.soft_max = soft_max
        self.per_element_max = per_element_max
        
    def limit(self, grad: torch.Tensor) -> torch.Tensor:
        grad_norm = grad.norm().item()
        
        # 1. Hard clip
        if grad_norm > self.hard_max:
            grad = grad * (self.hard_max / (grad_norm + 1e-8))
        
        # 2. Soft clip with tanh (smoothly limits large values)
        grad_norm = grad.norm().item()
        if grad_norm > self.soft_max:
            # Use tanh to smoothly limit
            scale = math.tanh(self.soft_max / grad_norm)
            grad = grad * scale
        
        # 3. Per-element clipping
        grad = torch.clamp(grad, -self.per_element_max, self.per_element_max)
        
        return grad


class OverkillGradientController:
    """
    Master controller that applies ALL overkill algorithms in sequence.
    
    Pipeline:
    1. Per-Layer Clipping (max 1.0)
    2. Global Rescaling (target 0.5)
    3. Exponential Decay (threshold 0.5, alpha 2.0)
    4. Magnitude Normalization (target 0.1)
    5. Ultimate Limiter (hard 1.0, soft 0.5, element 0.01)
    """
    
    def __init__(self, aggressive: bool = True):
        if aggressive:
            self.per_layer = PerLayerGradientClipper(max_norm=0.5)
            self.global_rescaler = GlobalGradientRescaler(target_norm=0.3)
            self.exp_decay = ExponentialGradientDecay(threshold=0.3, alpha=3.0)
            self.normalizer = GradientMagnitudeNormalizer(target_magnitude=0.05)
            self.ultimate = UltimateGradientLimiter(
                hard_max=0.5,
                soft_max=0.3,
                per_element_max=0.001,
            )
        else:
            self.per_layer = PerLayerGradientClipper(max_norm=1.0)
            self.global_rescaler = GlobalGradientRescaler(target_norm=0.5)
            self.exp_decay = ExponentialGradientDecay(threshold=0.5, alpha=2.0)
            self.normalizer = GradientMagnitudeNormalizer(target_magnitude=0.1)
            self.ultimate = UltimateGradientLimiter()
        
        self.step_count = 0
        self.use_normalizer = False  # Only enable if other methods fail
        
    def process(self, grad: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Apply all gradient control algorithms in sequence.
        
        Returns:
            processed_grad: Gradient after all processing
            metrics: Processing metrics
        """
        self.step_count += 1
        original_norm = grad.norm().item()
        
        # 1. Per-Layer Clipping
        grad = self.per_layer.clip(grad)
        after_clip = grad.norm().item()
        
        # 2. Global Rescaling
        grad = self.global_rescaler.rescale(grad)
        after_rescale = grad.norm().item()
        
        # 3. Exponential Decay
        grad = self.exp_decay.decay(grad)
        after_decay = grad.norm().item()
        
        # 4. Magnitude Normalization (only if still too large)
        if self.use_normalizer or after_decay > 1.0:
            grad = self.normalizer.normalize(grad)
            self.use_normalizer = True  # Enable permanently once triggered
        after_norm = grad.norm().item()
        
        # 5. Ultimate Limiter (always apply as final safety)
        grad = self.ultimate.limit(grad)
        final_norm = grad.norm().item()
        
        metrics = {
            'original_norm': original_norm,
            'after_clip': after_clip,
            'after_rescale': after_rescale,
            'after_decay': after_decay,
            'after_norm': after_norm,
            'final_norm': final_norm,
            'reduction_ratio': original_norm / (final_norm + 1e-8),
        }
        
        return grad, metrics


def create_overkill_controller(aggressive: bool = True) -> OverkillGradientController:
    """Factory function to create OverkillGradientController."""
    return OverkillGradientController(aggressive=aggressive)
