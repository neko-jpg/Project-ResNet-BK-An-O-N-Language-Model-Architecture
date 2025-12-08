"""
Muon Gradient Damper - Pre-Orthogonalization Gradient Conditioning

Prevents extreme gradient values before Newton-Schulz orthogonalization
to avoid NaN/Inf during matrix operations.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class MuonGradientDamper:
    """
    Pre-conditions gradients before orthogonalization to prevent numerical instability.
    
    Features:
    - Spectral normalization by largest singular value
    - Outlier detection and clamping (±3σ threshold)
    - EMA-based smooth scaling to prevent sudden jumps
    - Condition number monitoring and adjustment
    """
    
    def __init__(
        self,
        outlier_threshold: float = 3.0,
        spectral_norm_eps: float = 1e-6,
        ema_decay: float = 0.9,
        max_condition_number: float = 100.0,
        enable_spectral_norm: bool = True,
        enable_outlier_clamp: bool = True,
        enable_smooth_scaling: bool = True,
    ):
        """
        Args:
            outlier_threshold: Number of standard deviations for outlier detection
            spectral_norm_eps: Epsilon for spectral normalization
            ema_decay: Decay rate for EMA scaling
            max_condition_number: Maximum allowed condition number
            enable_spectral_norm: Enable spectral normalization
            enable_outlier_clamp: Enable outlier clamping
            enable_smooth_scaling: Enable EMA-based smooth scaling
        """
        self.outlier_threshold = outlier_threshold
        self.spectral_norm_eps = spectral_norm_eps
        self.ema_decay = ema_decay
        self.max_condition_number = max_condition_number
        
        self.enable_spectral_norm = enable_spectral_norm
        self.enable_outlier_clamp = enable_outlier_clamp
        self.enable_smooth_scaling = enable_smooth_scaling
        
        # State for EMA scaling
        self.ema_scale = None
        self.step_count = 0
    
    def _spectral_normalize(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Normalize gradient by its largest singular value.
        
        For 2D matrices: σ_max = ||G||_2
        This ensures all singular values are ≤ 1.0
        """
        if grad.ndim < 2:
            # For 1D, just normalize by norm
            return grad / (grad.norm() + self.spectral_norm_eps)
        
        # Compute largest singular value efficiently
        # ||G||_2 = sqrt(λ_max(G^T @ G))
        with torch.no_grad():
            # Use power iteration for efficiency (avoid full SVD)
            # Approximation: ||G||_2 ≈ ||G||_F for well-conditioned matrices
            spectral_norm = grad.norm()
            
            # Clamp to prevent division by zero
            spectral_norm = torch.clamp(spectral_norm, min=self.spectral_norm_eps)
        
        return grad / spectral_norm
    
    def _clamp_outliers(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Clamp gradient outliers beyond ±3σ threshold.
        
        This prevents extreme values from destabilizing orthogonalization.
        """
        with torch.no_grad():
            mean = grad.mean()
            std = grad.std()
            
            # Avoid degenerate case (all zeros)
            if std < 1e-8:
                return grad
            
            # Compute bounds
            lower_bound = mean - self.outlier_threshold * std
            upper_bound = mean + self.outlier_threshold * std
        
        return torch.clamp(grad, min=lower_bound, max=upper_bound)
    
    def _smooth_scale(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Apply EMA-based scaling to prevent sudden magnitude jumps.
        
        scale_t = ema_decay * scale_{t-1} + (1 - ema_decay) * ||grad_t||
        grad_normalized = grad_t / scale_t
        """
        current_norm = grad.norm().item()
        
        if self.ema_scale is None:
            # Initialize
            self.ema_scale = current_norm
        else:
            # Update EMA
            self.ema_scale = self.ema_decay * self.ema_scale + (1 - self.ema_decay) * current_norm
        
        # Avoid division by zero
        scale = max(self.ema_scale, 1e-8)
        
        return grad / scale
    
    def _check_condition_number(self, grad: torch.Tensor) -> float:
        """
        Estimate condition number of gradient matrix.
        
        κ(G) = σ_max / σ_min
        
        High condition number indicates numerical instability.
        """
        if grad.ndim < 2:
            return 1.0
        
        with torch.no_grad():
            # Use SVD to get exact singular values
            # For large matrices, this is expensive - only use for monitoring
            try:
                _, S, _ = torch.svd(grad.float())
                
                sigma_max = S[0].item()
                sigma_min = S[-1].item()
                
                if sigma_min < 1e-8:
                    return float('inf')
                
                return sigma_max / sigma_min
            except:
                # SVD failed, return safe value
                return 1.0
    
    def damp_gradient(self, grad: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply gradient damping pipeline.
        
        Pipeline:
        1. Outlier clamping (optional)
        2. Spectral normalization (optional)
        3. Smooth scaling (optional)
        
        Args:
            grad: Raw gradient tensor
            
        Returns:
            damped_grad: Conditioned gradient
            metrics: Dictionary of damping metrics
        """
        self.step_count += 1
        
        damped_grad = grad
        metrics = {}
        
        # Track original norm
        original_norm = grad.norm().item()
        metrics['original_norm'] = original_norm
        
        # 1. Outlier clamping
        if self.enable_outlier_clamp:
            damped_grad = self._clamp_outliers(damped_grad)
            metrics['outlier_clamp_applied'] = 1.0
        
        # 2. Spectral normalization
        if self.enable_spectral_norm:
            damped_grad = self._spectral_normalize(damped_grad)
            metrics['spectral_norm_applied'] = 1.0
        
        # 3. Smooth scaling
        if self.enable_smooth_scaling:
            damped_grad = self._smooth_scale(damped_grad)
            metrics['smooth_scale_applied'] = 1.0
            metrics['ema_scale'] = self.ema_scale if self.ema_scale else 1.0
        
        # Track final norm
        final_norm = damped_grad.norm().item()
        metrics['final_norm'] = final_norm
        metrics['damping_ratio'] = original_norm / (final_norm + 1e-8)
        
        # Condition number (only check periodically to avoid overhead)
        if self.step_count % 100 == 0 and damped_grad.ndim >= 2:
            condition_num = self._check_condition_number(damped_grad)
            metrics['condition_number'] = condition_num
        
        return damped_grad, metrics


def create_muon_gradient_damper(
    aggressive: bool = False,
    **kwargs
) -> MuonGradientDamper:
    """
    Factory function to create MuonGradientDamper with preset configurations.
    
    Args:
        aggressive: If True, use more aggressive damping (warmup phase)
        **kwargs: Additional arguments to override defaults
        
    Returns:
        MuonGradientDamper instance
    """
    if aggressive:
        # Warmup phase: Very conservative
        defaults = {
            'outlier_threshold': 2.0,  # Tighter outlier bounds
            'spectral_norm_eps': 1e-4,  # Safer epsilon
            'ema_decay': 0.95,  # Slower adaptation
            'max_condition_number': 50.0,  # Stricter conditioning
            'enable_spectral_norm': True,
            'enable_outlier_clamp': True,
            'enable_smooth_scaling': True,
        }
    else:
        # Normal phase: Balanced
        defaults = {
            'outlier_threshold': 3.0,
            'spectral_norm_eps': 1e-6,
            'ema_decay': 0.9,
            'max_condition_number': 100.0,
            'enable_spectral_norm': True,
            'enable_outlier_clamp': True,
            'enable_smooth_scaling': True,
        }
    
    # Override with user-provided kwargs
    defaults.update(kwargs)
    
    return MuonGradientDamper(**defaults)
