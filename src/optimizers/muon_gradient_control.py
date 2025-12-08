"""
Muon Gradient Control Suite - Complete Gradient Explosion Prevention

This module contains multiple algorithms to prevent gradient explosion
in the Muon optimizer:

1. MomentumBufferScaler - Prevents momentum accumulation from exploding
2. AdaptiveUpdateController - Limits update magnitude dynamically
3. EmergencyGradientBrake - Last-resort safety mechanism
"""

import torch
import math
from typing import Dict, Optional, Tuple


class MomentumBufferScaler:
    """
    Algorithm 4: Momentum Buffer Scaler
    
    The Muon momentum formula: buf = buf * momentum + grad
    This can cause buf to grow unboundedly if gradients are consistently large.
    
    Solution: Periodically rescale the momentum buffer to prevent explosion.
    
    Target: ||buf||_F should stay within [0.1, 10.0] range
    """
    
    def __init__(
        self,
        target_norm: float = 1.0,
        max_norm: float = 10.0,
        min_norm: float = 0.01,
        scale_frequency: int = 1,  # Scale every N steps
        soft_clamp: bool = True,
    ):
        self.target_norm = target_norm
        self.max_norm = max_norm
        self.min_norm = min_norm
        self.scale_frequency = scale_frequency
        self.soft_clamp = soft_clamp
        
        self.step_count = 0
        self.total_rescales = 0
        
    def scale_buffer(self, buf: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Scale momentum buffer to keep norm in safe range.
        
        Args:
            buf: Momentum buffer tensor
            
        Returns:
            scaled_buf: Scaled momentum buffer
            metrics: Scaling metrics
        """
        self.step_count += 1
        metrics = {'buffer_scaled': False, 'original_norm': 0.0}
        
        # Only scale at specified frequency
        if self.step_count % self.scale_frequency != 0:
            return buf, metrics
        
        buf_norm = buf.norm().item()
        metrics['original_norm'] = buf_norm
        
        # Skip if norm is in acceptable range
        if self.min_norm <= buf_norm <= self.max_norm:
            return buf, metrics
        
        # Compute scaling factor
        if buf_norm > self.max_norm:
            # Too large - scale down
            if self.soft_clamp:
                # Soft clamp: asymptotically approach target
                scale = self.target_norm / (buf_norm + 1e-8)
                scale = max(scale, 0.1)  # Don't scale down too aggressively
            else:
                # Hard clamp to max
                scale = self.max_norm / (buf_norm + 1e-8)
        elif buf_norm < self.min_norm and buf_norm > 1e-8:
            # Too small - scale up (prevents dead gradients)
            scale = self.min_norm / (buf_norm + 1e-8)
            scale = min(scale, 10.0)  # Don't scale up too aggressively
        else:
            return buf, metrics
        
        # Apply scaling
        scaled_buf = buf * scale
        
        self.total_rescales += 1
        metrics['buffer_scaled'] = True
        metrics['scale_factor'] = scale
        metrics['new_norm'] = scaled_buf.norm().item()
        metrics['total_rescales'] = self.total_rescales
        
        return scaled_buf, metrics


class AdaptiveUpdateController:
    """
    Algorithm 5: Adaptive Update Magnitude Controller
    
    Directly controls the magnitude of weight updates to prevent
    any single update from being too large.
    
    Target: ||update||_F should be proportional to ||weight||_F
    Typical ratio: update_norm / weight_norm ≈ 0.01 (1% change per step)
    """
    
    def __init__(
        self,
        max_update_ratio: float = 0.05,  # Max 5% weight change per step
        min_update_ratio: float = 0.0001,  # Min 0.01% weight change
        use_weight_relative_scaling: bool = True,
        absolute_max_update: float = 1.0,  # Absolute cap on update norm
    ):
        self.max_update_ratio = max_update_ratio
        self.min_update_ratio = min_update_ratio
        self.use_weight_relative_scaling = use_weight_relative_scaling
        self.absolute_max_update = absolute_max_update
        
        self.step_count = 0
        self.total_clips = 0
        
    def control_update(
        self,
        update: torch.Tensor,
        weight: torch.Tensor,
        lr: float,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Control the magnitude of weight update.
        
        Args:
            update: Proposed update (before lr scaling)
            weight: Current weights
            lr: Learning rate
            
        Returns:
            controlled_update: Magnitude-controlled update
            metrics: Control metrics
        """
        self.step_count += 1
        metrics = {'update_clipped': False}
        
        # Compute norms
        update_norm = (update * lr).norm().item()
        weight_norm = weight.norm().item()
        
        metrics['update_norm'] = update_norm
        metrics['weight_norm'] = weight_norm
        
        # Skip for zero-ish weights (embeddings initialization phase)
        if weight_norm < 1e-6:
            return update, metrics
        
        # Compute current update ratio
        current_ratio = update_norm / (weight_norm + 1e-8)
        metrics['update_ratio'] = current_ratio
        
        # Check if update needs to be scaled
        need_scale = False
        target_ratio = current_ratio
        
        if current_ratio > self.max_update_ratio:
            target_ratio = self.max_update_ratio
            need_scale = True
        
        # Also enforce absolute maximum
        if update_norm > self.absolute_max_update:
            target_norm = self.absolute_max_update
            target_ratio_from_abs = target_norm / (lr * update.norm().item() + 1e-8)
            if target_ratio_from_abs < target_ratio:
                target_ratio = target_ratio_from_abs
                need_scale = True
        
        if need_scale:
            scale = target_ratio / (current_ratio + 1e-8)
            controlled_update = update * scale
            
            self.total_clips += 1
            metrics['update_clipped'] = True
            metrics['clip_scale'] = scale
            metrics['new_update_norm'] = (controlled_update * lr).norm().item()
            metrics['total_clips'] = self.total_clips
            
            return controlled_update, metrics
        
        return update, metrics


class EmergencyGradientBrake:
    """
    Algorithm 6: Emergency Gradient Brake
    
    A last-resort safety mechanism that monitors the overall health
    of the training and applies emergency braking when needed.
    
    Triggers:
    - Gradient norm exceeds safe threshold
    - Loss spikes suddenly
    - Too many consecutive updates skipped
    
    Actions:
    - Reduce update magnitude drastically
    - Reset momentum buffers
    - Temporarily increase NS iterations
    """
    
    def __init__(
        self,
        grad_threshold: float = 50.0,  # Trigger if grad_norm > this
        brake_strength: float = 0.1,   # Scale updates by this when braking
        cooldown_steps: int = 10,      # Steps to stay in brake mode
        consecutive_skip_trigger: int = 3,  # Trigger if N consecutive skips
    ):
        self.grad_threshold = grad_threshold
        self.brake_strength = brake_strength
        self.cooldown_steps = cooldown_steps
        self.consecutive_skip_trigger = consecutive_skip_trigger
        
        self.brake_active = False
        self.brake_remaining = 0
        self.consecutive_skips = 0
        self.total_activations = 0
        
    def check_and_brake(
        self,
        grad_norm: float,
        update: torch.Tensor,
        was_skipped: bool = False,
    ) -> Tuple[torch.Tensor, bool, Dict]:
        """
        Check if emergency braking is needed and apply if so.
        
        Args:
            grad_norm: Current gradient norm
            update: Proposed update
            was_skipped: Whether previous step was skipped
            
        Returns:
            braked_update: Potentially braked update
            reset_momentum: Whether to reset momentum buffers
            metrics: Brake metrics
        """
        metrics = {
            'brake_active': self.brake_active,
            'brake_remaining': self.brake_remaining,
        }
        
        # Track consecutive skips
        if was_skipped:
            self.consecutive_skips += 1
        else:
            self.consecutive_skips = 0
        
        # Check if we should activate brake
        should_activate = False
        
        if grad_norm > self.grad_threshold:
            should_activate = True
            metrics['trigger_reason'] = 'grad_threshold'
        
        if self.consecutive_skips >= self.consecutive_skip_trigger:
            should_activate = True
            metrics['trigger_reason'] = 'consecutive_skips'
        
        # Activate brake if needed
        if should_activate and not self.brake_active:
            self.brake_active = True
            self.brake_remaining = self.cooldown_steps
            self.total_activations += 1
            self.consecutive_skips = 0  # Reset counter
            
            metrics['brake_activated'] = True
            metrics['total_activations'] = self.total_activations
        
        # Apply brake if active
        if self.brake_active:
            braked_update = update * self.brake_strength
            self.brake_remaining -= 1
            
            metrics['brake_applied'] = True
            metrics['brake_scale'] = self.brake_strength
            
            # Deactivate if cooldown over
            if self.brake_remaining <= 0:
                self.brake_active = False
                metrics['brake_deactivated'] = True
            
            # Signal to reset momentum on first activation step
            reset_momentum = (self.brake_remaining == self.cooldown_steps - 1)
            
            return braked_update, reset_momentum, metrics
        
        return update, False, metrics


class MuonGradientControlSuite:
    """
    Unified control suite combining all gradient control algorithms.
    
    This class provides a single interface to apply all gradient
    control mechanisms in the correct order:
    
    1. Momentum Buffer Scaling (pre-update)
    2. Adaptive Update Control (post-orthogonalization)
    3. Emergency Brake (final safety check)
    """
    
    def __init__(
        self,
        momentum_scaler: Optional[MomentumBufferScaler] = None,
        update_controller: Optional[AdaptiveUpdateController] = None,
        emergency_brake: Optional[EmergencyGradientBrake] = None,
        enable_all: bool = True,
    ):
        if enable_all:
            self.momentum_scaler = momentum_scaler or MomentumBufferScaler()
            self.update_controller = update_controller or AdaptiveUpdateController()
            self.emergency_brake = emergency_brake or EmergencyGradientBrake()
        else:
            self.momentum_scaler = momentum_scaler
            self.update_controller = update_controller
            self.emergency_brake = emergency_brake
        
        self.step_count = 0
        
    def apply_momentum_scaling(self, buf: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Apply momentum buffer scaling."""
        if self.momentum_scaler is None:
            return buf, {}
        return self.momentum_scaler.scale_buffer(buf)
    
    def apply_update_control(
        self,
        update: torch.Tensor,
        weight: torch.Tensor,
        lr: float,
    ) -> Tuple[torch.Tensor, Dict]:
        """Apply update magnitude control."""
        if self.update_controller is None:
            return update, {}
        return self.update_controller.control_update(update, weight, lr)
    
    def apply_emergency_brake(
        self,
        grad_norm: float,
        update: torch.Tensor,
        was_skipped: bool = False,
    ) -> Tuple[torch.Tensor, bool, Dict]:
        """Apply emergency brake if needed."""
        if self.emergency_brake is None:
            return update, False, {}
        return self.emergency_brake.check_and_brake(grad_norm, update, was_skipped)
    
    def get_metrics(self) -> Dict:
        """Get combined metrics from all components."""
        metrics = {'step_count': self.step_count}
        
        if self.momentum_scaler:
            metrics['momentum_scaler'] = {
                'total_rescales': self.momentum_scaler.total_rescales,
            }
        
        if self.update_controller:
            metrics['update_controller'] = {
                'total_clips': self.update_controller.total_clips,
            }
        
        if self.emergency_brake:
            metrics['emergency_brake'] = {
                'total_activations': self.emergency_brake.total_activations,
                'brake_active': self.emergency_brake.brake_active,
            }
        
        return metrics


def create_gradient_control_suite(
    aggressive: bool = False,
) -> MuonGradientControlSuite:
    """
    Factory function to create MuonGradientControlSuite with preset configurations.
    
    Args:
        aggressive: If True, use more aggressive control settings
        
    Returns:
        MuonGradientControlSuite instance
    """
    if aggressive:
        momentum_scaler = MomentumBufferScaler(
            target_norm=0.5,
            max_norm=5.0,
            scale_frequency=1,
        )
        update_controller = AdaptiveUpdateController(
            max_update_ratio=0.01,  # Max 1% weight change
            absolute_max_update=0.5,
        )
        emergency_brake = EmergencyGradientBrake(
            grad_threshold=20.0,
            brake_strength=0.05,
            cooldown_steps=20,
        )
    else:
        momentum_scaler = MomentumBufferScaler()
        update_controller = AdaptiveUpdateController()
        emergency_brake = EmergencyGradientBrake()
    
    return MuonGradientControlSuite(
        momentum_scaler=momentum_scaler,
        update_controller=update_controller,
        emergency_brake=emergency_brake,
    )
