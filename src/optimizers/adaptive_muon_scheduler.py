"""
Adaptive Muon Scheduler - Dynamic Parameter Adjustment

Dynamically adjusts Muon optimizer parameters during training
to maintain stability during warmup and maximize performance
during stable training.
"""

import torch
from typing import Dict, Optional


class AdaptiveMuonScheduler:
    """
    Scheduler for Muon optimizer parameters.
    
    Adjusts the following based on training progress:
    - Newton-Schulz iterations (ns_steps)
    - Momentum coefficient
    - Epsilon for normalization
    - AdamW learning rate for 1D params
    
    Training Phases:
    1. Warmup (0-2000 steps): Conservative settings for stability
    2. Ramp-up (2000-4000 steps): Gradual transition to full performance
    3. Stable (>4000 steps): Full performance settings
    """
    
    def __init__(
        self,
        warmup_steps: int = 2000,
        rampup_steps: int = 2000,
        base_ns_steps: int = 5,
        base_momentum: float = 0.95,
        base_eps: float = 1e-7,
        base_adamw_lr: float = 1e-4,
        health_tracking: bool = True,
    ):
        """
        Args:
            warmup_steps: Number of warmup steps (conservative phase)
            rampup_steps: Number of ramp-up steps (transition phase)
            base_ns_steps: Base Newton-Schulz iterations (stable phase)
            base_momentum: Base momentum (stable phase)
            base_eps: Base epsilon (stable phase)
            base_adamw_lr: Base AdamW LR for 1D params (stable phase)
            health_tracking: Track gradient health metrics
        """
        self.warmup_steps = warmup_steps
        self.rampup_steps = rampup_steps
        self.base_ns_steps = base_ns_steps
        self.base_momentum = base_momentum
        self.base_eps = base_eps
        self.base_adamw_lr = base_adamw_lr
        self.health_tracking = health_tracking
        
        # State tracking
        self.global_step = 0
        self.phase = "warmup"  # warmup, rampup, stable
        
        # Health metrics
        self.recent_grad_norms = []
        self.max_history = 100
        self.nan_count = 0
        self.skip_count = 0
        
    def step(self, metrics: Optional[Dict] = None) -> Dict[str, float]:
        """
        Advance scheduler by one step and return current parameters.
        
        Args:
            metrics: Optional metrics from current training step
                - grad_norm: Gradient norm
                - nan_detected: Whether NaN was detected
                - update_skipped: Whether update was skipped
                
        Returns:
            Scheduler parameters for current step
        """
        self.global_step += 1
        
        # Update health tracking
        if self.health_tracking and metrics:
            if 'grad_norm' in metrics:
                self.recent_grad_norms.append(metrics['grad_norm'])
                if len(self.recent_grad_norms) > self.max_history:
                    self.recent_grad_norms.pop(0)
            
            if metrics.get('nan_detected', False):
                self.nan_count += 1
            
            if metrics.get('update_skipped', False):
                self.skip_count += 1
        
        # Determine current phase
        if self.global_step <= self.warmup_steps:
            self.phase = "warmup"
        elif self.global_step <= self.warmup_steps + self.rampup_steps:
            self.phase = "rampup"
        else:
            self.phase = "stable"
        
        # Compute parameters based on phase
        return self._get_phase_parameters()
    
    def _get_phase_parameters(self) -> Dict[str, float]:
        """
        Get optimizer parameters for current phase.
        
        Returns:
            Dictionary of parameters
        """
        if self.phase == "warmup":
            return self._warmup_parameters()
        elif self.phase == "rampup":
            return self._rampup_parameters()
        else:
            return self._stable_parameters()
    
    def _warmup_parameters(self) -> Dict[str, float]:
        """
        Conservative parameters for warmup phase.
        
        Goals:
        - Prevent NaN/Inf during initialization
        - Strong orthogonalization (more NS iterations)
        - Reduced momentum (less aggressive updates)
        - Larger epsilon (safer normalization)
        """
        return {
            'ns_steps': self.base_ns_steps + 5,  # 5 -> 10 (stronger orthogonalization)
            'momentum': self.base_momentum - 0.10,  # 0.95 -> 0.85 (less aggressive)
            'eps': 1e-4,  # 1e-7 -> 1e-4 (safer)
            'adamw_lr': self.base_adamw_lr * 0.5,  # Reduce AdamW LR too
            'phase': 'warmup',
            'global_step': self.global_step,
        }
    
    def _rampup_parameters(self) -> Dict[str, float]:
        """
        Transition parameters for ramp-up phase.
        
        Linear interpolation from warmup to stable settings.
        """
        # Progress through ramp-up (0.0 to 1.0)
        progress = (self.global_step - self.warmup_steps) / self.rampup_steps
        progress = max(0.0, min(1.0, progress))
        
        # Linear interpolation
        warmup_params = self._warmup_parameters()
        stable_params = self._stable_parameters()
        
        # Interpolate NS steps (discrete, round to int)
        ns_steps = int(
            warmup_params['ns_steps'] * (1 - progress) +
            stable_params['ns_steps'] * progress
        )
        
        return {
            'ns_steps': ns_steps,
            'momentum': warmup_params['momentum'] * (1 - progress) + stable_params['momentum'] * progress,
            'eps': warmup_params['eps'] * (1 - progress) + stable_params['eps'] * progress,
            'adamw_lr': warmup_params['adamw_lr'] * (1 - progress) + stable_params['adamw_lr'] * progress,
            'phase': 'rampup',
            'progress': progress,
            'global_step': self.global_step,
        }
    
    def _stable_parameters(self) -> Dict[str, float]:
        """
        Full performance parameters for stable phase.
        
        Goals:
        - Maximum training efficiency
        - Standard NS iterations
        - Full momentum
        - Adaptive epsilon based on health
        """
        # Adaptive epsilon based on recent gradient health
        adaptive_eps = self.base_eps
        if self.health_tracking and self.recent_grad_norms:
            avg_grad_norm = sum(self.recent_grad_norms) / len(self.recent_grad_norms)
            
            # If gradients are consistently large, use larger eps
            if avg_grad_norm > 5.0:
                adaptive_eps = 1e-6
            # If gradients are very small, use smaller eps
            elif avg_grad_norm < 0.1:
                adaptive_eps = 1e-8
        
        return {
            'ns_steps': self.base_ns_steps,
            'momentum': self.base_momentum,
            'eps': adaptive_eps,
            'adamw_lr': self.base_adamw_lr,
            'phase': 'stable',
            'global_step': self.global_step,
            'avg_grad_norm': sum(self.recent_grad_norms) / len(self.recent_grad_norms) if self.recent_grad_norms else 0.0,
        }
    
    def get_health_report(self) -> Dict[str, float]:
        """
        Get health report for monitoring.
        
        Returns:
            Dictionary of health metrics
        """
        report = {
            'global_step': self.global_step,
            'phase': self.phase,
            'nan_count': self.nan_count,
            'skip_count': self.skip_count,
        }
        
        if self.recent_grad_norms:
            report['avg_grad_norm'] = sum(self.recent_grad_norms) / len(self.recent_grad_norms)
            report['max_grad_norm'] = max(self.recent_grad_norms)
            report['min_grad_norm'] = min(self.recent_grad_norms)
        
        return report
    
    def should_use_aggressive_damping(self) -> bool:
        """
        Determine if aggressive gradient damping should be used.
        
        Returns True during warmup or if recent health is poor.
        """
        if self.phase == "warmup":
            return True
        
        # Check recent health
        if self.health_tracking and self.recent_grad_norms:
            recent_window = self.recent_grad_norms[-10:]  # Last 10 steps
            if len(recent_window) >= 5:
                avg_recent = sum(recent_window) / len(recent_window)
                # If gradients are exploding, use aggressive damping
                if avg_recent > 10.0:
                    return True
        
        return False


def create_adaptive_muon_scheduler(
    warmup_steps: int = 2000,
    **kwargs
) -> AdaptiveMuonScheduler:
    """
    Factory function to create AdaptiveMuonScheduler.
    
    Args:
        warmup_steps: Number of warmup steps (from config)
        **kwargs: Additional arguments to override defaults
        
    Returns:
        AdaptiveMuonScheduler instance
    """
    defaults = {
        'warmup_steps': warmup_steps,
        'rampup_steps': 2000,
        'base_ns_steps': 5,
        'base_momentum': 0.95,
        'base_eps': 1e-7,
        'base_adamw_lr': 1e-4,
        'health_tracking': True,
    }
    
    defaults.update(kwargs)
    
    return AdaptiveMuonScheduler(**defaults)
