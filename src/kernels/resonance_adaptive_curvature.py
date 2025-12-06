"""
Resonance Adaptive Curvature Optimizer - Phase 8 Optimization

ResonanceDetectorの曲率調整を訓練ループに統合し、
共鳴状態に応じて双曲空間の曲率を動的に調整。

効果: 訓練安定性向上 + NaN発生率低下
適用: 全双曲コンポーネント（BK-Core, AR-SSM, Hyperbolic Attention）
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
import math


class ResonanceAdaptiveCurvature:
    """
    Adaptive curvature optimizer based on resonance detection.
    
    Monitors G_ii (Green function diagonal) to detect resonance states
    and adjusts curvature parameters across all hyperbolic components.
    
    Resonance detection:
    - High |G_ii.real| → Strong resonance → Reduce curvature (flatten space)
    - Low |G_ii.real| → Weak resonance → Increase curvature (curve space)
    
    Usage:
        curvature_opt = ResonanceAdaptiveCurvature(model)
        
        for step in training:
            G_ii = model.get_green_function(x)
            curvature_opt.step(G_ii)
    """
    
    def __init__(
        self,
        model: nn.Module,
        initial_curvature: float = 1.0,
        min_curvature: float = 0.1,
        max_curvature: float = 2.0,
        resonance_threshold: float = 0.5,
        adjustment_rate: float = 0.01,
        momentum: float = 0.9,
        warmup_steps: int = 100
    ):
        """
        Args:
            model: Model containing hyperbolic components
            initial_curvature: Starting curvature value
            min_curvature: Minimum allowed curvature
            max_curvature: Maximum allowed curvature
            resonance_threshold: Threshold for resonance detection
            adjustment_rate: Rate of curvature adjustment
            momentum: Momentum for smooth adjustment
            warmup_steps: Steps before enabling adjustment
        """
        self.model = model
        self.curvature = initial_curvature
        self.min_curvature = min_curvature
        self.max_curvature = max_curvature
        self.resonance_threshold = resonance_threshold
        self.adjustment_rate = adjustment_rate
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        
        self.step_count = 0
        self.velocity = 0.0
        self.resonance_history: List[float] = []
        self.curvature_history: List[float] = []
        
        # Find all curvature parameters in model
        self.curvature_params = self._find_curvature_params()
    
    def _find_curvature_params(self) -> List[Tuple[str, nn.Module]]:
        """Find all modules with curvature parameters."""
        params = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'curvature'):
                params.append((name, module))
            elif hasattr(module, 'c') and isinstance(getattr(module, 'c'), (float, torch.Tensor)):
                params.append((name, module))
        return params
    
    def compute_resonance(self, G_ii: torch.Tensor) -> float:
        """
        Compute resonance score from G_ii.
        
        Args:
            G_ii: Green function diagonal (complex or real tensor)
        
        Returns:
            resonance: Scalar resonance score
        """
        with torch.no_grad():
            if torch.is_complex(G_ii):
                # Use real part for resonance detection
                resonance = G_ii.real.abs().mean().item()
            else:
                resonance = G_ii.abs().mean().item()
        
        return resonance
    
    def step(self, G_ii: torch.Tensor) -> Dict[str, float]:
        """
        Perform one optimization step.
        
        Args:
            G_ii: Green function diagonal
        
        Returns:
            diagnostics: Dict with resonance and curvature info
        """
        self.step_count += 1
        
        # Compute resonance
        resonance = self.compute_resonance(G_ii)
        self.resonance_history.append(resonance)
        
        # Keep only recent history
        if len(self.resonance_history) > 1000:
            self.resonance_history = self.resonance_history[-100:]
        
        # Skip adjustment during warmup
        if self.step_count < self.warmup_steps:
            self.curvature_history.append(self.curvature)
            return {
                'resonance': resonance,
                'curvature': self.curvature,
                'adjusted': False
            }
        
        # Compute adjustment
        if resonance > self.resonance_threshold:
            # High resonance → reduce curvature
            target_adjustment = -self.adjustment_rate
        elif resonance < self.resonance_threshold * 0.5:
            # Low resonance → increase curvature
            target_adjustment = self.adjustment_rate
        else:
            # Normal range → no adjustment
            target_adjustment = 0.0
        
        # Apply momentum
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * target_adjustment
        
        # Update curvature
        new_curvature = self.curvature * (1 + self.velocity)
        new_curvature = max(self.min_curvature, min(self.max_curvature, new_curvature))
        
        adjusted = abs(new_curvature - self.curvature) > 1e-6
        self.curvature = new_curvature
        self.curvature_history.append(self.curvature)
        
        # Update model curvature parameters
        if adjusted:
            self._update_model_curvature()
        
        return {
            'resonance': resonance,
            'curvature': self.curvature,
            'velocity': self.velocity,
            'adjusted': adjusted
        }
    
    def _update_model_curvature(self):
        """Update curvature in all model components."""
        for name, module in self.curvature_params:
            if hasattr(module, 'curvature'):
                if isinstance(module.curvature, nn.Parameter):
                    module.curvature.data.fill_(self.curvature)
                else:
                    module.curvature = self.curvature
            elif hasattr(module, 'c'):
                if isinstance(module.c, nn.Parameter):
                    module.c.data.fill_(self.curvature)
                else:
                    module.c = self.curvature
    
    def get_stats(self) -> Dict[str, float]:
        """Get optimization statistics."""
        if not self.resonance_history:
            return {}
        
        recent_resonance = self.resonance_history[-100:]
        recent_curvature = self.curvature_history[-100:] if self.curvature_history else [self.curvature]
        
        return {
            'mean_resonance': sum(recent_resonance) / len(recent_resonance),
            'max_resonance': max(recent_resonance),
            'min_resonance': min(recent_resonance),
            'current_curvature': self.curvature,
            'mean_curvature': sum(recent_curvature) / len(recent_curvature),
            'curvature_std': (sum((c - sum(recent_curvature)/len(recent_curvature))**2 for c in recent_curvature) / len(recent_curvature)) ** 0.5,
        }
    
    def state_dict(self) -> Dict:
        """Return state for checkpointing."""
        return {
            'curvature': self.curvature,
            'velocity': self.velocity,
            'step_count': self.step_count,
            'resonance_history': self.resonance_history[-100:],
            'curvature_history': self.curvature_history[-100:],
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint."""
        self.curvature = state_dict['curvature']
        self.velocity = state_dict['velocity']
        self.step_count = state_dict['step_count']
        self.resonance_history = state_dict.get('resonance_history', [])
        self.curvature_history = state_dict.get('curvature_history', [])
        self._update_model_curvature()


class StabilityMonitor:
    """
    Monitor training stability and detect potential issues.
    
    Tracks:
    - Loss trends
    - Gradient norms
    - Resonance levels
    - NaN/Inf occurrences
    
    Can trigger automatic adjustments or early stopping.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        nan_threshold: int = 5,
        exploding_grad_threshold: float = 100.0,
        loss_spike_threshold: float = 10.0
    ):
        self.window_size = window_size
        self.nan_threshold = nan_threshold
        self.exploding_grad_threshold = exploding_grad_threshold
        self.loss_spike_threshold = loss_spike_threshold
        
        self.loss_history: List[float] = []
        self.grad_norm_history: List[float] = []
        self.nan_count = 0
        self.warnings: List[str] = []
    
    def update(
        self,
        loss: float,
        grad_norm: float,
        had_nan: bool = False
    ) -> Dict[str, any]:
        """
        Update monitor with current step info.
        
        Returns:
            status: Dict with stability status and any warnings
        """
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        
        if had_nan:
            self.nan_count += 1
        
        # Keep only recent history
        if len(self.loss_history) > self.window_size * 2:
            self.loss_history = self.loss_history[-self.window_size:]
            self.grad_norm_history = self.grad_norm_history[-self.window_size:]
        
        # Check for issues
        issues = []
        
        # Check NaN frequency
        if self.nan_count >= self.nan_threshold:
            issues.append(f"High NaN frequency: {self.nan_count}")
        
        # Check gradient explosion
        if grad_norm > self.exploding_grad_threshold:
            issues.append(f"Exploding gradient: {grad_norm:.2f}")
        
        # Check loss spike
        if len(self.loss_history) > 10:
            recent_mean = sum(self.loss_history[-10:]) / 10
            older_mean = sum(self.loss_history[-20:-10]) / 10 if len(self.loss_history) >= 20 else recent_mean
            if recent_mean > older_mean * self.loss_spike_threshold:
                issues.append(f"Loss spike: {recent_mean:.4f} vs {older_mean:.4f}")
        
        # Compute stability score (0 = unstable, 1 = stable)
        stability_score = 1.0
        if issues:
            stability_score = max(0.0, 1.0 - len(issues) * 0.3)
        
        return {
            'stable': len(issues) == 0,
            'stability_score': stability_score,
            'issues': issues,
            'nan_count': self.nan_count,
            'mean_grad_norm': sum(self.grad_norm_history[-10:]) / max(1, len(self.grad_norm_history[-10:])),
        }
    
    def reset_nan_count(self):
        """Reset NaN counter after recovery."""
        self.nan_count = 0
    
    def should_reduce_lr(self) -> bool:
        """Check if LR should be reduced for stability."""
        if len(self.grad_norm_history) < 20:
            return False
        
        recent = self.grad_norm_history[-10:]
        older = self.grad_norm_history[-20:-10]
        
        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)
        
        return recent_mean > older_mean * 2.0
