"""
Gradient Feeder V2.1 - Hybrid Control (Explosion vs Vanishing)

V2 flaw identified:
- Adjusting clipping threshold can REDUCE big gradients
- But it CANNOT INCREASE small gradients!
- If grad=0.05, threshold=500 still gives grad=0.05

V2.1 Solution - Hybrid Control:
1. EXPLOSION (grad > target): Threshold adjustment (V2 approach)
2. VANISHING (grad < target): Emergency scaling (V1 approach, surgical)

V2.2 CUDA Optimization:
- Uses C++/CUDA extension for zero-overhead scaling
- Falls back to torch._foreach_mul_ if CUDA extension not available

Author: ResNet-BK Project
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import math

# Try to import C++ extension for maximum performance
# Priority: Pre-built C++ extension (instant load) > JIT-compiled version > PyTorch fallback
_CPP_FEEDER_AVAILABLE = False
_CUDA_FEEDER_AVAILABLE = False  # Initialize to prevent NameError
_cpp_ext = None
_cuda_ext = None

# 1. Try pre-built C++ extension first (NO startup delay!)
try:
    import torch  # Ensure libc10.so is loaded
    import sys
    import os
    # Add kernels directory to path for pre-built .so
    _kernels_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _kernels_path = os.path.join(_kernels_dir, 'kernels')
    if _kernels_path not in sys.path:
        sys.path.insert(0, _kernels_path)
    
    import gradient_feeder_cpp as _cpp_ext
    _CPP_FEEDER_AVAILABLE = True
    # Create compatibility wrapper
    _cuda_ext = type('CppExt', (), {'scale_all_gradients': _cpp_ext.scale_all_gradients})()
    _CUDA_FEEDER_AVAILABLE = True  # Keep old name for compatibility
except ImportError:
    pass

# 2. Try JIT-compiled CUDA version (has ~2-3s startup delay)
if not _CPP_FEEDER_AVAILABLE:
    try:
        from src.kernels.gradient_feeder_jit import (
            scale_all_gradients_cuda as _cuda_scale,
            CUDA_AVAILABLE as _jit_available
        )
        if _jit_available:
            _CUDA_FEEDER_AVAILABLE = True
            _cuda_ext = type('CudaExt', (), {'scale_all_gradients': _cuda_scale})()
    except ImportError:
        pass

# 3. Try pre-built CUDA extension (legacy fallback)
if not _CPP_FEEDER_AVAILABLE and not _CUDA_FEEDER_AVAILABLE:
    try:
        import gradient_feeder_cuda as _cuda_ext
        _CUDA_FEEDER_AVAILABLE = True
    except ImportError:
        pass


@dataclass  
class FeederV21Stats:
    """Statistics from V2.1 hybrid gradient feeding."""
    grad_norm_input: float
    grad_norm_output: float  # After any modifications
    velocity: float
    predicted_next: float
    clip_threshold: float
    scale_factor: float  # 1.0 = no scaling, >1.0 = boost
    action: str  # "threshold_lower", "emergency_scale", "hold"
    health_score: float


class GradientFeederV21:
    """
    Gradient Feeder V2.1 - Hybrid Control
    
    Two separate control mechanisms:
    1. Upper Bound (Explosion): Adjust clipping THRESHOLD
       - velocity > 0 (rising) → lower threshold (restrict)
       - Works because clip_grad_norm_ can reduce gradients
       
    2. Lower Bound (Vanishing): Emergency SCALING
       - velocity < 0 (falling) AND grad < critical → scale UP
       - This is the V1 approach, but ONLY in emergencies
       - Preserves direction by scaling ALL gradients equally
       - Only used when gradient is dying (< 0.3)
    
    Why hybrid?
    - Threshold adjustment can't increase gradients
    - Scaling can increase gradients but risks direction corruption
    - Solution: Use each tool for what it's good at
    """
    
    def __init__(
        self,
        # Target range
        target_low: float = 0.5,
        target_high: float = 3.0,
        
        # Threshold control (for explosion)
        initial_threshold: float = 50.0,
        min_threshold: float = 5.0,
        max_threshold: float = 200.0,
        
        # Scaling control (for vanishing)
        max_scale: float = 3.0,       # Never scale more than 3x
        critical_threshold: float = 0.2,  # Below this = critical
        emergency_threshold: float = 0.1,  # Below this = emergency
        
        # Dynamics
        reaction_speed: float = 0.4,
        prediction_weight: float = 0.6,
        history_size: int = 5,
    ):
        self.target_low = target_low
        self.target_high = target_high
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.max_scale = max_scale
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        self.reaction_speed = reaction_speed
        self.prediction_weight = prediction_weight
        
        # State
        self.clip_threshold = initial_threshold
        self.scale_factor = 1.0
        self.history = deque(maxlen=history_size)
        self.velocity = 0.0
        self.step_count = 0
        self.consecutive_low = 0
        
    def feed(self, grad_norm: float) -> Tuple[float, float, FeederV21Stats]:
        """
        Compute clipping threshold AND scale factor.
        
        Returns:
            clip_threshold: For torch.nn.utils.clip_grad_norm_()
            scale_factor: For manual gradient scaling (1.0 = no scaling)
            stats: Diagnostic information
        """
        self.step_count += 1
        action = "hold"
        
        # === Velocity Calculation ===
        if len(self.history) >= 1:
            self.velocity = grad_norm - self.history[-1]
        else:
            self.velocity = 0.0
        
        self.history.append(grad_norm)
        
        # === Prediction ===
        predicted = grad_norm + self.velocity * self.prediction_weight
        
        # === Health Assessment ===
        if grad_norm < self.emergency_threshold:
            health = 0.0
            self.consecutive_low += 1
        elif grad_norm < self.critical_threshold:
            health = 0.2
            self.consecutive_low += 1
        elif grad_norm < self.target_low:
            health = 0.5 + 0.3 * (grad_norm / self.target_low)
            self.consecutive_low = 0
        elif grad_norm <= self.target_high:
            health = 1.0
            self.consecutive_low = 0
        else:
            health = max(0.3, 1.0 - (grad_norm - self.target_high) / (self.target_high * 2))
            self.consecutive_low = 0
        
        # ============================================
        # UPPER BOUND CONTROL (Explosion Prevention)
        # ============================================
        if grad_norm > self.target_high or predicted > self.target_high:
            # Gradient too high or rising towards danger
            # Action: LOWER the clipping threshold
            excess = max(grad_norm, predicted) / self.target_high
            reduction = 1.0 - self.reaction_speed * (excess - 1.0) * 0.5
            reduction = max(reduction, 0.7)
            self.clip_threshold *= reduction
            action = "threshold_lower"
            self.scale_factor = 1.0  # No scaling when exploding
            
        # ============================================
        # LOWER BOUND CONTROL (Vanishing Prevention)
        # ============================================
        elif grad_norm < self.critical_threshold or predicted < self.emergency_threshold:
            # Gradient too low or predicted to die
            # Action: EMERGENCY SCALING (V1 approach, but surgical)
            
            if grad_norm < self.emergency_threshold:
                # EMERGENCY: Aggressive boost
                target_grad = self.target_low * 1.5  # Aim for healthy level
                needed_boost = target_grad / (grad_norm + 1e-8)
                self.scale_factor = min(needed_boost, self.max_scale)
                action = "EMERGENCY_SCALE"
                
            elif grad_norm < self.critical_threshold:
                # Critical but not emergency: Moderate boost
                deficit_ratio = self.target_low / (grad_norm + 1e-8)
                self.scale_factor = 1.0 + (deficit_ratio - 1.0) * 0.5
                self.scale_factor = min(self.scale_factor, self.max_scale * 0.7)
                action = "scale_boost"
                
            elif predicted < self.critical_threshold and self.velocity < 0:
                # Currently OK but predicted to drop: Pre-emptive boost
                self.scale_factor = 1.0 + (-self.velocity / grad_norm) * 2.0
                self.scale_factor = min(self.scale_factor, 2.0)
                action = "preemptive_scale"
            else:
                self.scale_factor = 1.0
                
            # Don't lower threshold when vanishing
            self.clip_threshold = min(self.clip_threshold * 1.05, self.max_threshold)
            
        # ============================================
        # HEALTHY RANGE
        # ============================================
        else:
            # In healthy range: gradually normalize
            self.scale_factor = self.scale_factor * 0.9 + 1.0 * 0.1
            self.clip_threshold = self.clip_threshold * 0.98 + 50.0 * 0.02
            action = "hold"
        
        # Clamp values
        self.clip_threshold = max(self.min_threshold, min(self.max_threshold, self.clip_threshold))
        self.scale_factor = max(1.0, min(self.max_scale, self.scale_factor))  # min 1.0 = never reduce
        
        # Compute expected output
        expected_output = grad_norm * self.scale_factor
        
        stats = FeederV21Stats(
            grad_norm_input=grad_norm,
            grad_norm_output=expected_output,
            velocity=self.velocity,
            predicted_next=predicted,
            clip_threshold=self.clip_threshold,
            scale_factor=self.scale_factor,
            action=action,
            health_score=health,
        )
        
        return self.clip_threshold, self.scale_factor, stats
    
    def apply_scaling(self, model: nn.Module, scale: float):
        """
        Apply scale factor to all gradients.
        Only called when scale > 1.0 (emergency boost).
        
        OPTIMIZED (V2.2):
        - Priority 1: C++/CUDA extension (zero Python overhead)
        - Priority 2: torch._foreach_mul_ (single GPU kernel)
        """
        if scale <= 1.0 + 1e-6:
            return  # No scaling needed
        
        # Collect all gradients (no GPU ops yet)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        
        if len(grads) == 0:
            return
        
        # Use CUDA extension if available (fastest)
        if _CUDA_FEEDER_AVAILABLE:
            _cuda_ext.scale_all_gradients(grads, scale)
        else:
            # Fallback: PyTorch fused operation
            torch._foreach_mul_(grads, scale)
    
    def get_status(self) -> str:
        """Human-readable status."""
        return (f"GradFeederV2.1[thr={self.clip_threshold:.1f}, "
                f"scale={self.scale_factor:.2f}, v={self.velocity:+.3f}]")


# Keep V2 for backward compatibility
class GradientFeederV2(GradientFeederV21):
    """Alias for V2.1 (backward compatible)"""
    pass


def test_gradient_feeder_v21():
    """Test V2.1 with real gradient pattern."""
    print("Testing GradientFeederV2.1 - Hybrid Control")
    print("="*80)
    
    feeder = GradientFeederV21(
        target_low=0.5,
        target_high=3.0,
        reaction_speed=0.4,
    )
    
    # Real pattern from training_log.json
    simulated_grads = [
        0.63, 0.66, 0.68, 0.68, 0.81,  # Healthy start
        0.36,  # Drop!
        0.81, 0.49, 0.75,  # Recovery
        0.22,  # Drop below critical!
        0.87, 1.10, 1.06, 0.87,  # Healthy
        0.46, 0.24,  # Decay pattern
        0.058,  # EMERGENCY!
        0.25, 0.45,  # Recovery attempt
        5.0, 6.0,  # EXPLOSION test
    ]
    
    print(f"\n{'Step':<5} {'Grad':<8} {'Vel':<8} {'Pred':<8} {'Thresh':<8} {'Scale':<8} {'Output':<8} {'Action':<18} {'Health':<8}")
    print("-"*80)
    
    for i, grad in enumerate(simulated_grads):
        threshold, scale, stats = feeder.feed(grad)
        
        # Indicators
        if "EMERGENCY" in stats.action:
            indicator = "🚨"
        elif stats.health_score >= 0.8:
            indicator = "✅"
        elif stats.health_score >= 0.5:
            indicator = "⚠️"
        else:
            indicator = "❌"
        
        print(f"{i+1:<5} {grad:<8.3f} {stats.velocity:<+8.3f} {stats.predicted_next:<8.3f} "
              f"{threshold:<8.1f} {scale:<8.2f} {stats.grad_norm_output:<8.2f} "
              f"{stats.action:<18} {stats.health_score:<8.2f} {indicator}")
    
    print("-"*80)
    print("\nV2.1 Hybrid Control:")
    print("  ✓ EXPLOSION: Threshold adjustment (clips high gradients)")
    print("  ✓ VANISHING: Emergency scaling (boosts dying gradients)")
    print("  ✓ Each mechanism handles what it's good at!")
    print("="*80)
    
    return True


if __name__ == "__main__":
    test_gradient_feeder_v21()
