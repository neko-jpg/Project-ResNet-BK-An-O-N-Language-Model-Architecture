"""
Orthogonalization Stabilizer - Safe Newton-Schulz Implementation

Enhanced Newton-Schulz iteration with numerical stability guarantees
for Muon optimizer's gradient orthogonalization.
"""

import torch
import math
from typing import Optional, Tuple


def stable_zeropower_via_newtonschulz(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    adaptive_eps: bool = True,
    monitor_convergence: bool = True,
    max_norm_ratio: float = 10.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Stabilized Newton-Schulz iteration to compute zeroth power (orthogonal matrix).
    
    Original Muon implementation has several instability sources:
    1. Fixed eps=1e-7 too small for large gradients
    2. No NaN/Inf checks during iteration
    3. bfloat16 conversion loses precision at critical steps
    4. No convergence monitoring
    
    This version addresses all of these issues.
    
    Args:
        G: Input gradient matrix (2D)
        steps: Number of Newton-Schulz iterations
        eps: Base epsilon for normalization
        adaptive_eps: If True, scale eps based on gradient magnitude
        monitor_convergence: If True, track iteration error and early stop
        max_norm_ratio: Maximum allowed norm ratio before clamping
        
    Returns:
        X: Orthogonalized matrix
        metrics: Dictionary of convergence metrics
    """
    assert len(G.shape) == 2, "Input must be 2D matrix"
    
    metrics = {
        'iterations_used': steps,
        'converged': False,
        'early_stopped': False,
        'nan_detected': False,
        'final_error': 0.0,
    }
    
    # Adaptive epsilon based on gradient magnitude
    if adaptive_eps:
        grad_magnitude = G.abs().mean().item()
        if grad_magnitude > 1.0:
            # Large gradients need larger eps for stability
            eps = max(eps, 1e-4)
            metrics['adaptive_eps_used'] = eps
        elif grad_magnitude < 0.01:
            # Small gradients can use smaller eps
            eps = min(eps, 1e-6)
            metrics['adaptive_eps_used'] = eps
    
    # Newton-Schulz coefficients (pre-computed)
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # CRITICAL: Keep computation in FP32 for numerical stability
    # Only convert to bfloat16 at the very end
    X = G.float()  # Changed from G.bfloat16()
    
    # Compute initial norm with safety checks
    initial_norm = X.norm()
    if torch.isnan(initial_norm) or torch.isinf(initial_norm):
        metrics['nan_detected'] = True
        # Fallback: Return normalized identity-like matrix
        return torch.eye(G.size(0), G.size(1), device=G.device, dtype=G.dtype), metrics
    
    # Normalize: X /= (X.norm() + eps)
    # Add clamping to prevent extreme normalization
    norm_divisor = torch.clamp(initial_norm, min=eps, max=initial_norm * max_norm_ratio)
    X = X / (norm_divisor + eps)
    
    # Handle transpose for tall matrices
    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
    
    # Newton-Schulz iterations with convergence monitoring
    prev_error = float('inf')
    for iter_idx in range(steps):
        # Compute A = X @ X.T
        A = X @ X.T
        
        # NaN/Inf check after each critical operation
        if torch.isnan(A).any() or torch.isinf(A).any():
            metrics['nan_detected'] = True
            metrics['iterations_used'] = iter_idx
            # Fallback to previous X (before this iteration)
            break
        
        # Compute B = b * A + c * A @ A
        B = b * A + c * (A @ A)
        
        # NaN/Inf check
        if torch.isnan(B).any() or torch.isinf(B).any():
            metrics['nan_detected'] = True
            metrics['iterations_used'] = iter_idx
            break
        
        # Update: X = a * X + B @ X
        X_new = a * X + B @ X
        
        # NaN/Inf check
        if torch.isnan(X_new).any() or torch.isinf(X_new).any():
            metrics['nan_detected'] = True
            metrics['iterations_used'] = iter_idx
            break
        
        # Convergence monitoring (optional, adds overhead)
        if monitor_convergence:
            # Error: ||X_new - X||_F
            error = (X_new - X).norm().item()
            metrics['final_error'] = error
            
            # Check convergence (error decreasing and small)
            if error < 1e-6:
                metrics['converged'] = True
                metrics['iterations_used'] = iter_idx + 1
                X = X_new
                break
            
            # Check divergence (error increasing)
            if error > prev_error * 1.5:
                # Diverging, stop early
                metrics['early_stopped'] = True
                metrics['iterations_used'] = iter_idx
                # Don't update X, use previous iteration
                break
            
            prev_error = error
        
        X = X_new
    
    # Transpose back if needed
    if transposed:
        X = X.T
    
    # === POST-ORTHOGONALIZATION SCALING ===
    # The Newton-Schulz method normalizes singular values to ~1.0, but the
    # Frobenius norm ||X||_F ≈ sqrt(min(m,n)) can still be large.
    # For large matrices (e.g., 4096×4096), ||X||_F ≈ 64, causing gradient explosion.
    # 
    # Solution: Scale by matrix dimensions to keep updates reasonable
    # Target: ||X_scaled||_F ≈ 1.0 regardless of matrix size
    scaling_factor = 1.0 / math.sqrt(min(G.size(0), G.size(1)))
    X = X * scaling_factor
    
    # Track scaling in metrics
    metrics['scaling_factor'] = scaling_factor
    
    # Convert back to original dtype
    X = X.to(G.dtype)
    
    return X, metrics


def safe_orthogonalize_gradient(
    grad: torch.Tensor,
    ns_steps: int = 5,
    warmup_mode: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """
    High-level wrapper for safe gradient orthogonalization.
    
    Automatically handles different gradient shapes and applies
    appropriate stabilization strategies.
    
    Args:
        grad: Gradient tensor (any shape)
        ns_steps: Number of Newton-Schulz iterations
        warmup_mode: If True, use extra-conservative settings
        
    Returns:
        ortho_grad: Orthogonalized gradient
        metrics: Orthogonalization metrics
    """
    # For 1D or 0D gradients, just normalize
    if grad.ndim < 2:
        norm = grad.norm()
        if norm < 1e-8 or torch.isnan(norm) or torch.isinf(norm):
            # Degenerate case
            return torch.zeros_like(grad), {'1d_fallback': True}
        return grad / norm, {'1d_normalized': True}
    
    # For 3D+ tensors, reshape to 2D, orthogonalize, then reshape back
    original_shape = grad.shape
    if grad.ndim > 2:
        # Reshape to 2D: (first_dim, rest)
        grad_2d = grad.view(grad.size(0), -1)
        metrics_prefix = {'reshaped_from': list(original_shape)}
    else:
        grad_2d = grad
        metrics_prefix = {}
    
    # For 2D gradients, use stabilized Newton-Schulz
    if warmup_mode:
        # Warmup: Extra conservative
        ortho_grad, metrics = stable_zeropower_via_newtonschulz(
            grad_2d,
            steps=ns_steps,
            eps=1e-4,  # Safer epsilon
            adaptive_eps=True,
            monitor_convergence=True,
            max_norm_ratio=5.0,  # Stricter clamping
        )
    else:
        # Normal: Balanced
        ortho_grad, metrics = stable_zeropower_via_newtonschulz(
            grad_2d,
            steps=ns_steps,
            eps=1e-7,
            adaptive_eps=True,
            monitor_convergence=True,
            max_norm_ratio=10.0,
        )
    
    # Reshape back to original shape if needed
    if grad.ndim > 2:
        ortho_grad = ortho_grad.view(original_shape)
    
    metrics.update(metrics_prefix)
    return ortho_grad, metrics


class OrthogonalizationStabilizer:
    """
    Stateful wrapper for gradient orthogonalization with monitoring.
    
    Tracks orthogonalization health over time and adjusts parameters
    dynamically.
    """
    
    def __init__(
        self,
        base_ns_steps: int = 5,
        warmup_steps: int = 2000,
        track_health: bool = True,
    ):
        self.base_ns_steps = base_ns_steps
        self.warmup_steps = warmup_steps
        self.track_health = track_health
        
        self.step_count = 0
        self.nan_count = 0
        self.convergence_failures = 0
        
    def orthogonalize(self, grad: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Orthogonalize gradient with adaptive settings.
        
        Args:
            grad: Gradient tensor
            
        Returns:
            ortho_grad: Orthogonalized gradient
            metrics: Orthogonalization metrics
        """
        self.step_count += 1
        
        # Determine if in warmup
        in_warmup = self.step_count <= self.warmup_steps
        
        # Adaptive NS steps
        # Warmup: More iterations for stronger orthogonalization
        # Post-warmup: Fewer iterations for efficiency
        ns_steps = self.base_ns_steps + 5 if in_warmup else self.base_ns_steps
        
        # Orthogonalize
        ortho_grad, metrics = safe_orthogonalize_gradient(
            grad,
            ns_steps=ns_steps,
            warmup_mode=in_warmup,
        )
        
        # Track health
        if self.track_health:
            if metrics.get('nan_detected', False):
                self.nan_count += 1
            if metrics.get('early_stopped', False):
                self.convergence_failures += 1
        
        # Add global metrics
        metrics['global_step'] = self.step_count
        metrics['total_nan_count'] = self.nan_count
        metrics['total_convergence_failures'] = self.convergence_failures
        
        return ortho_grad, metrics
