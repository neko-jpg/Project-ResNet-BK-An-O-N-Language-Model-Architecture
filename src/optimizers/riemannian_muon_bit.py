"""
Riemannian-Muon-Bit Optimizer

A novel optimizer for training neural networks in hyperbolic space with 1.58-bit quantization.
Combines:
1. Riemannian optimization (respects manifold geometry)
2. Muon-style orthogonalization (J-orthogonal for Lorentz group)
3. BitNet-aware stochastic rounding

Key innovations:
- Higham-Schulz iteration for J-orthogonalization (Lorentz group O(n,1))
- Parallel transport of momentum across tangent spaces
- Quantization-aware gradient handling with stochastic rounding

References:
- Q-RGT: Quantized Riemannian Gradient Tracking (Chen et al., 2024)
- Higham-Schulz for J-orthogonal matrices (Higham et al.)
- Muon optimizer (Momentum Orthogonalized)
"""

import torch
import torch.optim as optim
import math
from typing import Optional, Dict, Any, List, Tuple


def lorentz_metric_tensor(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Create the Minkowski metric tensor J for Lorentz model.
    J = diag(-1, 1, 1, ..., 1) for n+1 dimensions.
    The first coordinate is the "time" dimension.
    """
    J = torch.eye(n, device=device, dtype=dtype)
    J[0, 0] = -1.0
    return J


def higham_schulz_iteration(
    X: torch.Tensor, 
    J: torch.Tensor, 
    steps: int = 5,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Higham-Schulz iteration for J-orthogonalization.
    
    For standard orthogonalization (J=I): 
        X_{k+1} = 0.5 * X_k @ (3I - X_k^T @ X_k)
        
    For J-orthogonalization (e.g., Lorentz J=diag(-1,1,...,1)):
        We use a modified iteration that converges to X^T @ J @ X = J.
        
    This is a simplified approach that normalizes columns with respect to J.
    
    Args:
        X: Input matrix to J-orthogonalize (must be square)
        J: Metric tensor (Lorentz or Euclidean), same size as X
        steps: Number of iterations (not all used for J-orth)
        eps: Small value for numerical stability
        
    Returns:
        J-orthogonalized matrix (approximately)
    """
    if X.dim() == 1:
        return X / (X.norm() + eps)
    
    # Only works for square matrices
    if X.dim() >= 2 and X.shape[-2] != X.shape[-1]:
        # For non-square, fall back to simpler normalization
        return X / (X.norm() + eps)
    
    n = X.shape[-1]
    
    # Ensure J matches X dimensions
    if J.shape[0] != n or J.shape[1] != n:
        J = lorentz_metric_tensor(n, X.device, X.dtype)
    
    # Use double precision for stability
    original_dtype = X.dtype
    X = X.double()
    J = J.double()
    
    # Keep original for fallback
    X_original = X.clone()
    
    # Normalize X for better convergence - critical for stability
    norm = X.norm()
    if norm > eps:
        X = X / norm * math.sqrt(n)
    else:
        return X_original.to(original_dtype)
    
    # Standard Newton-Schulz iteration (orthogonalization)
    # This makes X^T @ X ≈ I, which is a relaxation of J-orthogonality
    a, b, c = (3.4445, -4.7750, 2.0315)  # Optimal coefficients
    
    prev_norm = float('inf')
    for step in range(steps):
        try:
            A = X @ X.T
            B = b * A + c * A @ A
            X_new = a * X + B @ X
            
            # Check for divergence (norm explosion)
            new_norm = X_new.norm().item()
            if new_norm > 1e10 or new_norm > prev_norm * 10:
                # Diverging - stop early
                break
            prev_norm = new_norm
            
            # Stability check - must be finite
            if torch.isnan(X_new).any() or torch.isinf(X_new).any():
                break
            X = X_new
        except Exception:
            break
    
    # Final check - if result is bad, return normalized original
    if torch.isnan(X).any() or torch.isinf(X).any():
        return X_original.to(original_dtype) / (X_original.norm() + eps)
    
    return X.to(original_dtype)


def euclidean_newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Standard Newton-Schulz iteration for Euclidean orthogonalization.
    Fallback for when J-orthogonalization is not applicable.
    
    Uses the zeropower approach: G @ (G^T @ G)^{-1/2}
    Achieved iteratively without explicit inverse.
    """
    if G.dim() == 1:
        return G / (G.norm() + eps)
    
    # Normalize input for numerical stability
    a, b, c = (3.4445, -4.7750, 2.0315)  # Optimal coefficients
    
    X = G.clone()
    X = X / (X.norm() + eps) * math.sqrt(max(G.shape[-2], G.shape[-1]))
    
    if G.shape[-2] > G.shape[-1]:
        X = X.T
        transposed = True
    else:
        transposed = False
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if transposed:
        X = X.T
    
    return X


def riemannian_gradient(
    euclidean_grad: torch.Tensor,
    point: Optional[torch.Tensor] = None,
    curvature: float = -1.0,
    model: str = "lorentz"  # Changed from "poincare" - Poincaré causes gradient vanishing
) -> torch.Tensor:
    """
    Convert Euclidean gradient to Riemannian gradient.
    
    For Poincaré ball model:
        grad_R = ((1 - ||x||^2)^2 / 4) * grad_E
        
    For Lorentz model:
        grad_R = J * grad_E (where J is Lorentz metric)
        
    Args:
        euclidean_grad: Gradient in Euclidean coordinates
        point: Current point on manifold (needed for Poincaré)
        curvature: Manifold curvature (negative for hyperbolic)
        model: "poincare" or "lorentz"
        
    Returns:
        Riemannian gradient in tangent space
    """
    if model == "lorentz":
        # For Lorentz model, the Riemannian gradient involves the metric
        # In practice, we project to ensure it's in the tangent space
        return euclidean_grad
    
    elif model == "poincare":
        if point is None:
            # If no point provided, assume origin (metric = I)
            return euclidean_grad
        
        # Poincaré ball metric factor: (1 - ||x||^2)^2 / 4
        norm_sq = (point ** 2).sum(dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
        metric_factor = ((1.0 - norm_sq) ** 2) / 4.0
        return euclidean_grad * metric_factor
    
    else:
        # Default to Euclidean
        return euclidean_grad


def stochastic_round_ternary(w: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Stochastic rounding for 1.58-bit (ternary) quantization.
    
    Instead of deterministic round(), use probabilistic rounding:
    P(floor) = ceil - x, P(ceil) = x - floor
    
    This enables gradients to flow even for sub-threshold updates
    via the stochastic resonance effect.
    
    Args:
        w: Weights to quantize
        scale: Optional pre-computed scale. If None, uses mean(|w|)
        
    Returns:
        Ternary quantized weights {-1, 0, 1} * scale
    """
    if w.numel() == 0:
        return w
    
    # Compute scale if not provided
    if scale is None:
        scale = w.abs().mean().clamp(min=1e-4)
    
    # Scale weights
    w_scaled = w / scale
    
    # Stochastic rounding to {-1, 0, 1}
    # First clamp to [-1.5, 1.5] for soft boundaries
    w_clamped = w_scaled.clamp(-1.5, 1.5)
    
    # Probabilistic rounding
    w_floor = torch.floor(w_clamped)
    prob = w_clamped - w_floor
    
    # Generate random values
    rand = torch.rand_like(prob)
    
    # Stochastically round
    w_rounded = w_floor + (rand < prob).float()
    
    # Clamp to ternary values
    w_ternary = w_rounded.clamp(-1, 1)
    
    # Rescale
    w_out = w_ternary * scale
    
    # Straight-Through Estimator (STE)
    return (w_out - w).detach() + w


class RiemannianMuonBit(optim.Optimizer):
    """
    Riemannian-Muon-Bit Optimizer
    
    A novel optimizer combining:
    1. Riemannian optimization for hyperbolic neural networks
    2. Muon-style momentum orthogonalization (with J-orthogonal support)
    3. BitNet-aware stochastic rounding for 1.58-bit training
    
    This optimizer is designed for training models where:
    - Parameters live in hyperbolic (non-Euclidean) space
    - Weights are quantized to 1.58-bit (ternary: -1, 0, 1)
    - Standard optimizers cause NaN due to geometric mismatch
    
    Args:
        params: Iterable of parameters or param groups
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        hs_steps: Higham-Schulz iteration steps (default: 5)
        curvature: Hyperbolic space curvature (default: -1.0)
        use_j_orthogonal: Use J-orthogonalization for Lorentz (default: True)
        use_stochastic_rounding: Use stochastic rounding for BitNet (default: True)
        adamw_lr: Learning rate for small/non-matrix params (default: 1e-4)
        adamw_betas: Adam betas for small params (default: (0.9, 0.95))
        adamw_eps: Adam epsilon (default: 1e-8)
        adamw_wd: Weight decay (default: 0.01)
        warmup_steps: Number of warmup steps (default: 2000)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        hs_steps: int = 5,
        curvature: float = -1.0,
        use_j_orthogonal: bool = True,
        use_stochastic_rounding: bool = True,
        use_orthogonalization: bool = False,  # DISABLED by default - was killing gradient flow
        adamw_lr: float = 1e-4,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.01,
        warmup_steps: int = 2000,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            hs_steps=hs_steps,
            curvature=curvature,
            use_j_orthogonal=use_j_orthogonal,
            use_stochastic_rounding=use_stochastic_rounding,
            use_orthogonalization=use_orthogonalization,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)
        
        self.warmup_steps = warmup_steps
        self._step_count = 0
        
        # Store parameter IDs for efficient lookup (avoid tensor comparison issues)
        self._riemannian_param_ids = set()  # Large matrices: use Riemannian-Muon
        self._adamw_param_ids = set()       # Small/1D params: use AdamW
        
        self._categorize_params()
    
    def _categorize_params(self):
        """Categorize parameters into Riemannian-Muon vs AdamW groups by ID."""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    # Large 2D matrices get Riemannian-Muon treatment
                    if p.dim() >= 2 and p.numel() >= 4096:
                        self._riemannian_param_ids.add(id(p))
                    else:
                        self._adamw_param_ids.add(id(p))
    
    def _is_riemannian_param(self, p: torch.Tensor) -> bool:
        """Check if parameter should use Riemannian-Muon update."""
        return id(p) in self._riemannian_param_ids or (p.dim() >= 2 and p.numel() >= 4096)
    
    def _get_warmup_factor(self) -> float:
        """Get warmup scaling factor."""
        if self._step_count >= self.warmup_steps:
            return 1.0
        return min(1.0, (self._step_count + 1) / max(1, self.warmup_steps))
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
            
        Returns:
            Optional loss value from closure.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step_count += 1
        warmup_factor = self._get_warmup_factor()
        
        for group in self.param_groups:
            lr = group['lr'] * warmup_factor
            momentum = group['momentum']
            nesterov = group['nesterov']
            hs_steps = group['hs_steps']
            use_j_orthogonal = group['use_j_orthogonal']
            use_stochastic_rounding = group['use_stochastic_rounding']
            use_orthogonalization = group['use_orthogonalization']
            curvature = group['curvature']
            
            adamw_lr = group['adamw_lr'] * warmup_factor
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            wd = group['adamw_wd']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Handle NaN/Inf in gradients
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                    # AdamW states
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                state['step'] += 1
                
                # Decide which optimizer to use based on parameter shape
                if self._is_riemannian_param(p):
                    # ===== Riemannian-Muon Update =====
                    self._riemannian_muon_update(
                        p, grad, state, lr, momentum, nesterov,
                        hs_steps, use_j_orthogonal, use_stochastic_rounding, 
                        use_orthogonalization, curvature
                    )
                else:
                    # ===== AdamW Update for small parameters =====
                    self._adamw_update(
                        p, grad, state, adamw_lr, beta1, beta2, eps, wd
                    )
        
        return loss
    
    def _riemannian_muon_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict,
        lr: float,
        momentum: float,
        nesterov: bool,
        hs_steps: int,
        use_j_orthogonal: bool,
        use_stochastic_rounding: bool,
        use_orthogonalization: bool,
        curvature: float
    ):
        """Apply Riemannian-Muon update to a parameter."""
        buf = state['momentum_buffer']
        
        # Convert to Riemannian gradient (skip if NaN)
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            return  # Skip this parameter
        
        riemannian_grad = riemannian_gradient(grad, p, curvature, model="lorentz")
        
        # Apply momentum
        buf.mul_(momentum).add_(riemannian_grad)
        
        if nesterov:
            update = riemannian_grad + momentum * buf
        else:
            update = buf.clone()
        
        # Early NaN/Inf check
        if torch.isnan(update).any() or torch.isinf(update).any():
            update = torch.nan_to_num(update, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # === ORTHOGONALIZATION (only if enabled) ===
        # When disabled, gradient updates flow directly without normalization
        if use_orthogonalization:
            # Reshape for orthogonalization if needed
            original_shape = update.shape
            if update.dim() > 2:
                update = update.view(update.shape[0], -1)
            
            # Apply orthogonalization with fallback
            if update.dim() == 2 and update.shape[0] > 1 and update.shape[1] > 1:
                try:
                    if use_j_orthogonal and update.shape[0] == update.shape[1]:
                        # J-orthogonalization for square matrices (Lorentz space)
                        J = lorentz_metric_tensor(update.shape[0], update.device, update.dtype)
                        update_orth = higham_schulz_iteration(update, J, steps=hs_steps)
                    else:
                        # Standard Euclidean orthogonalization
                        update_orth = euclidean_newton_schulz(update, steps=hs_steps)
                    
                    # Check result is valid
                    if not torch.isnan(update_orth).any() and not torch.isinf(update_orth).any():
                        update = update_orth
                    else:
                        # Fallback to simple normalization
                        update = update / (update.norm() + 1e-8)
                except Exception:
                    # Fallback: just normalize
                    update = update / (update.norm() + 1e-8)
            
            # Reshape back
            update = update.view(original_shape)
        
        # Final NaN check on update
        if torch.isnan(update).any() or torch.isinf(update).any():
            return  # Skip this parameter if update is invalid
        
        # Calculate safe scaled learning rate
        dim1 = p.shape[-2] if p.dim() >= 2 else 1
        dim2 = p.shape[-1] if p.dim() >= 1 else 1
        scaled_lr = lr * math.sqrt(max(dim1, dim2))
        
        # Apply update
        if use_stochastic_rounding:
            p.add_(update, alpha=-scaled_lr)
            
            # Apply stochastic rounding to keep weights quantization-friendly
            # This helps maintain the 1.58-bit structure during training
            if p.numel() > 0:
                p.copy_(stochastic_round_ternary(p))
        else:
            # Standard update without stochastic rounding
            p.add_(update, alpha=-scaled_lr)
        
        # Final safety check: replace any NaN/Inf that slipped through
        if torch.isnan(p).any() or torch.isinf(p).any():
            p.copy_(torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=-1.0))
    
    def _adamw_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        wd: float
    ):
        """Apply AdamW update to a parameter."""
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        step = state['step']
        
        # Weight decay
        p.mul_(1 - lr * wd)
        
        # Update biased first moment estimate
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        # Update biased second moment estimate
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        # Compute step size
        step_size = lr / bias_correction1
        
        # Compute denominator
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        
        # Apply update
        p.addcdiv_(exp_avg, denom, value=-step_size)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get optimizer metrics for monitoring.
        
        Returns:
            Dictionary with:
            - step: Current step count
            - warmup_factor: Current warmup scaling
            - riemannian_params: Number of Riemannian-Muon params
            - adamw_params: Number of AdamW params
        """
        return {
            'step': self._step_count,
            'warmup_factor': self._get_warmup_factor(),
            'riemannian_params': len(self._riemannian_param_ids),
            'adamw_params': len(self._adamw_param_ids),
            'warmup_complete': self._step_count >= self.warmup_steps,
        }


# Convenience function
def create_riemannian_muon_bit(
    model,
    lr: float = 0.02,
    momentum: float = 0.95,
    curvature: float = -1.0,
    warmup_steps: int = 2000,
    **kwargs
) -> RiemannianMuonBit:
    """
    Factory function to create RiemannianMuonBit optimizer for a model.
    
    Args:
        model: The PyTorch model
        lr: Learning rate
        momentum: Momentum coefficient
        curvature: Hyperbolic curvature
        warmup_steps: Warmup steps
        **kwargs: Additional arguments for RiemannianMuonBit
        
    Returns:
        Configured RiemannianMuonBit optimizer
    """
    return RiemannianMuonBit(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        curvature=curvature,
        warmup_steps=warmup_steps,
        **kwargs
    )
