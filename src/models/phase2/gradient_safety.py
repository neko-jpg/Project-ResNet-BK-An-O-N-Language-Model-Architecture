"""
Gradient Safety Mechanisms for Phase 2

Implements gradient clipping and NaN/Inf handling for complex gradients.
Ensures numerical stability during backpropagation through complex potentials.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class GradientSafetyModule(nn.Module):
    """
    Gradient safety wrapper for complex potential layers.
    
    Provides:
    1. Gradient clipping to prevent explosion
    2. NaN/Inf detection and replacement
    3. Gradient statistics monitoring
    
    Args:
        max_grad_norm: Maximum gradient norm (default: 1000.0)
        replace_nan_with_zero: Replace NaN/Inf with zero (default: True)
        monitor_stats: Track gradient statistics (default: True)
    """
    
    def __init__(
        self,
        max_grad_norm: float = 1000.0,
        replace_nan_with_zero: bool = True,
        monitor_stats: bool = True,
    ):
        super().__init__()
        
        if max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}")
        
        self.max_grad_norm = max_grad_norm
        self.replace_nan_with_zero = replace_nan_with_zero
        self.monitor_stats = monitor_stats
        
        # Statistics buffers
        if monitor_stats:
            self.register_buffer('grad_norm_history', torch.zeros(1000))
            self.register_buffer('nan_count_history', torch.zeros(1000))
            self.register_buffer('clip_count_history', torch.zeros(1000))
            self.register_buffer('history_idx', torch.tensor(0, dtype=torch.long))
    
    def apply_safety(
        self,
        grad: torch.Tensor,
        param_name: str = "unknown"
    ) -> torch.Tensor:
        """
        Apply gradient safety mechanisms.
        
        Args:
            grad: Gradient tensor (any shape)
            param_name: Parameter name for logging
        
        Returns:
            safe_grad: Gradient with safety mechanisms applied
        """
        if grad is None:
            return None
        
        original_grad = grad.clone() if self.monitor_stats else None
        
        # Step 1: NaN/Inf detection and replacement
        if self.replace_nan_with_zero:
            has_nan = torch.isnan(grad).any()
            has_inf = torch.isinf(grad).any()
            
            if has_nan or has_inf:
                grad = torch.where(
                    torch.isfinite(grad),
                    grad,
                    torch.zeros_like(grad)
                )
                
                if self.monitor_stats:
                    nan_count = torch.isnan(original_grad).sum().item()
                    inf_count = torch.isinf(original_grad).sum().item()
                    print(f"Warning: {param_name} gradient has {nan_count} NaN and {inf_count} Inf values. Replaced with zeros.")
        
        # Step 2: Gradient clipping
        grad_norm = torch.norm(grad)
        clipped = False
        
        if grad_norm > self.max_grad_norm:
            grad = grad * (self.max_grad_norm / (grad_norm + 1e-6))
            clipped = True
            
            if self.monitor_stats:
                print(f"Warning: {param_name} gradient norm {grad_norm:.2f} exceeds threshold {self.max_grad_norm}. Clipped.")
        
        # Step 3: Update statistics
        if self.monitor_stats:
            self._update_statistics(original_grad, grad, clipped)
        
        return grad
    
    def _update_statistics(
        self,
        original_grad: torch.Tensor,
        safe_grad: torch.Tensor,
        was_clipped: bool
    ):
        """Update gradient statistics buffers."""
        with torch.no_grad():
            idx = self.history_idx.item() % 1000
            
            # Gradient norm
            self.grad_norm_history[idx] = torch.norm(original_grad).item()
            
            # NaN count
            nan_count = torch.isnan(original_grad).sum().item()
            self.nan_count_history[idx] = nan_count
            
            # Clip count
            self.clip_count_history[idx] = 1.0 if was_clipped else 0.0
            
            self.history_idx += 1
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get gradient statistics.
        
        Returns:
            Dictionary with gradient statistics
        """
        if not self.monitor_stats:
            return {}
        
        with torch.no_grad():
            valid_len = min(self.history_idx.item(), 1000)
            if valid_len == 0:
                return {
                    'mean_grad_norm': 0.0,
                    'max_grad_norm': 0.0,
                    'nan_rate': 0.0,
                    'clip_rate': 0.0,
                }
            
            valid_norms = self.grad_norm_history[:valid_len]
            valid_nans = self.nan_count_history[:valid_len]
            valid_clips = self.clip_count_history[:valid_len]
            
            return {
                'mean_grad_norm': valid_norms.mean().item(),
                'max_grad_norm': valid_norms.max().item(),
                'std_grad_norm': valid_norms.std().item() if valid_len > 1 else 0.0,
                'nan_rate': (valid_nans > 0).float().mean().item(),
                'clip_rate': valid_clips.mean().item(),
                'total_samples': valid_len,
            }
    
    def reset_statistics(self):
        """Reset gradient statistics buffers."""
        if self.monitor_stats:
            self.grad_norm_history.zero_()
            self.nan_count_history.zero_()
            self.clip_count_history.zero_()
            self.history_idx.zero_()


def safe_complex_backward(
    module: nn.Module,
    max_grad_norm: float = 1000.0,
    replace_nan: bool = True
) -> None:
    """
    Apply gradient safety to all parameters in a module.
    
    This function should be called after loss.backward() but before optimizer.step().
    
    Args:
        module: PyTorch module
        max_grad_norm: Maximum gradient norm
        replace_nan: Replace NaN/Inf with zero
    
    Example:
        >>> loss.backward()
        >>> safe_complex_backward(model, max_grad_norm=1000.0)
        >>> optimizer.step()
    """
    for name, param in module.named_parameters():
        if param.grad is not None:
            # NaN/Inf replacement
            if replace_nan:
                has_nan_inf = ~torch.isfinite(param.grad)
                if has_nan_inf.any():
                    param.grad = torch.where(
                        torch.isfinite(param.grad),
                        param.grad,
                        torch.zeros_like(param.grad)
                    )
            
            # Gradient clipping
            grad_norm = torch.norm(param.grad)
            if grad_norm > max_grad_norm:
                param.grad = param.grad * (max_grad_norm / (grad_norm + 1e-6))


def clip_grad_norm_safe(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False
) -> torch.Tensor:
    """
    Safe version of torch.nn.utils.clip_grad_norm_ with NaN/Inf handling.
    
    Args:
        parameters: Iterable of parameters
        max_norm: Maximum gradient norm
        norm_type: Type of norm (default: 2.0)
        error_if_nonfinite: Raise error if non-finite (default: False)
    
    Returns:
        Total norm of the parameters (before clipping)
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = [p for p in parameters if p.grad is not None]
    
    if len(parameters) == 0:
        return torch.tensor(0.0)
    
    device = parameters[0].grad.device
    
    # Replace NaN/Inf with zero
    for p in parameters:
        if not torch.isfinite(p.grad).all():
            if error_if_nonfinite:
                raise RuntimeError("Non-finite gradient detected")
            p.grad = torch.where(
                torch.isfinite(p.grad),
                p.grad,
                torch.zeros_like(p.grad)
            )
    
    # Compute total norm
    if norm_type == float('inf'):
        total_norm = max(p.grad.abs().max() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad, norm_type) for p in parameters]),
            norm_type
        )
    
    # Clip gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.mul_(clip_coef)
    
    return total_norm
