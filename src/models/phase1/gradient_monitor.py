"""
Gradient Monitoring and Clipping for Phase 1 Components

Task 8.4: Implement gradient norm tracking and automatic clipping

物理的直観 (Physical Intuition):
勾配爆発は、物理系における「共鳴」や「発散」に相当します。
適切な監視とクリッピングにより、学習の安定性を保ちます。

This module provides utilities for:
- Tracking gradient norms for all Phase 1 components
- Automatic gradient clipping when norms exceed thresholds
- Logging gradient statistics for debugging
- Integration with training loops

Requirements: 10.4, 10.5
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class GradientStatistics:
    """
    Statistics for gradient monitoring.
    
    Attributes:
        component_name: Name of the component
        grad_norm: L2 norm of gradients
        grad_max: Maximum absolute gradient value
        grad_min: Minimum absolute gradient value
        grad_mean: Mean absolute gradient value
        grad_std: Standard deviation of gradients
        num_nan: Number of NaN values in gradients
        num_inf: Number of Inf values in gradients
        clipped: Whether gradients were clipped
    """
    component_name: str
    grad_norm: float
    grad_max: float
    grad_min: float
    grad_mean: float
    grad_std: float
    num_nan: int = 0
    num_inf: int = 0
    clipped: bool = False
    
    def is_healthy(self, max_norm: float = 10.0) -> bool:
        """
        Check if gradients are healthy.
        
        Args:
            max_norm: Maximum acceptable gradient norm
        
        Returns:
            True if gradients are healthy
        """
        return (
            self.num_nan == 0 and
            self.num_inf == 0 and
            self.grad_norm < max_norm and
            torch.isfinite(torch.tensor(self.grad_norm))
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            'component': self.component_name,
            'grad_norm': self.grad_norm,
            'grad_max': self.grad_max,
            'grad_min': self.grad_min,
            'grad_mean': self.grad_mean,
            'grad_std': self.grad_std,
            'num_nan': self.num_nan,
            'num_inf': self.num_inf,
            'clipped': self.clipped,
        }


class GradientMonitor:
    """
    Monitor and clip gradients for Phase 1 components.
    
    Task 8.4: Implement gradient norm tracking for all Phase 1 components
    
    物理的直観:
    勾配ノルムは「系のエネルギー変化率」に相当。
    異常に大きい勾配は、系が不安定な状態にあることを示します。
    
    This class provides:
    - Per-component gradient tracking
    - Automatic clipping when thresholds exceeded
    - Statistics logging for debugging
    - Integration with Phase 1 components
    
    Args:
        max_norm: Maximum gradient norm threshold (default: 10.0)
        clip_mode: Clipping mode ('norm' or 'value')
        log_interval: Logging interval in steps (default: 100)
        track_components: List of component names to track
                         If None, tracks all components
    
    Example:
        >>> monitor = GradientMonitor(max_norm=10.0)
        >>> 
        >>> # In training loop
        >>> loss.backward()
        >>> stats = monitor.track_and_clip(model)
        >>> if not stats['all_healthy']:
        >>>     logger.warning(f"Gradient issues: {stats}")
        >>> optimizer.step()
    
    Requirements: 10.4, 10.5
    """
    
    def __init__(
        self,
        max_norm: float = 10.0,
        clip_mode: str = 'norm',
        log_interval: int = 100,
        track_components: Optional[List[str]] = None,
    ):
        assert clip_mode in ['norm', 'value'], \
            f"Invalid clip_mode: {clip_mode}. Must be 'norm' or 'value'"
        
        self.max_norm = max_norm
        self.clip_mode = clip_mode
        self.log_interval = log_interval
        self.track_components = track_components
        
        # Statistics tracking
        self.step = 0
        self.history: Dict[str, List[GradientStatistics]] = {}
        
        # Component name patterns for Phase 1
        self.phase1_patterns = [
            'ar_ssm',
            'htt_embedding',
            'lns_linear',
            'complexity_gate',
            'U_proj',
            'V_proj',
            'T_conv',
            'core1',
            'core2',
            'phase_shift',
        ]
    
    def _is_phase1_component(self, name: str) -> bool:
        """Check if parameter belongs to Phase 1 component."""
        if self.track_components is not None:
            return any(comp in name for comp in self.track_components)
        return any(pattern in name for pattern in self.phase1_patterns)
    
    def compute_gradient_statistics(
        self,
        name: str,
        parameter: nn.Parameter,
    ) -> Optional[GradientStatistics]:
        """
        Compute gradient statistics for a parameter.
        
        Task 8.4: Implement gradient norm tracking
        
        Args:
            name: Parameter name
            parameter: Parameter with gradients
        
        Returns:
            GradientStatistics or None if no gradient
        """
        if parameter.grad is None:
            return None
        
        grad = parameter.grad
        
        # Compute statistics
        grad_norm = grad.norm().item()
        grad_max = grad.abs().max().item()
        grad_min = grad.abs().min().item()
        grad_mean = grad.abs().mean().item()
        grad_std = grad.std().item()
        num_nan = torch.isnan(grad).sum().item()
        num_inf = torch.isinf(grad).sum().item()
        
        return GradientStatistics(
            component_name=name,
            grad_norm=grad_norm,
            grad_max=grad_max,
            grad_min=grad_min,
            grad_mean=grad_mean,
            grad_std=grad_std,
            num_nan=num_nan,
            num_inf=num_inf,
            clipped=False,
        )
    
    def clip_gradients(
        self,
        parameters: List[nn.Parameter],
        stats: List[GradientStatistics],
    ) -> List[GradientStatistics]:
        """
        Clip gradients based on configured mode.
        
        Task 8.4: Add automatic gradient clipping when norms exceed threshold
        
        Args:
            parameters: List of parameters to clip
            stats: List of gradient statistics
        
        Returns:
            Updated statistics with clipping information
        """
        if self.clip_mode == 'norm':
            # Clip by global norm
            # Task 8.4: Automatic gradient clipping
            total_norm = torch.sqrt(
                sum(p.grad.norm() ** 2 for p in parameters if p.grad is not None)
            ).item()
            
            if total_norm > self.max_norm:
                clip_coef = self.max_norm / (total_norm + 1e-6)
                for p in parameters:
                    if p.grad is not None:
                        p.grad.mul_(clip_coef)
                
                # Mark all as clipped
                for stat in stats:
                    stat.clipped = True
                    stat.grad_norm *= clip_coef
                
                logger.debug(
                    f"Clipped gradients: total_norm={total_norm:.4f} -> {self.max_norm:.4f}"
                )
        
        elif self.clip_mode == 'value':
            # Clip by value
            for p, stat in zip(parameters, stats):
                if p.grad is not None:
                    original_norm = p.grad.norm().item()
                    p.grad.clamp_(-self.max_norm, self.max_norm)
                    new_norm = p.grad.norm().item()
                    
                    if new_norm < original_norm:
                        stat.clipped = True
                        stat.grad_norm = new_norm
        
        return stats
    
    def track_and_clip(
        self,
        model: nn.Module,
        clip: bool = True,
    ) -> Dict[str, any]:
        """
        Track gradient statistics and optionally clip.
        
        Task 8.4: Implement gradient norm tracking and automatic clipping
        
        Args:
            model: Model with gradients
            clip: Whether to clip gradients (default: True)
        
        Returns:
            Dictionary with:
                - statistics: List of GradientStatistics
                - all_healthy: Whether all gradients are healthy
                - num_clipped: Number of components clipped
                - max_norm_component: Component with largest gradient norm
        """
        self.step += 1
        
        # Collect statistics for Phase 1 components
        statistics = []
        parameters = []
        
        for name, param in model.named_parameters():
            if not self._is_phase1_component(name):
                continue
            
            if param.grad is None:
                continue
            
            stat = self.compute_gradient_statistics(name, param)
            if stat is not None:
                statistics.append(stat)
                parameters.append(param)
        
        # Clip gradients if requested
        if clip and len(parameters) > 0:
            statistics = self.clip_gradients(parameters, statistics)
        
        # Store history
        for stat in statistics:
            if stat.component_name not in self.history:
                self.history[stat.component_name] = []
            self.history[stat.component_name].append(stat)
        
        # Compute summary
        all_healthy = all(stat.is_healthy(self.max_norm) for stat in statistics)
        num_clipped = sum(1 for stat in statistics if stat.clipped)
        
        max_norm_component = None
        if statistics:
            max_norm_component = max(statistics, key=lambda s: s.grad_norm)
        
        # Log if interval reached
        if self.step % self.log_interval == 0 and statistics:
            self._log_statistics(statistics)
        
        return {
            'statistics': statistics,
            'all_healthy': all_healthy,
            'num_clipped': num_clipped,
            'max_norm_component': max_norm_component,
            'step': self.step,
        }
    
    def _log_statistics(self, statistics: List[GradientStatistics]):
        """
        Log gradient statistics.
        
        Task 8.4: Log gradient statistics for debugging
        """
        logger.info(f"=== Gradient Statistics (Step {self.step}) ===")
        
        for stat in statistics:
            status = "✓" if stat.is_healthy(self.max_norm) else "✗"
            clipped_str = " [CLIPPED]" if stat.clipped else ""
            
            logger.info(
                f"{status} {stat.component_name}: "
                f"norm={stat.grad_norm:.4f}, "
                f"max={stat.grad_max:.4f}, "
                f"mean={stat.grad_mean:.4f}"
                f"{clipped_str}"
            )
            
            if stat.num_nan > 0 or stat.num_inf > 0:
                logger.warning(
                    f"  ⚠ NaN/Inf detected: nan={stat.num_nan}, inf={stat.num_inf}"
                )
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get summary of gradient monitoring history.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.history:
            return {'num_components': 0, 'total_steps': self.step}
        
        summary = {
            'num_components': len(self.history),
            'total_steps': self.step,
            'components': {},
        }
        
        for component_name, stats_list in self.history.items():
            if not stats_list:
                continue
            
            norms = [s.grad_norm for s in stats_list]
            num_clipped = sum(1 for s in stats_list if s.clipped)
            num_unhealthy = sum(1 for s in stats_list if not s.is_healthy(self.max_norm))
            
            summary['components'][component_name] = {
                'mean_norm': sum(norms) / len(norms),
                'max_norm': max(norms),
                'min_norm': min(norms),
                'num_clipped': num_clipped,
                'num_unhealthy': num_unhealthy,
                'clip_rate': num_clipped / len(stats_list),
            }
        
        return summary
    
    def reset(self):
        """Reset monitoring history."""
        self.step = 0
        self.history.clear()


def create_gradient_monitor_from_config(config) -> GradientMonitor:
    """
    Create GradientMonitor from Phase1Config.
    
    Args:
        config: Phase1Config instance
    
    Returns:
        GradientMonitor instance
    """
    return GradientMonitor(
        max_norm=config.gradient_norm_threshold,
        clip_mode='norm',
        log_interval=100,
    )


def check_gradient_health(
    model: nn.Module,
    max_norm: float = 10.0,
) -> Tuple[bool, List[str]]:
    """
    Quick check for gradient health without full monitoring.
    
    Args:
        model: Model with gradients
        max_norm: Maximum acceptable gradient norm
    
    Returns:
        (is_healthy, warnings): Tuple of health status and warning messages
    """
    warnings = []
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        
        grad = param.grad
        
        # Check for NaN/Inf
        if torch.isnan(grad).any():
            warnings.append(f"NaN gradient in {name}")
        
        if torch.isinf(grad).any():
            warnings.append(f"Inf gradient in {name}")
        
        # Check norm
        grad_norm = grad.norm().item()
        if grad_norm > max_norm:
            warnings.append(
                f"Large gradient in {name}: norm={grad_norm:.4f} > {max_norm}"
            )
    
    is_healthy = len(warnings) == 0
    
    return is_healthy, warnings


__all__ = [
    'GradientStatistics',
    'GradientMonitor',
    'create_gradient_monitor_from_config',
    'check_gradient_health',
]
