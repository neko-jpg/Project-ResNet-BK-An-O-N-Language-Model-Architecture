"""
Stability Monitor for Mamba-Killer ResNet-BK

Implements comprehensive failure detection and monitoring for training stability.
Based on Requirement 12: 失敗モード分析と自動リカバリ
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


@dataclass
class StabilityMetrics:
    """Container for stability metrics."""
    
    # NaN/Inf detection
    has_nan: bool = False
    has_inf: bool = False
    nan_tensors: List[str] = field(default_factory=list)
    inf_tensors: List[str] = field(default_factory=list)
    
    # Gradient metrics
    gradient_norm: float = 0.0
    gradient_median: float = 0.0
    gradient_explosion: bool = False
    
    # Loss metrics
    current_loss: float = 0.0
    loss_divergence: bool = False
    loss_increase_pct: float = 0.0
    
    # Numerical stability
    condition_numbers: Dict[str, float] = field(default_factory=dict)
    schatten_norms: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    eigenvalue_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Health status
    overall_health: str = "healthy"  # healthy, warning, critical
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


class StabilityMonitor:
    """
    Real-time monitoring of numerical health and training stability.
    
    Implements:
    - NaN/Inf detection (Requirement 12.1)
    - Gradient explosion detection (Requirement 12.3)
    - Loss divergence detection (Requirement 12.5)
    - Numerical stability monitoring (Requirement 12.15)
    - Training health dashboard (Requirement 12.17)
    """
    
    def __init__(
        self,
        check_interval: int = 10,
        gradient_window: int = 100,
        loss_window: int = 100,
        gradient_explosion_threshold: float = 10.0,
        loss_divergence_threshold: float = 0.5,
        condition_number_threshold: float = 1e6,
        schatten_threshold: float = 100.0,
        enable_detailed_logging: bool = True
    ):
        """
        Initialize stability monitor.
        
        Args:
            check_interval: Check tensors every N steps
            gradient_window: Window size for gradient statistics
            loss_window: Window size for loss statistics
            gradient_explosion_threshold: Threshold for gradient explosion (×median)
            loss_divergence_threshold: Threshold for loss divergence (% increase)
            condition_number_threshold: Threshold for condition number
            schatten_threshold: Threshold for Schatten norms
            enable_detailed_logging: Enable detailed logging
        """
        self.check_interval = check_interval
        self.gradient_window = gradient_window
        self.loss_window = loss_window
        self.gradient_explosion_threshold = gradient_explosion_threshold
        self.loss_divergence_threshold = loss_divergence_threshold
        self.condition_number_threshold = condition_number_threshold
        self.schatten_threshold = schatten_threshold
        self.enable_detailed_logging = enable_detailed_logging
        
        # History tracking
        self.gradient_norms = deque(maxlen=gradient_window)
        self.losses = deque(maxlen=loss_window)
        self.condition_numbers_history = deque(maxlen=100)
        self.schatten_norms_history = deque(maxlen=100)
        
        # Failure counts
        self.nan_count = 0
        self.inf_count = 0
        self.gradient_explosion_count = 0
        self.loss_divergence_count = 0
        self.oom_count = 0
        
        # Step counter
        self.step = 0
        
        # Last stable state
        self.last_stable_step = 0
        self.last_stable_metrics = None
        
        logger.info("✓ StabilityMonitor initialized")
    
    def check_step(
        self,
        model: torch.nn.Module,
        loss: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: Optional[int] = None
    ) -> StabilityMetrics:
        """
        Check stability at current step.
        
        Args:
            model: Model to check
            loss: Current loss value
            optimizer: Optimizer (for gradient checking)
            step: Current step number
        
        Returns:
            StabilityMetrics with current health status
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1
        
        metrics = StabilityMetrics()
        
        # Only check every N steps
        if self.step % self.check_interval != 0:
            return metrics
        
        # Check for NaN/Inf in model parameters
        nan_inf_check = self.check_nan_inf(model)
        metrics.has_nan = nan_inf_check['has_nan']
        metrics.has_inf = nan_inf_check['has_inf']
        metrics.nan_tensors = nan_inf_check['nan_tensors']
        metrics.inf_tensors = nan_inf_check['inf_tensors']
        
        if metrics.has_nan:
            self.nan_count += 1
            metrics.errors.append(f"NaN detected in {len(metrics.nan_tensors)} tensors")
            metrics.overall_health = "critical"
            logger.error(f"❌ NaN detected at step {self.step}: {metrics.nan_tensors[:5]}")
        
        if metrics.has_inf:
            self.inf_count += 1
            metrics.errors.append(f"Inf detected in {len(metrics.inf_tensors)} tensors")
            metrics.overall_health = "critical"
            logger.error(f"❌ Inf detected at step {self.step}: {metrics.inf_tensors[:5]}")
        
        # Check gradient explosion
        if optimizer is not None:
            grad_norm = self.compute_gradient_norm(model)
            self.gradient_norms.append(grad_norm)
            metrics.gradient_norm = grad_norm
            
            if len(self.gradient_norms) >= 10:
                grad_median = np.median(list(self.gradient_norms))
                metrics.gradient_median = grad_median
                
                if grad_norm > self.gradient_explosion_threshold * grad_median:
                    self.gradient_explosion_count += 1
                    metrics.gradient_explosion = True
                    metrics.warnings.append(
                        f"Gradient explosion: {grad_norm:.2f} > {self.gradient_explosion_threshold}× median ({grad_median:.2f})"
                    )
                    if metrics.overall_health == "healthy":
                        metrics.overall_health = "warning"
                    logger.warning(f"⚠️ Gradient explosion at step {self.step}: {grad_norm:.2f} > {grad_median:.2f}")
        
        # Check loss divergence
        loss_value = loss.item() if torch.is_tensor(loss) else loss
        self.losses.append(loss_value)
        metrics.current_loss = loss_value
        
        if len(self.losses) >= self.loss_window:
            loss_100_steps_ago = self.losses[0]
            if loss_100_steps_ago > 0:
                loss_increase = (loss_value - loss_100_steps_ago) / loss_100_steps_ago
                metrics.loss_increase_pct = loss_increase * 100
                
                if loss_increase > self.loss_divergence_threshold:
                    self.loss_divergence_count += 1
                    metrics.loss_divergence = True
                    metrics.warnings.append(
                        f"Loss divergence: {loss_increase*100:.1f}% increase over {self.loss_window} steps"
                    )
                    if metrics.overall_health == "healthy":
                        metrics.overall_health = "warning"
                    logger.warning(f"⚠️ Loss divergence at step {self.step}: {loss_increase*100:.1f}% increase")
        
        # Check numerical stability (condition numbers, Schatten norms)
        stability_check = self.check_numerical_stability(model)
        metrics.condition_numbers = stability_check['condition_numbers']
        metrics.schatten_norms = stability_check['schatten_norms']
        metrics.eigenvalue_stats = stability_check['eigenvalue_stats']
        
        # Check condition numbers
        for name, cond_num in metrics.condition_numbers.items():
            if cond_num > self.condition_number_threshold:
                metrics.warnings.append(
                    f"High condition number in {name}: {cond_num:.2e} > {self.condition_number_threshold:.2e}"
                )
                if metrics.overall_health == "healthy":
                    metrics.overall_health = "warning"
        
        # Check Schatten norms
        for name, (s1_norm, s2_norm) in metrics.schatten_norms.items():
            if s2_norm > self.schatten_threshold:
                metrics.warnings.append(
                    f"High Schatten-2 norm in {name}: {s2_norm:.2f} > {self.schatten_threshold:.2f}"
                )
                if metrics.overall_health == "healthy":
                    metrics.overall_health = "warning"
        
        # Update last stable state if healthy
        if metrics.overall_health == "healthy":
            self.last_stable_step = self.step
            self.last_stable_metrics = metrics
        
        # Log summary
        if self.enable_detailed_logging and (metrics.warnings or metrics.errors):
            self._log_metrics_summary(metrics)
        
        return metrics
    
    def check_nan_inf(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Check for NaN/Inf in all model tensors.
        
        Implements Requirement 12.1: NaN/Inf detection
        
        Args:
            model: Model to check
        
        Returns:
            Dictionary with detection results
        """
        nan_tensors = []
        inf_tensors = []
        
        for name, param in model.named_parameters():
            if param is None or param.data is None:
                continue
            
            # Check for NaN
            if torch.isnan(param.data).any():
                nan_tensors.append(name)
            
            # Check for Inf
            if torch.isinf(param.data).any():
                inf_tensors.append(name)
            
            # Check gradients if available
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_tensors.append(f"{name}.grad")
                if torch.isinf(param.grad).any():
                    inf_tensors.append(f"{name}.grad")
        
        return {
            'has_nan': len(nan_tensors) > 0,
            'has_inf': len(inf_tensors) > 0,
            'nan_tensors': nan_tensors,
            'inf_tensors': inf_tensors
        }
    
    def compute_gradient_norm(self, model: torch.nn.Module) -> float:
        """
        Compute total gradient norm.
        
        Implements Requirement 12.3: Gradient explosion detection
        
        Args:
            model: Model to compute gradient norm for
        
        Returns:
            Total gradient norm
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        return total_norm ** 0.5
    
    def check_numerical_stability(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Check numerical stability metrics.
        
        Implements Requirement 12.15: Numerical stability monitoring
        
        Args:
            model: Model to check
        
        Returns:
            Dictionary with stability metrics
        """
        condition_numbers = {}
        schatten_norms = {}
        eigenvalue_stats = {}
        
        for name, module in model.named_modules():
            # Check for BK-Core modules
            if hasattr(module, 'compute_condition_number'):
                try:
                    cond_num = module.compute_condition_number()
                    condition_numbers[name] = cond_num
                except Exception as e:
                    logger.debug(f"Failed to compute condition number for {name}: {e}")
            
            # Check for Schatten norms
            if hasattr(module, 'compute_schatten_norms'):
                try:
                    s1_norm, s2_norm = module.compute_schatten_norms()
                    schatten_norms[name] = (s1_norm, s2_norm)
                except Exception as e:
                    logger.debug(f"Failed to compute Schatten norms for {name}: {e}")
            
            # Check eigenvalue statistics
            if hasattr(module, 'get_eigenvalue_stats'):
                try:
                    eig_stats = module.get_eigenvalue_stats()
                    eigenvalue_stats[name] = eig_stats
                except Exception as e:
                    logger.debug(f"Failed to get eigenvalue stats for {name}: {e}")
        
        return {
            'condition_numbers': condition_numbers,
            'schatten_norms': schatten_norms,
            'eigenvalue_stats': eigenvalue_stats
        }
    
    def suggest_recovery(self, metrics: StabilityMetrics) -> str:
        """
        Suggest recovery action based on failure mode.
        
        Implements Requirement 12.18: Suggest corrective actions
        
        Args:
            metrics: Current stability metrics
        
        Returns:
            Suggested recovery action
        """
        if metrics.has_nan or metrics.has_inf:
            return "rollback_checkpoint"
        
        if metrics.gradient_explosion:
            return "reduce_lr_10x"
        
        if metrics.loss_divergence:
            return "increase_epsilon"
        
        # Check condition numbers
        for name, cond_num in metrics.condition_numbers.items():
            if cond_num > self.condition_number_threshold:
                return "upgrade_precision"
        
        return "continue"
    
    def get_health_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive health dashboard.
        
        Implements Requirement 12.17: Training health dashboard
        
        Returns:
            Dictionary with 20+ health metrics
        """
        dashboard = {
            # Step info
            'current_step': self.step,
            'last_stable_step': self.last_stable_step,
            'steps_since_stable': self.step - self.last_stable_step,
            
            # Failure counts
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'gradient_explosion_count': self.gradient_explosion_count,
            'loss_divergence_count': self.loss_divergence_count,
            'oom_count': self.oom_count,
            
            # Gradient statistics
            'current_gradient_norm': self.gradient_norms[-1] if self.gradient_norms else 0.0,
            'gradient_norm_mean': np.mean(list(self.gradient_norms)) if self.gradient_norms else 0.0,
            'gradient_norm_std': np.std(list(self.gradient_norms)) if self.gradient_norms else 0.0,
            'gradient_norm_median': np.median(list(self.gradient_norms)) if self.gradient_norms else 0.0,
            'gradient_norm_max': max(self.gradient_norms) if self.gradient_norms else 0.0,
            'gradient_norm_min': min(self.gradient_norms) if self.gradient_norms else 0.0,
            
            # Loss statistics
            'current_loss': self.losses[-1] if self.losses else 0.0,
            'loss_mean': np.mean(list(self.losses)) if self.losses else 0.0,
            'loss_std': np.std(list(self.losses)) if self.losses else 0.0,
            'loss_median': np.median(list(self.losses)) if self.losses else 0.0,
            'loss_trend': self._compute_loss_trend(),
            
            # Condition numbers
            'condition_numbers_history': list(self.condition_numbers_history),
            'max_condition_number': max(self.condition_numbers_history) if self.condition_numbers_history else 0.0,
            
            # Schatten norms
            'schatten_norms_history': list(self.schatten_norms_history),
            
            # Overall health
            'overall_health': self.last_stable_metrics.overall_health if self.last_stable_metrics else "unknown",
            'health_score': self._compute_health_score(),
        }
        
        return dashboard
    
    def _compute_loss_trend(self) -> str:
        """Compute loss trend (improving, stable, degrading)."""
        if len(self.losses) < 20:
            return "insufficient_data"
        
        recent_losses = list(self.losses)[-20:]
        early_mean = np.mean(recent_losses[:10])
        late_mean = np.mean(recent_losses[10:])
        
        if late_mean < early_mean * 0.95:
            return "improving"
        elif late_mean > early_mean * 1.05:
            return "degrading"
        else:
            return "stable"
    
    def _compute_health_score(self) -> float:
        """
        Compute overall health score (0-100).
        
        Returns:
            Health score where 100 is perfect health
        """
        score = 100.0
        
        # Penalize failures
        score -= self.nan_count * 20
        score -= self.inf_count * 20
        score -= self.gradient_explosion_count * 5
        score -= self.loss_divergence_count * 5
        score -= self.oom_count * 10
        
        # Penalize instability
        if len(self.gradient_norms) >= 10:
            grad_std = np.std(list(self.gradient_norms))
            grad_mean = np.mean(list(self.gradient_norms))
            if grad_mean > 0:
                cv = grad_std / grad_mean  # Coefficient of variation
                score -= min(cv * 10, 20)  # Penalize high variability
        
        return max(0.0, min(100.0, score))
    
    def _log_metrics_summary(self, metrics: StabilityMetrics):
        """Log summary of metrics."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Stability Check - Step {self.step}")
        logger.info(f"{'='*60}")
        logger.info(f"Overall Health: {metrics.overall_health.upper()}")
        
        if metrics.warnings:
            logger.warning("Warnings:")
            for warning in metrics.warnings:
                logger.warning(f"  - {warning}")
        
        if metrics.errors:
            logger.error("Errors:")
            for error in metrics.errors:
                logger.error(f"  - {error}")
        
        logger.info(f"Gradient Norm: {metrics.gradient_norm:.4f}")
        logger.info(f"Current Loss: {metrics.current_loss:.4f}")
        logger.info(f"{'='*60}\n")
    
    def reset_failure_counts(self):
        """Reset all failure counters."""
        self.nan_count = 0
        self.inf_count = 0
        self.gradient_explosion_count = 0
        self.loss_divergence_count = 0
        self.oom_count = 0
        logger.info("✓ Failure counts reset")
    
    def export_metrics(self, filepath: str):
        """
        Export metrics history to file.
        
        Args:
            filepath: Path to save metrics
        """
        import json
        
        metrics_data = {
            'gradient_norms': list(self.gradient_norms),
            'losses': list(self.losses),
            'condition_numbers_history': list(self.condition_numbers_history),
            'failure_counts': {
                'nan': self.nan_count,
                'inf': self.inf_count,
                'gradient_explosion': self.gradient_explosion_count,
                'loss_divergence': self.loss_divergence_count,
                'oom': self.oom_count
            },
            'dashboard': self.get_health_dashboard()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"✓ Metrics exported to {filepath}")


if __name__ == '__main__':
    # Test stability monitor
    import torch.nn as nn
    
    # Create dummy model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create monitor
    monitor = StabilityMonitor(check_interval=1)
    
    # Simulate training
    for step in range(10):
        # Forward pass
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check stability
        metrics = monitor.check_step(model, loss, optimizer, step)
        
        if step == 5:
            # Inject NaN to test detection
            with torch.no_grad():
                model[0].weight[0, 0] = float('nan')
        
        optimizer.step()
    
    # Get dashboard
    dashboard = monitor.get_health_dashboard()
    print("\n" + "="*60)
    print("Health Dashboard:")
    print("="*60)
    for key, value in dashboard.items():
        if not isinstance(value, (list, dict)):
            print(f"{key}: {value}")
    
    print("\n✓ StabilityMonitor test passed")
