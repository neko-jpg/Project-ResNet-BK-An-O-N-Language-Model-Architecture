"""
Birman-Schwinger Stability Monitor

Monitors the mathematical stability of the Birman-Schwinger operator K_ε during training
to prevent divergence due to operator singularities.

Physical Intuition (物理的直観):
- det(I - K_ε) は「系の安定性マージン」を表す
- ゼロに近づくと、物理系が特異点（共鳴）に達する
- 学習中に監視し、発散を事前に防ぐ

Mathematical Foundation:
    K_ε(z) = |V_ε|^(1/2) · R₀(z) · |V_ε|^(1/2)
    
    Stability Condition:
        |det(I - K_ε)| > δ > 0  (δ = stability margin)
    
    Schatten Norm Bounds:
        ||K_ε||_S1 ≤ C₁ · ||V||_L1 / ε
        ||K_ε||_S2 ≤ C₂ · ||V||_L2 / √ε
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


@dataclass
class StabilityThresholds:
    """Configuration for stability monitoring thresholds."""
    
    # Determinant condition threshold
    det_threshold: float = 1e-6
    
    # Schatten norm bounds
    schatten_s1_bound: float = 100.0
    schatten_s2_bound: float = 50.0
    
    # Eigenvalue threshold
    min_eigenvalue_threshold: float = 1e-8
    
    # Gradient norm threshold
    gradient_norm_threshold: float = 10.0
    
    # Condition number threshold
    condition_number_threshold: float = 1e6
    
    def validate(self):
        """Validate threshold values."""
        assert self.det_threshold > 0, "det_threshold must be positive"
        assert self.schatten_s1_bound > 0, "schatten_s1_bound must be positive"
        assert self.schatten_s2_bound > 0, "schatten_s2_bound must be positive"
        assert self.min_eigenvalue_threshold > 0, "min_eigenvalue_threshold must be positive"
        assert self.gradient_norm_threshold > 0, "gradient_norm_threshold must be positive"
        assert self.condition_number_threshold > 1, "condition_number_threshold must be > 1"


@dataclass
class StabilityMetrics:
    """Stability metrics computed at each monitoring step."""
    
    # Determinant condition
    det_condition: float
    
    # Schatten norms
    schatten_s1: float
    schatten_s2: float
    
    # Eigenvalue statistics
    min_eigenvalue: float
    max_eigenvalue: float
    eigenvalue_ratio: float  # max/min (condition number proxy)
    
    # Gradient statistics
    gradient_norm: float
    
    # Flags
    is_stable: bool
    warnings: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'det_condition': self.det_condition,
            'schatten_s1': self.schatten_s1,
            'schatten_s2': self.schatten_s2,
            'min_eigenvalue': self.min_eigenvalue,
            'max_eigenvalue': self.max_eigenvalue,
            'eigenvalue_ratio': self.eigenvalue_ratio,
            'gradient_norm': self.gradient_norm,
            'is_stable': self.is_stable,
            'num_warnings': len(self.warnings),
            'num_actions': len(self.recommended_actions),
        }


class BKStabilityMonitor:
    """
    Monitor Birman-Schwinger operator stability during training.
    
    Tracks:
    - Determinant condition: |det(I - K_ε)|
    - Schatten norms: ||K||_S1, ||K||_S2
    - Minimum eigenvalue of (I - K_ε)
    - Gradient norms
    
    Actions:
    - Log warnings when stability threshold violated
    - Trigger gradient clipping
    - Suggest learning rate reduction
    - Optionally halt training if critical
    
    Example:
        >>> monitor = BKStabilityMonitor()
        >>> metrics = monitor.check_stability(G_ii, potential, epsilon)
        >>> if not metrics.is_stable:
        ...     logger.warning(f"Stability warnings: {metrics.warnings}")
        ...     for action in metrics.recommended_actions:
        ...         logger.info(f"Recommended: {action}")
    """
    
    def __init__(
        self,
        thresholds: Optional[StabilityThresholds] = None,
        enable_history: bool = True,
        history_size: int = 1000,
    ):
        """
        Initialize stability monitor.
        
        Args:
            thresholds: Stability thresholds configuration
            enable_history: Whether to track history of metrics
            history_size: Maximum number of history entries to keep
        """
        self.thresholds = thresholds or StabilityThresholds()
        self.thresholds.validate()
        
        self.enable_history = enable_history
        self.history_size = history_size
        
        # History tracking
        self.det_history: List[float] = []
        self.schatten_s1_history: List[float] = []
        self.schatten_s2_history: List[float] = []
        self.min_eigenvalue_history: List[float] = []
        self.gradient_norm_history: List[float] = []
        
        # Statistics
        self.total_checks = 0
        self.stability_violations = 0
        self.last_stable_step: Optional[int] = None
    
    def _update_history(self, metrics: StabilityMetrics):
        """Update history with new metrics."""
        if not self.enable_history:
            return
        
        # Add new values
        self.det_history.append(metrics.det_condition)
        self.schatten_s1_history.append(metrics.schatten_s1)
        self.schatten_s2_history.append(metrics.schatten_s2)
        self.min_eigenvalue_history.append(metrics.min_eigenvalue)
        self.gradient_norm_history.append(metrics.gradient_norm)
        
        # Trim to history_size
        if len(self.det_history) > self.history_size:
            self.det_history = self.det_history[-self.history_size:]
            self.schatten_s1_history = self.schatten_s1_history[-self.history_size:]
            self.schatten_s2_history = self.schatten_s2_history[-self.history_size:]
            self.min_eigenvalue_history = self.min_eigenvalue_history[-self.history_size:]
            self.gradient_norm_history = self.gradient_norm_history[-self.history_size:]
    
    def get_history_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics from history."""
        if not self.enable_history or len(self.det_history) == 0:
            return {}
        
        def compute_stats(values: List[float]) -> Dict[str, float]:
            tensor = torch.tensor(values)
            return {
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item(),
            }
        
        return {
            'det_condition': compute_stats(self.det_history),
            'schatten_s1': compute_stats(self.schatten_s1_history),
            'schatten_s2': compute_stats(self.schatten_s2_history),
            'min_eigenvalue': compute_stats(self.min_eigenvalue_history),
            'gradient_norm': compute_stats(self.gradient_norm_history),
        }
    
    def reset_history(self):
        """Reset all history tracking."""
        self.det_history.clear()
        self.schatten_s1_history.clear()
        self.schatten_s2_history.clear()
        self.min_eigenvalue_history.clear()
        self.gradient_norm_history.clear()
        self.total_checks = 0
        self.stability_violations = 0
        self.last_stable_step = None

    def _compute_determinant_condition(
        self,
        G_ii: torch.Tensor,
        epsilon: float,
    ) -> float:
        """
        Compute determinant condition: |det(I - K_ε)|
        
        For the Birman-Schwinger operator K_ε, we approximate the determinant
        condition using the resolvent diagonal G_ii = diag((H - zI)^-1).
        
        Approximation:
            det(I - K_ε) ≈ ∏ᵢ (1 - λᵢ)
            where λᵢ are eigenvalues of K_ε
        
        For numerical stability, we compute in log space:
            log|det(I - K_ε)| = Σᵢ log|1 - λᵢ|
        
        Args:
            G_ii: (B, N) complex resolvent diagonal
            epsilon: Regularization parameter
        
        Returns:
            det_condition: |det(I - K_ε)| approximation
        """
        # Use magnitude of resolvent as proxy for operator norm
        # |det(I - K_ε)| ≈ exp(-||K_ε||_trace)
        G_mag = G_ii.abs()
        
        # Compute trace-based approximation
        # Clamp to avoid log(0)
        G_mag_clamped = torch.clamp(G_mag, min=1e-10, max=1e10)
        
        # log|det| ≈ -Σ log(1 + ε·|G_ii|)
        log_det_approx = -torch.sum(torch.log(1.0 + epsilon * G_mag_clamped), dim=-1)
        
        # Take mean over batch and convert to magnitude
        det_condition = torch.exp(log_det_approx.mean()).item()
        
        return det_condition
    
    def _compute_schatten_norms(
        self,
        G_ii: torch.Tensor,
        potential: torch.Tensor,
        epsilon: float,
    ) -> Tuple[float, float]:
        """
        Compute Schatten norms S1 and S2 for the operator K_ε.
        
        Schatten S1 norm (nuclear norm):
            ||K||_S1 = Σᵢ σᵢ (sum of singular values)
        
        Schatten S2 norm (Frobenius norm):
            ||K||_S2 = √(Σᵢ σᵢ²)
        
        Theoretical bounds:
            ||K_ε||_S1 ≤ C₁ · ||V||_L1 / ε
            ||K_ε||_S2 ≤ C₂ · ||V||_L2 / √ε
        
        Args:
            G_ii: (B, N) complex resolvent diagonal
            potential: (B, N) potential values
            epsilon: Regularization parameter
        
        Returns:
            schatten_s1: S1 norm estimate
            schatten_s2: S2 norm estimate
        """
        # Compute operator magnitude: |K_ε| ≈ |V_ε|^(1/2) · |G| · |V_ε|^(1/2)
        V_mag = potential.abs()
        G_mag = G_ii.abs()
        
        # K_ε magnitude approximation
        K_mag = torch.sqrt(V_mag + epsilon) * G_mag * torch.sqrt(V_mag + epsilon)
        
        # S1 norm: sum of singular values ≈ sum of magnitudes
        schatten_s1 = K_mag.sum(dim=-1).mean().item()
        
        # S2 norm: Frobenius norm ≈ sqrt(sum of squared magnitudes)
        schatten_s2 = torch.sqrt((K_mag ** 2).sum(dim=-1).mean()).item()
        
        return schatten_s1, schatten_s2
    
    def _compute_eigenvalue_stats(
        self,
        G_ii: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """
        Compute eigenvalue statistics from resolvent diagonal.
        
        The resolvent G = (H - zI)^-1 has eigenvalues 1/(λᵢ - z),
        where λᵢ are eigenvalues of H.
        
        Args:
            G_ii: (B, N) complex resolvent diagonal
        
        Returns:
            min_eigenvalue: Minimum eigenvalue magnitude
            max_eigenvalue: Maximum eigenvalue magnitude
            eigenvalue_ratio: max/min (condition number proxy)
        """
        # Eigenvalue magnitude from resolvent: |λ - z| ≈ 1/|G|
        G_mag = G_ii.abs()
        
        # Clamp to avoid division by zero
        G_mag_clamped = torch.clamp(G_mag, min=1e-10, max=1e10)
        
        # Approximate eigenvalue magnitudes
        lambda_mag = 1.0 / G_mag_clamped
        
        # Compute statistics
        min_eigenvalue = lambda_mag.min().item()
        max_eigenvalue = lambda_mag.max().item()
        
        # Condition number proxy
        eigenvalue_ratio = max_eigenvalue / (min_eigenvalue + 1e-10)
        
        return min_eigenvalue, max_eigenvalue, eigenvalue_ratio
    
    def _check_thresholds(
        self,
        det_condition: float,
        schatten_s1: float,
        schatten_s2: float,
        min_eigenvalue: float,
        eigenvalue_ratio: float,
        gradient_norm: float,
    ) -> Tuple[bool, List[str]]:
        """
        Check if metrics violate stability thresholds.
        
        Args:
            det_condition: Determinant condition value
            schatten_s1: S1 norm value
            schatten_s2: S2 norm value
            min_eigenvalue: Minimum eigenvalue
            eigenvalue_ratio: Eigenvalue ratio (condition number)
            gradient_norm: Gradient norm
        
        Returns:
            is_stable: True if all thresholds satisfied
            warnings: List of warning messages
        """
        warnings = []
        
        # Check determinant condition
        if det_condition < self.thresholds.det_threshold:
            warnings.append(
                f"Determinant condition {det_condition:.2e} below threshold "
                f"{self.thresholds.det_threshold:.2e} - operator approaching singularity"
            )
        
        # Check Schatten S1 norm
        if schatten_s1 > self.thresholds.schatten_s1_bound:
            warnings.append(
                f"Schatten S1 norm {schatten_s1:.2f} exceeds bound "
                f"{self.thresholds.schatten_s1_bound:.2f} - operator too large"
            )
        
        # Check Schatten S2 norm
        if schatten_s2 > self.thresholds.schatten_s2_bound:
            warnings.append(
                f"Schatten S2 norm {schatten_s2:.2f} exceeds bound "
                f"{self.thresholds.schatten_s2_bound:.2f} - operator energy too high"
            )
        
        # Check minimum eigenvalue
        if min_eigenvalue < self.thresholds.min_eigenvalue_threshold:
            warnings.append(
                f"Minimum eigenvalue {min_eigenvalue:.2e} below threshold "
                f"{self.thresholds.min_eigenvalue_threshold:.2e} - near-zero eigenvalue detected"
            )
        
        # Check condition number
        if eigenvalue_ratio > self.thresholds.condition_number_threshold:
            warnings.append(
                f"Eigenvalue ratio {eigenvalue_ratio:.2e} exceeds threshold "
                f"{self.thresholds.condition_number_threshold:.2e} - ill-conditioned operator"
            )
        
        # Check gradient norm
        if gradient_norm > self.thresholds.gradient_norm_threshold:
            warnings.append(
                f"Gradient norm {gradient_norm:.2f} exceeds threshold "
                f"{self.thresholds.gradient_norm_threshold:.2f} - potential gradient explosion"
            )
        
        is_stable = len(warnings) == 0
        
        return is_stable, warnings
    
    def check_stability(
        self,
        G_ii: torch.Tensor,
        potential: torch.Tensor,
        epsilon: float,
        gradient_norm: Optional[float] = None,
    ) -> StabilityMetrics:
        """
        Check stability conditions and return comprehensive diagnostics.
        
        Args:
            G_ii: (B, N) complex resolvent diagonal from BK-Core
            potential: (B, N) potential values
            epsilon: Regularization parameter
            gradient_norm: Optional gradient norm (computed if not provided)
        
        Returns:
            StabilityMetrics with all diagnostic information
        
        Example:
            >>> G_ii = model.bk_core(x)  # (B, N) complex
            >>> potential = model.potential(x)  # (B, N) real
            >>> metrics = monitor.check_stability(G_ii, potential, epsilon=0.1)
            >>> if not metrics.is_stable:
            ...     print(f"Warnings: {metrics.warnings}")
            ...     print(f"Actions: {metrics.recommended_actions}")
        """
        self.total_checks += 1
        
        # Compute determinant condition
        det_condition = self._compute_determinant_condition(G_ii, epsilon)
        
        # Compute Schatten norms
        schatten_s1, schatten_s2 = self._compute_schatten_norms(G_ii, potential, epsilon)
        
        # Compute eigenvalue statistics
        min_eigenvalue, max_eigenvalue, eigenvalue_ratio = self._compute_eigenvalue_stats(G_ii)
        
        # Compute gradient norm if not provided
        if gradient_norm is None:
            if potential.grad is not None:
                gradient_norm = potential.grad.norm().item()
            else:
                gradient_norm = 0.0
        
        # Check thresholds
        is_stable, warnings = self._check_thresholds(
            det_condition,
            schatten_s1,
            schatten_s2,
            min_eigenvalue,
            eigenvalue_ratio,
            gradient_norm,
        )
        
        # Generate recommended actions
        recommended_actions = []
        if not is_stable:
            self.stability_violations += 1
            recommended_actions = self._generate_recovery_actions(
                det_condition,
                schatten_s1,
                schatten_s2,
                min_eigenvalue,
                eigenvalue_ratio,
                gradient_norm,
            )
        else:
            self.last_stable_step = self.total_checks
        
        # Create metrics object
        metrics = StabilityMetrics(
            det_condition=det_condition,
            schatten_s1=schatten_s1,
            schatten_s2=schatten_s2,
            min_eigenvalue=min_eigenvalue,
            max_eigenvalue=max_eigenvalue,
            eigenvalue_ratio=eigenvalue_ratio,
            gradient_norm=gradient_norm,
            is_stable=is_stable,
            warnings=warnings,
            recommended_actions=recommended_actions,
        )
        
        # Update history
        self._update_history(metrics)
        
        return metrics

    def _generate_recovery_actions(
        self,
        det_condition: float,
        schatten_s1: float,
        schatten_s2: float,
        min_eigenvalue: float,
        eigenvalue_ratio: float,
        gradient_norm: float,
    ) -> List[str]:
        """
        Generate recommended recovery actions based on stability violations.
        
        Args:
            det_condition: Determinant condition value
            schatten_s1: S1 norm value
            schatten_s2: S2 norm value
            min_eigenvalue: Minimum eigenvalue
            eigenvalue_ratio: Eigenvalue ratio
            gradient_norm: Gradient norm
        
        Returns:
            List of recommended action strings
        """
        actions = []
        
        # Critical: determinant approaching zero
        if det_condition < self.thresholds.det_threshold:
            actions.append("CRITICAL: Apply gradient clipping (max_norm=1.0)")
            actions.append("CRITICAL: Reduce learning rate by 50%")
            actions.append("Consider increasing epsilon regularization")
        
        # Schatten norm violations
        if schatten_s1 > self.thresholds.schatten_s1_bound:
            actions.append("Apply spectral clipping to operator (clip S1 norm)")
            actions.append("Reduce potential magnitude or increase epsilon")
        
        if schatten_s2 > self.thresholds.schatten_s2_bound:
            actions.append("Apply spectral clipping to operator (clip S2 norm)")
            actions.append("Consider weight decay or L2 regularization")
        
        # Near-zero eigenvalue
        if min_eigenvalue < self.thresholds.min_eigenvalue_threshold:
            actions.append("Increase epsilon to avoid singular operator")
            actions.append("Check for numerical precision issues (use float64)")
        
        # Ill-conditioned operator
        if eigenvalue_ratio > self.thresholds.condition_number_threshold:
            actions.append("Operator is ill-conditioned - apply preconditioning")
            actions.append("Consider reducing model complexity or rank")
        
        # Gradient explosion
        if gradient_norm > self.thresholds.gradient_norm_threshold:
            actions.append("Apply gradient clipping immediately")
            actions.append("Reduce learning rate")
            actions.append("Check for NaN/Inf in inputs")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_actions = []
        for action in actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)
        
        return unique_actions
    
    def apply_gradient_clipping(
        self,
        parameters: torch.nn.Parameter,
        max_norm: float = 1.0,
    ) -> float:
        """
        Apply gradient clipping to model parameters.
        
        Args:
            parameters: Model parameters to clip
            max_norm: Maximum gradient norm
        
        Returns:
            Total gradient norm before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        logger.info(f"Applied gradient clipping: norm {total_norm:.4f} -> {max_norm}")
        return total_norm
    
    def apply_spectral_clipping(
        self,
        operator: torch.Tensor,
        max_s1_norm: Optional[float] = None,
        max_s2_norm: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply spectral clipping to operator to enforce Schatten norm bounds.
        
        Clips the singular values of the operator to satisfy:
            ||K||_S1 ≤ max_s1_norm
            ||K||_S2 ≤ max_s2_norm
        
        Args:
            operator: (B, N, N) or (N, N) operator matrix
            max_s1_norm: Maximum S1 norm (nuclear norm)
            max_s2_norm: Maximum S2 norm (Frobenius norm)
        
        Returns:
            Clipped operator with same shape
        """
        if max_s1_norm is None:
            max_s1_norm = self.thresholds.schatten_s1_bound
        if max_s2_norm is None:
            max_s2_norm = self.thresholds.schatten_s2_bound
        
        # Handle batched or single operator
        original_shape = operator.shape
        if operator.ndim == 2:
            operator = operator.unsqueeze(0)  # (1, N, N)
        
        clipped_operators = []
        for op in operator:
            # Compute SVD
            U, S, Vh = torch.linalg.svd(op, full_matrices=False)
            
            # Clip S1 norm (sum of singular values)
            s1_norm = S.sum()
            if s1_norm > max_s1_norm:
                S = S * (max_s1_norm / s1_norm)
                logger.debug(f"Clipped S1 norm: {s1_norm:.2f} -> {max_s1_norm:.2f}")
            
            # Clip S2 norm (Frobenius norm = sqrt(sum of squared singular values))
            s2_norm = torch.sqrt((S ** 2).sum())
            if s2_norm > max_s2_norm:
                S = S * (max_s2_norm / s2_norm)
                logger.debug(f"Clipped S2 norm: {s2_norm:.2f} -> {max_s2_norm:.2f}")
            
            # Reconstruct operator
            clipped_op = U @ torch.diag(S) @ Vh
            clipped_operators.append(clipped_op)
        
        result = torch.stack(clipped_operators)
        
        # Restore original shape
        if len(original_shape) == 2:
            result = result.squeeze(0)
        
        return result
    
    def suggest_learning_rate_reduction(
        self,
        current_lr: float,
        reduction_factor: float = 0.5,
    ) -> float:
        """
        Suggest learning rate reduction based on stability violations.
        
        Args:
            current_lr: Current learning rate
            reduction_factor: Factor to reduce by (default 0.5 = 50% reduction)
        
        Returns:
            Suggested new learning rate
        """
        new_lr = current_lr * reduction_factor
        logger.warning(
            f"Stability violation detected. Suggest reducing LR: "
            f"{current_lr:.2e} -> {new_lr:.2e}"
        )
        return new_lr
    
    def log_stability_event(
        self,
        metrics: StabilityMetrics,
        step: int,
        severity: str = "WARNING",
    ):
        """
        Log stability event with full diagnostic information.
        
        Args:
            metrics: Stability metrics to log
            step: Training step number
            severity: Log severity level (INFO, WARNING, ERROR, CRITICAL)
        """
        log_func = getattr(logger, severity.lower(), logger.warning)
        
        log_func(f"=== Stability Event at Step {step} ===")
        log_func(f"Status: {'STABLE' if metrics.is_stable else 'UNSTABLE'}")
        log_func(f"Determinant condition: {metrics.det_condition:.2e}")
        log_func(f"Schatten S1 norm: {metrics.schatten_s1:.2f}")
        log_func(f"Schatten S2 norm: {metrics.schatten_s2:.2f}")
        log_func(f"Min eigenvalue: {metrics.min_eigenvalue:.2e}")
        log_func(f"Max eigenvalue: {metrics.max_eigenvalue:.2e}")
        log_func(f"Eigenvalue ratio: {metrics.eigenvalue_ratio:.2e}")
        log_func(f"Gradient norm: {metrics.gradient_norm:.2f}")
        
        if not metrics.is_stable:
            log_func(f"\nWarnings ({len(metrics.warnings)}):")
            for i, warning in enumerate(metrics.warnings, 1):
                log_func(f"  {i}. {warning}")
            
            log_func(f"\nRecommended Actions ({len(metrics.recommended_actions)}):")
            for i, action in enumerate(metrics.recommended_actions, 1):
                log_func(f"  {i}. {action}")
        
        log_func("=" * 50)
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get summary statistics of monitoring session.
        
        Returns:
            Dictionary with summary statistics
        """
        stability_rate = (
            (self.total_checks - self.stability_violations) / self.total_checks
            if self.total_checks > 0
            else 1.0
        )
        
        summary = {
            'total_checks': self.total_checks,
            'stability_violations': self.stability_violations,
            'stability_rate': stability_rate,
            'last_stable_step': self.last_stable_step,
        }
        
        # Add history statistics if available
        if self.enable_history and len(self.det_history) > 0:
            summary['history_stats'] = self.get_history_stats()
        
        return summary
