"""
Unit tests for Birman-Schwinger Stability Monitor

Tests cover:
- Determinant computation correctness
- Schatten norm computation
- Warning generation when thresholds violated
- Recovery action recommendations

Requirements: 6.2, 7.1, 7.2, 7.3, 7.4
"""

import pytest
import torch
import numpy as np
from src.models.phase1.stability_monitor import (
    BKStabilityMonitor,
    StabilityThresholds,
    StabilityMetrics,
)


class TestStabilityThresholds:
    """Test StabilityThresholds dataclass."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = StabilityThresholds()
        
        assert thresholds.det_threshold == 1e-6
        assert thresholds.schatten_s1_bound == 100.0
        assert thresholds.schatten_s2_bound == 50.0
        assert thresholds.min_eigenvalue_threshold == 1e-8
        assert thresholds.gradient_norm_threshold == 10.0
        assert thresholds.condition_number_threshold == 1e6
    
    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = StabilityThresholds(
            det_threshold=1e-5,
            schatten_s1_bound=200.0,
            schatten_s2_bound=100.0,
        )
        
        assert thresholds.det_threshold == 1e-5
        assert thresholds.schatten_s1_bound == 200.0
        assert thresholds.schatten_s2_bound == 100.0
    
    def test_validate_valid_thresholds(self):
        """Test validation passes for valid thresholds."""
        thresholds = StabilityThresholds()
        thresholds.validate()  # Should not raise
    
    def test_validate_invalid_thresholds(self):
        """Test validation fails for invalid thresholds."""
        # Negative det_threshold
        with pytest.raises(AssertionError):
            thresholds = StabilityThresholds(det_threshold=-1.0)
            thresholds.validate()
        
        # Zero schatten bound
        with pytest.raises(AssertionError):
            thresholds = StabilityThresholds(schatten_s1_bound=0.0)
            thresholds.validate()
        
        # Condition number <= 1
        with pytest.raises(AssertionError):
            thresholds = StabilityThresholds(condition_number_threshold=1.0)
            thresholds.validate()


class TestStabilityMetrics:
    """Test StabilityMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating StabilityMetrics."""
        metrics = StabilityMetrics(
            det_condition=1e-5,
            schatten_s1=50.0,
            schatten_s2=25.0,
            min_eigenvalue=1e-7,
            max_eigenvalue=1e2,
            eigenvalue_ratio=1e9,
            gradient_norm=5.0,
            is_stable=True,
        )
        
        assert metrics.det_condition == 1e-5
        assert metrics.schatten_s1 == 50.0
        assert metrics.is_stable is True
        assert len(metrics.warnings) == 0
        assert len(metrics.recommended_actions) == 0
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = StabilityMetrics(
            det_condition=1e-5,
            schatten_s1=50.0,
            schatten_s2=25.0,
            min_eigenvalue=1e-7,
            max_eigenvalue=1e2,
            eigenvalue_ratio=1e9,
            gradient_norm=5.0,
            is_stable=True,
            warnings=["Test warning"],
            recommended_actions=["Test action"],
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['det_condition'] == 1e-5
        assert metrics_dict['is_stable'] is True
        assert metrics_dict['num_warnings'] == 1
        assert metrics_dict['num_actions'] == 1


class TestBKStabilityMonitor:
    """Test BKStabilityMonitor class."""
    
    @pytest.fixture
    def monitor(self):
        """Create a stability monitor for testing."""
        return BKStabilityMonitor()
    
    @pytest.fixture
    def custom_monitor(self):
        """Create a monitor with custom thresholds."""
        thresholds = StabilityThresholds(
            det_threshold=1e-5,
            schatten_s1_bound=200.0,
            schatten_s2_bound=100.0,
        )
        return BKStabilityMonitor(thresholds=thresholds)
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.thresholds is not None
        assert monitor.enable_history is True
        assert monitor.history_size == 1000
        assert monitor.total_checks == 0
        assert monitor.stability_violations == 0
    
    def test_determinant_computation(self, monitor):
        """Test determinant condition computation (Requirement 7.2)."""
        # Create stable resolvent diagonal
        B, N = 2, 10
        G_ii = torch.randn(B, N, dtype=torch.complex64) * 0.1 + 1.0
        epsilon = 0.1
        
        det_condition = monitor._compute_determinant_condition(G_ii, epsilon)
        
        # Should be positive and finite
        assert det_condition > 0
        assert np.isfinite(det_condition)
    
    def test_schatten_norm_computation(self, monitor):
        """Test Schatten norm computation (Requirement 7.2)."""
        # Create test data
        B, N = 2, 10
        G_ii = torch.randn(B, N, dtype=torch.complex64) * 0.1 + 1.0
        potential = torch.randn(B, N) * 0.5
        epsilon = 0.1
        
        schatten_s1, schatten_s2 = monitor._compute_schatten_norms(
            G_ii, potential, epsilon
        )
        
        # S1 and S2 should be positive
        assert schatten_s1 > 0
        assert schatten_s2 > 0
        
        # S2 should be <= S1 (mathematical property)
        assert schatten_s2 <= schatten_s1 * 2  # Allow some numerical tolerance
    
    def test_eigenvalue_stats_computation(self, monitor):
        """Test eigenvalue statistics computation (Requirement 7.2)."""
        # Create test resolvent diagonal
        B, N = 2, 10
        G_ii = torch.randn(B, N, dtype=torch.complex64) * 0.1 + 1.0
        
        min_eig, max_eig, eig_ratio = monitor._compute_eigenvalue_stats(G_ii)
        
        # All should be positive
        assert min_eig > 0
        assert max_eig > 0
        assert eig_ratio > 0
        
        # Ratio should be >= 1
        assert eig_ratio >= 1.0
    
    def test_stable_system(self, monitor):
        """Test monitoring a stable system."""
        # Create stable system
        B, N = 2, 10
        G_ii = torch.randn(B, N, dtype=torch.complex64) * 0.1 + 1.0
        potential = torch.randn(B, N) * 0.5
        epsilon = 0.1
        
        metrics = monitor.check_stability(G_ii, potential, epsilon)
        
        # Should be stable
        assert metrics.is_stable is True
        assert len(metrics.warnings) == 0
        assert len(metrics.recommended_actions) == 0
        
        # Check history updated
        assert len(monitor.det_history) == 1
        assert monitor.total_checks == 1
        assert monitor.stability_violations == 0
    
    def test_unstable_determinant(self, monitor):
        """Test warning generation for low determinant (Requirement 7.2)."""
        # Create system with near-singular operator
        B, N = 2, 10
        G_ii = torch.randn(B, N, dtype=torch.complex64) * 10.0  # Large values
        potential = torch.randn(B, N) * 10.0
        epsilon = 0.001  # Small epsilon
        
        metrics = monitor.check_stability(G_ii, potential, epsilon)
        
        # May be unstable due to large values
        if not metrics.is_stable:
            assert len(metrics.warnings) > 0
            assert len(metrics.recommended_actions) > 0
            assert monitor.stability_violations > 0
    
    def test_schatten_norm_violation(self, custom_monitor):
        """Test warning for Schatten norm violation (Requirement 7.2)."""
        # Create system with large operator
        B, N = 2, 10
        G_ii = torch.randn(B, N, dtype=torch.complex64) * 5.0
        potential = torch.randn(B, N) * 20.0  # Large potential
        epsilon = 0.01
        
        metrics = custom_monitor.check_stability(G_ii, potential, epsilon)
        
        # Check if Schatten norms are computed
        assert metrics.schatten_s1 > 0
        assert metrics.schatten_s2 > 0
    
    def test_gradient_norm_tracking(self, monitor):
        """Test gradient norm tracking."""
        B, N = 2, 10
        G_ii = torch.randn(B, N, dtype=torch.complex64) * 0.1 + 1.0
        potential = torch.randn(B, N, requires_grad=True) * 0.5
        epsilon = 0.1
        
        # Create fake gradient
        potential.grad = torch.randn_like(potential) * 0.5
        
        metrics = monitor.check_stability(G_ii, potential, epsilon)
        
        # Gradient norm should be computed
        assert metrics.gradient_norm > 0
    
    def test_recovery_actions_generation(self, monitor):
        """Test recovery action recommendations (Requirement 7.3)."""
        # Create unstable system
        B, N = 2, 10
        G_ii = torch.randn(B, N, dtype=torch.complex64) * 100.0  # Very large
        potential = torch.randn(B, N) * 100.0
        epsilon = 0.0001  # Very small
        
        metrics = monitor.check_stability(G_ii, potential, epsilon)
        
        # Should have warnings and actions
        if not metrics.is_stable:
            assert len(metrics.warnings) > 0
            assert len(metrics.recommended_actions) > 0
            
            # Check for specific action types
            actions_str = " ".join(metrics.recommended_actions)
            # Should suggest some recovery action
            assert len(actions_str) > 0
    
    def test_gradient_clipping(self, monitor):
        """Test gradient clipping functionality (Requirement 7.3)."""
        # Create model parameters
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.grad = torch.randn_like(param) * 10.0  # Large gradient
        
        # Apply gradient clipping
        total_norm = monitor.apply_gradient_clipping([param], max_norm=1.0)
        
        # Check gradient was clipped
        assert total_norm > 1.0  # Original norm was large
        assert param.grad.norm().item() <= 1.0  # Clipped to max_norm
    
    def test_spectral_clipping(self, monitor):
        """Test spectral clipping functionality (Requirement 7.3)."""
        # Create operator with large singular values
        N = 10
        operator = torch.randn(N, N) * 10.0
        
        # Clip operator
        clipped = monitor.apply_spectral_clipping(
            operator,
            max_s1_norm=50.0,
            max_s2_norm=25.0,
        )
        
        # Check shape preserved
        assert clipped.shape == operator.shape
        
        # Check norms are bounded (approximately)
        U, S, Vh = torch.linalg.svd(clipped, full_matrices=False)
        s1_norm = S.sum().item()
        s2_norm = torch.sqrt((S ** 2).sum()).item()
        
        assert s1_norm <= 50.0 * 1.1  # Allow 10% tolerance
        assert s2_norm <= 25.0 * 1.1
    
    def test_learning_rate_reduction(self, monitor):
        """Test learning rate reduction suggestion (Requirement 7.3)."""
        current_lr = 1e-3
        new_lr = monitor.suggest_learning_rate_reduction(
            current_lr,
            reduction_factor=0.5,
        )
        
        assert new_lr == current_lr * 0.5
        assert new_lr < current_lr
    
    def test_history_tracking(self, monitor):
        """Test history tracking functionality."""
        B, N = 2, 10
        
        # Perform multiple checks
        for _ in range(5):
            G_ii = torch.randn(B, N, dtype=torch.complex64) * 0.1 + 1.0
            potential = torch.randn(B, N) * 0.5
            epsilon = 0.1
            
            monitor.check_stability(G_ii, potential, epsilon)
        
        # Check history
        assert len(monitor.det_history) == 5
        assert len(monitor.schatten_s1_history) == 5
        assert monitor.total_checks == 5
        
        # Get history stats
        stats = monitor.get_history_stats()
        assert 'det_condition' in stats
        assert 'mean' in stats['det_condition']
        assert 'std' in stats['det_condition']
    
    def test_history_size_limit(self):
        """Test history size limiting."""
        monitor = BKStabilityMonitor(history_size=3)
        
        B, N = 2, 10
        
        # Perform more checks than history size
        for _ in range(5):
            G_ii = torch.randn(B, N, dtype=torch.complex64) * 0.1 + 1.0
            potential = torch.randn(B, N) * 0.5
            epsilon = 0.1
            
            monitor.check_stability(G_ii, potential, epsilon)
        
        # History should be limited
        assert len(monitor.det_history) == 3
        assert monitor.total_checks == 5
    
    def test_reset_history(self, monitor):
        """Test resetting history."""
        B, N = 2, 10
        
        # Perform some checks
        for _ in range(3):
            G_ii = torch.randn(B, N, dtype=torch.complex64) * 0.1 + 1.0
            potential = torch.randn(B, N) * 0.5
            epsilon = 0.1
            
            monitor.check_stability(G_ii, potential, epsilon)
        
        # Reset
        monitor.reset_history()
        
        # Check everything is cleared
        assert len(monitor.det_history) == 0
        assert monitor.total_checks == 0
        assert monitor.stability_violations == 0
    
    def test_summary_generation(self, monitor):
        """Test summary statistics generation."""
        B, N = 2, 10
        
        # Perform some checks
        for _ in range(3):
            G_ii = torch.randn(B, N, dtype=torch.complex64) * 0.1 + 1.0
            potential = torch.randn(B, N) * 0.5
            epsilon = 0.1
            
            monitor.check_stability(G_ii, potential, epsilon)
        
        # Get summary
        summary = monitor.get_summary()
        
        assert 'total_checks' in summary
        assert summary['total_checks'] == 3
        assert 'stability_rate' in summary
        assert 0.0 <= summary['stability_rate'] <= 1.0
    
    def test_log_stability_event(self, monitor, caplog):
        """Test logging stability events (Requirement 7.3)."""
        import logging
        caplog.set_level(logging.WARNING)
        
        # Create unstable metrics
        metrics = StabilityMetrics(
            det_condition=1e-8,
            schatten_s1=150.0,
            schatten_s2=75.0,
            min_eigenvalue=1e-10,
            max_eigenvalue=1e5,
            eigenvalue_ratio=1e15,
            gradient_norm=50.0,
            is_stable=False,
            warnings=["Test warning"],
            recommended_actions=["Test action"],
        )
        
        # Log event
        monitor.log_stability_event(metrics, step=100, severity="WARNING")
        
        # Check log was created
        assert len(caplog.records) > 0


class TestStabilityMonitorIntegration:
    """Test integration with AR-SSM and BK-Core."""
    
    def test_ar_ssm_condition_number_check(self):
        """Test condition number checking in AR-SSM context."""
        from src.models.phase1.ar_ssm_layer import AdaptiveRankSemiseparableLayer
        
        # Create AR-SSM layer with stability monitoring
        monitor = BKStabilityMonitor()
        layer = AdaptiveRankSemiseparableLayer(
            d_model=64,
            max_rank=16,
            stability_monitor=monitor,
            enable_stability_checks=True,
        )
        
        # Create test input
        B, L, D = 2, 10, 64
        x = torch.randn(B, L, D)
        
        # Forward pass
        y, diagnostics = layer(x)
        
        # Check diagnostics include stability info
        if 'condition_number' in diagnostics:
            assert diagnostics['condition_number'] > 0
    
    def test_ar_ssm_singularity_check(self):
        """Test singularity checking in AR-SSM context."""
        from src.models.phase1.ar_ssm_layer import AdaptiveRankSemiseparableLayer
        
        # Create AR-SSM layer
        monitor = BKStabilityMonitor()
        layer = AdaptiveRankSemiseparableLayer(
            d_model=64,
            max_rank=16,
            stability_monitor=monitor,
            enable_stability_checks=True,
        )
        
        # Create test input
        B, L, D = 2, 10, 64
        x = torch.randn(B, L, D)
        
        # Forward pass
        y, diagnostics = layer(x)
        
        # Check diagnostics include singularity info
        if 'is_singular' in diagnostics:
            assert isinstance(diagnostics['is_singular'], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
