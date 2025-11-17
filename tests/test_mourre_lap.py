"""
Tests for Mourre Estimate and LAP Verification

Tests verify:
1. Mourre estimate: [H_0, iA] = I
2. LAP: Uniform resolvent bounds
3. Stability dashboard functionality
"""

import pytest
import torch
import numpy as np
from src.models.mourre_lap import (
    MourreEstimateVerifier,
    LAPVerifier,
    StabilityDashboard,
    StabilityMetrics,
    verify_birman_schwinger_stability,
)


class TestMourreEstimateVerifier:
    """Test Mourre estimate verification."""
    
    def test_initialization(self):
        """Test verifier initialization."""
        n_seq = 64
        verifier = MourreEstimateVerifier(n_seq)
        
        assert verifier.n_seq == n_seq
        assert verifier.H_0.shape == (n_seq, n_seq)
        assert verifier.A.shape == (n_seq, n_seq)
        assert verifier.commutator.shape == (n_seq, n_seq)
    
    def test_free_hamiltonian_structure(self):
        """Test that H_0 has correct tridiagonal structure."""
        n_seq = 64
        verifier = MourreEstimateVerifier(n_seq)
        H_0 = verifier.H_0
        
        # Check main diagonal is -2
        assert torch.allclose(H_0.diagonal(), torch.full((n_seq,), -2.0))
        
        # Check off-diagonals are 1
        if n_seq > 1:
            assert torch.allclose(H_0.diagonal(1), torch.ones(n_seq - 1))
            assert torch.allclose(H_0.diagonal(-1), torch.ones(n_seq - 1))
    
    def test_position_operator_structure(self):
        """Test that A is diagonal with positions."""
        n_seq = 64
        verifier = MourreEstimateVerifier(n_seq)
        A = verifier.A
        
        # Check diagonal contains positions
        expected_positions = torch.arange(n_seq, dtype=torch.float32)
        assert torch.allclose(A.diagonal(), expected_positions)
        
        # Check off-diagonals are zero
        if n_seq > 1:
            assert torch.allclose(A.diagonal(1), torch.zeros(n_seq - 1))
            assert torch.allclose(A.diagonal(-1), torch.zeros(n_seq - 1))
    
    def test_mourre_estimate_verification(self):
        """Test Mourre estimate: [H_0, iA] has positive commutator."""
        n_seq = 64
        verifier = MourreEstimateVerifier(n_seq)
        results = verifier.verify()
        
        # Check that verification passes
        assert results['verified'], f"Mourre estimate not verified: commutator_norm={results['commutator_norm']}"
        
        # Check that commutator is non-trivial (not all zeros)
        assert results['commutator_norm'] > 0.5, f"Commutator norm {results['commutator_norm']} too small"
        
        # Check hermitian error is small
        assert results['hermitian_error'] < 1e-5, f"Hermitian error {results['hermitian_error']} too large"
    
    def test_mourre_estimate_different_sizes(self):
        """Test Mourre estimate for different sequence lengths."""
        for n_seq in [16, 32, 64, 128]:
            verifier = MourreEstimateVerifier(n_seq)
            results = verifier.verify()
            
            assert results['verified'], f"Mourre estimate failed for n_seq={n_seq}, commutator_norm={results['commutator_norm']}"
            assert results['commutator_norm'] > 0.5, f"Commutator norm too small for n_seq={n_seq}"


class TestLAPVerifier:
    """Test LAP verification."""
    
    def test_initialization(self):
        """Test LAP verifier initialization."""
        n_seq = 64
        s = 1.0
        verifier = LAPVerifier(n_seq, s)
        
        assert verifier.n_seq == n_seq
        assert verifier.s == s
        assert verifier.weight.shape == (n_seq,)
    
    def test_weight_function(self):
        """Test weight function ⟨x⟩^{-s} = (1 + x²)^{-s/2}."""
        n_seq = 64
        s = 1.0
        verifier = LAPVerifier(n_seq, s)
        
        # Check weight decreases with position
        assert verifier.weight[0] > verifier.weight[-1]
        
        # Check weight at origin is 1
        assert torch.isclose(verifier.weight[0], torch.tensor(1.0))
        
        # Check weight formula
        positions = torch.arange(n_seq, dtype=torch.float32)
        expected_weight = (1.0 + positions ** 2) ** (-s / 2.0)
        assert torch.allclose(verifier.weight, expected_weight)
    
    def test_invalid_s_parameter(self):
        """Test that s <= 0.5 raises error."""
        n_seq = 64
        
        with pytest.raises(ValueError, match="LAP requires s > 1/2"):
            LAPVerifier(n_seq, s=0.5)
        
        with pytest.raises(ValueError, match="LAP requires s > 1/2"):
            LAPVerifier(n_seq, s=0.0)
    
    def test_weighted_resolvent_computation(self):
        """Test weighted resolvent computation."""
        n_seq = 32
        verifier = LAPVerifier(n_seq, s=1.0)
        
        # Create simple Hamiltonian
        H = torch.eye(n_seq) * 2.0
        
        # Compute weighted resolvent
        lambda_ = 0.0
        eta = 0.1
        weighted_resolvent = verifier.compute_weighted_resolvent(H, lambda_, eta)
        
        assert weighted_resolvent.shape == (n_seq, n_seq)
        assert torch.isfinite(weighted_resolvent).all()
    
    def test_uniform_bounds_verification(self):
        """Test uniform bounds as η → 0."""
        n_seq = 32
        verifier = LAPVerifier(n_seq, s=1.0)
        
        # Create free Hamiltonian
        mourre_verifier = MourreEstimateVerifier(n_seq)
        H_0 = mourre_verifier.H_0
        
        # Test uniform bounds
        eta_values = [1.0, 0.5, 0.1, 0.05, 0.01]
        results = verifier.verify_uniform_bounds(H_0, lambda_=0.0, eta_values=eta_values, C_bound=100.0)
        
        assert 'norms' in results
        assert len(results['norms']) == len(eta_values)
        assert all(norm < 100.0 for norm in results['norms']), "Norms exceed bound"
    
    def test_continuity_at_boundary(self):
        """Test continuity as η → 0."""
        n_seq = 32
        verifier = LAPVerifier(n_seq, s=1.0)
        
        # Create free Hamiltonian
        mourre_verifier = MourreEstimateVerifier(n_seq)
        H_0 = mourre_verifier.H_0
        
        # Test continuity
        eta_sequence = [1.0, 0.5, 0.1, 0.05, 0.01]
        results = verifier.verify_continuity_at_boundary(H_0, lambda_=0.0, eta_sequence=eta_sequence)
        
        assert 'differences' in results
        assert len(results['differences']) == len(eta_sequence) - 1


class TestStabilityDashboard:
    """Test stability dashboard."""
    
    def test_initialization(self):
        """Test dashboard initialization."""
        n_seq = 64
        dashboard = StabilityDashboard(n_seq)
        
        assert dashboard.n_seq == n_seq
        assert len(dashboard.metrics_history) == 0
        assert len(dashboard.alerts) == 0
    
    def test_update_basic(self):
        """Test basic metric update."""
        n_seq = 64
        dashboard = StabilityDashboard(n_seq)
        
        metrics = dashboard.update(step=0)
        
        assert metrics.step == 0
        assert metrics.mourre_verified
        assert len(dashboard.metrics_history) == 1
    
    def test_update_with_hamiltonian(self):
        """Test update with Hamiltonian matrix."""
        n_seq = 32
        dashboard = StabilityDashboard(n_seq)
        
        # Create simple Hamiltonian
        H = torch.eye(n_seq) * 2.0
        
        metrics = dashboard.update(step=0, H=H)
        
        assert metrics.lap_verified
        assert metrics.condition_number_Heps > 0
    
    def test_update_with_birman_schwinger_operator(self):
        """Test update with Birman-Schwinger operator."""
        n_seq = 32
        dashboard = StabilityDashboard(n_seq)
        
        # Create mock K and V
        K = torch.randn(2, n_seq, n_seq, dtype=torch.complex64) * 0.1
        V = torch.randn(2, n_seq) * 0.5
        
        metrics = dashboard.update(step=0, K=K, V=V)
        
        assert metrics.schatten_s1 > 0
        assert metrics.schatten_s2 > 0
        assert metrics.schatten_s1_bound > 0
        assert metrics.schatten_s2_bound > 0
    
    def test_nan_detection(self):
        """Test NaN detection."""
        n_seq = 32
        dashboard = StabilityDashboard(n_seq)
        
        # Create tensor with NaN
        tensors = {
            'test': torch.tensor([1.0, 2.0, float('nan'), 4.0])
        }
        
        metrics = dashboard.update(step=0, tensors=tensors)
        
        assert metrics.has_nan
        assert not metrics.all_finite
        assert len(dashboard.alerts) > 0
    
    def test_inf_detection(self):
        """Test Inf detection."""
        n_seq = 32
        dashboard = StabilityDashboard(n_seq)
        
        # Create tensor with Inf
        tensors = {
            'test': torch.tensor([1.0, 2.0, float('inf'), 4.0])
        }
        
        metrics = dashboard.update(step=0, tensors=tensors)
        
        assert metrics.has_inf
        assert not metrics.all_finite
        assert len(dashboard.alerts) > 0
    
    def test_condition_number_alert(self):
        """Test condition number alert."""
        n_seq = 32
        dashboard = StabilityDashboard(n_seq)
        
        # Set low threshold
        dashboard.set_threshold('condition_number_max', 10.0)
        
        # Create ill-conditioned matrix
        H = torch.eye(n_seq)
        H[0, 0] = 1e-6  # Make it ill-conditioned
        
        metrics = dashboard.update(step=0, H=H)
        
        # Should trigger alert
        assert len(dashboard.alerts) > 0
    
    def test_get_summary(self):
        """Test summary statistics."""
        n_seq = 32
        dashboard = StabilityDashboard(n_seq)
        
        # Add multiple metrics
        for step in range(10):
            dashboard.update(step=step)
        
        summary = dashboard.get_summary()
        
        assert summary['total_steps'] == 10
        assert 'condition_number' in summary
        assert 'mourre_error' in summary
        assert summary['mourre_verified_rate'] > 0
    
    def test_export_metrics(self):
        """Test metric export."""
        n_seq = 32
        dashboard = StabilityDashboard(n_seq)
        
        # Add metrics
        for step in range(5):
            dashboard.update(step=step)
        
        exported = dashboard.export_metrics()
        
        assert len(exported) == 5
        assert all('step' in m for m in exported)
        assert all('mourre_constant' in m for m in exported)
    
    def test_alert_management(self):
        """Test alert management."""
        n_seq = 32
        dashboard = StabilityDashboard(n_seq)
        
        # Trigger some alerts
        tensors = {'test': torch.tensor([float('nan')])}
        dashboard.update(step=0, tensors=tensors)
        dashboard.update(step=1, tensors=tensors)
        
        assert len(dashboard.alerts) >= 2
        
        recent = dashboard.get_recent_alerts(n=1)
        assert len(recent) == 1
        
        dashboard.clear_alerts()
        assert len(dashboard.alerts) == 0


class TestComprehensiveVerification:
    """Test comprehensive stability verification."""
    
    def test_verify_birman_schwinger_stability(self):
        """Test comprehensive verification function."""
        n_seq = 64
        epsilon = 1.0
        
        results = verify_birman_schwinger_stability(n_seq, epsilon)
        
        assert 'mourre' in results
        assert 'lap_uniform_bounds' in results
        assert 'lap_continuity' in results
        assert results['epsilon'] == epsilon
        assert results['n_seq'] == n_seq
    
    def test_verification_different_epsilon(self):
        """Test verification with different epsilon values."""
        n_seq = 32
        
        for epsilon in [1.0, 0.75, 0.5]:
            results = verify_birman_schwinger_stability(n_seq, epsilon)
            
            # Mourre estimate should always hold
            assert results['mourre']['verified']
    
    def test_verification_different_sizes(self):
        """Test verification for different sequence lengths."""
        for n_seq in [16, 32, 64]:
            results = verify_birman_schwinger_stability(n_seq, epsilon=1.0)
            
            # Basic checks should pass
            assert results['mourre']['verified']


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_dashboard_with_full_workflow(self):
        """Test dashboard with complete workflow."""
        n_seq = 32
        dashboard = StabilityDashboard(n_seq)
        
        # Simulate training loop
        for step in range(20):
            # Create mock data
            H = torch.eye(n_seq) * (2.0 + 0.1 * step)
            K = torch.randn(2, n_seq, n_seq, dtype=torch.complex64) * 0.1
            V = torch.randn(2, n_seq) * 0.5
            tensors = {
                'activations': torch.randn(2, n_seq, 64),
                'gradients': torch.randn(2, n_seq, 64) * 0.01,
            }
            
            metrics = dashboard.update(
                step=step,
                H=H,
                K=K,
                V=V,
                tensors=tensors
            )
            
            assert metrics.all_finite
        
        # Check summary
        summary = dashboard.get_summary()
        assert summary['total_steps'] == 20
        assert summary['nan_count'] == 0
        assert summary['inf_count'] == 0
    
    def test_stability_monitoring_over_time(self):
        """Test stability monitoring over extended period."""
        n_seq = 32
        dashboard = StabilityDashboard(n_seq)
        
        condition_numbers = []
        
        for step in range(50):
            metrics = dashboard.update(step=step)
            condition_numbers.append(metrics.condition_number)
        
        # Condition numbers should remain bounded
        assert all(cn < 1e6 for cn in condition_numbers)
        
        # Summary should show stable behavior
        summary = dashboard.get_summary()
        assert summary['condition_number']['max'] < 1e6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
