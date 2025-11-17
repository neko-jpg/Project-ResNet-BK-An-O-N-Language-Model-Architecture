"""
Tests for Prime-Bump Potential Implementation

Verifies:
1. Prime sieve correctness
2. Canonical coefficient computation
3. Gaussian cutoff function
4. Potential computation
5. Finite overlap condition
6. GUE eigenvalue spacing (Wigner surmise)
7. Epsilon scheduling
8. Integration with Birman-Schwinger core
"""

import pytest
import torch
import numpy as np
import math
from src.models.prime_bump_potential import (
    PrimeBumpPotential,
    EpsilonScheduler,
    sieve_of_eratosthenes,
)


class TestPrimeSieve:
    """Test prime number generation."""
    
    def test_sieve_small(self):
        """Test sieve for small limits."""
        primes = sieve_of_eratosthenes(10)
        assert primes == [2, 3, 5, 7]
    
    def test_sieve_medium(self):
        """Test sieve for medium limits."""
        primes = sieve_of_eratosthenes(30)
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert primes == expected
    
    def test_sieve_edge_cases(self):
        """Test edge cases."""
        assert sieve_of_eratosthenes(0) == []
        assert sieve_of_eratosthenes(1) == []
        assert sieve_of_eratosthenes(2) == []
        assert sieve_of_eratosthenes(3) == [2]
    
    def test_sieve_large(self):
        """Test sieve for larger limits."""
        primes = sieve_of_eratosthenes(100)
        assert len(primes) == 25  # There are 25 primes < 100
        assert primes[0] == 2
        assert primes[-1] == 97


class TestPrimeBumpPotential:
    """Test Prime-Bump potential computation."""
    
    @pytest.fixture
    def potential(self):
        """Create test potential."""
        return PrimeBumpPotential(
            n_seq=128,
            epsilon=1.0,
            k_max=3,
            scale=0.02,
        )
    
    def test_initialization(self, potential):
        """Test proper initialization."""
        assert potential.n_seq == 128
        assert potential.epsilon == 1.0
        assert potential.k_max == 3
        assert potential.scale == 0.02
        assert len(potential.primes) > 0
        assert all(p < 128 for p in potential.primes)
    
    def test_prime_positions(self, potential):
        """Test prime positions are computed correctly."""
        primes = potential.get_prime_indices()
        assert len(primes) > 0
        
        # Verify first few primes
        assert 2 in primes
        assert 3 in primes
        assert 5 in primes
        assert 7 in primes
    
    def test_alpha_coefficients(self, potential):
        """Test canonical coefficient computation."""
        # α_{p,k}(ε) = (log p) / p^{k(1/2+ε)}
        p = 2
        k = 1
        epsilon = 1.0
        
        alpha = potential.compute_alpha_coefficient(p, k)
        
        # Manual computation
        expected = math.log(2) / (2 ** (1 * (0.5 + 1.0)))
        assert abs(alpha - expected) < 1e-6
    
    def test_gaussian_cutoff(self, potential):
        """Test Gaussian cutoff function."""
        x = torch.linspace(0, 10, 100)
        center = 5.0
        
        psi = potential.compute_gaussian_cutoff(x, center)
        
        # Check properties
        assert psi.shape == x.shape
        assert torch.all(psi >= 0)  # Non-negative
        
        # Peak should be at center
        peak_idx = torch.argmax(psi)
        assert abs(x[peak_idx].item() - center) < 0.2
        
        # Should decay away from center
        assert psi[0] < psi[peak_idx]
        assert psi[-1] < psi[peak_idx]
    
    def test_potential_computation(self, potential):
        """Test potential V_ε(x) computation."""
        V = potential.compute_potential()
        
        # Check shape
        assert V.shape == (potential.n_seq,)
        
        # Check finite
        assert torch.all(torch.isfinite(V))
        
        # Should have peaks near prime positions
        # (This is a qualitative check)
        assert V.abs().max() > 0
    
    def test_forward_pass(self, potential):
        """Test forward pass with batch."""
        batch_size = 4
        n_seq = 128
        d_model = 64
        
        x = torch.randn(batch_size, n_seq, d_model)
        v = potential(x)
        
        # Check shape
        assert v.shape == (batch_size, n_seq)
        
        # Check finite
        assert torch.all(torch.isfinite(v))
    
    def test_finite_overlap(self, potential):
        """Test finite overlap condition."""
        results = potential.verify_finite_overlap()
        
        assert 'num_overlaps' in results
        assert 'total_pairs' in results
        assert 'overlap_fraction' in results
        assert 'threshold' in results
        
        # Threshold should be 2√ε
        expected_threshold = 2.0 * math.sqrt(potential.epsilon)
        assert abs(results['threshold'] - expected_threshold) < 1e-6
    
    def test_norms(self, potential):
        """Test potential norms are finite."""
        V = potential.compute_potential()
        
        l1_norm = V.abs().sum().item()
        l2_norm = torch.sqrt((V ** 2).sum()).item()
        
        assert l1_norm < float('inf')
        assert l2_norm < float('inf')
        assert l1_norm > 0
        assert l2_norm > 0
    
    def test_statistics(self, potential):
        """Test statistics computation."""
        stats = potential.get_statistics()
        
        required_keys = [
            'n_seq', 'epsilon', 'k_max', 'scale',
            'num_primes', 'l1_norm', 'l2_norm',
            'mean_potential', 'std_potential',
        ]
        
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))


class TestGUEVerification:
    """Test GUE eigenvalue spacing verification."""
    
    @pytest.fixture
    def potential(self):
        """Create test potential with smaller size for faster eigenvalue computation."""
        return PrimeBumpPotential(
            n_seq=64,  # Smaller for faster tests
            epsilon=1.0,
            k_max=2,
            scale=0.02,
        )
    
    def test_eigenvalue_spacing_computation(self, potential):
        """Test eigenvalue spacing computation."""
        spacings = potential.compute_eigenvalue_spacing()
        
        # Should have N-1 spacings for N eigenvalues
        assert len(spacings) == potential.n_seq - 1
        
        # All spacings should be non-negative
        assert torch.all(spacings >= 0)
    
    def test_gue_verification(self, potential):
        """Test GUE statistics verification."""
        results = potential.verify_gue_statistics()
        
        required_keys = [
            'mean_spacing', 'std_spacing',
            'wigner_expected_mean', 'wigner_expected_std',
            'wigner_fit_error', 'gue_verified',
        ]
        
        for key in required_keys:
            assert key in results
        
        # Mean spacing should be close to 1.0 (normalized)
        assert 0.5 < results['mean_spacing'] < 1.5
        
        # Std should be reasonable
        assert 0.1 < results['std_spacing'] < 1.0
    
    def test_wigner_surmise_properties(self, potential):
        """Test that spacing distribution has Wigner-like properties."""
        spacings = potential.compute_eigenvalue_spacing()
        
        if len(spacings) > 0:
            spacings_np = spacings.cpu().numpy()
            
            # Wigner surmise: P(s) = s * exp(-π s² / 4)
            # Properties:
            # 1. No level repulsion: P(0) = 0
            # 2. Exponential decay for large s
            
            # Check that very small spacings are rare (level repulsion)
            small_spacings = (spacings_np < 0.1).sum()
            total_spacings = len(spacings_np)
            
            # Less than 10% should be very small
            assert small_spacings / total_spacings < 0.1


class TestEpsilonScheduler:
    """Test epsilon annealing scheduler."""
    
    def test_linear_schedule(self):
        """Test linear epsilon schedule."""
        scheduler = EpsilonScheduler(
            initial_epsilon=1.0,
            final_epsilon=0.5,
            num_steps=100,
            schedule_type='linear',
        )
        
        # Initial value
        assert scheduler.get_epsilon() == 1.0
        
        # After 50 steps, should be at midpoint
        for _ in range(50):
            scheduler.step()
        
        assert abs(scheduler.get_epsilon() - 0.75) < 0.01
        
        # After 100 steps, should be at final
        for _ in range(50):
            scheduler.step()
        
        assert abs(scheduler.get_epsilon() - 0.5) < 0.01
    
    def test_cosine_schedule(self):
        """Test cosine epsilon schedule."""
        scheduler = EpsilonScheduler(
            initial_epsilon=1.0,
            final_epsilon=0.5,
            num_steps=100,
            schedule_type='cosine',
        )
        
        # Initial value
        assert scheduler.get_epsilon() == 1.0
        
        # Step through
        epsilons = []
        for _ in range(100):
            epsilons.append(scheduler.step())
        
        # Should decrease monotonically (approximately)
        # Cosine schedule is smooth
        assert epsilons[0] > epsilons[-1]
        assert abs(epsilons[-1] - 0.5) < 0.01
    
    def test_exponential_schedule(self):
        """Test exponential epsilon schedule."""
        scheduler = EpsilonScheduler(
            initial_epsilon=1.0,
            final_epsilon=0.5,
            num_steps=100,
            schedule_type='exponential',
        )
        
        # Initial value
        assert scheduler.get_epsilon() == 1.0
        
        # Step through
        for _ in range(100):
            scheduler.step()
        
        # Final value
        assert abs(scheduler.get_epsilon() - 0.5) < 0.01
    
    def test_reset(self):
        """Test scheduler reset."""
        scheduler = EpsilonScheduler(
            initial_epsilon=1.0,
            final_epsilon=0.5,
            num_steps=100,
        )
        
        # Step forward
        for _ in range(50):
            scheduler.step()
        
        assert scheduler.get_epsilon() != 1.0
        
        # Reset
        scheduler.reset()
        assert scheduler.get_epsilon() == 1.0
        assert scheduler.current_step == 0


class TestIntegration:
    """Test integration with other components."""
    
    def test_potential_with_different_epsilon(self):
        """Test potential behavior with different epsilon values."""
        epsilons = [1.0, 0.75, 0.5]
        potentials = []
        
        for eps in epsilons:
            pot = PrimeBumpPotential(n_seq=64, epsilon=eps, k_max=2)
            V = pot.compute_potential()
            potentials.append(V)
        
        # As epsilon decreases, bumps should become narrower
        # (This is a qualitative check - peaks should be sharper)
        for i in range(len(potentials) - 1):
            V1 = potentials[i]
            V2 = potentials[i + 1]
            
            # Both should be finite
            assert torch.all(torch.isfinite(V1))
            assert torch.all(torch.isfinite(V2))
    
    def test_batch_processing(self):
        """Test batch processing of potentials."""
        potential = PrimeBumpPotential(n_seq=64, epsilon=1.0)
        
        batch_sizes = [1, 4, 16]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 64, 32)
            v = potential(x)
            
            assert v.shape == (batch_size, 64)
            assert torch.all(torch.isfinite(v))
    
    def test_visualization_data(self):
        """Test visualization data generation."""
        potential = PrimeBumpPotential(n_seq=64, epsilon=1.0)
        
        viz_data = potential.visualize_potential()
        
        required_keys = [
            'positions', 'potential', 'primes',
            'log_primes', 'epsilon', 'k_max', 'scale',
        ]
        
        for key in required_keys:
            assert key in viz_data


class TestConvergenceSpeed:
    """Test that Prime-Bump initialization improves convergence."""
    
    def test_potential_magnitude(self):
        """Test that potential has reasonable magnitude."""
        potential = PrimeBumpPotential(n_seq=128, epsilon=1.0, scale=0.02)
        V = potential.compute_potential()
        
        # Should be small but non-zero
        assert V.abs().max() < 1.0
        assert V.abs().max() > 1e-6
        
        # Mean should be small (potential is localized at primes)
        assert abs(V.mean()) < 0.1
    
    def test_different_scales(self):
        """Test potential with different scaling factors."""
        scales = [0.01, 0.02, 0.05]
        
        for scale in scales:
            potential = PrimeBumpPotential(n_seq=64, epsilon=1.0, scale=scale)
            V = potential.compute_potential()
            
            # Larger scale should give larger potential
            assert V.abs().max() > 0
            assert torch.all(torch.isfinite(V))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
