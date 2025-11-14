"""
Unit Tests for BK-Core
Tests theta/phi recursion correctness and numerical stability.
"""

import torch
import pytest
import sys
sys.path.insert(0, '.')

from src.models.bk_core import (
    get_tridiagonal_inverse_diagonal,
    vmapped_get_diag,
    BKCoreFunction
)


class TestBKCore:
    """Test suite for BK-Core theta/phi recursion."""
    
    def test_theta_phi_correctness_small(self):
        """Test theta/phi recursion on small matrix (N=4)."""
        N = 4
        device = torch.device('cpu')
        
        # Create simple tridiagonal matrix
        a = torch.tensor([-2.0, -2.0, -2.0, -2.0], device=device)
        b = torch.tensor([1.0, 1.0, 1.0], device=device)
        c = torch.tensor([1.0, 1.0, 1.0], device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        
        # Compute using BK-Core
        G_ii = get_tridiagonal_inverse_diagonal(a, b, c, z)
        
        # Verify output shape and type
        assert G_ii.shape == (N,), f"Expected shape ({N},), got {G_ii.shape}"
        assert G_ii.dtype == torch.complex64, f"Expected complex64, got {G_ii.dtype}"
        
        # Verify no NaN or Inf
        assert torch.all(torch.isfinite(G_ii)), "G_ii contains NaN or Inf"
        
        # Verify magnitude is reasonable
        mag = G_ii.abs()
        assert torch.all(mag < 50.0), f"G_ii magnitude too large: max={mag.max()}"
        assert torch.all(mag > 0.0), f"G_ii magnitude too small: min={mag.min()}"
    
    def test_theta_phi_vs_direct_inverse(self):
        """Compare BK-Core result with direct matrix inversion."""
        N = 8
        device = torch.device('cpu')
        
        # Create tridiagonal matrix
        a = torch.randn(N, device=device) * 0.5 - 2.0
        b = torch.ones(N-1, device=device)
        c = torch.ones(N-1, device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        
        # Compute using BK-Core
        G_ii_bk = get_tridiagonal_inverse_diagonal(a, b, c, z)
        
        # Compute using direct inversion
        T = torch.zeros(N, N, dtype=torch.complex128, device=device)
        for i in range(N):
            T[i, i] = a[i].to(torch.complex128)
        for i in range(N-1):
            T[i, i+1] = b[i].to(torch.complex128)
            T[i+1, i] = c[i].to(torch.complex128)
        
        T_shifted = T - z.to(torch.complex128) * torch.eye(N, dtype=torch.complex128, device=device)
        T_inv = torch.linalg.inv(T_shifted)
        G_ii_direct = torch.diag(T_inv).to(torch.complex64)
        
        # Compare results (allow some numerical error)
        rel_error = torch.abs(G_ii_bk - G_ii_direct) / (torch.abs(G_ii_direct) + 1e-6)
        max_rel_error = rel_error.max().item()
        
        assert max_rel_error < 1e-3, f"BK-Core vs direct inversion: max relative error = {max_rel_error}"
    
    def test_batched_bk_core(self):
        """Test batched BK-Core computation."""
        B = 4
        N = 16
        device = torch.device('cpu')
        
        # Create batched inputs
        a = torch.randn(B, N, device=device) * 0.5 - 2.0
        b = torch.ones(B, N-1, device=device)
        c = torch.ones(B, N-1, device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        
        # Compute batched
        G_ii_batched = vmapped_get_diag(a, b, c, z)
        
        # Verify shape
        assert G_ii_batched.shape == (B, N), f"Expected shape ({B}, {N}), got {G_ii_batched.shape}"
        
        # Verify no NaN or Inf
        assert torch.all(torch.isfinite(G_ii_batched)), "Batched G_ii contains NaN or Inf"
        
        # Compare with sequential computation
        G_ii_sequential = []
        for i in range(B):
            G_ii_i = get_tridiagonal_inverse_diagonal(a[i], b[i], c[i], z)
            G_ii_sequential.append(G_ii_i)
        G_ii_sequential = torch.stack(G_ii_sequential)
        
        # Should be identical
        assert torch.allclose(G_ii_batched, G_ii_sequential, rtol=1e-5, atol=1e-6), \
            "Batched and sequential results differ"
    
    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme potential values."""
        N = 16
        device = torch.device('cpu')
        
        # Test with large diagonal values
        a_large = torch.full((N,), 100.0, device=device)
        b = torch.ones(N-1, device=device)
        c = torch.ones(N-1, device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        
        G_ii_large = get_tridiagonal_inverse_diagonal(a_large, b, c, z)
        assert torch.all(torch.isfinite(G_ii_large)), "Failed with large diagonal values"
        
        # Test with small diagonal values
        a_small = torch.full((N,), -0.01, device=device)
        G_ii_small = get_tridiagonal_inverse_diagonal(a_small, b, c, z)
        assert torch.all(torch.isfinite(G_ii_small)), "Failed with small diagonal values"
    
    def test_bk_core_autograd_function(self):
        """Test BKCoreFunction forward and backward."""
        B = 2
        N = 8
        device = torch.device('cpu')
        
        # Create inputs
        he_diag = torch.randn(B, N, device=device, requires_grad=True)
        h0_super = torch.ones(B, N-1, device=device)
        h0_sub = torch.ones(B, N-1, device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        
        # Forward pass
        features = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
        
        # Verify output shape
        assert features.shape == (B, N, 2), f"Expected shape ({B}, {N}, 2), got {features.shape}"
        assert features.dtype == torch.float32, f"Expected float32, got {features.dtype}"
        
        # Backward pass
        loss = features.sum()
        loss.backward()
        
        # Verify gradients exist and are finite
        assert he_diag.grad is not None, "No gradient for he_diag"
        assert torch.all(torch.isfinite(he_diag.grad)), "Gradient contains NaN or Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
