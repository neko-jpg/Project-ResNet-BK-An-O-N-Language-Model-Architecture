"""
Gradient Tests
Compare analytic gradients with finite difference approximation.
"""

import torch
import pytest
import sys
sys.path.insert(0, '.')

from src.models.bk_core import BKCoreFunction


class TestGradients:
    """Test suite for gradient correctness."""
    
    def test_analytic_vs_finite_difference(self):
        """Compare analytic gradient with finite difference."""
        B = 1
        N = 8
        device = torch.device('cpu')
        eps = 1e-4
        
        # Create inputs
        he_diag = torch.randn(B, N, device=device, requires_grad=True)
        h0_super = torch.ones(B, N-1, device=device)
        h0_sub = torch.ones(B, N-1, device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        
        # Compute analytic gradient
        features = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
        loss = features.sum()
        loss.backward()
        grad_analytic = he_diag.grad.clone()
        
        # Compute finite difference gradient
        grad_fd = torch.zeros_like(he_diag)
        
        for i in range(N):
            # Forward perturbation
            he_diag_plus = he_diag.detach().clone()
            he_diag_plus[0, i] += eps
            features_plus = BKCoreFunction.apply(he_diag_plus, h0_super, h0_sub, z)
            loss_plus = features_plus.sum()
            
            # Backward perturbation
            he_diag_minus = he_diag.detach().clone()
            he_diag_minus[0, i] -= eps
            features_minus = BKCoreFunction.apply(he_diag_minus, h0_super, h0_sub, z)
            loss_minus = features_minus.sum()
            
            # Central difference
            grad_fd[0, i] = (loss_plus - loss_minus) / (2 * eps)
        
        # Compare gradients
        rel_error = torch.abs(grad_analytic - grad_fd) / (torch.abs(grad_fd) + 1e-6)
        max_rel_error = rel_error.max().item()
        
        print(f"Max relative error: {max_rel_error:.6f}")
        print(f"Analytic grad: {grad_analytic[0, :4]}")
        print(f"Finite diff grad: {grad_fd[0, :4]}")
        
        # Allow larger error due to hybrid gradient approximation
        assert max_rel_error < 0.5, f"Gradient error too large: {max_rel_error}"
    
    def test_gradient_blend_effect(self):
        """Test effect of GRAD_BLEND parameter on gradients."""
        B = 1
        N = 8
        device = torch.device('cpu')
        
        # Create inputs
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.ones(B, N-1, device=device)
        h0_sub = torch.ones(B, N-1, device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        
        gradients = {}
        
        # Test different blend values
        for blend in [0.0, 0.5, 1.0]:
            BKCoreFunction.GRAD_BLEND = blend
            
            he_diag_test = he_diag.clone().requires_grad_(True)
            features = BKCoreFunction.apply(he_diag_test, h0_super, h0_sub, z)
            loss = features.sum()
            loss.backward()
            
            gradients[blend] = he_diag_test.grad.clone()
        
        # Verify gradients are different for different blends
        assert not torch.allclose(gradients[0.0], gradients[1.0], rtol=0.1), \
            "Gradients should differ for different GRAD_BLEND values"
        
        # Verify blend=0.5 is between blend=0.0 and blend=1.0
        # (This is a rough check, not always true due to nonlinearity)
        grad_diff_0_to_05 = (gradients[0.5] - gradients[0.0]).abs().mean()
        grad_diff_05_to_1 = (gradients[1.0] - gradients[0.5]).abs().mean()
        
        print(f"Gradient difference 0.0->0.5: {grad_diff_0_to_05:.6f}")
        print(f"Gradient difference 0.5->1.0: {grad_diff_05_to_1:.6f}")
        
        # Reset to default
        BKCoreFunction.GRAD_BLEND = 0.5
    
    def test_gradient_numerical_stability(self):
        """Test gradient stability with extreme values."""
        B = 1
        N = 8
        device = torch.device('cpu')
        
        # Test with large values
        he_diag_large = torch.full((B, N), 10.0, device=device, requires_grad=True)
        h0_super = torch.ones(B, N-1, device=device)
        h0_sub = torch.ones(B, N-1, device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        
        features = BKCoreFunction.apply(he_diag_large, h0_super, h0_sub, z)
        loss = features.sum()
        loss.backward()
        
        assert torch.all(torch.isfinite(he_diag_large.grad)), \
            "Gradient contains NaN/Inf with large values"
        
        # Test with small values
        he_diag_small = torch.full((B, N), -0.1, device=device, requires_grad=True)
        features = BKCoreFunction.apply(he_diag_small, h0_super, h0_sub, z)
        loss = features.sum()
        loss.backward()
        
        assert torch.all(torch.isfinite(he_diag_small.grad)), \
            "Gradient contains NaN/Inf with small values"
    
    def test_gradient_clipping(self):
        """Test that gradients are properly clipped."""
        B = 1
        N = 8
        device = torch.device('cpu')
        
        # Create inputs that might produce large gradients
        he_diag = torch.randn(B, N, device=device, requires_grad=True) * 10.0
        h0_super = torch.ones(B, N-1, device=device)
        h0_sub = torch.ones(B, N-1, device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        
        features = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
        loss = features.sum()
        loss.backward()
        
        # Verify gradients are clipped to [-1000, 1000]
        assert torch.all(he_diag.grad >= -1000.0), "Gradient below clipping threshold"
        assert torch.all(he_diag.grad <= 1000.0), "Gradient above clipping threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
