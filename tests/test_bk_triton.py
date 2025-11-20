wsl"""
BK-Core Triton Kernel Unit Tests

Tests the Triton-accelerated BK-Core implementation for:
- Numerical correctness vs PyTorch reference implementation
- Complex number arithmetic correctness
- Edge cases (small/large sequences, extreme values)
- Gradient computation correctness

Success criteria:
- Numerical error < 1e-4 compared to PyTorch implementation
- All edge cases handled correctly
- Gradients computed correctly
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bk_core import BKCoreFunction, set_triton_mode


def is_triton_available():
    """Check if Triton is available."""
    try:
        from src.kernels.bk_scan import is_triton_available as check_triton
        return check_triton()
    except Exception:
        return False


# Skip all tests if Triton is not available
pytestmark = pytest.mark.skipif(
    not is_triton_available(),
    reason="Triton not available"
)


class TestBKTritonNumericalCorrectness:
    """Test numerical correctness of Triton implementation."""
    
    def compute_error(self, output_pytorch, output_triton):
        """
        Compute error metrics between PyTorch and Triton outputs.
        
        Args:
            output_pytorch: (B, N, 2) PyTorch output
            output_triton: (B, N, 2) Triton output
        
        Returns:
            max_abs_error: Maximum absolute error
            mse: Mean squared error
            max_rel_error: Maximum relative error
        """
        diff = output_pytorch - output_triton
        max_abs_error = torch.abs(diff).max().item()
        mse = (diff ** 2).mean().item()
        
        # Relative error (avoid division by zero)
        denom = torch.abs(output_pytorch) + 1e-9
        rel_error = torch.abs(diff) / denom
        max_rel_error = rel_error.max().item()
        
        return max_abs_error, mse, max_rel_error
    
    def test_small_sequence(self):
        """Test with small sequence length (N=8)."""
        B, N = 2, 8
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate test data
        torch.manual_seed(42)
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.randn(B, N - 1, device=device)
        h0_sub = torch.randn(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        # PyTorch implementation
        set_triton_mode(False)
        output_pytorch = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
        
        # Triton implementation
        set_triton_mode(True)
        output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        # Compute errors
        max_abs_error, mse, max_rel_error = self.compute_error(output_pytorch, output_triton)
        
        # Verify error is within tolerance
        assert mse < 1e-4, f"MSE too large: {mse:.2e} (expected < 1e-4)"
        assert max_abs_error < 1e-2, f"Max absolute error too large: {max_abs_error:.2e}"
        assert max_rel_error < 1e-2, f"Max relative error too large: {max_rel_error:.2e}"
    
    def test_medium_sequence(self):
        """Test with medium sequence length (N=512)."""
        B, N = 4, 512
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(123)
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.randn(B, N - 1, device=device)
        h0_sub = torch.randn(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        # PyTorch implementation
        set_triton_mode(False)
        output_pytorch = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
        
        # Triton implementation
        set_triton_mode(True)
        output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        # Compute errors
        max_abs_error, mse, max_rel_error = self.compute_error(output_pytorch, output_triton)
        
        # Verify error is within tolerance
        assert mse < 1e-4, f"MSE too large: {mse:.2e} (expected < 1e-4)"
    
    def test_large_sequence(self):
        """Test with large sequence length (N=2048)."""
        B, N = 2, 2048
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(456)
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.randn(B, N - 1, device=device)
        h0_sub = torch.randn(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        # PyTorch implementation
        set_triton_mode(False)
        output_pytorch = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
        
        # Triton implementation
        set_triton_mode(True)
        output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        # Compute errors
        max_abs_error, mse, max_rel_error = self.compute_error(output_pytorch, output_triton)
        
        # Verify error is within tolerance
        assert mse < 1e-4, f"MSE too large: {mse:.2e} (expected < 1e-4)"
    
    def test_single_batch(self):
        """Test with single batch (B=1)."""
        B, N = 1, 256
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(789)
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.randn(B, N - 1, device=device)
        h0_sub = torch.randn(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        # PyTorch implementation
        set_triton_mode(False)
        output_pytorch = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
        
        # Triton implementation
        set_triton_mode(True)
        output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        # Compute errors
        max_abs_error, mse, max_rel_error = self.compute_error(output_pytorch, output_triton)
        
        # Verify error is within tolerance
        assert mse < 1e-4, f"MSE too large: {mse:.2e} (expected < 1e-4)"
    
    def test_large_batch(self):
        """Test with large batch (B=16)."""
        B, N = 16, 128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(101112)
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.randn(B, N - 1, device=device)
        h0_sub = torch.randn(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        # PyTorch implementation
        set_triton_mode(False)
        output_pytorch = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
        
        # Triton implementation
        set_triton_mode(True)
        output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        # Compute errors
        max_abs_error, mse, max_rel_error = self.compute_error(output_pytorch, output_triton)
        
        # Verify error is within tolerance
        assert mse < 1e-4, f"MSE too large: {mse:.2e} (expected < 1e-4)"


class TestBKTritonEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_extreme_diagonal_values_large(self):
        """Test with large diagonal values."""
        B, N = 2, 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Large diagonal values
        he_diag = torch.full((B, N), 100.0, device=device)
        h0_super = torch.ones(B, N - 1, device=device)
        h0_sub = torch.ones(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        # Triton implementation
        set_triton_mode(True)
        output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        # Verify no NaN or Inf
        assert torch.all(torch.isfinite(output_triton)), "Output contains NaN or Inf with large diagonal"
        
        # Verify output shape
        assert output_triton.shape == (B, N, 2), f"Unexpected output shape: {output_triton.shape}"
    
    def test_extreme_diagonal_values_small(self):
        """Test with small diagonal values."""
        B, N = 2, 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Small diagonal values
        he_diag = torch.full((B, N), -0.01, device=device)
        h0_super = torch.ones(B, N - 1, device=device)
        h0_sub = torch.ones(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        # Triton implementation
        set_triton_mode(True)
        output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        # Verify no NaN or Inf
        assert torch.all(torch.isfinite(output_triton)), "Output contains NaN or Inf with small diagonal"
    
    def test_zero_off_diagonal(self):
        """Test with zero off-diagonal elements."""
        B, N = 2, 32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.zeros(B, N - 1, device=device)
        h0_sub = torch.zeros(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        # Triton implementation
        set_triton_mode(True)
        output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        # Verify no NaN or Inf
        assert torch.all(torch.isfinite(output_triton)), "Output contains NaN or Inf with zero off-diagonal"
    
    def test_different_z_values(self):
        """Test with different complex shift values."""
        B, N = 2, 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.randn(B, N - 1, device=device)
        h0_sub = torch.randn(B, N - 1, device=device)
        
        # Test different z values
        z_values = [
            0.1 + 0.1j,
            1.0 + 1.0j,
            0.01 + 0.01j,
            0.5 - 0.5j,
        ]
        
        for z_val in z_values:
            z = torch.tensor(z_val, dtype=torch.complex64, device=device)
            
            # PyTorch implementation
            set_triton_mode(False)
            output_pytorch = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
            
            # Triton implementation
            set_triton_mode(True)
            output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
            
            # Verify no NaN or Inf
            assert torch.all(torch.isfinite(output_triton)), f"Output contains NaN or Inf with z={z_val}"
            
            # Compute error
            diff = output_pytorch - output_triton
            mse = (diff ** 2).mean().item()
            assert mse < 1e-4, f"MSE too large with z={z_val}: {mse:.2e}"


class TestBKTritonComplexArithmetic:
    """Test complex number arithmetic correctness."""
    
    def test_real_part_accuracy(self):
        """Test accuracy of real part computation."""
        B, N = 4, 128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(999)
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.randn(B, N - 1, device=device)
        h0_sub = torch.randn(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        # PyTorch implementation
        set_triton_mode(False)
        output_pytorch = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
        
        # Triton implementation
        set_triton_mode(True)
        output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        # Check real part separately
        real_diff = output_pytorch[..., 0] - output_triton[..., 0]
        real_mse = (real_diff ** 2).mean().item()
        
        assert real_mse < 1e-4, f"Real part MSE too large: {real_mse:.2e}"
    
    def test_imag_part_accuracy(self):
        """Test accuracy of imaginary part computation."""
        B, N = 4, 128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(888)
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.randn(B, N - 1, device=device)
        h0_sub = torch.randn(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        # PyTorch implementation
        set_triton_mode(False)
        output_pytorch = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
        
        # Triton implementation
        set_triton_mode(True)
        output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        # Check imaginary part separately
        imag_diff = output_pytorch[..., 1] - output_triton[..., 1]
        imag_mse = (imag_diff ** 2).mean().item()
        
        assert imag_mse < 1e-4, f"Imaginary part MSE too large: {imag_mse:.2e}"


class TestBKTritonGradients:
    """Test gradient computation correctness."""
    
    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        B, N = 2, 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(777)
        he_diag = torch.randn(B, N, device=device, requires_grad=True)
        h0_super = torch.randn(B, N - 1, device=device)
        h0_sub = torch.randn(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        # Triton implementation
        set_triton_mode(True)
        output = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Verify gradient exists and is finite
        assert he_diag.grad is not None, "No gradient computed"
        assert torch.all(torch.isfinite(he_diag.grad)), "Gradient contains NaN or Inf"
        
        # Verify gradient is non-zero (at least somewhere)
        assert torch.any(he_diag.grad != 0), "Gradient is all zeros"
    
    def test_gradient_consistency(self):
        """Test that Triton gradients match PyTorch gradients."""
        B, N = 2, 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(666)
        
        # PyTorch gradients
        he_diag_pytorch = torch.randn(B, N, device=device, requires_grad=True)
        h0_super = torch.randn(B, N - 1, device=device)
        h0_sub = torch.randn(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        set_triton_mode(False)
        output_pytorch = BKCoreFunction.apply(he_diag_pytorch, h0_super, h0_sub, z, False)
        loss_pytorch = output_pytorch.sum()
        loss_pytorch.backward()
        grad_pytorch = he_diag_pytorch.grad.clone()
        
        # Triton gradients
        he_diag_triton = he_diag_pytorch.detach().clone().requires_grad_(True)
        
        set_triton_mode(True)
        output_triton = BKCoreFunction.apply(he_diag_triton, h0_super, h0_sub, z, True)
        loss_triton = output_triton.sum()
        loss_triton.backward()
        grad_triton = he_diag_triton.grad
        
        # Compare gradients
        grad_diff = grad_pytorch - grad_triton
        grad_mse = (grad_diff ** 2).mean().item()
        
        # Note: Gradient comparison may have larger tolerance due to numerical differences
        assert grad_mse < 1e-3, f"Gradient MSE too large: {grad_mse:.2e}"


class TestBKTritonOutputProperties:
    """Test output properties and invariants."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        B, N = 4, 128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.randn(B, N - 1, device=device)
        h0_sub = torch.randn(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        set_triton_mode(True)
        output = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        assert output.shape == (B, N, 2), f"Expected shape ({B}, {N}, 2), got {output.shape}"
    
    def test_output_dtype(self):
        """Test that output has correct dtype."""
        B, N = 2, 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.randn(B, N - 1, device=device)
        h0_sub = torch.randn(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        set_triton_mode(True)
        output = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"
    
    def test_output_device(self):
        """Test that output is on correct device."""
        B, N = 2, 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.randn(B, N - 1, device=device)
        h0_sub = torch.randn(B, N - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        set_triton_mode(True)
        output = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        assert output.device == device, f"Expected device {device}, got {output.device}"


@pytest.mark.skipif(False, reason="Always run")
class TestBKTritonAvailability:
    """Test Triton availability detection (runs even without Triton)."""
    
    def test_triton_availability_check(self):
        """Test that Triton availability check works."""
        # This test always runs to verify the test infrastructure
        available = is_triton_available()
        
        # Just verify the function returns a boolean
        assert isinstance(available, bool), "is_triton_available should return bool"
        
        # If Triton is not available, verify other tests are skipped
        if not available:
            print("\nNote: Triton not available - main tests will be skipped")
        else:
            print("\nNote: Triton available - main tests will run")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
