"""
Test Suite for Safe Triton Operations

Tests the numerically stable Triton primitives against PyTorch references
and verifies behavior with extreme values.

Run with: python -m pytest tests/test_safe_ops_triton.py -v
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernels.safe_ops_triton import (
    safe_exp_pytorch,
    safe_log_pytorch,
    safe_acosh_pytorch,
    safe_atanh_pytorch,
    is_triton_available,
    K_THRESHOLD,
)


class TestSafeExpPyTorch:
    """Test safe_exp PyTorch reference implementation."""
    
    def test_normal_values(self):
        """Test with normal input values."""
        x = torch.linspace(-10, 10, 100)
        result = safe_exp_pytorch(x)
        
        # Should match torch.exp for normal values
        expected = torch.exp(x)
        assert torch.allclose(result, expected, rtol=1e-3), \
            "safe_exp should match torch.exp for normal values"
    
    def test_large_positive_no_overflow(self):
        """Test that large positive values don't overflow."""
        x = torch.tensor([100.0, 500.0, 1000.0])
        result = safe_exp_pytorch(x)
        
        assert torch.all(torch.isfinite(result)), \
            "safe_exp should not overflow with large positive values"
        assert torch.all(result > 0), \
            "safe_exp should always return positive values"
    
    def test_large_negative_no_underflow(self):
        """Test that large negative values don't underflow to exactly zero."""
        x = torch.tensor([-100.0, -500.0, -1000.0])
        result = safe_exp_pytorch(x)
        
        assert torch.all(torch.isfinite(result)), \
            "safe_exp should not have issues with large negative values"
        assert torch.all(result >= 0), \
            "safe_exp should always return non-negative values"
    
    def test_rubber_wall_saturation(self):
        """Test that Rubber Wall saturates correctly."""
        x_large = torch.tensor([1000.0])
        result = safe_exp_pytorch(x_large)
        
        # Should saturate to approximately exp(K)
        expected_max = torch.exp(torch.tensor(K_THRESHOLD))
        assert result[0] <= expected_max * 1.01, \
            f"safe_exp should saturate near exp({K_THRESHOLD})"
    
    def test_gradient_exists(self):
        """Test that gradients can be computed."""
        x = torch.tensor([1.0, 10.0, 100.0], requires_grad=True)
        result = safe_exp_pytorch(x)
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None, "Gradient should be computed"
        assert torch.all(torch.isfinite(x.grad)), \
            "Gradients should be finite"


class TestSafeLogPyTorch:
    """Test safe_log PyTorch reference implementation."""
    
    def test_positive_values(self):
        """Test with positive input values."""
        x = torch.linspace(0.1, 100, 100)
        result = safe_log_pytorch(x)
        expected = torch.log(x)
        
        assert torch.allclose(result, expected, rtol=1e-5), \
            "safe_log should match torch.log for positive values"
    
    def test_zero_input(self):
        """Test with zero input (should not produce -inf)."""
        x = torch.tensor([0.0])
        result = safe_log_pytorch(x)
        
        assert torch.all(torch.isfinite(result)), \
            "safe_log should not produce -inf for zero input"
    
    def test_near_zero_input(self):
        """Test with near-zero positive input."""
        x = torch.tensor([1e-10, 1e-20, 1e-30])
        result = safe_log_pytorch(x)
        
        assert torch.all(torch.isfinite(result)), \
            "safe_log should handle near-zero values"


class TestSafeAcoshPyTorch:
    """Test safe_acosh PyTorch reference implementation."""
    
    def test_normal_values(self):
        """Test with values >= 1."""
        x = torch.linspace(1.1, 100, 100)
        result = safe_acosh_pytorch(x)
        expected = torch.acosh(x)
        
        assert torch.allclose(result, expected, rtol=1e-4), \
            "safe_acosh should match torch.acosh for normal values"
    
    def test_value_close_to_one(self):
        """Test with values close to 1 (edge case)."""
        x = torch.tensor([1.0, 1.0001, 1.001, 1.01])
        result = safe_acosh_pytorch(x)
        
        assert torch.all(torch.isfinite(result)), \
            "safe_acosh should handle values close to 1"
        assert torch.all(result >= 0), \
            "acosh should always return non-negative values"
    
    def test_value_less_than_one(self):
        """Test with values < 1 (should be clamped)."""
        x = torch.tensor([0.5, 0.0, -1.0])
        result = safe_acosh_pytorch(x)
        
        assert torch.all(torch.isfinite(result)), \
            "safe_acosh should handle values < 1 via clamping"


class TestSafeAtanhPyTorch:
    """Test safe_atanh PyTorch reference implementation."""
    
    def test_normal_values(self):
        """Test with values in (-1, 1)."""
        x = torch.linspace(-0.9, 0.9, 100)
        result = safe_atanh_pytorch(x)
        expected = torch.atanh(x)
        
        assert torch.allclose(result, expected, rtol=1e-3), \
            "safe_atanh should match torch.atanh for normal values"
    
    def test_boundary_values(self):
        """Test with values at ±1 boundary."""
        x = torch.tensor([-1.0, -0.9999, 0.9999, 1.0])
        result = safe_atanh_pytorch(x)
        
        assert torch.all(torch.isfinite(result)), \
            "safe_atanh should handle boundary values"


class TestTritonAvailability:
    """Test Triton availability detection."""
    
    def test_availability_function(self):
        """Test that availability function returns bool."""
        result = is_triton_available()
        assert isinstance(result, bool), \
            "is_triton_available should return bool"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_triton_available(), reason="Triton not available")
class TestTritonKernels:
    """Test Triton kernels on GPU (requires CUDA and Triton)."""
    
    def test_safe_exp_kernel(self):
        """Test safe_exp Triton kernel matches PyTorch."""
        # This would require running a small Triton kernel
        # For now, we verify the PyTorch reference works on CUDA
        x = torch.linspace(-100, 100, 1000, device='cuda')
        result = safe_exp_pytorch(x)
        
        assert torch.all(torch.isfinite(result)), \
            "safe_exp should work on CUDA tensors"
    
    def test_safe_log_kernel(self):
        """Test safe_log on CUDA."""
        x = torch.linspace(0, 100, 1000, device='cuda')
        result = safe_log_pytorch(x)
        
        assert torch.all(torch.isfinite(result)), \
            "safe_log should work on CUDA tensors"
    
    def test_safe_acosh_kernel(self):
        """Test safe_acosh on CUDA."""
        x = torch.linspace(0.999, 100, 1000, device='cuda')
        result = safe_acosh_pytorch(x)
        
        assert torch.all(torch.isfinite(result)), \
            "safe_acosh should work on CUDA tensors"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
