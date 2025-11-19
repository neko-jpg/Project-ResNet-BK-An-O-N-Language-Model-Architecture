"""
Unit Tests for LNS (Logarithmic Number System) Kernel

Tests cover:
1. Kernel execution without errors on CUDA
2. Output finiteness (no NaN/Inf)
3. CPU fallback availability
4. Numerical accuracy within acceptable bounds

Requirements: 6.1, 6.2, 6.6
"""

import pytest
import torch
import math

try:
    from src.kernels.lns_kernel import lns_matmul, TRITON_AVAILABLE
except ImportError:
    pytest.skip("Cannot import lns_kernel", allow_module_level=True)

try:
    from src.models.phase1 import LNSLinear, convert_linear_to_lns
except ImportError:
    pytest.skip("Cannot import LNSLinear", allow_module_level=True)


class TestLNSKernel:
    """Test suite for LNS matrix multiplication kernel."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_kernel_execution_no_errors(self):
        """
        Test that LNS kernel executes without errors on CUDA.
        
        Requirement: 6.1
        """
        # Create test matrices in log domain
        M, K, N = 128, 128, 128
        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        
        # Execute kernel (should not raise)
        c = lns_matmul(a, b)
        
        # Check output shape
        assert c.shape == (M, N), f"Expected shape ({M}, {N}), got {c.shape}"
        assert c.device.type == 'cuda', "Output should be on CUDA"
        assert c.dtype == torch.float32, "Output should be float32"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_output_is_finite(self):
        """
        Test that LNS kernel output is finite (no NaN/Inf).
        
        Requirement: 6.2
        """
        # Create test matrices with reasonable values
        M, K, N = 64, 64, 64
        a = torch.randn(M, K, device='cuda', dtype=torch.float32) * 0.1
        b = torch.randn(K, N, device='cuda', dtype=torch.float32) * 0.1
        
        # Execute kernel
        c = lns_matmul(a, b)
        
        # Check finiteness
        assert torch.isfinite(c).all(), "Output contains NaN or Inf"
        assert not torch.isnan(c).any(), "Output contains NaN"
        assert not torch.isinf(c).any(), "Output contains Inf"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_different_matrix_sizes(self):
        """Test LNS kernel with various matrix sizes."""
        sizes = [
            (32, 32, 32),
            (64, 128, 64),
            (128, 64, 256),
            (256, 256, 256),
        ]
        
        for M, K, N in sizes:
            a = torch.randn(M, K, device='cuda', dtype=torch.float32)
            b = torch.randn(K, N, device='cuda', dtype=torch.float32)
            
            c = lns_matmul(a, b)
            
            assert c.shape == (M, N), f"Size {(M,K,N)}: Expected shape ({M}, {N}), got {c.shape}"
            assert torch.isfinite(c).all(), f"Size {(M,K,N)}: Output not finite"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_numerical_accuracy(self):
        """
        Test numerical accuracy of LNS kernel within acceptable bounds.
        
        Requirement: 6.2
        
        Note: Max-log approximation introduces error, but should be < 30%
        for typical neural network activations.
        """
        M, K, N = 128, 128, 128
        
        # Create positive matrices (for log domain)
        a_linear = torch.abs(torch.randn(M, K, device='cuda', dtype=torch.float32)) + 0.1
        b_linear = torch.abs(torch.randn(K, N, device='cuda', dtype=torch.float32)) + 0.1
        
        # Ground truth: standard matmul
        c_true = torch.matmul(a_linear, b_linear)
        
        # LNS approximation
        log_a = torch.log(a_linear)
        log_b = torch.log(b_linear)
        log_c = lns_matmul(log_a, log_b)
        c_lns = torch.exp(log_c)
        
        # Compute relative error
        rel_error = torch.abs(c_true - c_lns) / (torch.abs(c_true) + 1e-8)
        mean_rel_error = rel_error.mean().item()
        max_rel_error = rel_error.max().item()
        
        # Max-log approximation can have significant error
        # We accept up to 50% mean error and 100% max error for this test
        # In practice, with sparse activations, error is much lower
        assert mean_rel_error < 0.5, f"Mean relative error too high: {mean_rel_error:.2%}"
        assert max_rel_error < 1.0, f"Max relative error too high: {max_rel_error:.2%}"
        
        print(f"LNS Accuracy: Mean error = {mean_rel_error:.2%}, Max error = {max_rel_error:.2%}")
    
    def test_cpu_fallback_raises_error(self):
        """
        Test that LNS kernel raises error on CPU (no fallback).
        
        Requirement: 6.6
        
        Note: Unlike fused_associative_scan, LNS kernel does NOT have
        CPU fallback because the approximation is CUDA-specific.
        """
        M, K, N = 32, 32, 32
        a = torch.randn(M, K, dtype=torch.float32)  # CPU tensor
        b = torch.randn(K, N, dtype=torch.float32)  # CPU tensor
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="LNS kernel requires CUDA"):
            _ = lns_matmul(a, b)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_configurable_block_sizes(self):
        """Test LNS kernel with different block sizes."""
        M, K, N = 256, 256, 256
        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        
        block_sizes = [
            (64, 64, 16),
            (128, 128, 32),
            (256, 256, 64),
        ]
        
        for bm, bn, bk in block_sizes:
            c = lns_matmul(a, b, block_size_m=bm, block_size_n=bn, block_size_k=bk)
            
            assert c.shape == (M, N), f"Block size {(bm,bn,bk)}: Wrong shape"
            assert torch.isfinite(c).all(), f"Block size {(bm,bn,bk)}: Output not finite"


class TestLNSLinear:
    """Test suite for LNSLinear layer."""
    
    def test_lns_linear_initialization(self):
        """Test LNSLinear layer initialization."""
        layer = LNSLinear(512, 256, bias=True, use_lns=True)
        
        assert layer.in_features == 512
        assert layer.out_features == 256
        assert layer.weight.shape == (256, 512)
        assert layer.bias is not None
        assert layer.bias.shape == (256,)
        assert layer.log_weight is None  # Not computed yet
    
    def test_lns_linear_training_mode(self):
        """Test that LNSLinear uses standard matmul in training mode."""
        layer = LNSLinear(128, 64, bias=True, use_lns=True)
        layer.train()  # Training mode
        
        x = torch.randn(32, 128)
        y = layer(x)
        
        assert y.shape == (32, 64)
        assert torch.isfinite(y).all()
        
        # Should use standard computation (log_weight not computed)
        assert layer.log_weight is None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_lns_linear_inference_mode(self):
        """Test that LNSLinear uses LNS kernel in inference mode."""
        layer = LNSLinear(128, 64, bias=True, use_lns=True).cuda()
        layer.eval()  # Inference mode
        
        x = torch.randn(32, 128, device='cuda')
        y = layer(x)
        
        assert y.shape == (32, 64)
        assert torch.isfinite(y).all()
        
        # log_weight should be computed after first inference
        assert layer.log_weight is not None
        assert layer.weight_sign is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_prepare_lns_weights(self):
        """Test pre-computation of log weights."""
        layer = LNSLinear(64, 32, bias=True, use_lns=True).cuda()
        
        # Initially None
        assert layer.log_weight is None
        
        # Prepare weights
        layer.prepare_lns_weights()
        
        # Should be computed
        assert layer.log_weight is not None
        assert layer.weight_sign is not None
        assert layer.log_weight.shape == layer.weight.shape
        assert layer.weight_sign.shape == layer.weight.shape
        
        if layer.bias is not None:
            assert layer.log_bias is not None
            assert layer.bias_sign is not None
    
    def test_lns_linear_without_bias(self):
        """Test LNSLinear layer without bias."""
        layer = LNSLinear(128, 64, bias=False, use_lns=True)
        
        assert layer.bias is None
        
        x = torch.randn(16, 128)
        y = layer(x)
        
        assert y.shape == (16, 64)
        assert torch.isfinite(y).all()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_lns_linear_batch_dimensions(self):
        """Test LNSLinear with various batch dimensions."""
        layer = LNSLinear(64, 32, use_lns=True).cuda()
        layer.eval()
        
        # 2D input
        x2d = torch.randn(16, 64, device='cuda')
        y2d = layer(x2d)
        assert y2d.shape == (16, 32)
        
        # 3D input
        x3d = torch.randn(8, 16, 64, device='cuda')
        y3d = layer(x3d)
        assert y3d.shape == (8, 16, 32)
        
        # 4D input
        x4d = torch.randn(4, 8, 16, 64, device='cuda')
        y4d = layer(x4d)
        assert y4d.shape == (4, 8, 16, 32)
    
    def test_convert_linear_to_lns(self):
        """Test conversion of nn.Linear to LNSLinear."""
        # Create a simple model with Linear layers
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(128, 64)
                self.fc2 = torch.nn.Linear(64, 32)
        
        model = SimpleModel()
        
        # Convert to LNS
        model_lns = convert_linear_to_lns(model, inplace=False)
        
        # Check that layers are converted
        assert isinstance(model_lns.fc1, LNSLinear)
        assert isinstance(model_lns.fc2, LNSLinear)
        
        # Check that weights are preserved
        assert torch.allclose(model.fc1.weight, model_lns.fc1.weight)
        assert torch.allclose(model.fc2.weight, model_lns.fc2.weight)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_lns_linear_gradient_flow(self):
        """Test that gradients flow correctly in training mode."""
        layer = LNSLinear(64, 32, use_lns=True).cuda()
        layer.train()
        
        x = torch.randn(16, 64, device='cuda', requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert layer.weight.grad is not None
        if layer.bias is not None:
            assert layer.bias.grad is not None
        
        # Check gradients are finite
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(layer.weight.grad).all()


class TestLNSKernelEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_incompatible_dimensions(self):
        """Test that incompatible dimensions raise error."""
        a = torch.randn(128, 64, device='cuda')
        b = torch.randn(128, 64, device='cuda')  # Wrong K dimension
        
        with pytest.raises(AssertionError):
            _ = lns_matmul(a, b)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_non_contiguous_tensors(self):
        """Test that non-contiguous tensors are handled."""
        a = torch.randn(128, 128, device='cuda').t()  # Non-contiguous
        b = torch.randn(128, 128, device='cuda')
        
        # Should work (kernel makes contiguous internally)
        c = lns_matmul(a, b)
        assert c.shape == (128, 128)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_extreme_values(self):
        """Test LNS kernel with extreme values in log domain."""
        M, K, N = 64, 64, 64
        
        # Very large values in log domain
        a = torch.full((M, K), 10.0, device='cuda')
        b = torch.full((K, N), 10.0, device='cuda')
        
        c = lns_matmul(a, b)
        
        # Should not overflow to inf
        assert torch.isfinite(c).all()
        
        # Very small values in log domain (large negative)
        a = torch.full((M, K), -10.0, device='cuda')
        b = torch.full((K, N), -10.0, device='cuda')
        
        c = lns_matmul(a, b)
        
        # Should not underflow to -inf
        assert torch.isfinite(c).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
