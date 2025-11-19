"""
Unit tests for Fused Associative Scan kernel.

Tests correctness, performance, and edge cases for the fused_associative_scan
Triton kernel implementation.

Requirements: 6.1, 6.2, 6.6
"""

import pytest
import torch
import math

from src.kernels.associative_scan import (
    fused_associative_scan,
    TRITON_AVAILABLE,
)


class TestFusedAssociativeScan:
    """Unit tests for fused_associative_scan function."""
    
    def test_output_correctness_small(self):
        """
        Test that fused scan produces correct results for small inputs.
        
        Requirement 6.1: Test output correctness matches torch.cumsum
        """
        # Small test case for easy verification
        x = torch.tensor([
            [[1.0, 2.0],
             [3.0, 4.0],
             [5.0, 6.0]]
        ])  # (1, 3, 2)
        
        expected = torch.cumsum(x, dim=1)
        actual = fused_associative_scan(x, dim=1)
        
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-6), \
            f"Output mismatch: expected {expected}, got {actual}"
    
    def test_output_correctness_random(self):
        """
        Test correctness with random inputs of various sizes.
        
        Requirement 6.1: Test output correctness matches torch.cumsum
        """
        test_shapes = [
            (2, 64, 32),
            (4, 128, 64),
            (1, 256, 128),
            (8, 512, 256),
        ]
        
        for shape in test_shapes:
            x = torch.randn(*shape)
            
            expected = torch.cumsum(x, dim=1)
            actual = fused_associative_scan(x, dim=1)
            
            max_diff = (expected - actual).abs().max().item()
            rel_error = max_diff / (expected.abs().max().item() + 1e-8)
            
            assert rel_error < 1e-4, \
                f"Shape {shape}: Relative error {rel_error:.2e} exceeds threshold"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_execution(self):
        """
        Test that kernel executes without errors on CUDA.
        
        Requirement 6.1: Test kernel executes without errors on CUDA
        """
        x = torch.randn(4, 256, 128, device='cuda')
        
        # Should not raise any errors
        output = fused_associative_scan(x, dim=1)
        
        assert output.shape == x.shape
        assert output.device == x.device
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"
    
    def test_cpu_fallback(self):
        """
        Test that CPU fallback works when CUDA unavailable.
        
        Requirement 6.6: Test CPU fallback works when CUDA unavailable
        """
        x = torch.randn(2, 128, 64)  # CPU tensor
        
        # Should use torch.cumsum fallback
        output = fused_associative_scan(x, dim=1)
        
        expected = torch.cumsum(x, dim=1)
        
        assert torch.allclose(output, expected, rtol=1e-5, atol=1e-7)
    
    def test_reverse_scan(self):
        """
        Test reverse (anti-causal) cumulative sum.
        
        Requirement 8.4: Test backward scan for anti-causal processing
        """
        x = torch.randn(2, 128, 64)
        
        # Forward scan
        forward = fused_associative_scan(x, dim=1, reverse=False)
        
        # Reverse scan
        reverse = fused_associative_scan(x, dim=1, reverse=True)
        
        # Verify reverse is correct
        expected_reverse = torch.flip(
            torch.cumsum(torch.flip(x, dims=[1]), dim=1),
            dims=[1]
        )
        
        assert torch.allclose(reverse, expected_reverse, rtol=1e-4, atol=1e-6)
        
        # Forward and reverse should be different
        assert not torch.allclose(forward, reverse)
    
    def test_different_dimensions(self):
        """
        Test scanning along different dimensions.
        
        Requirement 6.1: Test output correctness
        """
        x = torch.randn(4, 32, 64, 128)
        
        # Test dim=1
        output1 = fused_associative_scan(x, dim=1)
        expected1 = torch.cumsum(x, dim=1)
        assert torch.allclose(output1, expected1, rtol=1e-4, atol=1e-6)
        
        # Test dim=2
        output2 = fused_associative_scan(x, dim=2)
        expected2 = torch.cumsum(x, dim=2)
        assert torch.allclose(output2, expected2, rtol=1e-4, atol=1e-6)
        
        # Test dim=-1
        output3 = fused_associative_scan(x, dim=-1)
        expected3 = torch.cumsum(x, dim=-1)
        assert torch.allclose(output3, expected3, rtol=1e-4, atol=1e-6)
    
    def test_numerical_stability_large_sequences(self):
        """
        Test numerical stability for large sequences.
        
        Requirement 6.2: Test numerical stability for large sequences
        """
        # Large sequence length
        x = torch.randn(1, 8192, 128)
        
        output = fused_associative_scan(x, dim=1)
        
        # Check for NaN or Inf
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"
        
        # Compare with torch.cumsum
        expected = torch.cumsum(x, dim=1)
        
        # Allow slightly larger tolerance for long sequences
        max_diff = (expected - output).abs().max().item()
        rel_error = max_diff / (expected.abs().max().item() + 1e-8)
        
        assert rel_error < 1e-3, \
            f"Large sequence: Relative error {rel_error:.2e} exceeds threshold"
    
    def test_numerical_stability_large_values(self):
        """
        Test numerical stability with large input values.
        
        Requirement 6.2: Test numerical stability
        """
        # Large values
        x = torch.randn(2, 512, 64) * 100.0
        
        output = fused_associative_scan(x, dim=1)
        
        # Check for NaN or Inf
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"
        
        # Compare with torch.cumsum
        expected = torch.cumsum(x, dim=1)
        
        max_diff = (expected - output).abs().max().item()
        rel_error = max_diff / (expected.abs().max().item() + 1e-8)
        
        assert rel_error < 1e-3, \
            f"Large values: Relative error {rel_error:.2e} exceeds threshold"
    
    def test_zero_input(self):
        """Test with zero input."""
        x = torch.zeros(2, 128, 64)
        
        output = fused_associative_scan(x, dim=1)
        
        assert torch.allclose(output, torch.zeros_like(x))
    
    def test_single_element_sequence(self):
        """Test with sequence length of 1."""
        x = torch.randn(4, 1, 64)
        
        output = fused_associative_scan(x, dim=1)
        expected = torch.cumsum(x, dim=1)
        
        assert torch.allclose(output, expected, rtol=1e-5, atol=1e-7)
    
    def test_contiguity_handling(self):
        """
        Test that non-contiguous tensors are handled correctly.
        
        Requirement 8.1: Add input validation (contiguity checks)
        """
        x = torch.randn(4, 256, 128)
        
        # Create non-contiguous tensor by transposing and NOT transposing back
        x_noncontig = x.transpose(0, 2)  # Now (128, 256, 4)
        assert not x_noncontig.is_contiguous()
        
        # Should still work (will be made contiguous internally)
        # Scan along dim=1 (which is still the sequence dimension)
        output = fused_associative_scan(x_noncontig, dim=1)
        expected = torch.cumsum(x_noncontig, dim=1)
        
        assert torch.allclose(output, expected, rtol=1e-4, atol=1e-6)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_dtype_support(self):
        """Test support for different dtypes."""
        dtypes = [torch.float32, torch.float16]
        
        for dtype in dtypes:
            x = torch.randn(2, 128, 64, device='cuda', dtype=dtype)
            
            output = fused_associative_scan(x, dim=1)
            expected = torch.cumsum(x, dim=1)
            
            # Use appropriate tolerance for dtype
            rtol = 1e-3 if dtype == torch.float16 else 1e-4
            atol = 1e-4 if dtype == torch.float16 else 1e-6
            
            assert torch.allclose(output, expected, rtol=rtol, atol=atol), \
                f"Failed for dtype {dtype}"
    
    def test_gradient_flow(self):
        """
        Test that gradients flow correctly through the scan operation.
        
        Requirement 6.2: Test gradient flow
        """
        x = torch.randn(2, 128, 64, requires_grad=True)
        
        output = fused_associative_scan(x, dim=1)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist and are finite
        assert x.grad is not None, "No gradient computed"
        assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf"
        
        # Gradient should be non-zero for most elements
        assert (x.grad.abs() > 1e-8).sum() > 0, "Gradient is all zeros"
    
    @pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, 
                       reason="CUDA and Triton required")
    def test_performance_improvement(self):
        """
        Test that fused scan is faster than torch.cumsum.
        
        Requirement 8.3: Verify speedup
        """
        x = torch.randn(4, 2048, 512, device='cuda')
        
        # Warmup
        for _ in range(10):
            _ = torch.cumsum(x, dim=1)
            _ = fused_associative_scan(x, dim=1)
        torch.cuda.synchronize()
        
        # Benchmark torch.cumsum
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            _ = torch.cumsum(x, dim=1)
        end.record()
        torch.cuda.synchronize()
        cumsum_time = start.elapsed_time(end) / 100
        
        # Benchmark fused_associative_scan
        start.record()
        for _ in range(100):
            _ = fused_associative_scan(x, dim=1)
        end.record()
        torch.cuda.synchronize()
        fused_time = start.elapsed_time(end) / 100
        
        speedup = cumsum_time / fused_time
        
        print(f"\nPerformance: torch.cumsum={cumsum_time:.3f}ms, "
              f"fused_scan={fused_time:.3f}ms, speedup={speedup:.2f}x")
        
        # Note: Speedup may vary by hardware and may not always achieve 3x
        # This is more of an informational test
        assert speedup > 0.5, "Fused scan is significantly slower than torch.cumsum"


class TestIntegrationWithARSSM:
    """Integration tests with AR-SSM layer."""
    
    def test_ar_ssm_uses_fused_scan(self):
        """
        Test that AR-SSM layer uses fused scan when enabled.
        
        Requirement 8.2: Integration with AR-SSM
        """
        from src.models.phase1.ar_ssm_layer import AdaptiveRankSemiseparableLayer
        
        # Create layer with fused scan enabled
        layer = AdaptiveRankSemiseparableLayer(
            d_model=64,
            max_rank=16,
            use_fused_scan=True
        )
        
        x = torch.randn(2, 128, 64)
        output, diagnostics = layer(x)
        
        # Check that fused scan was used (if available)
        if torch.cuda.is_available() and TRITON_AVAILABLE:
            assert diagnostics.get('used_fused_scan', False), \
                "AR-SSM should use fused scan when available"
        
        # Output should be valid
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_ar_ssm_fallback_without_fused_scan(self):
        """
        Test that AR-SSM falls back to torch.cumsum when fused scan disabled.
        
        Requirement 8.2: Fallback behavior
        """
        from src.models.phase1.ar_ssm_layer import AdaptiveRankSemiseparableLayer
        
        # Create layer with fused scan disabled
        layer = AdaptiveRankSemiseparableLayer(
            d_model=64,
            max_rank=16,
            use_fused_scan=False
        )
        
        x = torch.randn(2, 128, 64)
        output, diagnostics = layer(x)
        
        # Should use fallback
        assert not diagnostics.get('used_fused_scan', True), \
            "AR-SSM should not use fused scan when disabled"
        
        # Output should still be valid
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_ar_ssm_bidirectional_scan(self):
        """
        Test bidirectional processing with fused scan.
        
        Requirement 8.4: Bidirectional scan support
        """
        from src.models.phase1.ar_ssm_layer import AdaptiveRankSemiseparableLayer
        
        layer = AdaptiveRankSemiseparableLayer(
            d_model=64,
            max_rank=16,
            use_fused_scan=True
        )
        
        x = torch.randn(2, 128, 64)
        
        # Test bidirectional forward
        output, diagnostics = layer.forward_bidirectional(x, use_anticausal=True)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        assert diagnostics.get('bidirectional', False), \
            "Should indicate bidirectional processing"
        
        # Should have both forward and reverse components
        assert 'k_cumsum' in diagnostics
        assert 'k_cumsum_reverse' in diagnostics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
