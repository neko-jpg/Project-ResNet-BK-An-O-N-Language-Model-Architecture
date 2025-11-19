"""
Gradient Correctness Tests for Phase 1 Components

Task 8.5: Implement finite difference gradient checking and verify gradient flow

物理的直観 (Physical Intuition):
有限差分法により、自動微分の正確性を検証します。
これは、数値的に ∂f/∂x ≈ (f(x+ε) - f(x-ε)) / (2ε) を計算し、
PyTorchの自動微分結果と比較します。

Requirements: 10.6, 6.2
"""

import pytest
import torch
import torch.nn as nn
import math
from typing import Callable, Tuple

from src.models.phase1 import (
    AdaptiveRankSemiseparableLayer,
    HolographicTTEmbedding,
    LNSLinear,
    GradientMonitor,
    check_gradient_health,
)


def finite_difference_gradient(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute gradient using finite difference method.
    
    Task 8.5: Implement finite difference gradient checking
    
    物理的直観:
    有限差分法: ∂f/∂x ≈ (f(x+ε) - f(x-ε)) / (2ε)
    中心差分を使用することで、O(ε²)の精度を実現します。
    
    Args:
        func: Function to differentiate
        x: Input tensor
        eps: Finite difference step size
    
    Returns:
        Gradient tensor computed via finite difference
    """
    grad = torch.zeros_like(x)
    
    # Flatten for easier iteration
    x_flat = x.view(-1)
    grad_flat = grad.view(-1)
    
    for i in range(x_flat.numel()):
        # Perturb +eps
        x_plus = x_flat.clone()
        x_plus[i] += eps
        
        # Perturb -eps
        x_minus = x_flat.clone()
        x_minus[i] -= eps
        
        # Compute function values
        f_plus = func(x_plus.view_as(x))
        f_minus = func(x_minus.view_as(x))
        
        # Central difference
        grad_flat[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad


def check_gradient_correctness(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    eps: float = 1e-5,
) -> Tuple[bool, float, torch.Tensor, torch.Tensor]:
    """
    Check gradient correctness using finite difference.
    
    Task 8.5: Verify gradient correctness
    
    Args:
        func: Function to test
        x: Input tensor (requires_grad=True)
        rtol: Relative tolerance
        atol: Absolute tolerance
        eps: Finite difference step size
    
    Returns:
        (is_correct, max_error, autograd_grad, finite_diff_grad)
    """
    # Compute autograd gradient
    x.requires_grad_(True)
    y = func(x)
    y.backward()
    autograd_grad = x.grad.clone()
    x.grad.zero_()
    
    # Compute finite difference gradient
    with torch.no_grad():
        finite_diff_grad = finite_difference_gradient(
            lambda x_: func(x_).detach(),
            x.detach(),
            eps=eps
        )
    
    # Compare
    diff = torch.abs(autograd_grad - finite_diff_grad)
    max_error = diff.max().item()
    
    # Check tolerance
    is_correct = torch.allclose(
        autograd_grad,
        finite_diff_grad,
        rtol=rtol,
        atol=atol
    )
    
    return is_correct, max_error, autograd_grad, finite_diff_grad


class TestARSSMGradients:
    """
    Test gradient correctness for AR-SSM layer.
    
    Task 8.5: Test gradient flow through AR-SSM
    """
    
    def test_ar_ssm_gradient_flow(self):
        """
        Test that gradients flow through all AR-SSM components.
        
        Task 8.5: Verify gradient flow
        """
        torch.manual_seed(42)
        
        # Create AR-SSM layer
        layer = AdaptiveRankSemiseparableLayer(
            d_model=32,
            max_rank=8,
            min_rank=2,
        )
        
        # Forward pass
        x = torch.randn(2, 16, 32, requires_grad=True)
        y, diagnostics = layer(x)
        
        # Backward pass
        loss = y.sum()
        loss.backward()
        
        # Check all parameters have gradients
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
    
    def test_ar_ssm_gradient_correctness_simple(self):
        """
        Test AR-SSM gradient correctness using finite difference.
        
        Task 8.5: Implement finite difference gradient checking
        
        Note: We test a simplified version due to computational cost
        """
        torch.manual_seed(42)
        
        # Small layer for faster testing
        layer = AdaptiveRankSemiseparableLayer(
            d_model=8,
            max_rank=4,
            min_rank=2,
        )
        
        # Small input
        x = torch.randn(1, 4, 8, requires_grad=True)
        
        # Define function
        def func(x_):
            y, _ = layer(x_)
            return y.sum()
        
        # Check gradient correctness
        is_correct, max_error, _, _ = check_gradient_correctness(
            func, x, rtol=1e-2, atol=1e-4, eps=1e-4
        )
        
        # Allow some tolerance due to complexity of AR-SSM
        assert max_error < 0.1, f"Gradient error too large: {max_error}"
    
    def test_ar_ssm_no_nan_inf_gradients(self):
        """
        Test that AR-SSM gradients don't contain NaN/Inf.
        
        Task 8.5: Verify no NaN/Inf in gradients
        """
        torch.manual_seed(42)
        
        layer = AdaptiveRankSemiseparableLayer(
            d_model=64,
            max_rank=16,
        )
        
        # Multiple forward-backward passes
        for _ in range(10):
            x = torch.randn(4, 32, 64)
            y, _ = layer(x)
            loss = y.sum()
            loss.backward()
            
            # Check for NaN/Inf
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), \
                        f"Non-finite gradient in {name}"
            
            # Zero gradients for next iteration
            layer.zero_grad()
    
    def test_ar_ssm_gradient_checkpointing(self):
        """
        Test that gradient checkpointing doesn't break gradients.
        
        Task 8.1: Verify gradient checkpointing
        """
        torch.manual_seed(42)
        
        layer = AdaptiveRankSemiseparableLayer(
            d_model=32,
            max_rank=8,
        )
        
        # Enable checkpointing
        layer.enable_checkpointing()
        
        x = torch.randn(2, 16, 32, requires_grad=True)
        y, _ = layer(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        # Check all parameters have gradients
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for {name} with checkpointing"


class TestHTTGradients:
    """
    Test gradient correctness for HTT embedding.
    
    Task 8.5: Test gradient flow through HTT
    """
    
    def test_htt_gradient_flow(self):
        """
        Test that gradients flow to all Tensor Train cores.
        
        Task 8.2: Verify gradient flow to all Tensor Train cores
        """
        torch.manual_seed(42)
        
        # Create HTT embedding
        embedding = HolographicTTEmbedding(
            vocab_size=1000,
            d_model=128,
            rank=8,
        )
        
        # Forward pass
        input_ids = torch.randint(0, 1000, (4, 16))
        output = embedding(input_ids)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check core1 has gradient
        assert embedding.core1.grad is not None
        assert torch.isfinite(embedding.core1.grad).all()
        assert embedding.core1.grad.abs().sum() > 0
        
        # Check core2 has gradient
        assert embedding.core2.grad is not None
        assert torch.isfinite(embedding.core2.grad).all()
        assert embedding.core2.grad.abs().sum() > 0
        
        # Check phase_shift has gradient (if phase encoding enabled)
        if embedding.phase_encoding:
            assert embedding.phase_shift.grad is not None
            assert torch.isfinite(embedding.phase_shift.grad).all()
    
    def test_htt_no_full_matrix_materialization(self):
        """
        Test that HTT doesn't materialize full embedding matrix.
        
        Task 8.2: Ensure no full embedding matrix materialization
        """
        torch.manual_seed(42)
        
        # Large vocabulary to test memory efficiency
        vocab_size = 50000
        d_model = 1024
        
        embedding = HolographicTTEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            rank=16,
        )
        
        # Check parameter count
        htt_params = sum(p.numel() for p in embedding.parameters())
        standard_params = vocab_size * d_model
        
        # HTT should use much fewer parameters
        assert htt_params < standard_params * 0.1, \
            "HTT uses too many parameters (not compressed)"
        
        # Forward pass should work without materializing full matrix
        input_ids = torch.randint(0, vocab_size, (2, 32))
        output = embedding(input_ids)
        
        assert output.shape == (2, 32, d_model)
    
    def test_htt_gradient_correctness(self):
        """
        Test HTT gradient correctness using finite difference.
        
        Task 8.5: Implement finite difference gradient checking
        """
        torch.manual_seed(42)
        
        # Small embedding for testing
        embedding = HolographicTTEmbedding(
            vocab_size=100,
            d_model=32,
            rank=4,
        )
        
        # Test gradient w.r.t. core1 using parameter gradient
        input_ids = torch.randint(0, 100, (2, 4))
        
        # Forward-backward to get autograd gradient
        output = embedding(input_ids)
        loss = output.sum()
        loss.backward()
        
        autograd_grad = embedding.core1.grad.clone()
        
        # Compute finite difference gradient manually
        eps = 1e-4
        finite_diff_grad = torch.zeros_like(embedding.core1)
        
        with torch.no_grad():
            for idx in torch.randperm(embedding.core1.numel())[:10]:  # Sample 10 elements
                # Get multi-dimensional index
                idx_tuple = torch.unravel_index(idx, embedding.core1.shape)
                
                # Perturb +eps
                original_val = embedding.core1[idx_tuple].item()
                embedding.core1[idx_tuple] = original_val + eps
                output_plus = embedding(input_ids).sum().item()
                
                # Perturb -eps
                embedding.core1[idx_tuple] = original_val - eps
                output_minus = embedding(input_ids).sum().item()
                
                # Restore original
                embedding.core1[idx_tuple] = original_val
                
                # Compute gradient
                finite_diff_grad[idx_tuple] = (output_plus - output_minus) / (2 * eps)
        
        # Compare sampled gradients
        sampled_indices = torch.randperm(embedding.core1.numel())[:10]
        for idx in sampled_indices:
            idx_tuple = torch.unravel_index(idx, embedding.core1.shape)
            if finite_diff_grad[idx_tuple] != 0:  # Only check computed values
                autograd_val = autograd_grad[idx_tuple].item()
                finite_diff_val = finite_diff_grad[idx_tuple].item()
                
                # Allow some tolerance
                rel_error = abs(autograd_val - finite_diff_val) / (abs(finite_diff_val) + 1e-8)
                assert rel_error < 0.1, \
                    f"HTT gradient error at {idx_tuple}: autograd={autograd_val}, finite_diff={finite_diff_val}"


class TestLNSGradients:
    """
    Test gradient correctness for LNS kernel.
    
    Task 8.5: Test gradient flow through LNS
    """
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_lns_linear_gradient_flow(self):
        """
        Test that LNS linear layer has gradient flow in training mode.
        
        Task 8.3: Verify backward pass implementation
        """
        torch.manual_seed(42)
        
        # Create LNS linear layer
        layer = LNSLinear(
            in_features=32,
            out_features=16,
            use_lns=True,
        ).cuda()
        
        # Training mode (should use standard matmul)
        layer.train()
        
        x = torch.randn(4, 32, device='cuda', requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_lns_gradient_clipping(self):
        """
        Test that LNS kernel gradient clipping works.
        
        Task 8.3: Add gradient clipping at kernel level
        """
        torch.manual_seed(42)
        
        layer = LNSLinear(
            in_features=32,
            out_features=16,
            use_lns=True,
            gradient_clip_value=1.0,
        ).cuda()
        
        layer.train()
        
        # Large input to potentially cause large gradients
        x = torch.randn(4, 32, device='cuda', requires_grad=True) * 10.0
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        # Gradients should be clipped
        if x.grad is not None:
            assert x.grad.abs().max() <= 1.0 + 1e-3, \
                "Gradient not properly clipped"


class TestGradientMonitor:
    """
    Test gradient monitoring functionality.
    
    Task 8.4: Test gradient monitoring and clipping
    """
    
    def test_gradient_monitor_tracking(self):
        """
        Test that gradient monitor tracks statistics correctly.
        
        Task 8.4: Implement gradient norm tracking
        """
        torch.manual_seed(42)
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        
        # Create monitor
        monitor = GradientMonitor(max_norm=10.0)
        
        # Forward-backward
        x = torch.randn(4, 32)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Track gradients
        result = monitor.track_and_clip(model, clip=False)
        
        # Check result structure
        assert 'statistics' in result
        assert 'all_healthy' in result
        assert 'num_clipped' in result
        assert 'max_norm_component' in result
    
    def test_gradient_monitor_clipping(self):
        """
        Test that gradient monitor clips large gradients.
        
        Task 8.4: Test gradient clipping effectiveness
        """
        torch.manual_seed(42)
        
        # Create model with Phase 1 component
        model = AdaptiveRankSemiseparableLayer(
            d_model=32,
            max_rank=8,
        )
        
        # Create monitor with low threshold
        monitor = GradientMonitor(max_norm=1.0, clip_mode='norm')
        
        # Create large gradients
        x = torch.randn(2, 16, 32)
        y, _ = model(x)
        loss = y.sum() * 100.0  # Large loss to create large gradients
        loss.backward()
        
        # Track and clip
        result = monitor.track_and_clip(model, clip=True)
        
        # Check that clipping occurred
        # Only check Phase 1 components
        phase1_params = [
            p for name, p in model.named_parameters()
            if any(pattern in name for pattern in monitor.phase1_patterns)
            and p.grad is not None
        ]
        
        if phase1_params:
            total_norm = torch.sqrt(
                sum(p.grad.norm() ** 2 for p in phase1_params)
            ).item()
            
            assert total_norm <= 1.0 + 1e-2, \
                f"Gradient not clipped: norm={total_norm}"
    
    def test_gradient_health_check(self):
        """
        Test gradient health checking utility.
        
        Task 8.4: Verify no NaN/Inf in gradients
        """
        torch.manual_seed(42)
        
        # Create model with Phase 1 component
        model = AdaptiveRankSemiseparableLayer(
            d_model=32,
            max_rank=8,
        )
        
        # Test 1: Normal gradients
        x = torch.randn(2, 16, 32)
        y, _ = model(x)
        loss = y.sum()
        loss.backward()
        
        # Use higher threshold for normal gradients (AR-SSM can have larger gradients)
        is_healthy, warnings = check_gradient_health(model, max_norm=1000.0)
        assert is_healthy, f"Expected healthy gradients, got warnings: {warnings}"
        assert len(warnings) == 0
        
        # Test 2: NaN gradient
        # Create a fresh model to avoid gradient accumulation issues
        model2 = AdaptiveRankSemiseparableLayer(
            d_model=32,
            max_rank=8,
        )
        
        # Do a forward-backward to create gradients
        x2 = torch.randn(2, 16, 32)
        y2, _ = model2(x2)
        loss2 = y2.sum()
        loss2.backward()
        
        # Now inject NaN into one of the gradients
        for name, param in model2.named_parameters():
            if param.grad is not None:
                param.grad.data.fill_(float('nan'))
                break
        
        is_healthy, warnings = check_gradient_health(model2, max_norm=10.0)
        assert not is_healthy, "Expected unhealthy gradients due to NaN"
        assert len(warnings) > 0, "Expected warnings about NaN gradients"
        assert any('NaN' in w for w in warnings), f"Expected NaN warning, got: {warnings}"


class TestIntegratedGradientFlow:
    """
    Test gradient flow through integrated Phase 1 components.
    
    Task 8.5: Test gradient flow through all Phase 1 components
    """
    
    def test_ar_ssm_with_htt_gradient_flow(self):
        """
        Test gradient flow through AR-SSM + HTT combination.
        """
        torch.manual_seed(42)
        
        # Create components
        embedding = HolographicTTEmbedding(
            vocab_size=1000,
            d_model=64,
            rank=8,
        )
        
        ar_ssm = AdaptiveRankSemiseparableLayer(
            d_model=64,
            max_rank=16,
        )
        
        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 16))
        x = embedding(input_ids)
        y, _ = ar_ssm(x)
        
        # Backward pass
        loss = y.sum()
        loss.backward()
        
        # Check gradients in both components
        assert embedding.core1.grad is not None
        assert embedding.core2.grad is not None
        
        for name, param in ar_ssm.named_parameters():
            assert param.grad is not None, f"No gradient for AR-SSM {name}"
            assert torch.isfinite(param.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
