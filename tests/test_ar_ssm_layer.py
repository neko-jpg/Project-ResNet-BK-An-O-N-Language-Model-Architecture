"""
Unit Tests for Adaptive Rank Semiseparable Layer (AR-SSM)

Tests cover:
- Output shape correctness
- Complexity gate range validation
- Effective rank reduction
- Gradient flow
- Memory efficiency (O(N log N) vs O(N²))

Requirements: 6.1, 6.2, 6.4
"""

import pytest
import torch
import torch.nn as nn
import math

from src.models.phase1 import AdaptiveRankSemiseparableLayer, Phase1Config
from src.models import SemiseparableMatrix


class TestAdaptiveRankSemiseparableLayer:
    """Unit tests for AR-SSM layer."""
    
    @pytest.fixture
    def layer(self):
        """Create a standard AR-SSM layer for testing."""
        return AdaptiveRankSemiseparableLayer(
            d_model=64,
            max_rank=16,
            min_rank=4,
            l1_regularization=0.001,
            use_fused_scan=False,  # Use torch.cumsum for CPU testing
        )
    
    @pytest.fixture
    def input_tensor(self):
        """Create a standard input tensor for testing."""
        return torch.randn(2, 128, 64)  # (B=2, L=128, D=64)
    
    def test_output_shape(self, layer, input_tensor):
        """
        Test that output shape matches input shape.
        
        Requirement 6.1: Test output shape correctness for various input dimensions
        """
        y, diagnostics = layer(input_tensor)
        assert y.shape == input_tensor.shape, \
            f"Expected shape {input_tensor.shape}, got {y.shape}"
    
    def test_output_shape_various_dimensions(self):
        """
        Test output shape correctness for various input dimensions.
        
        Requirement 6.1: Test output shape correctness for various input dimensions
        """
        test_cases = [
            (1, 64, 32),    # Small batch, short sequence
            (4, 256, 128),  # Medium batch, medium sequence
            (2, 512, 256),  # Large sequence
            (8, 128, 64),   # Large batch
        ]
        
        for B, L, D in test_cases:
            layer = AdaptiveRankSemiseparableLayer(
                d_model=D,
                max_rank=16,
                use_fused_scan=False,
            )
            x = torch.randn(B, L, D)
            y, _ = layer(x)
            assert y.shape == (B, L, D), \
                f"Failed for shape ({B}, {L}, {D}): got {y.shape}"
    
    def test_complexity_gate_range(self, layer, input_tensor):
        """
        Test that complexity gate outputs are in [0, 1] range.
        
        Requirement 6.2: Verify gate outputs are in [0, 1]
        """
        gates = layer.estimate_rank_gate(input_tensor)
        
        # Check shape
        B, L, _ = input_tensor.shape
        assert gates.shape == (B, L, layer.max_rank), \
            f"Expected gate shape ({B}, {L}, {layer.max_rank}), got {gates.shape}"
        
        # Check range [0, 1]
        assert (gates >= 0).all(), "Some gates are negative"
        assert (gates <= 1).all(), "Some gates exceed 1.0"
        
        # Check that gates are not all zeros or all ones (should have variation)
        assert gates.min() < 0.9, "Gates are all near 1.0 (no adaptation)"
        assert gates.max() > 0.1, "Gates are all near 0.0 (no activation)"
    
    def test_effective_rank_reduction(self, layer, input_tensor):
        """
        Test that gating reduces effective rank below max_rank.
        
        Requirement 6.2: Test effective rank reduction via gating
        """
        gates = layer.estimate_rank_gate(input_tensor)
        effective_rank = layer.get_effective_rank(gates)
        
        # Effective rank should be less than max_rank (due to gating)
        assert effective_rank < layer.max_rank, \
            f"Effective rank {effective_rank} should be < max_rank {layer.max_rank}"
        
        # Effective rank should be greater than min_rank (not completely collapsed)
        assert effective_rank > layer.min_rank, \
            f"Effective rank {effective_rank} should be > min_rank {layer.min_rank}"
        
        print(f"Effective rank: {effective_rank:.2f} / {layer.max_rank}")
    
    def test_gradient_flow(self, layer, input_tensor):
        """
        Test that gradients flow through all components.
        
        Requirement 6.2: Test gradient flow through all components
        """
        # Enable gradient tracking
        input_tensor.requires_grad = True
        
        # Forward pass
        y, diagnostics = layer(input_tensor)
        
        # Compute loss
        loss = y.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that input has gradients
        assert input_tensor.grad is not None, "No gradient for input"
        assert torch.isfinite(input_tensor.grad).all(), "Input gradient contains NaN/Inf"
        
        # Check that all layer parameters have gradients
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), \
                f"Parameter {name} gradient contains NaN/Inf"
        
        print(f"Gradient norm (input): {input_tensor.grad.norm().item():.4f}")
    
    def test_memory_efficiency(self):
        """
        Test that memory usage is O(N log N), not O(N²).
        
        Requirement 6.4: Test memory usage is O(N log N) not O(N²)
        """
        layer = AdaptiveRankSemiseparableLayer(
            d_model=64,
            max_rank=16,
            use_fused_scan=False,
        )
        
        # Test with sequence length 1024
        batch_size = 1
        seq_len = 1024
        x = torch.randn(batch_size, seq_len, 64)
        
        # Get memory usage estimate
        memory_info = layer.get_memory_usage(batch_size, seq_len)
        
        # Verify memory reduction vs attention
        assert memory_info['memory_reduction_vs_attention'] > 0.5, \
            f"Memory reduction {memory_info['memory_reduction_vs_attention']:.2%} should be > 50%"
        
        # Verify actual memory is much less than O(N²)
        # Attention would use seq_len² * 4 bytes ≈ 4MB for attention matrix alone
        attention_memory_mb = seq_len * seq_len * 4 / (1024 ** 2)
        assert memory_info['activation_memory_mb'] < attention_memory_mb * 0.3, \
            f"AR-SSM memory {memory_info['activation_memory_mb']:.2f}MB should be < 30% of " \
            f"attention memory {attention_memory_mb:.2f}MB"
        
        print(f"AR-SSM memory: {memory_info['activation_memory_mb']:.2f}MB")
        print(f"Attention memory: {attention_memory_mb:.2f}MB")
        print(f"Reduction: {memory_info['memory_reduction_vs_attention']:.2%}")
    
    def test_memory_complexity_verification(self):
        """
        Test memory complexity verification method.
        
        Requirement 6.4: Verify O(N log N) complexity scaling
        """
        layer = AdaptiveRankSemiseparableLayer(
            d_model=128,
            max_rank=32,
            use_fused_scan=False,
        )
        
        batch_size = 2
        seq_len = 2048
        
        verification = layer.verify_memory_complexity(batch_size, seq_len)
        
        # Should be subquadratic
        assert verification['is_subquadratic'], \
            "Memory usage is not subquadratic (should be < 50% of attention)"
        
        # Should be near theoretical O(N log N) bound
        assert verification['is_near_theoretical'], \
            "Memory usage exceeds 2x theoretical O(N log N) bound"
        
        # Overall verification should pass
        assert verification['complexity_verified'], \
            "Memory complexity verification failed"
        
        print(f"Actual: {verification['actual_memory_mb']:.2f}MB")
        print(f"Theoretical: {verification['theoretical_memory_mb']:.2f}MB")
        print(f"Attention: {verification['attention_memory_mb']:.2f}MB")
    
    def test_l1_regularization(self, layer, input_tensor):
        """
        Test L1 regularization for gate sparsity.
        
        Requirement 6.2: Test L1 regularization support
        """
        gates = layer.estimate_rank_gate(input_tensor)
        l1_loss = layer.get_gate_l1_loss(gates)
        
        # L1 loss should be positive
        assert l1_loss > 0, "L1 loss should be positive"
        
        # L1 loss should be finite
        assert torch.isfinite(l1_loss), "L1 loss is NaN/Inf"
        
        # L1 loss should be scaled by regularization coefficient
        expected_magnitude = layer.l1_regularization
        assert l1_loss < expected_magnitude * 2, \
            f"L1 loss {l1_loss:.6f} seems too large"
        
        print(f"L1 loss: {l1_loss:.6f}")
    
    def test_rank_scheduling(self, layer):
        """
        Test rank scheduling for curriculum learning.
        
        Requirement 6.2: Test rank scheduling functionality
        """
        # Initially at max rank
        assert layer.current_max_rank == layer.max_rank
        
        # Update schedule at step 0 (should be at min_rank)
        layer.update_rank_schedule(step=0, warmup_steps=1000)
        assert layer.current_max_rank == layer.min_rank, \
            f"At step 0, should be at min_rank {layer.min_rank}, got {layer.current_max_rank}"
        
        # Update schedule at step 500 (should be halfway)
        layer.update_rank_schedule(step=500, warmup_steps=1000)
        expected_rank = int(layer.min_rank + 0.5 * (layer.max_rank - layer.min_rank))
        assert layer.current_max_rank == expected_rank, \
            f"At step 500, expected rank {expected_rank}, got {layer.current_max_rank}"
        
        # Update schedule at step 1000 (should be at max_rank)
        layer.update_rank_schedule(step=1000, warmup_steps=1000)
        assert layer.current_max_rank == layer.max_rank, \
            f"At step 1000, should be at max_rank {layer.max_rank}, got {layer.current_max_rank}"
        
        print(f"Rank schedule: {layer.min_rank} → {expected_rank} → {layer.max_rank}")
    
    def test_from_config(self):
        """
        Test creating AR-SSM layer from Phase1Config.
        
        Requirement 6.1: Test configuration-based initialization
        """
        config = Phase1Config(
            ar_ssm_max_rank=32,
            ar_ssm_min_rank=8,
            ar_ssm_l1_regularization=0.002,
            ar_ssm_use_fused_scan=False,
        )
        
        layer = AdaptiveRankSemiseparableLayer.from_config(
            config=config,
            d_model=128,
        )
        
        assert layer.max_rank == 32
        assert layer.min_rank == 8
        assert layer.l1_regularization == 0.002
        assert layer.use_fused_scan == False
        assert layer.d_model == 128
    
    def test_from_semiseparable_matrix(self):
        """
        Test creating AR-SSM layer from existing SemiseparableMatrix.
        
        Requirement 6.1: Test integration with SemiseparableMatrix
        """
        # Create a SemiseparableMatrix
        n_seq = 128
        semisep = SemiseparableMatrix(n_seq=n_seq, rank=8)
        
        # Create a dummy matrix and factorize
        H = torch.randn(n_seq, n_seq)
        H = (H + H.T) / 2  # Make symmetric
        semisep.factorize(H)
        
        # Create AR-SSM from semiseparable matrix
        layer = AdaptiveRankSemiseparableLayer.from_semiseparable_matrix(
            semisep=semisep,
            d_model=64,
            max_rank=16,
        )
        
        assert layer.max_rank == 16
        assert layer.base_semisep is semisep
        
        # Test forward pass
        x = torch.randn(2, n_seq, 64)
        y, _ = layer(x)
        assert y.shape == x.shape
    
    def test_checkpointing(self, layer, input_tensor):
        """
        Test gradient checkpointing functionality.
        
        Requirement 6.2: Test gradient checkpointing support
        """
        # Enable checkpointing
        layer.enable_checkpointing()
        assert layer._checkpointing_enabled
        
        # Forward pass with checkpointing
        input_tensor.requires_grad = True
        y, diagnostics = layer.forward_with_checkpointing(input_tensor)
        
        # Backward pass
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist
        assert input_tensor.grad is not None
        assert torch.isfinite(input_tensor.grad).all()
        
        # Disable checkpointing
        layer.disable_checkpointing()
        assert not layer._checkpointing_enabled
    
    def test_diagnostics_output(self, layer, input_tensor):
        """
        Test that forward pass returns proper diagnostics.
        
        Requirement 6.1: Test diagnostic output
        """
        y, diagnostics = layer(input_tensor)
        
        # Check required diagnostic keys
        required_keys = [
            't_component',
            'gates',
            'effective_rank',
            'u_gated',
            'v_gated',
            'k_cumsum',
            'uv_component',
            'gate_l1_loss',
        ]
        
        for key in required_keys:
            assert key in diagnostics, f"Missing diagnostic key: {key}"
        
        # Check diagnostic shapes
        B, L, D = input_tensor.shape
        assert diagnostics['t_component'].shape == (B, L, D)
        assert diagnostics['gates'].shape == (B, L, layer.max_rank)
        assert diagnostics['effective_rank'].dim() == 0  # Scalar
        assert diagnostics['u_gated'].shape == (B, L, layer.max_rank)
        assert diagnostics['v_gated'].shape == (B, L, layer.max_rank)
        assert diagnostics['k_cumsum'].shape == (B, L, layer.max_rank)
        assert diagnostics['uv_component'].shape == (B, L, D)
        assert diagnostics['gate_l1_loss'].dim() == 0  # Scalar
    
    def test_numerical_stability(self, layer):
        """
        Test numerical stability with extreme inputs.
        
        Requirement 6.2: Test numerical stability
        """
        # Test with very small values
        x_small = torch.randn(2, 128, 64) * 1e-6
        y_small, _ = layer(x_small)
        assert torch.isfinite(y_small).all(), "Output contains NaN/Inf for small inputs"
        
        # Test with very large values
        x_large = torch.randn(2, 128, 64) * 1e3
        y_large, _ = layer(x_large)
        assert torch.isfinite(y_large).all(), "Output contains NaN/Inf for large inputs"
        
        # Test with mixed signs
        x_mixed = torch.randn(2, 128, 64)
        x_mixed[0] *= -1
        y_mixed, _ = layer(x_mixed)
        assert torch.isfinite(y_mixed).all(), "Output contains NaN/Inf for mixed signs"
    
    def test_bk_core_integration(self, layer):
        """
        Test integration with BK-Core features.
        
        Requirement 6.1: Test BK-Core integration
        """
        # Simulate BK-Core output
        bk_features = torch.randn(2, 128, 64)
        
        # Process through AR-SSM
        output = layer.integrate_with_bk_core(bk_features)
        
        # Check output shape
        assert output.shape == bk_features.shape
        
        # Check output is finite
        assert torch.isfinite(output).all()


class TestARSSMScaling:
    """Test O(N) complexity scaling of AR-SSM layer."""
    
    def test_linear_time_complexity(self):
        """
        Test that forward pass time scales linearly with sequence length.
        
        Requirement 6.4: Verify O(N) complexity scaling
        """
        import time
        
        layer = AdaptiveRankSemiseparableLayer(
            d_model=128,
            max_rank=32,
            use_fused_scan=False,
        )
        
        sequence_lengths = [128, 256, 512, 1024]
        times = []
        
        for seq_len in sequence_lengths:
            x = torch.randn(1, seq_len, 128)
            
            # Warmup
            _ = layer(x)
            
            # Measure time
            start = time.time()
            for _ in range(10):
                _ = layer(x)
            elapsed = time.time() - start
            
            times.append(elapsed / 10)  # Average time per forward pass
        
        # Check that time scales roughly linearly
        # time[i+1] / time[i] should be close to seq_len[i+1] / seq_len[i]
        for i in range(len(times) - 1):
            time_ratio = times[i + 1] / times[i]
            seq_ratio = sequence_lengths[i + 1] / sequence_lengths[i]
            
            # Allow 50% tolerance (due to overhead and caching)
            assert time_ratio < seq_ratio * 1.5, \
                f"Time scaling not linear: {time_ratio:.2f}x vs {seq_ratio:.2f}x sequence length"
        
        print(f"Sequence lengths: {sequence_lengths}")
        print(f"Times (ms): {[t * 1000 for t in times]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
