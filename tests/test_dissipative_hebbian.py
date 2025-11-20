"""
Unit tests for Dissipative Hebbian Layer

Tests:
1. Basic instantiation and forward pass
2. Dissipative Hebbian update equation correctness
3. Lyapunov stability monitoring
4. Gradient computation (forward and backward)
5. Potential feedback mechanism
6. Sequential inference (forward_step)
7. Memory efficiency

Author: Project MUSE Team
Date: 2025-01-20
"""

import pytest
import torch
import torch.nn as nn
from src.models.phase2.dissipative_hebbian import (
    DissipativeHebbianLayer,
    LyapunovStabilityMonitor,
)


class TestLyapunovStabilityMonitor:
    """Test Lyapunov Stability Monitor"""
    
    def test_initialization(self):
        """Test monitor initialization"""
        monitor = LyapunovStabilityMonitor(gamma_adjust_rate=0.01)
        assert monitor.gamma_adjust_rate == 0.01
        assert monitor.prev_energy is None
        assert len(monitor.energy_history) == 0
        assert monitor.violation_count == 0
    
    def test_stability_check_stable(self):
        """Test stability check with decreasing energy"""
        monitor = LyapunovStabilityMonitor()
        
        # First state (higher energy)
        state1 = torch.randn(2, 4, 8, 8) * 2.0
        decay1 = torch.tensor(0.9).view(1, 1, 1, 1)
        update1 = torch.randn(2, 4, 8, 8) * 0.1
        
        metrics1 = monitor.check(state1, decay1, update1)
        assert metrics1['is_stable'] == True  # First check always stable
        
        # Second state (lower energy - stable)
        state2 = torch.randn(2, 4, 8, 8) * 1.0
        decay2 = torch.tensor(0.9).view(1, 1, 1, 1)
        update2 = torch.randn(2, 4, 8, 8) * 0.1
        
        metrics2 = monitor.check(state2, decay2, update2)
        assert metrics2['is_stable'] == True
        assert metrics2['dE_dt'] < 0  # Energy decreased
        assert metrics2['suggested_gamma_adjust'] < 0  # Suggest decreasing gamma
    
    def test_stability_check_unstable(self):
        """Test stability check with increasing energy"""
        monitor = LyapunovStabilityMonitor()
        
        # First state (lower energy)
        state1 = torch.randn(2, 4, 8, 8) * 1.0
        decay1 = torch.tensor(0.9).view(1, 1, 1, 1)
        update1 = torch.randn(2, 4, 8, 8) * 0.1
        
        monitor.check(state1, decay1, update1)
        
        # Second state (higher energy - unstable)
        state2 = torch.randn(2, 4, 8, 8) * 3.0
        decay2 = torch.tensor(0.9).view(1, 1, 1, 1)
        update2 = torch.randn(2, 4, 8, 8) * 0.1
        
        metrics2 = monitor.check(state2, decay2, update2)
        assert metrics2['is_stable'] == False
        assert metrics2['dE_dt'] > 0  # Energy increased
        assert metrics2['suggested_gamma_adjust'] > 0  # Suggest increasing gamma
        assert metrics2['violation_count'] == 1
    
    def test_reset(self):
        """Test monitor reset"""
        monitor = LyapunovStabilityMonitor()
        
        # Add some history
        state = torch.randn(2, 4, 8, 8)
        decay = torch.tensor(0.9).view(1, 1, 1, 1)
        update = torch.randn(2, 4, 8, 8)
        monitor.check(state, decay, update)
        
        # Reset
        monitor.reset()
        assert monitor.prev_energy is None
        assert len(monitor.energy_history) == 0
        assert monitor.violation_count == 0
    
    def test_statistics(self):
        """Test statistics computation"""
        monitor = LyapunovStabilityMonitor()
        
        # Add multiple checks
        for _ in range(10):
            state = torch.randn(2, 4, 8, 8)
            decay = torch.tensor(0.9).view(1, 1, 1, 1)
            update = torch.randn(2, 4, 8, 8)
            monitor.check(state, decay, update)
        
        stats = monitor.get_statistics()
        assert 'mean_energy' in stats
        assert 'std_energy' in stats
        assert 'min_energy' in stats
        assert 'max_energy' in stats
        assert stats['mean_energy'] > 0


class TestDissipativeHebbianLayer:
    """Test Dissipative Hebbian Layer"""
    
    @pytest.fixture
    def layer(self):
        """Create a test layer"""
        return DissipativeHebbianLayer(
            d_model=64,
            head_dim=16,
            num_heads=4,
            eta=0.1,
            dt=1.0,
            enable_potential_feedback=True,
        )
    
    def test_initialization(self, layer):
        """Test layer initialization"""
        assert layer.d_model == 64
        assert layer.head_dim == 16
        assert layer.num_heads == 4
        assert layer.eta == 0.1
        assert layer.dt == 1.0
        assert layer.enable_potential_feedback == True
        
        # Check projections
        assert isinstance(layer.q_proj, nn.Linear)
        assert isinstance(layer.k_proj, nn.Linear)
        assert isinstance(layer.v_proj, nn.Linear)
        assert isinstance(layer.out_proj, nn.Linear)
        
        # Check stability monitor
        assert isinstance(layer.stability_monitor, LyapunovStabilityMonitor)
    
    def test_forward_pass_basic(self, layer):
        """Test basic forward pass"""
        B, N, D = 2, 8, 64
        x = torch.randn(B, N, D)
        gamma = torch.rand(B, N) * 0.1  # Small gamma values
        
        output, state, potential_feedback = layer(x, gamma, return_potential_feedback=True)
        
        # Check output shape
        assert output.shape == (B, N, D)
        
        # Check state shape
        assert state.shape == (B, layer.num_heads, layer.head_dim, layer.head_dim)
        
        # Check potential feedback
        assert potential_feedback.shape == (B, N)
    
    def test_forward_pass_with_state(self, layer):
        """Test forward pass with existing state"""
        B, N, D = 2, 8, 64
        x = torch.randn(B, N, D)
        gamma = torch.rand(B, N) * 0.1
        
        # Initial state
        initial_state = torch.randn(B, layer.num_heads, layer.head_dim, layer.head_dim)
        
        output, new_state, _ = layer(x, gamma, state=initial_state)
        
        # State should be updated
        assert not torch.allclose(new_state, initial_state)
        assert new_state.shape == initial_state.shape
    
    def test_dissipative_hebbian_equation(self, layer):
        """Test that the dissipative Hebbian equation is correctly implemented"""
        B, N, D = 1, 1, 64  # Single timestep for easy verification
        x = torch.randn(B, N, D)
        gamma = torch.tensor([[0.1]])  # Fixed gamma
        
        # Initial state
        W_old = torch.randn(B, layer.num_heads, layer.head_dim, layer.head_dim)
        
        # Forward pass
        output, W_new, _ = layer(x, gamma, state=W_old)
        
        # Manually compute expected update
        q = layer.q_proj(x).view(B, N, layer.num_heads, layer.head_dim).transpose(1, 2)
        k = layer.k_proj(x).view(B, N, layer.num_heads, layer.head_dim).transpose(1, 2)
        v = layer.v_proj(x).view(B, N, layer.num_heads, layer.head_dim).transpose(1, 2)
        
        k_t = k[:, :, 0, :]
        v_t = v[:, :, 0, :]
        
        # Decay term
        decay = torch.exp(-gamma[0, 0] * layer.dt)
        
        # Hebbian update
        hebbian_update = layer.eta * torch.einsum('bhi,bhj->bhij', k_t, v_t)
        
        # Expected: W_new = decay * W_old + hebbian_update
        W_expected = decay * W_old + hebbian_update
        
        # Check (allow some numerical error)
        assert torch.allclose(W_new, W_expected, atol=1e-5)
    
    def test_gradient_computation(self, layer):
        """Test that gradients flow correctly through forward and backward"""
        B, N, D = 2, 4, 64
        x = torch.randn(B, N, D, requires_grad=True)
        gamma = torch.rand(B, N) * 0.1
        
        # Use potential feedback to ensure all parameters are used
        output, state, feedback = layer(x, gamma, return_potential_feedback=True)
        
        # Compute loss (include feedback to use memory_to_potential)
        loss = output.sum() + (feedback.sum() if feedback is not None else 0.0)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        # Check that layer parameters have gradients
        for name, param in layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
    
    def test_gradient_stability_with_large_gamma(self, layer):
        """Test gradient stability with large gamma values"""
        B, N, D = 2, 4, 64
        x = torch.randn(B, N, D, requires_grad=True)
        gamma = torch.rand(B, N) * 5.0  # Large gamma values
        
        output, state, _ = layer(x, gamma)
        loss = output.sum()
        loss.backward()
        
        # Gradients should still be finite
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    def test_forward_step(self, layer):
        """Test sequential inference with forward_step"""
        B, D = 2, 64
        x_t = torch.randn(B, D)
        gamma_t = torch.rand(B) * 0.1
        
        # Initial state
        state = torch.randn(B, layer.num_heads, layer.head_dim, layer.head_dim)
        
        # Single step
        output_t, new_state = layer.forward_step(x_t, gamma_t, state)
        
        # Check shapes
        assert output_t.shape == (B, D)
        assert new_state.shape == state.shape
        
        # State should be updated
        assert not torch.allclose(new_state, state)
    
    def test_potential_feedback(self, layer):
        """Test potential feedback mechanism"""
        B, N, D = 2, 8, 64
        x = torch.randn(B, N, D)
        gamma = torch.rand(B, N) * 0.1
        
        # With feedback
        output, state, feedback = layer(x, gamma, return_potential_feedback=True)
        assert feedback is not None
        assert feedback.shape == (B, N)
        
        # Without feedback
        output2, state2, feedback2 = layer(x, gamma, return_potential_feedback=False)
        assert feedback2 is None
    
    def test_potential_feedback_disabled(self):
        """Test layer with potential feedback disabled"""
        layer = DissipativeHebbianLayer(
            d_model=64,
            head_dim=16,
            num_heads=4,
            enable_potential_feedback=False,
        )
        
        B, N, D = 2, 8, 64
        x = torch.randn(B, N, D)
        gamma = torch.rand(B, N) * 0.1
        
        output, state, feedback = layer(x, gamma, return_potential_feedback=True)
        assert feedback is None
    
    def test_statistics(self, layer):
        """Test statistics collection"""
        B, N, D = 2, 8, 64
        x = torch.randn(B, N, D)
        gamma = torch.rand(B, N) * 0.1
        
        # Training mode
        layer.train()
        output, state, _ = layer(x, gamma)
        
        stats = layer.get_statistics()
        assert 'mean_update_norm' in stats
        assert 'std_update_norm' in stats
        assert 'mean_decay' in stats
        assert 'std_decay' in stats
        assert 'mean_energy' in stats
    
    def test_reset_state(self, layer):
        """Test state reset"""
        B, N, D = 2, 8, 64
        x = torch.randn(B, N, D)
        gamma = torch.rand(B, N) * 0.1
        
        # Run forward to populate history
        layer.train()
        layer(x, gamma)
        
        # Reset
        layer.reset_state()
        
        # Check that history is cleared
        assert layer.history_idx == 0
        assert layer.update_norm_history.sum() == 0
        assert layer.decay_history.sum() == 0
    
    def test_memory_efficiency(self, layer):
        """Test that memory usage is reasonable"""
        B, N, D = 4, 128, 64
        x = torch.randn(B, N, D)
        gamma = torch.rand(B, N) * 0.1
        
        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            x = x.cuda()
            gamma = gamma.cuda()
            layer = layer.cuda()
            
            output, state, _ = layer(x, gamma)
            
            # Memory usage should be reasonable
            # Fast Weights: B * H * D_h * D_h * 4 bytes
            # Expected: 4 * 4 * 16 * 16 * 4 = 16KB (very small)
            max_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            # Should be well under 1.5GB (KPI requirement)
            assert max_memory_mb < 1500, f"Memory usage {max_memory_mb:.2f}MB exceeds 1.5GB limit"
    
    def test_numerical_stability_long_sequence(self, layer):
        """Test numerical stability with long sequences"""
        B, N, D = 2, 512, 64
        x = torch.randn(B, N, D)
        gamma = torch.rand(B, N) * 0.1
        
        output, state, _ = layer(x, gamma)
        
        # Check for NaN or Inf
        assert torch.isfinite(output).all()
        assert torch.isfinite(state).all()
    
    def test_different_gamma_values(self, layer):
        """Test with different gamma values"""
        B, N, D = 2, 8, 64
        x = torch.randn(B, N, D)
        
        # Test with various gamma values
        gamma_values = [0.0, 0.01, 0.1, 1.0, 5.0]
        
        for gamma_val in gamma_values:
            gamma = torch.full((B, N), gamma_val)
            output, state, _ = layer(x, gamma)
            
            # Should not produce NaN or Inf
            assert torch.isfinite(output).all(), f"Non-finite output with gamma={gamma_val}"
            assert torch.isfinite(state).all(), f"Non-finite state with gamma={gamma_val}"


class TestIntegration:
    """Integration tests"""
    
    def test_multiple_forward_passes(self):
        """Test multiple forward passes with state persistence"""
        layer = DissipativeHebbianLayer(d_model=64, head_dim=16, num_heads=4)
        
        B, N, D = 2, 8, 64
        state = None
        
        # Multiple passes
        for _ in range(5):
            x = torch.randn(B, N, D)
            gamma = torch.rand(B, N) * 0.1
            
            output, state, _ = layer(x, gamma, state=state)
            
            # State should evolve
            assert torch.isfinite(state).all()
    
    def test_training_loop_simulation(self):
        """Simulate a training loop"""
        layer = DissipativeHebbianLayer(d_model=64, head_dim=16, num_heads=4)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
        
        B, N, D = 2, 8, 64
        
        # Training steps
        for step in range(10):
            x = torch.randn(B, N, D)
            gamma = torch.rand(B, N) * 0.1
            target = torch.randn(B, N, D)
            
            optimizer.zero_grad()
            output, state, _ = layer(x, gamma)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            
            # Loss should be finite
            assert torch.isfinite(loss).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
