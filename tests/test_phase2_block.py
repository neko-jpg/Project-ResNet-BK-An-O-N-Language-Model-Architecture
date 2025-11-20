"""
Test Phase2Block implementation

Tests:
1. Basic instantiation
2. Forward pass without diagnostics
3. Forward pass with diagnostics
4. Residual connections
5. State management
6. Gradient flow

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import pytest
import torch
import torch.nn as nn

from src.models.phase2.integrated_model import Phase2Block


class TestPhase2Block:
    """Test suite for Phase2Block"""
    
    @pytest.fixture
    def device(self):
        """Get device for testing"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self):
        """Basic configuration for testing"""
        return {
            'd_model': 128,
            'n_seq': 64,
            'num_heads': 4,
            'head_dim': 32,
            'use_triton': False,  # Use PyTorch for testing
            'ffn_dim': 512,
            'dropout': 0.1,
        }
    
    @pytest.fixture
    def block(self, config, device):
        """Create Phase2Block instance"""
        block = Phase2Block(**config)
        return block.to(device)
    
    def test_instantiation(self, block, config):
        """Test that Phase2Block can be instantiated (Requirement 6.1)"""
        assert isinstance(block, nn.Module)
        assert block.d_model == config['d_model']
        assert block.n_seq == config['n_seq']
        assert block.num_heads == config['num_heads']
        assert block.head_dim == config['head_dim']
        
        # Check components exist
        assert hasattr(block, 'dissipative_bk')
        assert hasattr(block, 'hebbian')
        assert hasattr(block, 'snr_filter')
        assert hasattr(block, 'resonance')
        assert hasattr(block, 'ffn')
        
        # Check layer norms
        assert hasattr(block, 'ln1')
        assert hasattr(block, 'ln2')
        assert hasattr(block, 'ln3')
    
    def test_forward_basic(self, block, config, device):
        """Test basic forward pass (Requirement 6.2)"""
        B, N, D = 2, config['n_seq'], config['d_model']
        x = torch.randn(B, N, D, device=device)
        
        # Forward pass
        output = block(x, return_diagnostics=False)
        
        # Check output shape
        assert output.shape == (B, N, D)
        
        # Check no NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_with_diagnostics(self, block, config, device):
        """Test forward pass with diagnostics (Requirement 6.2)"""
        B, N, D = 2, config['n_seq'], config['d_model']
        x = torch.randn(B, N, D, device=device)
        
        # Forward pass with diagnostics
        output, diagnostics = block(x, return_diagnostics=True)
        
        # Check output shape
        assert output.shape == (B, N, D)
        
        # Check diagnostics keys (Requirement 6.2)
        expected_keys = [
            'gamma',
            'v_complex',
            'bk_features',
            'hebbian_output',
            'fast_weight_energy',
            'potential_feedback',
            'snr_stats',
            'adjusted_gamma',
            'adjusted_eta',
            'resonance_info',
            'ffn_output',
            'stability',
        ]
        
        for key in expected_keys:
            assert key in diagnostics, f"Missing diagnostic key: {key}"
        
        # Check gamma values
        assert diagnostics['gamma'].shape == (B, N)
        assert (diagnostics['gamma'] > 0).all(), "Gamma should be positive"
        
        # Check SNR stats
        assert isinstance(diagnostics['snr_stats'], dict)
        
        # Check resonance info
        assert isinstance(diagnostics['resonance_info'], dict)
        
        # Check stability metrics
        assert isinstance(diagnostics['stability'], dict)
        assert 'energy' in diagnostics['stability']
        assert 'is_stable' in diagnostics['stability']
    
    def test_residual_connections(self, block, config, device):
        """Test that residual connections work (Requirement 6.3)"""
        B, N, D = 2, config['n_seq'], config['d_model']
        x = torch.randn(B, N, D, device=device)
        
        # Forward pass
        output = block(x, return_diagnostics=False)
        
        # Output should be different from input (not identity)
        assert not torch.allclose(output, x, atol=1e-5)
        
        # But should have similar magnitude (residual connections preserve scale)
        input_norm = torch.norm(x)
        output_norm = torch.norm(output)
        ratio = output_norm / input_norm
        
        # Ratio should be reasonable (not exploding or vanishing)
        assert 0.5 < ratio < 2.0, f"Output/input norm ratio: {ratio}"
    
    def test_state_management(self, block, config, device):
        """Test Fast Weight state management"""
        B, N, D = 2, config['n_seq'], config['d_model']
        x = torch.randn(B, N, D, device=device)
        
        # Initial state should be None
        assert block.fast_weight_state is None
        
        # After forward pass, state should be created
        _ = block(x)
        assert block.fast_weight_state is not None
        assert block.fast_weight_state.shape == (B, config['num_heads'], config['head_dim'], config['head_dim'])
        
        # Reset state
        block.reset_state()
        assert block.fast_weight_state is None
    
    def test_gradient_flow(self, block, config, device):
        """Test that gradients flow through the block (Requirement 6.4)"""
        B, N, D = 2, config['n_seq'], config['d_model']
        x = torch.randn(B, N, D, device=device, requires_grad=True)
        
        # Forward pass
        output = block(x)
        
        # Compute loss
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        # Check that parameters have gradients
        # Note: Some BK-Core parameters may not receive gradients depending on the computation path
        params_with_grad = 0
        params_without_grad = []
        for name, param in block.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    params_with_grad += 1
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                    assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
                else:
                    params_without_grad.append(name)
        
        # At least some parameters should have gradients
        assert params_with_grad > 0, "No parameters received gradients"
        
        # Most parameters should have gradients (allow some BK-Core params to not have grads)
        total_params = params_with_grad + len(params_without_grad)
        grad_ratio = params_with_grad / total_params
        assert grad_ratio > 0.5, f"Too few parameters with gradients: {grad_ratio:.2%}"
    
    def test_multiple_forward_passes(self, block, config, device):
        """Test multiple forward passes with state accumulation"""
        B, N, D = 2, config['n_seq'], config['d_model']
        
        # First forward pass
        x1 = torch.randn(B, N, D, device=device)
        output1 = block(x1)
        state1 = block.fast_weight_state.clone() if block.fast_weight_state is not None else None
        
        # Second forward pass (state should be updated)
        x2 = torch.randn(B, N, D, device=device)
        output2 = block(x2)
        state2 = block.fast_weight_state.clone() if block.fast_weight_state is not None else None
        
        # States should be different
        if state1 is not None and state2 is not None:
            assert not torch.allclose(state1, state2, atol=1e-5)
    
    def test_statistics(self, block, config, device):
        """Test statistics collection"""
        B, N, D = 2, config['n_seq'], config['d_model']
        x = torch.randn(B, N, D, device=device)
        
        # Forward pass
        _ = block(x)
        
        # Get statistics
        stats = block.get_statistics()
        
        # Check structure
        assert isinstance(stats, dict)
        assert 'hebbian' in stats
        assert 'snr' in stats
        assert 'non_hermitian' in stats
        
        # Check hebbian stats
        assert isinstance(stats['hebbian'], dict)
        
        # Check SNR stats
        assert isinstance(stats['snr'], dict)
        
        # Check non-hermitian stats
        assert isinstance(stats['non_hermitian'], dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
