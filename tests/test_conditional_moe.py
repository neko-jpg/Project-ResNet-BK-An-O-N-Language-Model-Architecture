"""
Tests for Conditional MoE Layer
"""

import torch
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.conditional_moe import ConditionalMoELayer, ConditionalMoEWithLoadBalancing


class TestConditionalMoELayer:
    """Test basic conditional MoE functionality."""
    
    def test_initialization(self):
        """Test model initialization."""
        d_model = 64
        max_experts = 4
        min_experts = 1
        
        model = ConditionalMoELayer(
            d_model=d_model,
            max_experts=max_experts,
            min_experts=min_experts
        )
        
        assert model.d_model == d_model
        assert model.max_experts == max_experts
        assert model.min_experts == min_experts
        assert len(model.experts) == max_experts
        
        # Check expert structure
        for expert in model.experts:
            assert isinstance(expert, torch.nn.Sequential)
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        d_model = 64
        batch_size = 4
        seq_len = 16
        
        model = ConditionalMoELayer(d_model=d_model, max_experts=4, min_experts=1)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output, stats = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
        
        # Check stats
        assert 'avg_entropy' in stats
        assert 'avg_num_experts' in stats
        assert 'min_num_experts' in stats
        assert 'max_num_experts' in stats
    
    def test_entropy_computation(self):
        """Test entropy computation."""
        d_model = 64
        batch_size = 2
        seq_len = 8
        
        model = ConditionalMoELayer(d_model=d_model, max_experts=4, min_experts=1)
        
        x = torch.randn(batch_size, seq_len, d_model)
        entropy = model.compute_input_entropy(x)
        
        # Check shape
        assert entropy.shape == (batch_size, seq_len)
        
        # Check non-negative (entropy is always >= 0)
        assert (entropy >= 0).all()
    
    def test_num_experts_determination(self):
        """Test expert count determination from entropy."""
        d_model = 64
        model = ConditionalMoELayer(
            d_model=d_model,
            max_experts=4,
            min_experts=1,
            entropy_threshold_low=0.5,
            entropy_threshold_high=2.0
        )
        
        # Low entropy → min experts
        low_entropy = torch.tensor([[0.1, 0.2, 0.3]])
        num_experts_low = model.determine_num_experts(low_entropy)
        assert (num_experts_low == 1).all()
        
        # High entropy → max experts
        high_entropy = torch.tensor([[2.5, 3.0, 3.5]])
        num_experts_high = model.determine_num_experts(high_entropy)
        assert (num_experts_high == 4).all()
        
        # Medium entropy → intermediate
        medium_entropy = torch.tensor([[1.0, 1.25, 1.5]])
        num_experts_medium = model.determine_num_experts(medium_entropy)
        assert (num_experts_medium >= 1).all()
        assert (num_experts_medium <= 4).all()
    
    def test_easy_vs_hard_inputs(self):
        """Test that easy inputs use fewer experts than hard inputs."""
        d_model = 64
        batch_size = 4
        seq_len = 16
        
        model = ConditionalMoELayer(
            d_model=d_model,
            max_experts=4,
            min_experts=1,
            entropy_threshold_low=0.3,
            entropy_threshold_high=1.5
        )
        
        # Easy input: low variance
        easy_input = torch.randn(batch_size, seq_len, d_model) * 0.1
        
        # Hard input: high variance
        hard_input = torch.randn(batch_size, seq_len, d_model) * 2.0
        
        with torch.no_grad():
            _, stats_easy = model(easy_input)
            _, stats_hard = model(hard_input)
        
        # Easy inputs should use fewer experts
        assert stats_easy['avg_num_experts'] < stats_hard['avg_num_experts']
        
        # Easy inputs should have lower entropy
        assert stats_easy['avg_entropy'] < stats_hard['avg_entropy']
    
    def test_statistics_tracking(self):
        """Test statistics tracking across multiple forward passes."""
        d_model = 64
        model = ConditionalMoELayer(d_model=d_model, max_experts=4, min_experts=1)
        
        # Initial statistics
        initial_stats = model.get_routing_statistics()
        assert initial_stats['num_forward_calls'] == 0
        
        # Run forward passes
        x = torch.randn(2, 8, d_model)
        
        with torch.no_grad():
            for _ in range(5):
                model(x)
        
        # Check updated statistics
        final_stats = model.get_routing_statistics()
        assert final_stats['num_forward_calls'] == 5
        assert final_stats['avg_num_experts_used'] > 0
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        d_model = 64
        batch_size = 2
        seq_len = 8
        
        model = ConditionalMoELayer(d_model=d_model, max_experts=4, min_experts=1)
        
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        output, _ = model(x)
        
        # Compute loss
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Check that at least some model parameters have gradients
        # (not all parameters may have gradients depending on routing)
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert params_with_grad > 0, "No parameters received gradients"
        assert params_with_grad >= total_params * 0.5, f"Only {params_with_grad}/{total_params} parameters received gradients"


class TestConditionalMoEWithLoadBalancing:
    """Test conditional MoE with load balancing."""
    
    def test_initialization(self):
        """Test model initialization with load balancing."""
        d_model = 64
        model = ConditionalMoEWithLoadBalancing(
            d_model=d_model,
            max_experts=4,
            load_balance_weight=0.01
        )
        
        assert model.load_balance_weight == 0.01
        assert hasattr(model, 'expert_usage_count')
    
    def test_load_balance_loss(self):
        """Test load balance loss computation."""
        d_model = 64
        model = ConditionalMoEWithLoadBalancing(
            d_model=d_model,
            max_experts=4,
            load_balance_weight=0.01
        )
        
        # Create router logits
        router_logits = torch.randn(32, 4)  # 32 tokens, 4 experts
        num_experts_per_token = torch.randint(1, 5, (32,))
        
        loss = model.compute_load_balance_loss(router_logits, num_experts_per_token)
        
        # Check loss is scalar
        assert loss.dim() == 0
        
        # Check loss is non-negative
        assert loss >= 0
    
    def test_forward_with_load_balancing(self):
        """Test forward pass with load balancing."""
        d_model = 64
        batch_size = 4
        seq_len = 16
        
        model = ConditionalMoEWithLoadBalancing(
            d_model=d_model,
            max_experts=4,
            load_balance_weight=0.01
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        output, stats = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
        
        # Check load balance loss in stats
        assert 'load_balance_loss' in stats
        assert stats['load_balance_loss'] >= 0
    
    def test_expert_usage_tracking(self):
        """Test expert usage tracking."""
        d_model = 64
        model = ConditionalMoEWithLoadBalancing(
            d_model=d_model,
            max_experts=4
        )
        
        # Initial usage should be zero
        initial_usage = model.get_expert_usage_distribution()
        assert (initial_usage == 0).all()
        
        # Run forward passes
        x = torch.randn(4, 16, d_model)
        
        with torch.no_grad():
            for _ in range(10):
                model(x)
        
        # Check usage is tracked
        final_usage = model.get_expert_usage_distribution()
        assert final_usage.sum() > 0
        
        # Check usage sums to 1 (normalized)
        assert torch.isclose(final_usage.sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_training_with_load_balancing(self):
        """Test training with load balancing loss."""
        d_model = 64
        model = ConditionalMoEWithLoadBalancing(
            d_model=d_model,
            max_experts=4,
            load_balance_weight=0.01
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training step
        x = torch.randn(2, 8, d_model)
        output, stats = model(x)
        
        # Compute loss
        loss = output.pow(2).mean()
        total_loss = loss + model.load_balance_weight * stats['load_balance_loss']
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Check that at least some parameters received gradients
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert params_with_grad > 0, "No parameters received gradients"
        assert params_with_grad >= total_params * 0.5, f"Only {params_with_grad}/{total_params} parameters received gradients"


class TestConditionalMoEIntegration:
    """Integration tests for conditional MoE."""
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        d_model = 64
        model = ConditionalMoELayer(d_model=d_model, max_experts=4, min_experts=1)
        
        batch_sizes = [1, 2, 4, 8, 16]
        seq_len = 16
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, seq_len, d_model)
            
            with torch.no_grad():
                output, stats = model(x)
            
            assert output.shape == (batch_size, seq_len, d_model)
            assert stats['avg_num_experts'] >= 1
            assert stats['avg_num_experts'] <= 4
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        d_model = 64
        model = ConditionalMoELayer(d_model=d_model, max_experts=4, min_experts=1)
        
        batch_size = 4
        seq_lengths = [8, 16, 32, 64, 128]
        
        for seq_len in seq_lengths:
            x = torch.randn(batch_size, seq_len, d_model)
            
            with torch.no_grad():
                output, stats = model(x)
            
            assert output.shape == (batch_size, seq_len, d_model)
    
    def test_mixed_precision_compatibility(self):
        """Test compatibility with mixed precision training."""
        d_model = 64
        model = ConditionalMoELayer(d_model=d_model, max_experts=4, min_experts=1)
        
        if torch.cuda.is_available():
            model = model.cuda()
            x = torch.randn(2, 8, d_model).cuda()
            
            # Test with autocast
            with torch.cuda.amp.autocast():
                output, stats = model(x)
            
            assert output.shape == (2, 8, d_model)
        else:
            # CPU test
            x = torch.randn(2, 8, d_model)
            output, stats = model(x)
            assert output.shape == (2, 8, d_model)
    
    def test_deterministic_behavior(self):
        """Test deterministic behavior with fixed seed."""
        torch.manual_seed(42)
        
        d_model = 64
        # Disable dropout for deterministic behavior
        model1 = ConditionalMoELayer(d_model=d_model, max_experts=4, min_experts=1, dropout_p=0.0)
        
        torch.manual_seed(42)
        model2 = ConditionalMoELayer(d_model=d_model, max_experts=4, min_experts=1, dropout_p=0.0)
        
        # Set to eval mode to disable any remaining stochasticity
        model1.eval()
        model2.eval()
        
        # Same input
        torch.manual_seed(42)
        x = torch.randn(2, 8, d_model)
        
        with torch.no_grad():
            output1, stats1 = model1(x)
            output2, stats2 = model2(x)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)
        assert stats1['avg_num_experts'] == stats2['avg_num_experts']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
