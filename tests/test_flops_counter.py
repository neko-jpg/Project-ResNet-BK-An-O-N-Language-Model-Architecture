"""
Tests for FLOPs Counter Infrastructure
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.benchmarks.flops_counter import FLOPsCounter, FLOPsCount, compare_models
from src.models.configurable_resnet_bk import ConfigurableResNetBK, BASELINE_CONFIG


class TestFLOPsCount:
    """Test FLOPsCount dataclass."""
    
    def test_initialization(self):
        """Test FLOPsCount initialization."""
        flops = FLOPsCount(forward=1000, backward=2000, optimizer=500)
        assert flops.forward == 1000
        assert flops.backward == 2000
        assert flops.optimizer == 500
        assert flops.total == 3500
    
    def test_addition(self):
        """Test FLOPsCount addition."""
        flops1 = FLOPsCount(forward=1000, backward=2000, optimizer=500)
        flops2 = FLOPsCount(forward=500, backward=1000, optimizer=250)
        flops_sum = flops1 + flops2
        
        assert flops_sum.forward == 1500
        assert flops_sum.backward == 3000
        assert flops_sum.optimizer == 750
        assert flops_sum.total == 5250
    
    def test_multiplication(self):
        """Test FLOPsCount multiplication."""
        flops = FLOPsCount(forward=1000, backward=2000, optimizer=500)
        flops_scaled = flops * 2
        
        assert flops_scaled.forward == 2000
        assert flops_scaled.backward == 4000
        assert flops_scaled.optimizer == 1000
        assert flops_scaled.total == 7000
    
    def test_to_dict(self):
        """Test FLOPsCount to_dict conversion."""
        flops = FLOPsCount(forward=1000, backward=2000, optimizer=500)
        flops_dict = flops.to_dict()
        
        assert flops_dict['forward'] == 1000
        assert flops_dict['backward'] == 2000
        assert flops_dict['optimizer'] == 500
        assert flops_dict['total'] == 3500


class TestFLOPsCounter:
    """Test FLOPsCounter class."""
    
    @pytest.fixture
    def model(self):
        """Create a small test model."""
        config = BASELINE_CONFIG
        config.d_model = 64
        config.n_layers = 2
        config.n_seq = 128
        config.num_experts = 4
        config.vocab_size = 1000
        return ConfigurableResNetBK(config)
    
    @pytest.fixture
    def counter(self, model):
        """Create FLOPs counter."""
        return FLOPsCounter(model, batch_size=8, seq_len=128)
    
    def test_initialization(self, counter):
        """Test FLOPs counter initialization."""
        assert counter.batch_size == 8
        assert counter.seq_len == 128
        assert counter.d_model == 64
        assert counter.n_layers == 2
        assert counter.num_experts == 4
    
    def test_count_bk_core_flops(self, counter):
        """Test BK-Core FLOPs counting."""
        flops = counter.count_bk_core_flops()
        
        # Should have forward and backward FLOPs
        assert flops.forward > 0
        assert flops.backward > 0
        
        # Backward should be comparable to forward (analytic gradient)
        assert flops.backward > 0.1 * flops.forward
        assert flops.backward < 10 * flops.forward
    
    def test_count_moe_flops(self, counter):
        """Test MoE FLOPs counting."""
        flops = counter.count_moe_flops()
        
        # Should have forward and backward FLOPs
        assert flops.forward > 0
        assert flops.backward > 0
        
        # Backward should be approximately 2× forward
        assert flops.backward > flops.forward
        assert flops.backward < 3 * flops.forward
    
    def test_count_linear_flops(self, counter):
        """Test linear layer FLOPs counting."""
        flops = counter.count_linear_flops(64, 128)
        
        # Forward: B*N*in*out*2
        expected_forward = 8 * 128 * 64 * 128 * 2
        assert flops.forward == expected_forward
        
        # Backward: 2× forward
        assert flops.backward == 2 * expected_forward
    
    def test_count_embedding_flops(self, counter):
        """Test embedding FLOPs counting."""
        flops = counter.count_embedding_flops()
        
        # Should have forward and backward FLOPs
        assert flops.forward > 0
        assert flops.backward > 0
    
    def test_count_layernorm_flops(self, counter):
        """Test LayerNorm FLOPs counting."""
        flops = counter.count_layernorm_flops()
        
        # Should have forward and backward FLOPs
        assert flops.forward > 0
        assert flops.backward > 0
    
    def test_count_forward_flops(self, counter):
        """Test total forward FLOPs counting."""
        flops = counter.count_forward_flops()
        
        # Should have only forward FLOPs
        assert flops.forward > 0
        assert flops.backward == 0
        assert flops.optimizer == 0
        
        # Should have component breakdown
        assert len(counter.component_flops) > 0
        assert 'embedding' in counter.component_flops
        assert 'lm_head' in counter.component_flops
    
    def test_count_backward_flops(self, counter):
        """Test total backward FLOPs counting."""
        flops = counter.count_backward_flops()
        
        # Should have only backward FLOPs
        assert flops.forward == 0
        assert flops.backward > 0
        assert flops.optimizer == 0
    
    def test_count_optimizer_flops(self, counter):
        """Test optimizer FLOPs counting."""
        # Test AdamW
        flops_adamw = counter.count_optimizer_flops('adamw')
        assert flops_adamw.optimizer > 0
        
        # Test SGD
        flops_sgd = counter.count_optimizer_flops('sgd')
        assert flops_sgd.optimizer > 0
        
        # AdamW should have more FLOPs than SGD
        assert flops_adamw.optimizer > flops_sgd.optimizer
    
    def test_count_total_flops(self, counter):
        """Test total FLOPs counting."""
        flops = counter.count_total_flops()
        
        # Should have all three components
        assert flops.forward > 0
        assert flops.backward > 0
        assert flops.optimizer > 0
        
        # Total should be sum
        assert flops.total == flops.forward + flops.backward + flops.optimizer
    
    def test_get_breakdown(self, counter):
        """Test FLOPs breakdown."""
        counter.count_forward_flops()
        breakdown = counter.get_breakdown()
        
        # Should have component breakdown
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0
        
        # Each component should have FLOPs dict
        for component, flops_dict in breakdown.items():
            assert 'forward' in flops_dict
            assert 'backward' in flops_dict
            assert 'optimizer' in flops_dict
            assert 'total' in flops_dict
    
    def test_print_summary(self, counter, capsys):
        """Test FLOPs summary printing."""
        counter.print_summary()
        
        captured = capsys.readouterr()
        assert "FLOPs Counter Summary" in captured.out
        assert "Forward Pass:" in captured.out
        assert "Backward Pass:" in captured.out
        assert "Optimizer Step:" in captured.out
        assert "Component Breakdown" in captured.out
    
    def test_save_to_json(self, counter, tmp_path):
        """Test saving FLOPs to JSON."""
        output_file = tmp_path / "flops_test.json"
        counter.save_to_json(str(output_file))
        
        # Check file exists
        assert output_file.exists()
        
        # Check file content
        import json
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert 'model_config' in data
        assert 'total_flops' in data
        assert 'component_breakdown' in data
        assert data['model_config']['d_model'] == 64
        assert data['model_config']['n_layers'] == 2


class TestCompareModels:
    """Test model comparison functionality."""
    
    def test_compare_models(self, capsys):
        """Test comparing two models."""
        # Create two models with different sizes
        config1 = BASELINE_CONFIG
        config1.d_model = 64
        config1.n_layers = 2
        config1.n_seq = 128
        model1 = ConfigurableResNetBK(config1)
        
        config2 = BASELINE_CONFIG
        config2.d_model = 128
        config2.n_layers = 4
        config2.n_seq = 128
        model2 = ConfigurableResNetBK(config2)
        
        # Compare models
        comparison = compare_models(
            model1, model2,
            batch_size=8, seq_len=128,
            model1_name="Small Model",
            model2_name="Large Model"
        )
        
        # Check comparison results
        assert 'Small Model' in comparison
        assert 'Large Model' in comparison
        assert 'speedup' in comparison
        
        # Large model should have more FLOPs
        assert comparison['Large Model']['total'] > comparison['Small Model']['total']
        
        # Speedup should be < 1 (model1 is faster)
        assert comparison['speedup']['total'] > 1.0
        
        # Check printed output
        captured = capsys.readouterr()
        assert "Model Comparison" in captured.out
        assert "Speedup:" in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
