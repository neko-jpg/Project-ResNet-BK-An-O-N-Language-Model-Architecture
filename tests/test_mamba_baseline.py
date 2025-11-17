"""
Tests for Mamba Baseline Implementation

Verifies:
- Model forward/backward pass
- FLOPs counting accuracy
- Memory estimation
- Fair comparison framework
- Identical hyperparameters

Requirements: 11.1, 11.2, 11.3, 11.4
"""

import pytest
import torch
import torch.nn as nn

from src.models.mamba_baseline import MambaLM, MambaConfig, MambaBlock, create_mamba_from_resnetbk_config
from src.benchmarks.mamba_flops_counter import MambaFLOPsCounter
from src.benchmarks.fair_comparison import FairComparison, ComparisonConfig, set_seed


class TestMambaBaseline:
    """Test suite for Mamba baseline model."""
    
    def test_mamba_config(self):
        """Test Mamba configuration."""
        config = MambaConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            d_state=16,
            max_seq_len=256
        )
        
        assert config.vocab_size == 1000
        assert config.d_model == 128
        assert config.n_layers == 4
        assert config.d_state == 16
        assert config.max_seq_len == 256
    
    def test_mamba_block_forward(self):
        """Test Mamba block forward pass."""
        config = MambaConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            max_seq_len=256
        )
        
        block = MambaBlock(config, layer_idx=0)
        
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, config.d_model)
        
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, config.d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_mamba_model_forward(self):
        """Test Mamba model forward pass."""
        config = MambaConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            max_seq_len=256
        )
        
        model = MambaLM(config)
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        logits, _ = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_mamba_model_with_loss(self):
        """Test Mamba model with loss computation."""
        config = MambaConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            max_seq_len=256
        )
        
        model = MambaLM(config)
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        logits, loss = model(input_ids, targets)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is not None
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_mamba_backward(self):
        """Test Mamba model backward pass."""
        config = MambaConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            max_seq_len=256
        )
        
        model = MambaLM(config)
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        logits, loss = model(input_ids, targets)
        loss.backward()
        
        # Check that gradients are computed
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_create_mamba_from_resnetbk_config(self):
        """Test creating Mamba config from ResNet-BK config."""
        # Mock ResNet-BK config
        class MockResNetBKConfig:
            vocab_size = 30000
            d_model = 256
            n_layers = 8
            n_seq = 2048
            dropout = 0.1
            tie_weights = True
        
        resnetbk_config = MockResNetBKConfig()
        mamba_config = create_mamba_from_resnetbk_config(resnetbk_config)
        
        # Verify identical hyperparameters
        assert mamba_config.vocab_size == resnetbk_config.vocab_size
        assert mamba_config.d_model == resnetbk_config.d_model
        assert mamba_config.n_layers == resnetbk_config.n_layers
        assert mamba_config.max_seq_len == resnetbk_config.n_seq
        assert mamba_config.dropout == resnetbk_config.dropout
        assert mamba_config.tie_weights == resnetbk_config.tie_weights


class TestMambaFLOPsCounter:
    """Test suite for Mamba FLOPs counter."""
    
    def test_flops_counter_initialization(self):
        """Test FLOPs counter initialization."""
        config = MambaConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            max_seq_len=256
        )
        
        model = MambaLM(config)
        counter = MambaFLOPsCounter(model, batch_size=32, seq_len=128)
        
        assert counter.batch_size == 32
        assert counter.seq_len == 128
        assert counter.d_model == 128
        assert counter.n_layers == 4
    
    def test_count_forward_flops(self):
        """Test forward FLOPs counting."""
        config = MambaConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            max_seq_len=256
        )
        
        model = MambaLM(config)
        counter = MambaFLOPsCounter(model, batch_size=32, seq_len=128)
        
        flops = counter.count_forward_flops()
        
        assert flops.forward > 0
        assert flops.backward == 0  # Only forward counted
        assert flops.optimizer == 0
    
    def test_count_total_flops(self):
        """Test total FLOPs counting."""
        config = MambaConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            max_seq_len=256
        )
        
        model = MambaLM(config)
        counter = MambaFLOPsCounter(model, batch_size=32, seq_len=128)
        
        flops = counter.count_total_flops('adamw')
        
        assert flops.forward > 0
        assert flops.backward > 0
        assert flops.optimizer > 0
        assert flops.total == flops.forward + flops.backward + flops.optimizer
    
    def test_count_memory_usage(self):
        """Test memory usage counting."""
        config = MambaConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            max_seq_len=256
        )
        
        model = MambaLM(config)
        counter = MambaFLOPsCounter(model, batch_size=32, seq_len=128)
        
        memory = counter.count_memory_usage('adamw', torch.float32)
        
        assert memory.parameters > 0
        assert memory.activations > 0
        assert memory.gradients > 0
        assert memory.optimizer_states > 0
        assert memory.total > 0


class TestFairComparison:
    """Test suite for fair comparison framework."""
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        
        x1 = torch.randn(10)
        
        set_seed(42)
        x2 = torch.randn(10)
        
        assert torch.allclose(x1, x2), "Random seed not working"
    
    def test_comparison_config(self):
        """Test comparison configuration."""
        config = ComparisonConfig(
            learning_rate=1e-3,
            batch_size=32,
            seq_len=128,
            seed=42
        )
        
        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
        assert config.seq_len == 128
        assert config.seed == 42
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fair_comparison_initialization(self):
        """Test fair comparison initialization."""
        # Create simple models for testing
        from src.models.configurable_resnet_bk import ConfigurableResNetBK, BASELINE_CONFIG
        
        # Create ResNet-BK model
        resnetbk_config = BASELINE_CONFIG
        resnetbk_model = ConfigurableResNetBK(resnetbk_config)
        
        # Create Mamba model
        mamba_config = create_mamba_from_resnetbk_config(resnetbk_config)
        mamba_model = MambaLM(mamba_config)
        
        # Create comparison
        comparison_config = ComparisonConfig(
            batch_size=8,
            seq_len=64,
            seed=42
        )
        
        comparison = FairComparison(resnetbk_model, mamba_model, comparison_config)
        
        assert comparison.resnetbk_model is not None
        assert comparison.mamba_model is not None
        assert comparison.resnetbk_flops.total > 0
        assert comparison.mamba_flops.total > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
