"""
Tests for Adaptive Computation Time (ACT) implementation.
"""

import torch
import pytest
from src.models.adaptive_computation import (
    AdaptiveResNetBKBlock,
    ACTLanguageModel,
    ACTTrainer
)


class TestAdaptiveResNetBKBlock:
    """Test AdaptiveResNetBKBlock functionality."""
    
    def test_block_initialization(self):
        """Test block initializes correctly."""
        d_model = 64
        n_seq = 128
        block = AdaptiveResNetBKBlock(d_model, n_seq, threshold=0.99)
        
        assert block.d_model == d_model
        assert block.n_seq == n_seq
        assert block.threshold == 0.99
        assert hasattr(block, 'halting_unit')
        assert hasattr(block, 'bk_layer')
    
    def test_forward_pass_first_layer(self):
        """Test forward pass on first layer (no previous state)."""
        d_model = 64
        n_seq = 128
        batch_size = 2
        
        block = AdaptiveResNetBKBlock(d_model, n_seq)
        x = torch.randn(batch_size, n_seq, d_model)
        
        output, halting_prob, still_running, weight = block(x)
        
        # Check output shapes
        assert output.shape == (batch_size, n_seq, d_model)
        assert halting_prob.shape == (batch_size, n_seq)
        assert still_running.shape == (batch_size, n_seq)
        assert weight.shape == (batch_size, n_seq)
        
        # Check halting probability is in valid range
        assert (halting_prob >= 0).all()
        assert (halting_prob <= 1.0).all()
        
        # Check weight is in valid range
        assert (weight >= 0).all()
        assert (weight <= 1.0).all()
    
    def test_forward_pass_with_state(self):
        """Test forward pass with previous state."""
        d_model = 64
        n_seq = 128
        batch_size = 2
        
        block = AdaptiveResNetBKBlock(d_model, n_seq, threshold=0.5)
        x = torch.randn(batch_size, n_seq, d_model)
        
        # Initial state
        halting_prob_init = torch.zeros(batch_size, n_seq)
        still_running_init = torch.ones(batch_size, n_seq, dtype=torch.bool)
        
        output, halting_prob, still_running, weight = block(
            x, halting_prob_init, still_running_init
        )
        
        # Some tokens should have halted (threshold is low)
        assert not still_running.all()
        
        # Halting probability should have increased
        assert (halting_prob >= halting_prob_init).all()
    
    def test_ponder_cost_tracking(self):
        """Test ponder cost is tracked correctly."""
        d_model = 64
        n_seq = 128
        batch_size = 2
        
        block = AdaptiveResNetBKBlock(d_model, n_seq)
        x = torch.randn(batch_size, n_seq, d_model)
        
        # Reset ponder cost
        block.reset_ponder_cost()
        assert block.ponder_cost == 0.0
        
        # Forward pass
        output, halting_prob, still_running, weight = block(x)
        
        # Ponder cost should be positive
        assert block.ponder_cost > 0
        
        # Ponder cost should equal sum of weights
        assert torch.isclose(block.ponder_cost, weight.sum(), atol=1e-5)
    
    def test_halting_threshold(self):
        """Test tokens halt when reaching threshold."""
        d_model = 64
        n_seq = 128
        batch_size = 2
        threshold = 0.5
        
        block = AdaptiveResNetBKBlock(d_model, n_seq, threshold=threshold)
        x = torch.randn(batch_size, n_seq, d_model)
        
        # Manually set high initial halting probability
        halting_prob_init = torch.full((batch_size, n_seq), 0.4)
        still_running_init = torch.ones(batch_size, n_seq, dtype=torch.bool)
        
        output, halting_prob, still_running, weight = block(
            x, halting_prob_init, still_running_init
        )
        
        # Tokens with cumulative prob >= threshold should have halted
        halted_tokens = halting_prob >= threshold
        assert (still_running[halted_tokens] == False).all()


class TestACTLanguageModel:
    """Test ACTLanguageModel functionality."""
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        vocab_size = 1000
        d_model = 64
        n_layers = 4
        n_seq = 128
        
        model = ACTLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=n_seq,
            act_lambda=0.01
        )
        
        assert model.d_model == d_model
        assert model.n_seq == n_seq
        assert model.n_layers == n_layers
        assert model.act_lambda == 0.01
        assert len(model.blocks) == n_layers
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shapes."""
        vocab_size = 1000
        d_model = 64
        n_layers = 4
        n_seq = 128
        batch_size = 2
        
        model = ACTLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=n_seq
        )
        
        x = torch.randint(0, vocab_size, (batch_size, n_seq))
        logits, ponder_cost = model(x, return_ponder_cost=True)
        
        # Check output shapes
        assert logits.shape == (batch_size, n_seq, vocab_size)
        assert isinstance(ponder_cost, torch.Tensor)
        assert ponder_cost.ndim == 0  # Scalar
        
        # Check ponder cost is positive
        assert ponder_cost > 0
    
    def test_forward_without_ponder_cost(self):
        """Test forward pass without returning ponder cost."""
        vocab_size = 1000
        d_model = 64
        n_seq = 128
        batch_size = 2
        
        model = ACTLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=4,
            n_seq=n_seq
        )
        
        x = torch.randint(0, vocab_size, (batch_size, n_seq))
        logits = model(x, return_ponder_cost=False)
        
        assert logits.shape == (batch_size, n_seq, vocab_size)
    
    def test_compute_loss(self):
        """Test loss computation with ponder cost."""
        vocab_size = 1000
        d_model = 64
        n_seq = 128
        batch_size = 2
        
        model = ACTLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=4,
            n_seq=n_seq,
            act_lambda=0.01
        )
        
        x = torch.randint(0, vocab_size, (batch_size, n_seq))
        targets = torch.randint(0, vocab_size, (batch_size * n_seq,))
        
        logits, ponder_cost = model(x, return_ponder_cost=True)
        total_loss, ce_loss, ponder_cost_val = model.compute_loss(
            logits, targets, ponder_cost
        )
        
        # Check loss components
        assert total_loss > 0
        assert ce_loss > 0
        assert ponder_cost_val > 0
        
        # Check total loss includes ponder cost
        expected_total = ce_loss + model.act_lambda * ponder_cost_val
        assert torch.isclose(total_loss, expected_total, atol=1e-5)
    
    def test_avg_layers_executed(self):
        """Test average layers executed tracking."""
        vocab_size = 1000
        d_model = 64
        n_seq = 128
        batch_size = 2
        n_layers = 4
        
        model = ACTLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=n_seq,
            act_threshold=0.99  # High threshold, most tokens use all layers
        )
        
        x = torch.randint(0, vocab_size, (batch_size, n_seq))
        logits, ponder_cost = model(x, return_ponder_cost=True)
        
        avg_layers = model.get_avg_layers_executed()
        
        # Should be between 0 and n_layers
        assert 0 < avg_layers <= n_layers
    
    def test_early_exit(self):
        """Test early exit when all tokens halt."""
        vocab_size = 1000
        d_model = 64
        n_seq = 128
        batch_size = 2
        
        model = ACTLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=4,
            n_seq=n_seq,
            act_threshold=0.1  # Very low threshold, tokens halt quickly
        )
        
        x = torch.randint(0, vocab_size, (batch_size, n_seq))
        logits, ponder_cost = model(x, return_ponder_cost=True)
        
        avg_layers = model.get_avg_layers_executed()
        
        # With low threshold, should use fewer layers
        assert avg_layers < 4.0


class TestACTTrainer:
    """Test ACTTrainer functionality."""
    
    def test_trainer_initialization(self):
        """Test trainer initializes correctly."""
        vocab_size = 1000
        model = ACTLanguageModel(vocab_size=vocab_size, d_model=64, n_layers=4, n_seq=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        trainer = ACTTrainer(model, optimizer)
        
        assert trainer.model is model
        assert trainer.optimizer is optimizer
        assert trainer.num_batches == 0
    
    def test_train_step(self):
        """Test training step executes correctly."""
        vocab_size = 1000
        d_model = 64
        n_seq = 128
        batch_size = 2
        
        model = ACTLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=4,
            n_seq=n_seq
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = ACTTrainer(model, optimizer)
        
        x_batch = torch.randint(0, vocab_size, (batch_size, n_seq))
        y_batch = torch.randint(0, vocab_size, (batch_size * n_seq,))
        
        metrics = trainer.train_step(x_batch, y_batch)
        
        # Check metrics returned
        assert 'total_loss' in metrics
        assert 'ce_loss' in metrics
        assert 'ponder_cost' in metrics
        assert 'avg_layers_executed' in metrics
        
        # Check values are reasonable
        assert metrics['total_loss'] > 0
        assert metrics['ce_loss'] > 0
        assert metrics['ponder_cost'] > 0
        assert 0 < metrics['avg_layers_executed'] <= 4
    
    def test_statistics_tracking(self):
        """Test statistics are tracked correctly."""
        vocab_size = 1000
        d_model = 64
        n_seq = 128
        batch_size = 2
        
        model = ACTLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=4,
            n_seq=n_seq
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = ACTTrainer(model, optimizer)
        
        # Run multiple training steps
        for _ in range(3):
            x_batch = torch.randint(0, vocab_size, (batch_size, n_seq))
            y_batch = torch.randint(0, vocab_size, (batch_size * n_seq,))
            trainer.train_step(x_batch, y_batch)
        
        # Check statistics
        assert trainer.num_batches == 3
        avg_metrics = trainer.get_average_metrics()
        assert avg_metrics['avg_ponder_cost'] > 0
        assert avg_metrics['avg_ce_loss'] > 0
        
        # Reset and check
        trainer.reset_statistics()
        assert trainer.num_batches == 0
        assert trainer.total_ponder_cost == 0.0
        assert trainer.total_ce_loss == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
