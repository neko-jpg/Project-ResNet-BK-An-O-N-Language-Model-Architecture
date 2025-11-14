"""
Tests for Koopman Operator Learning
"""

import torch
import torch.nn as nn
import pytest

from src.models.koopman_layer import (
    KoopmanResNetBKLayer,
    KoopmanResNetBKBlock,
    KoopmanLanguageModel
)
from src.training.hybrid_koopman_trainer import HybridKoopmanTrainer
from src.training.koopman_scheduler import KoopmanLossScheduler


class TestKoopmanLayer:
    """Test Koopman ResNet-BK layer."""
    
    def test_koopman_layer_initialization(self):
        """Test that Koopman layer initializes correctly."""
        d_model = 64
        n_seq = 128
        koopman_dim = 256
        
        layer = KoopmanResNetBKLayer(
            d_model=d_model,
            n_seq=n_seq,
            koopman_dim=koopman_dim,
            num_experts=4,
            top_k=1,
            dropout_p=0.1
        )
        
        # Check Koopman operator initialization (should be near identity)
        K = layer.K.data
        assert K.shape == (koopman_dim, koopman_dim)
        
        # Check that K is close to identity
        identity = torch.eye(koopman_dim)
        diff = (K - identity).abs().mean().item()
        assert diff < 0.1, f"K should be close to identity, but diff={diff}"
    
    def test_koopman_forward_standard(self):
        """Test standard forward pass."""
        batch_size = 4
        n_seq = 128
        d_model = 64
        koopman_dim = 256
        
        layer = KoopmanResNetBKLayer(
            d_model=d_model,
            n_seq=n_seq,
            koopman_dim=koopman_dim,
            num_experts=4,
            top_k=1,
            dropout_p=0.1
        )
        
        x = torch.randn(batch_size, n_seq, d_model)
        output = layer(x, use_koopman=False)
        
        assert output.shape == (batch_size, n_seq, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_koopman_forward_koopman_mode(self):
        """Test Koopman prediction mode."""
        batch_size = 4
        n_seq = 128
        d_model = 64
        koopman_dim = 256
        
        layer = KoopmanResNetBKLayer(
            d_model=d_model,
            n_seq=n_seq,
            koopman_dim=koopman_dim,
            num_experts=4,
            top_k=1,
            dropout_p=0.1
        )
        
        x = torch.randn(batch_size, n_seq, d_model)
        output = layer(x, use_koopman=True)
        
        assert output.shape == (batch_size, n_seq, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_koopman_loss(self):
        """Test Koopman auxiliary loss computation."""
        batch_size = 4
        n_seq = 128
        d_model = 64
        koopman_dim = 256
        
        layer = KoopmanResNetBKLayer(
            d_model=d_model,
            n_seq=n_seq,
            koopman_dim=koopman_dim,
            num_experts=4,
            top_k=1,
            dropout_p=0.1
        )
        
        x_current = torch.randn(batch_size, n_seq, d_model)
        x_next = torch.randn(batch_size, n_seq, d_model)
        
        loss = layer.koopman_loss(x_current, x_next)
        
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # MSE loss is non-negative
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_koopman_operator_update(self):
        """Test Koopman operator update via DMD."""
        batch_size = 4
        n_seq = 128
        d_model = 64
        koopman_dim = 256
        
        layer = KoopmanResNetBKLayer(
            d_model=d_model,
            n_seq=n_seq,
            koopman_dim=koopman_dim,
            num_experts=4,
            top_k=1,
            dropout_p=0.1
        )
        
        # Store initial K
        K_initial = layer.K.data.clone()
        
        # Perform multiple updates to fill buffer
        for _ in range(5):
            x_current = torch.randn(batch_size, n_seq, d_model)
            x_next = torch.randn(batch_size, n_seq, d_model)
            layer.update_koopman_operator(x_current, x_next)
        
        # Check that K has changed
        K_final = layer.K.data
        diff = (K_final - K_initial).abs().mean().item()
        
        # After buffer is filled, K should have changed
        assert diff > 0, "Koopman operator should have been updated"


class TestKoopmanLanguageModel:
    """Test Koopman language model."""
    
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        vocab_size = 1000
        d_model = 64
        n_layers = 4
        n_seq = 128
        koopman_dim = 256
        
        model = KoopmanLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=n_seq,
            koopman_dim=koopman_dim,
            num_experts=4,
            top_k=1,
            dropout_p=0.1
        )
        
        # Check model structure
        assert len(model.blocks) == n_layers
        assert model.d_model == d_model
        assert model.n_seq == n_seq
    
    def test_model_forward(self):
        """Test model forward pass."""
        vocab_size = 1000
        batch_size = 4
        n_seq = 128
        
        model = KoopmanLanguageModel(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=4,
            n_seq=n_seq,
            koopman_dim=256,
            num_experts=4,
            top_k=1,
            dropout_p=0.1
        )
        
        x = torch.randint(0, vocab_size, (batch_size, n_seq))
        logits = model(x, use_koopman=False)
        
        assert logits.shape == (batch_size, n_seq, vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_model_koopman_forward(self):
        """Test model forward with Koopman prediction."""
        vocab_size = 1000
        batch_size = 4
        n_seq = 128
        
        model = KoopmanLanguageModel(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=4,
            n_seq=n_seq,
            koopman_dim=256,
            num_experts=4,
            top_k=1,
            dropout_p=0.1
        )
        
        x = torch.randint(0, vocab_size, (batch_size, n_seq))
        logits = model(x, use_koopman=True)
        
        assert logits.shape == (batch_size, n_seq, vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


class TestKoopmanScheduler:
    """Test Koopman loss weight scheduler."""
    
    def test_scheduler_linear(self):
        """Test linear schedule."""
        scheduler = KoopmanLossScheduler(
            min_weight=0.0,
            max_weight=0.5,
            warmup_epochs=2,
            total_epochs=10,
            schedule_type='linear'
        )
        
        # During warmup
        scheduler.step(epoch=0)
        assert scheduler.get_weight() == 0.0
        
        scheduler.step(epoch=1)
        assert scheduler.get_weight() == 0.0
        
        # After warmup
        scheduler.step(epoch=2)
        assert scheduler.get_weight() == 0.0
        
        scheduler.step(epoch=6)  # Midpoint
        weight = scheduler.get_weight()
        assert 0.2 < weight < 0.3  # Should be around 0.25
        
        scheduler.step(epoch=10)  # End
        assert abs(scheduler.get_weight() - 0.5) < 0.01
    
    def test_scheduler_exponential(self):
        """Test exponential schedule."""
        scheduler = KoopmanLossScheduler(
            min_weight=0.01,
            max_weight=0.5,
            warmup_epochs=2,
            total_epochs=10,
            schedule_type='exponential'
        )
        
        # During warmup
        scheduler.step(epoch=0)
        assert scheduler.get_weight() == 0.01
        
        # After warmup
        scheduler.step(epoch=2)
        assert scheduler.get_weight() > 0.01
        
        scheduler.step(epoch=10)
        assert abs(scheduler.get_weight() - 0.5) < 0.01
    
    def test_scheduler_step(self):
        """Test step-wise schedule."""
        scheduler = KoopmanLossScheduler(
            min_weight=0.0,
            max_weight=0.5,
            warmup_epochs=2,
            total_epochs=10,
            schedule_type='step'
        )
        
        # During warmup
        scheduler.step(epoch=0)
        assert scheduler.get_weight() == 0.0
        
        # After warmup, first step
        scheduler.step(epoch=2)
        assert scheduler.get_weight() == 0.0
        
        # Second step
        scheduler.step(epoch=4)
        weight = scheduler.get_weight()
        assert 0.15 < weight < 0.2
        
        # Final step
        scheduler.step(epoch=8)
        assert scheduler.get_weight() == 0.5


class TestHybridKoopmanTrainer:
    """Test hybrid Koopman trainer."""
    
    def test_trainer_initialization(self):
        """Test that trainer initializes correctly."""
        vocab_size = 1000
        model = KoopmanLanguageModel(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=2,
            n_seq=128,
            koopman_dim=128,
            num_experts=4,
            top_k=1,
            dropout_p=0.1
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        trainer = HybridKoopmanTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            koopman_weight_min=0.0,
            koopman_weight_max=0.5,
            koopman_start_epoch=2,
            total_epochs=5,
            device='cpu'
        )
        
        assert trainer.current_epoch == 0
        assert not trainer.koopman_enabled
        assert not trainer.koopman_failed
    
    def test_trainer_step(self):
        """Test single training step."""
        vocab_size = 1000
        batch_size = 4
        n_seq = 128
        
        model = KoopmanLanguageModel(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=2,
            n_seq=n_seq,
            koopman_dim=128,
            num_experts=4,
            top_k=1,
            dropout_p=0.1
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        trainer = HybridKoopmanTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            koopman_weight_min=0.0,
            koopman_weight_max=0.5,
            koopman_start_epoch=2,
            total_epochs=5,
            device='cpu'
        )
        
        # Create dummy batch
        x_batch = torch.randint(0, vocab_size, (batch_size, n_seq))
        y_batch = torch.randint(0, vocab_size, (batch_size * n_seq,))
        
        # Training step
        metrics = trainer.train_step(x_batch, y_batch)
        
        # Check metrics
        assert 'loss_lm' in metrics
        assert 'loss_koopman' in metrics
        assert 'total_loss' in metrics
        assert 'koopman_weight' in metrics
        
        assert metrics['loss_lm'] > 0
        assert metrics['total_loss'] > 0
        assert not trainer.koopman_enabled  # Should be disabled initially


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
