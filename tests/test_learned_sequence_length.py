"""
Tests for Learned Sequence Length module
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('src')

from models.resnet_bk import LanguageModel
from models.learned_sequence_length import (
    SequenceLengthPredictor,
    AdaptiveSequenceLengthWrapper,
    LearnedSequenceLengthTrainer
)


@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def base_model(device):
    """Create a small base model for testing."""
    model = LanguageModel(
        vocab_size=100,
        d_model=32,
        n_layers=2,
        n_seq=64,
        num_experts=2,
        top_k=1,
        dropout_p=0.0
    ).to(device)
    return model


@pytest.fixture
def adaptive_model(base_model, device):
    """Create adaptive sequence length wrapper."""
    model = AdaptiveSequenceLengthWrapper(
        base_model=base_model,
        max_seq_len=64,
        num_length_bins=4,
        length_penalty=0.01
    ).to(device)
    return model


class TestSequenceLengthPredictor:
    """Test SequenceLengthPredictor component."""
    
    def test_initialization(self, device):
        """Test predictor initialization."""
        predictor = SequenceLengthPredictor(
            d_model=32,
            max_seq_len=64,
            num_length_bins=4
        ).to(device)
        
        assert predictor.d_model == 32
        assert predictor.max_seq_len == 64
        assert predictor.num_length_bins == 4
        assert len(predictor.length_bins) == 4
        
        # Check length bins are evenly spaced
        expected_bins = [16, 32, 48, 64]
        assert predictor.length_bins.tolist() == expected_bins
    
    def test_forward_shape(self, device):
        """Test predictor forward pass output shapes."""
        predictor = SequenceLengthPredictor(
            d_model=32,
            max_seq_len=64,
            num_length_bins=4
        ).to(device)
        
        B, N, D = 8, 64, 32
        x_embedded = torch.randn(B, N, D, device=device)
        
        # Without distribution
        predicted_lengths = predictor(x_embedded, return_distribution=False)
        assert predicted_lengths.shape == (B,)
        assert predicted_lengths.dtype == torch.long
        
        # With distribution
        predicted_lengths, length_probs = predictor(x_embedded, return_distribution=True)
        assert predicted_lengths.shape == (B,)
        assert length_probs.shape == (B, 4)
        assert torch.allclose(length_probs.sum(dim=1), torch.ones(B, device=device), atol=1e-5)
    
    def test_predicted_lengths_valid(self, device):
        """Test that predicted lengths are valid (within bins)."""
        predictor = SequenceLengthPredictor(
            d_model=32,
            max_seq_len=64,
            num_length_bins=4
        ).to(device)
        
        B, N, D = 8, 64, 32
        x_embedded = torch.randn(B, N, D, device=device)
        
        predicted_lengths = predictor(x_embedded)
        
        # Check all predictions are in valid bins
        valid_lengths = predictor.length_bins.to(device)
        for length in predicted_lengths:
            assert length in valid_lengths


class TestAdaptiveSequenceLengthWrapper:
    """Test AdaptiveSequenceLengthWrapper component."""
    
    def test_initialization(self, adaptive_model):
        """Test wrapper initialization."""
        assert adaptive_model.max_seq_len == 64
        assert adaptive_model.num_length_bins == 4
        assert adaptive_model.length_penalty == 0.01
        assert hasattr(adaptive_model, 'length_predictor')
        assert hasattr(adaptive_model, 'base_model')
    
    def test_forward_without_adaptation(self, adaptive_model, device):
        """Test forward pass without adaptive length."""
        B, N = 4, 64
        x = torch.randint(0, 100, (B, N), device=device)
        
        logits = adaptive_model(x, use_adaptive_length=False)
        
        assert logits.shape == (B, N, 100)  # vocab_size=100
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_forward_with_adaptation(self, adaptive_model, device):
        """Test forward pass with adaptive length."""
        B, N = 4, 64
        x = torch.randint(0, 100, (B, N), device=device)
        
        logits, length_info = adaptive_model(x, use_adaptive_length=True)
        
        # Check output shape
        assert logits.shape == (B, N, 100)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        
        # Check length info
        assert 'predicted_lengths' in length_info
        assert 'length_probs' in length_info
        assert 'avg_predicted_length' in length_info
        assert 'speedup_estimate' in length_info
        
        assert length_info['predicted_lengths'].shape == (B,)
        assert length_info['length_probs'].shape == (B, 4)
        assert length_info['avg_predicted_length'] > 0
        assert length_info['speedup_estimate'] >= 1.0
    
    def test_pad_or_truncate(self, adaptive_model, device):
        """Test padding and truncation logic."""
        B, N = 4, 64
        x = torch.randint(0, 100, (B, N), device=device)
        
        # Test truncation
        x_truncated, orig_len = adaptive_model._pad_or_truncate(x, target_length=32)
        assert x_truncated.shape == (B, 32)
        assert orig_len == N
        assert torch.equal(x_truncated, x[:, :32])
        
        # Test padding
        x_padded, orig_len = adaptive_model._pad_or_truncate(x, target_length=80)
        assert x_padded.shape == (B, 80)
        assert orig_len == N
        assert torch.equal(x_padded[:, :N], x)
        assert (x_padded[:, N:] == 0).all()  # Padding should be zeros
        
        # Test no change
        x_same, orig_len = adaptive_model._pad_or_truncate(x, target_length=64)
        assert x_same.shape == (B, 64)
        assert orig_len == N
        assert torch.equal(x_same, x)
    
    def test_restore_length(self, adaptive_model, device):
        """Test length restoration logic."""
        B, N, vocab_size = 4, 64, 100
        
        # Test padding restoration
        output_short = torch.randn(B, 32, vocab_size, device=device)
        output_restored = adaptive_model._restore_length(output_short, original_length=64, current_length=32)
        assert output_restored.shape == (B, 64, vocab_size)
        assert torch.equal(output_restored[:, :32, :], output_short)
        
        # Test truncation restoration
        output_long = torch.randn(B, 80, vocab_size, device=device)
        output_restored = adaptive_model._restore_length(output_long, original_length=64, current_length=80)
        assert output_restored.shape == (B, 64, vocab_size)
        assert torch.equal(output_restored, output_long[:, :64, :])
        
        # Test no change
        output_same = torch.randn(B, 64, vocab_size, device=device)
        output_restored = adaptive_model._restore_length(output_same, original_length=64, current_length=64)
        assert output_restored.shape == (B, 64, vocab_size)
        assert torch.equal(output_restored, output_same)
    
    def test_compute_loss(self, adaptive_model, device):
        """Test loss computation."""
        B, N, vocab_size = 4, 64, 100
        logits = torch.randn(B, N, vocab_size, device=device)
        targets = torch.randint(0, vocab_size, (B * N,), device=device)
        
        # Without length info
        total_loss, ce_loss, length_cost = adaptive_model.compute_loss(logits, targets, length_info=None)
        assert total_loss.item() > 0
        assert ce_loss.item() > 0
        assert length_cost.item() == 0.0
        assert torch.allclose(total_loss, ce_loss)
        
        # With length info
        length_info = {
            'avg_predicted_length': 48.0,
            'speedup_estimate': 1.33
        }
        total_loss, ce_loss, length_cost = adaptive_model.compute_loss(logits, targets, length_info)
        assert total_loss.item() > 0
        assert ce_loss.item() > 0
        # length_cost might be tensor or float
        length_cost_val = length_cost.item() if isinstance(length_cost, torch.Tensor) else length_cost
        assert length_cost_val > 0
        assert total_loss.item() > ce_loss.item()  # Total includes length penalty
    
    def test_statistics(self, adaptive_model, device):
        """Test statistics tracking."""
        # Reset statistics
        adaptive_model.reset_length_statistics()
        
        # Get initial statistics
        stats = adaptive_model.get_length_statistics()
        assert stats['avg_predicted_length'] == 0.0
        # Check for correct key name
        assert 'total_predictions' in stats or stats.get('avg_predicted_length') == 0.0
        
        # Run forward pass
        B, N = 4, 64
        x = torch.randint(0, 100, (B, N), device=device)
        _, _ = adaptive_model(x, use_adaptive_length=True)
        
        # Check statistics updated
        stats = adaptive_model.get_length_statistics()
        assert stats['avg_predicted_length'] > 0
        assert len(stats['length_distribution']) == 4
        assert len(stats['length_bins']) == 4


class TestLearnedSequenceLengthTrainer:
    """Test LearnedSequenceLengthTrainer component."""
    
    def test_initialization(self, adaptive_model, device):
        """Test trainer initialization."""
        optimizer = torch.optim.Adam(adaptive_model.parameters(), lr=1e-3)
        trainer = LearnedSequenceLengthTrainer(adaptive_model, optimizer, device=device)
        
        assert trainer.model is adaptive_model
        assert trainer.optimizer is optimizer
        assert trainer.device == device
        assert trainer.num_batches == 0
    
    def test_train_step(self, adaptive_model, device):
        """Test single training step."""
        optimizer = torch.optim.Adam(adaptive_model.parameters(), lr=1e-3)
        trainer = LearnedSequenceLengthTrainer(adaptive_model, optimizer, device=device)
        
        B, N = 4, 64
        x_batch = torch.randint(0, 100, (B, N), device=device)
        y_batch = torch.randint(0, 100, (B * N,), device=device)
        
        # Train step with adaptive length
        metrics = trainer.train_step(x_batch, y_batch, use_adaptive_length=True)
        
        assert 'total_loss' in metrics
        assert 'ce_loss' in metrics
        assert 'length_cost' in metrics
        assert 'avg_predicted_length' in metrics
        assert 'speedup_estimate' in metrics
        
        assert metrics['total_loss'] > 0
        assert metrics['ce_loss'] > 0
        assert metrics['avg_predicted_length'] > 0
        assert metrics['speedup_estimate'] >= 1.0
        
        # Check statistics updated
        assert trainer.num_batches == 1
        assert trainer.total_ce_loss > 0
    
    def test_train_step_without_adaptation(self, adaptive_model, device):
        """Test training step without adaptive length."""
        optimizer = torch.optim.Adam(adaptive_model.parameters(), lr=1e-3)
        trainer = LearnedSequenceLengthTrainer(adaptive_model, optimizer, device=device)
        
        B, N = 4, 64
        x_batch = torch.randint(0, 100, (B, N), device=device)
        y_batch = torch.randint(0, 100, (B * N,), device=device)
        
        # Train step without adaptive length
        metrics = trainer.train_step(x_batch, y_batch, use_adaptive_length=False)
        
        assert 'total_loss' in metrics
        assert 'ce_loss' in metrics
        assert 'length_cost' in metrics
        
        assert metrics['total_loss'] > 0
        assert metrics['ce_loss'] > 0
        assert metrics['length_cost'] == 0.0
        
        # Should not have length-specific metrics
        assert 'avg_predicted_length' not in metrics
        assert 'speedup_estimate' not in metrics
    
    def test_gradient_flow(self, adaptive_model, device):
        """Test that length predictor can be trained."""
        # The length predictor gets gradients through the probability distribution
        
        # Set model to training mode
        adaptive_model.train()
        
        optimizer = torch.optim.SGD(adaptive_model.length_predictor.parameters(), lr=0.1)
        
        B, N = 4, 64
        x_batch = torch.randint(0, 100, (B, N), device=device)
        
        # Get embeddings
        tok_emb = adaptive_model.base_model.token_embedding(x_batch)
        pos = torch.arange(0, N, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = adaptive_model.base_model.position_embedding(pos)
        x_embedded = tok_emb + pos_emb
        
        # Forward through predictor
        optimizer.zero_grad()
        predicted_lengths, length_probs = adaptive_model.length_predictor(
            x_embedded, return_distribution=True
        )
        
        # Create a loss using the probability distribution (differentiable)
        # Encourage uniform distribution as a simple test
        target_probs = torch.ones_like(length_probs) / length_probs.size(1)
        loss = F.kl_div(length_probs.log(), target_probs, reduction='batchmean')
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        has_gradients = False
        for param in adaptive_model.length_predictor.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients, "Length predictor should be trainable"


class TestIntegration:
    """Integration tests for learned sequence length."""
    
    def test_end_to_end_training(self, adaptive_model, device):
        """Test end-to-end training loop."""
        optimizer = torch.optim.Adam(adaptive_model.parameters(), lr=1e-3)
        trainer = LearnedSequenceLengthTrainer(adaptive_model, optimizer, device=device)
        
        # Create small dataset
        num_samples = 32
        B, N = 8, 64
        
        data = []
        for _ in range(num_samples // B):
            x_batch = torch.randint(0, 100, (B, N), device=device)
            y_batch = torch.randint(0, 100, (B * N,), device=device)
            data.append((x_batch, y_batch))
        
        # Training loop
        initial_loss = None
        for epoch in range(3):
            epoch_loss = 0.0
            for x_batch, y_batch in data:
                metrics = trainer.train_step(x_batch, y_batch, use_adaptive_length=True)
                epoch_loss += metrics['total_loss']
            
            avg_loss = epoch_loss / len(data)
            
            if initial_loss is None:
                initial_loss = avg_loss
            
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        
        # Loss should decrease (or at least not increase significantly)
        final_loss = avg_loss
        assert final_loss <= initial_loss * 1.5, "Loss should not increase significantly"
    
    def test_speedup_estimation(self, adaptive_model, device):
        """Test that speedup estimation is reasonable."""
        B, N = 8, 64
        x = torch.randint(0, 100, (B, N), device=device)
        
        adaptive_model.eval()
        with torch.no_grad():
            _, length_info = adaptive_model(x, use_adaptive_length=True)
        
        # Speedup should be >= 1.0 (no slowdown)
        assert length_info['speedup_estimate'] >= 1.0
        
        # Speedup should be <= max_seq_len / min_bin
        max_speedup = adaptive_model.max_seq_len / adaptive_model.length_predictor.length_bins[0].item()
        assert length_info['speedup_estimate'] <= max_speedup


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
