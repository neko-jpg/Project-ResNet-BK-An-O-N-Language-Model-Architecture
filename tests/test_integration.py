"""
Integration Tests
Test full model forward and backward passes.
"""

import torch
import pytest
import sys
sys.path.insert(0, '.')

from src.models.resnet_bk import LanguageModel
from src.models.configurable_resnet_bk import ConfigurableResNetBK, BASELINE_CONFIG


class TestIntegration:
    """Integration tests for full model."""
    
    def test_language_model_forward(self):
        """Test LanguageModel forward pass."""
        vocab_size = 1000
        d_model = 32
        n_layers = 2
        n_seq = 16
        batch_size = 4
        
        model = LanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=n_seq,
            num_experts=2,
            top_k=1,
        )
        
        # Create input
        x = torch.randint(0, vocab_size, (batch_size, n_seq))
        
        # Forward pass
        logits = model(x)
        
        # Verify output shape
        assert logits.shape == (batch_size, n_seq, vocab_size), \
            f"Expected shape ({batch_size}, {n_seq}, {vocab_size}), got {logits.shape}"
        
        # Verify no NaN or Inf
        assert torch.all(torch.isfinite(logits)), "Logits contain NaN or Inf"
    
    def test_language_model_backward(self):
        """Test LanguageModel backward pass."""
        vocab_size = 1000
        d_model = 32
        n_layers = 2
        n_seq = 16
        batch_size = 4
        
        model = LanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=n_seq,
            num_experts=2,
            top_k=1,
        )
        
        # Create input and target
        x = torch.randint(0, vocab_size, (batch_size, n_seq))
        y = torch.randint(0, vocab_size, (batch_size * n_seq,))
        
        # Forward pass
        logits = model(x)
        
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits.view(-1, vocab_size), y)
        
        # Backward pass
        loss.backward()
        
        # Verify gradients exist for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.all(torch.isfinite(param.grad)), \
                    f"Gradient for {name} contains NaN or Inf"
    
    def test_configurable_model(self):
        """Test ConfigurableResNetBK."""
        config = BASELINE_CONFIG
        config.vocab_size = 1000
        config.d_model = 32
        config.n_layers = 2
        config.n_seq = 16
        
        model = ConfigurableResNetBK(config)
        
        # Test forward pass
        batch_size = 4
        x = torch.randint(0, config.vocab_size, (batch_size, config.n_seq))
        logits = model(x)
        
        assert logits.shape == (batch_size, config.n_seq, config.vocab_size)
        
        # Test config summary
        summary = model.get_config_summary()
        assert "Model" in summary
        assert "Parameters" in summary
        assert "Enabled Features" in summary
    
    def test_training_step(self):
        """Test a complete training step."""
        vocab_size = 1000
        d_model = 32
        n_layers = 2
        n_seq = 16
        batch_size = 4
        
        model = LanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=n_seq,
            num_experts=2,
            top_k=1,
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Create batch
        x = torch.randint(0, vocab_size, (batch_size, n_seq))
        y = torch.randint(0, vocab_size, (batch_size * n_seq,))
        
        # Training step
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y)
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()
        
        # Verify loss is finite
        assert torch.isfinite(loss), "Loss is NaN or Inf"
        
        # Verify parameters updated
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_save_load(self):
        """Test model checkpoint save and load."""
        import tempfile
        import os
        
        vocab_size = 1000
        d_model = 32
        n_layers = 2
        n_seq = 16
        
        model = LanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=n_seq,
        )
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            
            # Load checkpoint
            model_loaded = LanguageModel(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_seq=n_seq,
            )
            model_loaded.load_state_dict(torch.load(checkpoint_path))
            
            # Verify parameters match
            for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
                assert torch.allclose(p1, p2), "Loaded parameters don't match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
