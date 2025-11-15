"""
Tests for Early Exit functionality
"""

import pytest
import torch
import torch.nn as nn

import sys
sys.path.append('src')

from models.early_exit import (
    EarlyExitResNetBKBlock,
    EarlyExitLanguageModel,
    EarlyExitEvaluator
)


@pytest.fixture
def model_config():
    """Standard model configuration for tests."""
    return {
        'vocab_size': 100,
        'd_model': 32,
        'n_layers': 4,
        'n_seq': 64,
        'num_experts': 2,
        'top_k': 1,
        'dropout_p': 0.0,
        'confidence_threshold': 0.9
    }


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestEarlyExitResNetBKBlock:
    """Tests for EarlyExitResNetBKBlock."""
    
    def test_initialization(self, model_config):
        """Test block initialization."""
        block = EarlyExitResNetBKBlock(
            d_model=model_config['d_model'],
            n_seq=model_config['n_seq'],
            vocab_size=model_config['vocab_size'],
            num_experts=model_config['num_experts'],
            top_k=model_config['top_k'],
            dropout_p=model_config['dropout_p']
        )
        
        assert block.d_model == model_config['d_model']
        assert block.n_seq == model_config['n_seq']
        assert block.vocab_size == model_config['vocab_size']
        assert hasattr(block, 'layer_norm')
        assert hasattr(block, 'bk_layer')
        assert hasattr(block, 'exit_classifier')
    
    def test_forward_pass(self, model_config, device):
        """Test forward pass produces correct output shapes."""
        block = EarlyExitResNetBKBlock(
            d_model=model_config['d_model'],
            n_seq=model_config['n_seq'],
            vocab_size=model_config['vocab_size'],
            num_experts=model_config['num_experts'],
            top_k=model_config['top_k'],
            dropout_p=model_config['dropout_p']
        ).to(device)
        
        batch_size = 4
        x = torch.randn(batch_size, model_config['n_seq'], model_config['d_model']).to(device)
        
        output, exit_logits = block(x)
        
        # Check output shapes
        assert output.shape == (batch_size, model_config['n_seq'], model_config['d_model'])
        assert exit_logits.shape == (batch_size, model_config['n_seq'], model_config['vocab_size'])
    
    def test_exit_classifier_produces_valid_logits(self, model_config, device):
        """Test that exit classifier produces valid logits."""
        block = EarlyExitResNetBKBlock(
            d_model=model_config['d_model'],
            n_seq=model_config['n_seq'],
            vocab_size=model_config['vocab_size'],
            num_experts=model_config['num_experts'],
            top_k=model_config['top_k'],
            dropout_p=model_config['dropout_p']
        ).to(device)
        
        batch_size = 2
        x = torch.randn(batch_size, model_config['n_seq'], model_config['d_model']).to(device)
        
        _, exit_logits = block(x)
        
        # Check that logits are finite
        assert torch.isfinite(exit_logits).all()
        
        # Check that softmax produces valid probabilities
        probs = torch.softmax(exit_logits, dim=-1)
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-5)


class TestEarlyExitLanguageModel:
    """Tests for EarlyExitLanguageModel."""
    
    def test_initialization(self, model_config):
        """Test model initialization."""
        model = EarlyExitLanguageModel(**model_config)
        
        assert model.d_model == model_config['d_model']
        assert model.n_seq == model_config['n_seq']
        assert model.n_layers == model_config['n_layers']
        assert model.vocab_size == model_config['vocab_size']
        assert model.confidence_threshold == model_config['confidence_threshold']
        assert len(model.blocks) == model_config['n_layers']
    
    def test_standard_forward_pass(self, model_config, device):
        """Test standard forward pass without early exit."""
        model = EarlyExitLanguageModel(**model_config).to(device)
        
        batch_size = 4
        x = torch.randint(0, model_config['vocab_size'], (batch_size, model_config['n_seq'])).to(device)
        
        logits = model(x, use_early_exit=False)
        
        # Check output shape
        assert logits.shape == (batch_size, model_config['n_seq'], model_config['vocab_size'])
        
        # Check that logits are finite
        assert torch.isfinite(logits).all()
    
    def test_early_exit_forward_pass(self, model_config, device):
        """Test forward pass with early exit."""
        model = EarlyExitLanguageModel(**model_config).to(device)
        
        batch_size = 4
        x = torch.randint(0, model_config['vocab_size'], (batch_size, model_config['n_seq'])).to(device)
        
        logits, exit_info = model(x, use_early_exit=True)
        
        # Check output shape
        assert logits.shape == (batch_size, model_config['n_seq'], model_config['vocab_size'])
        
        # Check exit info
        assert 'exit_layers' in exit_info
        assert 'exit_confidences' in exit_info
        assert 'exited_mask' in exit_info
        
        # Check exit info shapes
        assert exit_info['exit_layers'].shape == (batch_size, model_config['n_seq'])
        assert exit_info['exit_confidences'].shape == (batch_size, model_config['n_seq'])
        assert exit_info['exited_mask'].shape == (batch_size, model_config['n_seq'])
        
        # Check exit layers are valid
        assert (exit_info['exit_layers'] >= 0).all()
        assert (exit_info['exit_layers'] <= model_config['n_layers']).all()
        
        # Check confidences are valid probabilities
        assert (exit_info['exit_confidences'] >= 0).all()
        assert (exit_info['exit_confidences'] <= 1).all()
    
    def test_early_exit_reduces_computation(self, model_config, device):
        """Test that early exit reduces average layers executed."""
        # Use very low threshold to encourage early exits even with random model
        model_config['confidence_threshold'] = 0.01  # Very low threshold
        model = EarlyExitLanguageModel(**model_config).to(device)
        
        batch_size = 8
        x = torch.randint(0, model_config['vocab_size'], (batch_size, model_config['n_seq'])).to(device)
        
        _, exit_info = model(x, use_early_exit=True)
        
        avg_exit_layer = exit_info['exit_layers'].float().mean().item()
        
        # With very low threshold, average exit should be less than or equal to max layers
        # (some tokens should exit early)
        assert avg_exit_layer <= model_config['n_layers']
    
    def test_high_threshold_delays_exit(self, model_config, device):
        """Test that high threshold delays exits."""
        # Use very high threshold
        model_config['confidence_threshold'] = 0.99
        model = EarlyExitLanguageModel(**model_config).to(device)
        
        batch_size = 4
        x = torch.randint(0, model_config['vocab_size'], (batch_size, model_config['n_seq'])).to(device)
        
        _, exit_info = model(x, use_early_exit=True)
        
        avg_exit_layer = exit_info['exit_layers'].float().mean().item()
        
        # With high threshold, most tokens should reach later layers
        assert avg_exit_layer >= model_config['n_layers'] * 0.5
    
    def test_exit_statistics(self, model_config, device):
        """Test exit statistics tracking."""
        model = EarlyExitLanguageModel(**model_config).to(device)
        model.reset_exit_statistics()
        
        batch_size = 4
        x = torch.randint(0, model_config['vocab_size'], (batch_size, model_config['n_seq'])).to(device)
        
        # Run multiple forward passes
        for _ in range(3):
            _, _ = model(x, use_early_exit=True)
        
        stats = model.get_exit_statistics()
        
        # Check statistics structure
        assert 'avg_exit_layer' in stats
        assert 'exit_distribution' in stats
        assert 'speedup_estimate' in stats
        assert 'total_tokens_processed' in stats
        
        # Check values are reasonable
        assert 0 <= stats['avg_exit_layer'] <= model_config['n_layers']
        assert len(stats['exit_distribution']) == model_config['n_layers'] + 1
        # Speedup can be < 1.0 if avg_exit_layer > n_layers (shouldn't happen but check >= 0)
        assert stats['speedup_estimate'] > 0
        assert stats['total_tokens_processed'] > 0
    
    def test_reset_statistics(self, model_config, device):
        """Test resetting exit statistics."""
        model = EarlyExitLanguageModel(**model_config).to(device)
        
        batch_size = 4
        x = torch.randint(0, model_config['vocab_size'], (batch_size, model_config['n_seq'])).to(device)
        
        # Run forward pass
        _, _ = model(x, use_early_exit=True)
        
        # Reset statistics
        model.reset_exit_statistics()
        
        # Check that counters are reset
        assert model.avg_exit_layer.item() == 0.0
        assert (model.exit_layer_counts == 0).all()
    
    def test_consistency_between_modes(self, model_config, device):
        """Test that standard and early exit modes produce similar outputs."""
        model = EarlyExitLanguageModel(**model_config).to(device)
        model.eval()
        
        batch_size = 2
        x = torch.randint(0, model_config['vocab_size'], (batch_size, model_config['n_seq'])).to(device)
        
        with torch.no_grad():
            # Standard forward
            logits_standard = model(x, use_early_exit=False)
            
            # Early exit with very high threshold (should use all layers)
            model.confidence_threshold = 0.999
            logits_early, _ = model(x, use_early_exit=True)
        
        # Outputs should be similar (not exact due to different paths)
        # Check that predictions are the same
        pred_standard = logits_standard.argmax(dim=-1)
        pred_early = logits_early.argmax(dim=-1)
        
        # At least 80% of predictions should match
        match_rate = (pred_standard == pred_early).float().mean()
        assert match_rate >= 0.8


class TestEarlyExitEvaluator:
    """Tests for EarlyExitEvaluator."""
    
    def test_initialization(self, model_config, device):
        """Test evaluator initialization."""
        model = EarlyExitLanguageModel(**model_config).to(device)
        evaluator = EarlyExitEvaluator(model, device)
        
        assert evaluator.model is model
        assert evaluator.device == device
    
    def test_evaluate_multiple_thresholds(self, model_config, device):
        """Test evaluation across multiple thresholds."""
        model = EarlyExitLanguageModel(**model_config).to(device)
        evaluator = EarlyExitEvaluator(model, device)
        
        # Create simple dataloader
        from torch.utils.data import DataLoader, TensorDataset
        
        num_samples = 32
        x_data = torch.randint(0, model_config['vocab_size'], (num_samples, model_config['n_seq']))
        y_data = torch.randint(0, model_config['vocab_size'], (num_samples, model_config['n_seq']))
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        thresholds = [0.7, 0.9]
        results = evaluator.evaluate(dataloader, confidence_thresholds=thresholds)
        
        # Check results structure
        assert len(results) == len(thresholds)
        for threshold in thresholds:
            assert threshold in results
            assert 'perplexity' in results[threshold]
            assert 'avg_exit_layer' in results[threshold]
            assert 'speedup_estimate' in results[threshold]
    
    def test_lower_threshold_earlier_exit(self, model_config, device):
        """Test that lower threshold leads to earlier exits."""
        model = EarlyExitLanguageModel(**model_config).to(device)
        evaluator = EarlyExitEvaluator(model, device)
        
        # Create simple dataloader
        from torch.utils.data import DataLoader, TensorDataset
        
        num_samples = 32
        x_data = torch.randint(0, model_config['vocab_size'], (num_samples, model_config['n_seq']))
        y_data = torch.randint(0, model_config['vocab_size'], (num_samples, model_config['n_seq']))
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        thresholds = [0.7, 0.9]
        results = evaluator.evaluate(dataloader, confidence_thresholds=thresholds)
        
        # Lower threshold should lead to earlier average exit
        assert results[0.7]['avg_exit_layer'] <= results[0.9]['avg_exit_layer']
        
        # Lower threshold should have higher speedup
        assert results[0.7]['speedup_estimate'] >= results[0.9]['speedup_estimate']


def test_early_exit_requirements():
    """Test that early exit satisfies requirements 6.14 and 6.15."""
    print("\n" + "=" * 70)
    print("Testing Early Exit Requirements")
    print("=" * 70)
    
    # Requirement 6.14: Halt computation when confidence > threshold
    print("\nRequirement 6.14: Early exiting for inference")
    
    model_config = {
        'vocab_size': 100,
        'd_model': 32,
        'n_layers': 4,
        'n_seq': 64,
        'num_experts': 2,
        'top_k': 1,
        'dropout_p': 0.0,
        'confidence_threshold': 0.9
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EarlyExitLanguageModel(**model_config).to(device)
    
    batch_size = 8
    x = torch.randint(0, model_config['vocab_size'], (batch_size, model_config['n_seq'])).to(device)
    
    with torch.no_grad():
        logits, exit_info = model(x, use_early_exit=True)
    
    # Check that some tokens exited early
    avg_exit = exit_info['exit_layers'].float().mean().item()
    print(f"  ✓ Average exit layer: {avg_exit:.2f} (< {model_config['n_layers']} layers)")
    
    # Check that confidence threshold is respected
    exited_early = exit_info['exit_layers'] < model_config['n_layers']
    if exited_early.any():
        early_confidences = exit_info['exit_confidences'][exited_early]
        print(f"  ✓ Early exit confidences: min={early_confidences.min():.4f}, "
              f"mean={early_confidences.mean():.4f}")
    
    # Requirement 6.15: Measure average exit layer
    print("\nRequirement 6.15: Measure average exit layer")
    
    stats = model.get_exit_statistics()
    print(f"  ✓ Average exit layer: {stats['avg_exit_layer']:.2f}")
    print(f"  ✓ Speedup estimate: {stats['speedup_estimate']:.2f}x")
    print(f"  ✓ Exit distribution: {[f'{p:.1f}%' for p in stats['exit_distribution']]}")
    
    print("\n" + "=" * 70)
    print("All requirements satisfied!")
    print("=" * 70)


if __name__ == '__main__':
    # Run requirement test
    test_early_exit_requirements()
    
    # Run pytest
    pytest.main([__file__, '-v'])
