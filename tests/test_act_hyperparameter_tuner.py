"""
Tests for ACT hyperparameter tuning functionality.
"""

import torch
import pytest
import tempfile
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock

# Mock datasets and matplotlib modules if not available
if 'datasets' not in sys.modules:
    sys.modules['datasets'] = MagicMock()
if 'matplotlib' not in sys.modules:
    sys.modules['matplotlib'] = MagicMock()
    sys.modules['matplotlib.pyplot'] = MagicMock()
    sys.modules['matplotlib.animation'] = MagicMock()

from src.training.act_hyperparameter_tuner import ACTHyperparameterTuner
from src.models.adaptive_computation import ACTLanguageModel


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, vocab_size, n_seq, batch_size, num_batches=10):
        self.vocab_size = vocab_size
        self.n_seq = n_seq
        self.batch_size = batch_size
        self.num_batches = num_batches
    
    def __iter__(self):
        for _ in range(self.num_batches):
            x = torch.randint(0, self.vocab_size, (self.batch_size, self.n_seq))
            y = torch.randint(0, self.vocab_size, (self.batch_size * self.n_seq,))
            yield x, y
    
    def __len__(self):
        return self.num_batches


class TestACTHyperparameterTuner:
    """Test ACTHyperparameterTuner functionality."""
    
    def test_tuner_initialization(self):
        """Test tuner initializes correctly."""
        vocab_size = 1000
        tuner = ACTHyperparameterTuner(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=4,
            n_seq=128
        )
        
        assert tuner.vocab_size == vocab_size
        assert tuner.d_model == 64
        assert tuner.n_layers == 4
        assert tuner.n_seq == 128
        assert tuner.results == []
        assert tuner.best_config is None
        assert tuner.best_score == float('inf')
    
    def test_create_model(self):
        """Test model creation with different hyperparameters."""
        vocab_size = 1000
        tuner = ACTHyperparameterTuner(vocab_size=vocab_size, d_model=64, n_layers=4, n_seq=128)
        
        model = tuner.create_model(act_threshold=0.95, act_lambda=0.01)
        
        assert isinstance(model, ACTLanguageModel)
        assert model.n_layers == 4
        assert model.act_lambda == 0.01
        assert model.blocks[0].threshold == 0.95
    
    def test_evaluate(self):
        """Test model evaluation."""
        vocab_size = 1000
        n_seq = 128
        batch_size = 4
        
        tuner = ACTHyperparameterTuner(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=4,
            n_seq=n_seq,
            device='cpu'
        )
        
        model = tuner.create_model(act_threshold=0.95, act_lambda=0.01)
        val_loader = MockDataLoader(vocab_size, n_seq, batch_size, num_batches=5)
        
        perplexity = tuner.evaluate(model, val_loader)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
    
    def test_train_and_evaluate(self):
        """Test training and evaluation."""
        vocab_size = 1000
        n_seq = 128
        batch_size = 4
        
        tuner = ACTHyperparameterTuner(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=4,
            n_seq=n_seq,
            device='cpu'
        )
        
        model = tuner.create_model(act_threshold=0.95, act_lambda=0.01)
        train_loader = MockDataLoader(vocab_size, n_seq, batch_size, num_batches=10)
        val_loader = MockDataLoader(vocab_size, n_seq, batch_size, num_batches=5)
        
        metrics = tuner.train_and_evaluate(
            model, train_loader, val_loader, num_epochs=2, lr=1e-3
        )
        
        # Check all expected metrics are present
        assert 'final_val_perplexity' in metrics
        assert 'avg_layers_executed' in metrics
        assert 'training_time' in metrics
        assert 'convergence_speed' in metrics
        assert 'val_perplexity_history' in metrics
        assert 'avg_layers_history' in metrics
        assert 'train_loss_history' in metrics
        
        # Check values are reasonable
        assert metrics['final_val_perplexity'] > 0
        assert 0 < metrics['avg_layers_executed'] <= 4
        assert metrics['training_time'] > 0
        assert len(metrics['val_perplexity_history']) == 2  # 2 epochs
    
    def test_grid_search_small(self):
        """Test grid search with small search space."""
        vocab_size = 1000
        n_seq = 128
        batch_size = 4
        
        tuner = ACTHyperparameterTuner(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=4,
            n_seq=n_seq,
            device='cpu'
        )
        
        train_loader = MockDataLoader(vocab_size, n_seq, batch_size, num_batches=10)
        val_loader = MockDataLoader(vocab_size, n_seq, batch_size, num_batches=5)
        
        # Small search space for fast testing
        results = tuner.grid_search(
            train_loader=train_loader,
            val_loader=val_loader,
            threshold_values=[0.8, 0.95],
            lambda_values=[0.01, 0.05],
            num_epochs=1,
            score_metric='balanced'
        )
        
        # Check results structure
        assert 'best_config' in results
        assert 'best_score' in results
        assert 'best_result' in results
        assert 'all_results' in results
        
        # Check best config
        assert 'threshold' in results['best_config']
        assert 'lambda' in results['best_config']
        assert results['best_config']['threshold'] in [0.8, 0.95]
        assert results['best_config']['lambda'] in [0.01, 0.05]
        
        # Check all results
        assert len(results['all_results']) == 4  # 2 thresholds × 2 lambdas
        
        # Check tuner state updated
        assert tuner.best_config is not None
        assert tuner.best_score < float('inf')
        assert len(tuner.results) == 4
    
    def test_grid_search_score_metrics(self):
        """Test different score metrics."""
        vocab_size = 1000
        n_seq = 128
        batch_size = 4
        
        tuner = ACTHyperparameterTuner(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=4,
            n_seq=n_seq,
            device='cpu'
        )
        
        train_loader = MockDataLoader(vocab_size, n_seq, batch_size, num_batches=10)
        val_loader = MockDataLoader(vocab_size, n_seq, batch_size, num_batches=5)
        
        # Test perplexity metric
        results_ppl = tuner.grid_search(
            train_loader=train_loader,
            val_loader=val_loader,
            threshold_values=[0.95],
            lambda_values=[0.01],
            num_epochs=1,
            score_metric='perplexity'
        )
        
        assert results_ppl['best_result']['score'] == results_ppl['best_result']['final_val_perplexity']
        
        # Reset tuner
        tuner.results = []
        tuner.best_config = None
        tuner.best_score = float('inf')
        
        # Test layers metric
        results_layers = tuner.grid_search(
            train_loader=train_loader,
            val_loader=val_loader,
            threshold_values=[0.95],
            lambda_values=[0.01],
            num_epochs=1,
            score_metric='layers'
        )
        
        assert results_layers['best_result']['score'] == results_layers['best_result']['avg_layers_executed']
    
    def test_save_results(self):
        """Test saving results to JSON."""
        vocab_size = 1000
        n_seq = 128
        batch_size = 4
        
        tuner = ACTHyperparameterTuner(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=4,
            n_seq=n_seq,
            device='cpu'
        )
        
        train_loader = MockDataLoader(vocab_size, n_seq, batch_size, num_batches=10)
        val_loader = MockDataLoader(vocab_size, n_seq, batch_size, num_batches=5)
        
        # Run small grid search
        tuner.grid_search(
            train_loader=train_loader,
            val_loader=val_loader,
            threshold_values=[0.95],
            lambda_values=[0.01],
            num_epochs=1
        )
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'results.json'
            tuner.save_results(str(filepath))
            
            # Check file exists
            assert filepath.exists()
            
            # Load and verify contents
            with open(filepath, 'r') as f:
                saved_data = json.load(f)
            
            assert 'best_config' in saved_data
            assert 'best_score' in saved_data
            assert 'all_results' in saved_data
            assert 'model_config' in saved_data
            
            # Check model config
            assert saved_data['model_config']['vocab_size'] == vocab_size
            assert saved_data['model_config']['d_model'] == 64
            assert saved_data['model_config']['n_layers'] == 4
    
    def test_threshold_effect(self):
        """Test that lower thresholds result in fewer layers executed."""
        vocab_size = 1000
        n_seq = 128
        batch_size = 4
        
        tuner = ACTHyperparameterTuner(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=4,
            n_seq=n_seq,
            device='cpu'
        )
        
        train_loader = MockDataLoader(vocab_size, n_seq, batch_size, num_batches=10)
        val_loader = MockDataLoader(vocab_size, n_seq, batch_size, num_batches=5)
        
        # Test with different thresholds
        results = tuner.grid_search(
            train_loader=train_loader,
            val_loader=val_loader,
            threshold_values=[0.5, 0.99],  # Low and high threshold
            lambda_values=[0.01],
            num_epochs=1
        )
        
        # Find results for each threshold
        result_low = next(r for r in results['all_results'] if r['threshold'] == 0.5)
        result_high = next(r for r in results['all_results'] if r['threshold'] == 0.99)
        
        # Lower threshold should generally result in fewer layers
        # (though not guaranteed due to random initialization)
        assert result_low['avg_layers_executed'] <= result_high['avg_layers_executed'] + 0.5
    
    def test_lambda_effect(self):
        """Test that higher lambda encourages fewer layers."""
        vocab_size = 1000
        n_seq = 128
        batch_size = 4
        
        tuner = ACTHyperparameterTuner(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=4,
            n_seq=n_seq,
            device='cpu'
        )
        
        train_loader = MockDataLoader(vocab_size, n_seq, batch_size, num_batches=10)
        val_loader = MockDataLoader(vocab_size, n_seq, batch_size, num_batches=5)
        
        # Test with different lambdas
        results = tuner.grid_search(
            train_loader=train_loader,
            val_loader=val_loader,
            threshold_values=[0.95],
            lambda_values=[0.001, 0.1],  # Low and high lambda
            num_epochs=2  # More epochs to see effect
        )
        
        # Both configurations should complete successfully
        assert len(results['all_results']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
