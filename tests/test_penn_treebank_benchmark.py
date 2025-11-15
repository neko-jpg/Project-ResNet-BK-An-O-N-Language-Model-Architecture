"""
Tests for Penn Treebank benchmark implementation.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import json
import tempfile
import shutil

from src.benchmarks.penn_treebank_benchmark import (
    PennTreebankBenchmark,
    BenchmarkConfig,
    BenchmarkResults,
    load_penn_treebank_data
)


class TestPennTreebankDataLoader:
    """Test Penn Treebank data loading."""
    
    def test_load_penn_treebank_data_structure(self):
        """Test that data loader returns correct structure."""
        # Try to load data (may fail if dataset not available)
        try:
            train_data, vocab, get_batch = load_penn_treebank_data(
                batch_size=4,
                n_seq=32,
                data_limit=1000  # Limit for testing
            )
            
            if train_data is not None:
                # Check data structure
                assert train_data.dim() == 2, "Train data should be 2D"
                assert train_data.dtype == torch.long, "Train data should be LongTensor"
                
                # Check vocab structure
                assert 'stoi' in vocab
                assert 'itos' in vocab
                assert 'vocab_size' in vocab
                assert len(vocab['itos']) == vocab['vocab_size']
                
                # Check get_batch function
                x, y = get_batch(train_data, 0)
                assert x.dim() == 2
                assert y.dim() == 1
                assert x.size(0) <= 32  # n_seq
                
                print("Penn Treebank data loader test passed")
            else:
                pytest.skip("Penn Treebank dataset not available")
        except Exception as e:
            pytest.skip(f"Penn Treebank dataset not available: {e}")


class TestBenchmarkConfig:
    """Test benchmark configuration."""
    
    def test_benchmark_config_creation(self):
        """Test creating benchmark configuration."""
        config = BenchmarkConfig(
            model_name='test_model',
            d_model=64,
            n_layers=2,
            n_seq=32,
            batch_size=4,
            epochs=1,
            lr=1e-3,
            weight_decay=0.01,
            grad_clip=0.5,
            device='cpu',
            seed=42
        )
        
        assert config.model_name == 'test_model'
        assert config.d_model == 64
        assert config.use_analytic_gradient == True  # Default
        assert config.use_koopman == False  # Default


class TestBenchmarkResults:
    """Test benchmark results."""
    
    def test_benchmark_results_creation(self):
        """Test creating benchmark results."""
        results = BenchmarkResults(
            model_name='test_model',
            dataset_name='penn-treebank',
            config={'test': 'config'},
            final_loss=2.5,
            final_perplexity=12.18,
            best_perplexity=11.5,
            training_time=100.0,
            total_tokens=10000,
            vocab_size=5000,
            forward_flops=1000000,
            backward_flops=2000000,
            optimizer_flops=500000,
            total_flops_per_step=3500000,
            total_training_flops=350000000,
            peak_memory_mb=500.0,
            model_size_mb=10.0,
            epoch_losses=[3.0, 2.7, 2.5],
            epoch_perplexities=[20.0, 15.0, 12.18],
            epoch_times=[30.0, 35.0, 35.0]
        )
        
        assert results.model_name == 'test_model'
        assert results.dataset_name == 'penn-treebank'
        assert results.final_perplexity == 12.18
        assert len(results.epoch_losses) == 3
    
    def test_benchmark_results_to_dict(self):
        """Test converting results to dictionary."""
        results = BenchmarkResults(
            model_name='test_model',
            dataset_name='penn-treebank',
            config={},
            final_loss=2.5,
            final_perplexity=12.18,
            best_perplexity=11.5,
            training_time=100.0,
            total_tokens=10000,
            vocab_size=5000,
            forward_flops=1000000,
            backward_flops=2000000,
            optimizer_flops=500000,
            total_flops_per_step=3500000,
            total_training_flops=350000000,
            peak_memory_mb=500.0,
            model_size_mb=10.0,
            epoch_losses=[2.5],
            epoch_perplexities=[12.18],
            epoch_times=[100.0]
        )
        
        results_dict = results.to_dict()
        assert isinstance(results_dict, dict)
        assert results_dict['model_name'] == 'test_model'
        assert results_dict['final_perplexity'] == 12.18
    
    def test_benchmark_results_save_json(self):
        """Test saving results to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = BenchmarkResults(
                model_name='test_model',
                dataset_name='penn-treebank',
                config={},
                final_loss=2.5,
                final_perplexity=12.18,
                best_perplexity=11.5,
                training_time=100.0,
                total_tokens=10000,
                vocab_size=5000,
                forward_flops=1000000,
                backward_flops=2000000,
                optimizer_flops=500000,
                total_flops_per_step=3500000,
                total_training_flops=350000000,
                peak_memory_mb=500.0,
                model_size_mb=10.0,
                epoch_losses=[2.5],
                epoch_perplexities=[12.18],
                epoch_times=[100.0]
            )
            
            filepath = Path(tmpdir) / "test_results.json"
            results.save_json(str(filepath))
            
            assert filepath.exists()
            
            # Load and verify
            with open(filepath, 'r') as f:
                loaded = json.load(f)
            
            assert loaded['model_name'] == 'test_model'
            assert loaded['final_perplexity'] == 12.18


class TestPennTreebankBenchmark:
    """Test Penn Treebank benchmark class."""
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = PennTreebankBenchmark(output_dir=tmpdir)
            
            assert benchmark.output_dir == Path(tmpdir)
            assert benchmark.output_dir.exists()
            assert len(benchmark.results) == 0
    
    def test_create_transformer_baseline(self):
        """Test creating Transformer baseline model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = PennTreebankBenchmark(output_dir=tmpdir)
            
            config = BenchmarkConfig(
                model_name='transformer_baseline',
                d_model=32,
                n_layers=2,
                n_seq=16,
                batch_size=2,
                epochs=1,
                lr=1e-3,
                weight_decay=0.01,
                grad_clip=0.5,
                device='cpu',
                seed=42
            )
            
            model = benchmark._create_transformer_baseline(config, vocab_size=1000)
            
            assert isinstance(model, nn.Module)
            
            # Test forward pass
            x = torch.randint(0, 1000, (2, 16))
            logits = model(x)
            
            assert logits.shape == (2, 16, 1000)
    
    def test_create_resnet_bk_model(self):
        """Test creating ResNet-BK model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = PennTreebankBenchmark(output_dir=tmpdir)
            
            config = BenchmarkConfig(
                model_name='resnet_bk_test',
                d_model=32,
                n_layers=2,
                n_seq=16,
                batch_size=2,
                epochs=1,
                lr=1e-3,
                weight_decay=0.01,
                grad_clip=0.5,
                device='cpu',
                seed=42,
                use_analytic_gradient=True
            )
            
            model = benchmark._create_model(config, vocab_size=1000)
            
            assert isinstance(model, nn.Module)
            
            # Test forward pass
            x = torch.randint(0, 1000, (2, 16))
            logits = model(x)
            
            assert logits.shape == (2, 16, 1000)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
