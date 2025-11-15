"""
Tests for WikiText-103 Benchmark
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import json
import tempfile
import shutil

from src.benchmarks.wikitext103_benchmark import (
    WikiText103Benchmark,
    BenchmarkConfig,
    BenchmarkResults,
    load_wikitext103_data
)


class TestWikiText103DataLoading:
    """Test WikiText-103 data loading."""
    
    def test_load_wikitext103_data_with_limit(self):
        """Test loading WikiText-103 with data limit."""
        # Use small data limit for testing
        train_data, vocab, get_batch = load_wikitext103_data(
            batch_size=2,
            n_seq=16,
            data_limit=1000,
            vocab_size_limit=500
        )
        
        if train_data is None:
            pytest.skip("WikiText-103 dataset not available")
        
        # Check data shape
        assert train_data.dim() == 2
        assert train_data.size(1) == 2  # batch_size
        
        # Check vocabulary
        assert 'vocab_size' in vocab
        assert vocab['vocab_size'] > 0
        assert vocab['vocab_size'] <= 500
        
        # Check get_batch function
        x, y = get_batch(train_data, 0)
        # get_batch returns min(n_seq, remaining_length)
        assert x.shape[0] <= 16  # n_seq or less
        assert y.shape[0] <= 16  # n_seq or less
        assert x.shape[0] == y.shape[0]  # same length
    
    def test_vocab_size_limit(self):
        """Test vocabulary size limiting."""
        train_data, vocab, get_batch = load_wikitext103_data(
            batch_size=2,
            n_seq=16,
            data_limit=1000,
            vocab_size_limit=100
        )
        
        if train_data is None:
            pytest.skip("WikiText-103 dataset not available")
        
        assert vocab['vocab_size'] <= 100


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""
    
    def test_benchmark_config_creation(self):
        """Test creating benchmark configuration."""
        config = BenchmarkConfig(
            model_name='test_model',
            d_model=32,
            n_layers=2,
            n_seq=64,
            batch_size=4,
            epochs=1,
            lr=1e-3,
            weight_decay=0.01,
            grad_clip=0.5,
            device='cpu',
            seed=42,
            data_limit=500
        )
        
        assert config.model_name == 'test_model'
        assert config.d_model == 32
        assert config.data_limit == 500
        assert config.use_analytic_gradient is True  # default


class TestBenchmarkResults:
    """Test BenchmarkResults dataclass."""
    
    def test_benchmark_results_creation(self):
        """Test creating benchmark results."""
        results = BenchmarkResults(
            model_name='test_model',
            dataset_name='wikitext-103',
            config={},
            final_loss=2.5,
            final_perplexity=12.18,
            best_perplexity=11.5,
            training_time=100.0,
            total_tokens=10000,
            vocab_size=1000,
            forward_flops=1000000,
            backward_flops=2000000,
            optimizer_flops=500000,
            total_flops_per_step=3500000,
            total_training_flops=350000000,
            peak_memory_mb=512.0,
            model_size_mb=10.0,
            epoch_losses=[3.0, 2.7, 2.5],
            epoch_perplexities=[20.0, 15.0, 12.18],
            epoch_times=[30.0, 35.0, 35.0]
        )
        
        assert results.model_name == 'test_model'
        assert results.dataset_name == 'wikitext-103'
        assert results.final_perplexity == 12.18
        assert len(results.epoch_losses) == 3
    
    def test_results_to_dict(self):
        """Test converting results to dictionary."""
        results = BenchmarkResults(
            model_name='test_model',
            dataset_name='wikitext-103',
            config={},
            final_loss=2.5,
            final_perplexity=12.18,
            best_perplexity=11.5,
            training_time=100.0,
            total_tokens=10000,
            vocab_size=1000,
            forward_flops=1000000,
            backward_flops=2000000,
            optimizer_flops=500000,
            total_flops_per_step=3500000,
            total_training_flops=350000000,
            peak_memory_mb=512.0,
            model_size_mb=10.0,
            epoch_losses=[3.0, 2.7, 2.5],
            epoch_perplexities=[20.0, 15.0, 12.18],
            epoch_times=[30.0, 35.0, 35.0]
        )
        
        results_dict = results.to_dict()
        assert isinstance(results_dict, dict)
        assert results_dict['model_name'] == 'test_model'
        assert results_dict['final_perplexity'] == 12.18
    
    def test_results_save_json(self):
        """Test saving results to JSON."""
        results = BenchmarkResults(
            model_name='test_model',
            dataset_name='wikitext-103',
            config={},
            final_loss=2.5,
            final_perplexity=12.18,
            best_perplexity=11.5,
            training_time=100.0,
            total_tokens=10000,
            vocab_size=1000,
            forward_flops=1000000,
            backward_flops=2000000,
            optimizer_flops=500000,
            total_flops_per_step=3500000,
            total_training_flops=350000000,
            peak_memory_mb=512.0,
            model_size_mb=10.0,
            epoch_losses=[3.0, 2.7, 2.5],
            epoch_perplexities=[20.0, 15.0, 12.18],
            epoch_times=[30.0, 35.0, 35.0]
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            results.save_json(temp_path)
            
            # Load and verify
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded['model_name'] == 'test_model'
            assert loaded['final_perplexity'] == 12.18
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestWikiText103Benchmark:
    """Test WikiText103Benchmark class."""
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = WikiText103Benchmark(output_dir=tmpdir)
            
            assert benchmark.output_dir == Path(tmpdir)
            assert benchmark.output_dir.exists()
            assert len(benchmark.results) == 0
    
    def test_create_transformer_baseline(self):
        """Test creating Transformer baseline model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = WikiText103Benchmark(output_dir=tmpdir)
            
            config = BenchmarkConfig(
                model_name='transformer_baseline',
                d_model=32,
                n_layers=2,
                n_seq=64,
                batch_size=4,
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
            x = torch.randint(0, 1000, (4, 64))
            logits = model(x)
            assert logits.shape == (4, 64, 1000)
    
    def test_create_resnet_bk_model(self):
        """Test creating ResNet-BK model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = WikiText103Benchmark(output_dir=tmpdir)
            
            config = BenchmarkConfig(
                model_name='resnet_bk_test',
                d_model=32,
                n_layers=2,
                n_seq=64,
                batch_size=4,
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
            x = torch.randint(0, 1000, (4, 64))
            logits = model(x)
            assert logits.shape == (4, 64, 1000)
    
    @pytest.mark.slow
    def test_run_benchmark_small(self):
        """Test running a small benchmark (marked as slow)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = WikiText103Benchmark(output_dir=tmpdir)
            
            config = BenchmarkConfig(
                model_name='resnet_bk_tiny',
                d_model=16,
                n_layers=1,
                n_seq=32,
                batch_size=2,
                epochs=1,
                lr=1e-3,
                weight_decay=0.01,
                grad_clip=0.5,
                device='cpu',
                seed=42,
                data_limit=500,  # Very small for testing
                use_analytic_gradient=False
            )
            
            try:
                results = benchmark.run_benchmark(config)
                
                assert results.model_name == 'resnet_bk_tiny'
                assert results.dataset_name == 'wikitext-103'
                assert results.final_perplexity > 0
                assert results.training_time > 0
                assert len(results.epoch_losses) == 1
                
                # Check that results were saved
                results_file = Path(tmpdir) / "resnet_bk_tiny_wikitext103_results.json"
                assert results_file.exists()
                
            except Exception as e:
                if "Failed to load WikiText-103" in str(e):
                    pytest.skip("WikiText-103 dataset not available")
                else:
                    raise
    
    def test_compare_to_wikitext2_missing_file(self):
        """Test comparison when WikiText-2 results file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = WikiText103Benchmark(output_dir=tmpdir)
            
            # Should handle missing file gracefully
            benchmark.compare_to_wikitext2("nonexistent_file.json")
            # No exception should be raised


def test_integration_wikitext103_benchmark():
    """Integration test for WikiText-103 benchmark."""
    # This is a minimal integration test
    # Full benchmark is too slow for regular testing
    
    with tempfile.TemporaryDirectory() as tmpdir:
        benchmark = WikiText103Benchmark(output_dir=tmpdir)
        
        # Verify output directory was created
        assert benchmark.output_dir.exists()
        
        # Verify we can create models
        config = BenchmarkConfig(
            model_name='test_model',
            d_model=16,
            n_layers=1,
            n_seq=32,
            batch_size=2,
            epochs=1,
            lr=1e-3,
            weight_decay=0.01,
            grad_clip=0.5,
            device='cpu',
            seed=42
        )
        
        model = benchmark._create_model(config, vocab_size=100)
        assert isinstance(model, nn.Module)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
