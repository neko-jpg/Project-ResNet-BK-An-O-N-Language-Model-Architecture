"""
Tests for C4 Benchmark

This module tests the C4 benchmark implementation for task 9.5.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import json
import tempfile
import shutil

from src.benchmarks.c4_benchmark import (
    C4Benchmark,
    BenchmarkConfig,
    BenchmarkResults,
    load_c4_data
)


class TestC4DataLoading:
    """Test C4 data loading functionality."""
    
    def test_load_c4_data_small(self):
        """Test loading small amount of C4 data."""
        # Load very small amount for testing
        train_data, vocab, get_batch = load_c4_data(
            batch_size=2,
            n_seq=16,
            data_limit=1000,  # Only 1000 tokens for testing
            vocab_size_limit=500
        )
        
        if train_data is None:
            pytest.skip("C4 dataset not available (network issue)")
        
        # Check data shape
        assert train_data.dim() == 2
        assert train_data.size(1) == 2  # batch_size
        
        # Check vocabulary
        assert 'vocab_size' in vocab
        assert 'stoi' in vocab
        assert 'itos' in vocab
        assert vocab['vocab_size'] > 0
        assert vocab['vocab_size'] <= 500
        
        # Check get_batch function
        x, y = get_batch(train_data, 0)
        assert x.shape[0] == 16  # n_seq
        assert y.shape[0] == 16  # n_seq
    
    def test_vocab_special_tokens(self):
        """Test that vocabulary contains special tokens."""
        train_data, vocab, get_batch = load_c4_data(
            batch_size=2,
            n_seq=16,
            data_limit=500,
            vocab_size_limit=100
        )
        
        if train_data is None:
            pytest.skip("C4 dataset not available")
        
        # Check special tokens
        assert "<unk>" in vocab['stoi']
        assert "<pad>" in vocab['stoi']


class TestBenchmarkConfig:
    """Test benchmark configuration."""
    
    def test_config_creation(self):
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
            data_limit=1000
        )
        
        assert config.model_name == 'test_model'
        assert config.d_model == 32
        assert config.data_limit == 1000
        assert config.use_analytic_gradient == True  # default
    
    def test_config_with_optimizations(self):
        """Test configuration with optimization flags."""
        config = BenchmarkConfig(
            model_name='optimized_model',
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
            use_analytic_gradient=True,
            use_mixed_precision=True,
            use_act=True
        )
        
        assert config.use_analytic_gradient == True
        assert config.use_mixed_precision == True
        assert config.use_act == True


class TestBenchmarkResults:
    """Test benchmark results."""
    
    def test_results_creation(self):
        """Test creating benchmark results."""
        results = BenchmarkResults(
            model_name='test_model',
            dataset_name='c4',
            config={'test': 'config'},
            final_loss=2.5,
            final_perplexity=12.18,
            best_perplexity=11.5,
            training_time=100.0,
            total_tokens=1000,
            vocab_size=500,
            domain_perplexities={'general': 12.0, 'technical': 13.5},
            forward_flops=1000000,
            backward_flops=2000000,
            optimizer_flops=500000,
            total_flops_per_step=3500000,
            total_training_flops=350000000,
            peak_memory_mb=100.0,
            model_size_mb=5.0,
            epoch_losses=[3.0, 2.5],
            epoch_perplexities=[20.0, 12.18],
            epoch_times=[50.0, 50.0]
        )
        
        assert results.model_name == 'test_model'
        assert results.dataset_name == 'c4'
        assert results.final_perplexity == 12.18
        assert 'general' in results.domain_perplexities
        assert 'technical' in results.domain_perplexities
    
    def test_results_to_dict(self):
        """Test converting results to dictionary."""
        results = BenchmarkResults(
            model_name='test_model',
            dataset_name='c4',
            config={},
            final_loss=2.5,
            final_perplexity=12.18,
            best_perplexity=11.5,
            training_time=100.0,
            total_tokens=1000,
            vocab_size=500,
            domain_perplexities={'general': 12.0},
            forward_flops=1000000,
            backward_flops=2000000,
            optimizer_flops=500000,
            total_flops_per_step=3500000,
            total_training_flops=350000000,
            peak_memory_mb=100.0,
            model_size_mb=5.0,
            epoch_losses=[2.5],
            epoch_perplexities=[12.18],
            epoch_times=[100.0]
        )
        
        results_dict = results.to_dict()
        assert isinstance(results_dict, dict)
        assert results_dict['model_name'] == 'test_model'
        assert results_dict['dataset_name'] == 'c4'
        assert 'domain_perplexities' in results_dict
    
    def test_results_save_json(self):
        """Test saving results to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = BenchmarkResults(
                model_name='test_model',
                dataset_name='c4',
                config={},
                final_loss=2.5,
                final_perplexity=12.18,
                best_perplexity=11.5,
                training_time=100.0,
                total_tokens=1000,
                vocab_size=500,
                domain_perplexities={'general': 12.0},
                forward_flops=1000000,
                backward_flops=2000000,
                optimizer_flops=500000,
                total_flops_per_step=3500000,
                total_training_flops=350000000,
                peak_memory_mb=100.0,
                model_size_mb=5.0,
                epoch_losses=[2.5],
                epoch_perplexities=[12.18],
                epoch_times=[100.0]
            )
            
            filepath = Path(tmpdir) / "results.json"
            results.save_json(str(filepath))
            
            assert filepath.exists()
            
            # Load and verify
            with open(filepath, 'r') as f:
                loaded = json.load(f)
            
            assert loaded['model_name'] == 'test_model'
            assert loaded['dataset_name'] == 'c4'
            assert loaded['final_perplexity'] == 12.18


class TestC4Benchmark:
    """Test C4 benchmark class."""
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = C4Benchmark(output_dir=tmpdir)
            
            assert benchmark.output_dir == Path(tmpdir)
            assert benchmark.output_dir.exists()
            assert len(benchmark.results) == 0
    
    @pytest.mark.slow
    def test_run_small_benchmark(self):
        """Test running a very small benchmark (slow test)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = C4Benchmark(output_dir=tmpdir)
            
            config = BenchmarkConfig(
                model_name='tiny_test',
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
                
                assert results.model_name == 'tiny_test'
                assert results.dataset_name == 'c4'
                assert results.final_perplexity > 0
                assert results.training_time > 0
                assert 'tiny_test' in benchmark.results
                
                # Check that results file was created
                results_file = Path(tmpdir) / "tiny_test_c4_results.json"
                assert results_file.exists()
            
            except Exception as e:
                if "Failed to load C4" in str(e):
                    pytest.skip("C4 dataset not available")
                else:
                    raise
    
    def test_create_transformer_baseline(self):
        """Test creating Transformer baseline model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = C4Benchmark(output_dir=tmpdir)
            
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
            assert hasattr(model, 'token_embedding')
            assert hasattr(model, 'position_embedding')
            assert hasattr(model, 'transformer')
            assert hasattr(model, 'lm_head')
            
            # Test forward pass
            x = torch.randint(0, 1000, (4, 64))
            logits = model(x)
            assert logits.shape == (4, 64, 1000)
    
    def test_create_resnet_bk_model(self):
        """Test creating ResNet-BK model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = C4Benchmark(output_dir=tmpdir)
            
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


def test_c4_benchmark_imports():
    """Test that all necessary imports work."""
    from src.benchmarks.c4_benchmark import (
        C4Benchmark,
        BenchmarkConfig,
        BenchmarkResults,
        load_c4_data,
        main
    )
    
    assert C4Benchmark is not None
    assert BenchmarkConfig is not None
    assert BenchmarkResults is not None
    assert load_c4_data is not None
    assert main is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
