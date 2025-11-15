"""
Tests for WikiText-2 benchmark infrastructure.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import json
import tempfile
import shutil

from src.benchmarks.wikitext2_benchmark import (
    BenchmarkConfig,
    BenchmarkResults,
    WikiText2Benchmark,
)


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""
    
    def test_create_config(self):
        """Test creating benchmark configuration."""
        config = BenchmarkConfig(
            model_name='test_model',
            d_model=64,
            n_layers=2,
            n_seq=128,
            batch_size=16,
            epochs=1,
            lr=1e-3,
            weight_decay=0.01,
            grad_clip=0.5,
            device='cpu',
            seed=42,
        )
        
        assert config.model_name == 'test_model'
        assert config.d_model == 64
        assert config.n_layers == 2
        assert config.use_analytic_gradient == True  # default


class TestBenchmarkResults:
    """Test BenchmarkResults dataclass."""
    
    def test_create_results(self):
        """Test creating benchmark results."""
        results = BenchmarkResults(
            model_name='test_model',
            config={'d_model': 64},
            final_loss=2.5,
            final_perplexity=12.18,
            best_perplexity=11.5,
            training_time=100.0,
            forward_flops=1000000,
            backward_flops=2000000,
            optimizer_flops=500000,
            total_flops_per_step=3500000,
            total_training_flops=350000000,
            peak_memory_mb=512.0,
            model_size_mb=10.5,
            epoch_losses=[3.0, 2.7, 2.5],
            epoch_perplexities=[20.0, 15.0, 12.18],
            epoch_times=[30.0, 35.0, 35.0],
        )
        
        assert results.model_name == 'test_model'
        assert results.final_perplexity == 12.18
        assert len(results.epoch_losses) == 3
    
    def test_to_dict(self):
        """Test converting results to dictionary."""
        results = BenchmarkResults(
            model_name='test_model',
            config={'d_model': 64},
            final_loss=2.5,
            final_perplexity=12.18,
            best_perplexity=11.5,
            training_time=100.0,
            forward_flops=1000000,
            backward_flops=2000000,
            optimizer_flops=500000,
            total_flops_per_step=3500000,
            total_training_flops=350000000,
            peak_memory_mb=512.0,
            model_size_mb=10.5,
            epoch_losses=[3.0, 2.7, 2.5],
            epoch_perplexities=[20.0, 15.0, 12.18],
            epoch_times=[30.0, 35.0, 35.0],
        )
        
        result_dict = results.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['model_name'] == 'test_model'
        assert result_dict['final_perplexity'] == 12.18
    
    def test_save_json(self):
        """Test saving results to JSON."""
        results = BenchmarkResults(
            model_name='test_model',
            config={'d_model': 64},
            final_loss=2.5,
            final_perplexity=12.18,
            best_perplexity=11.5,
            training_time=100.0,
            forward_flops=1000000,
            backward_flops=2000000,
            optimizer_flops=500000,
            total_flops_per_step=3500000,
            total_training_flops=350000000,
            peak_memory_mb=512.0,
            model_size_mb=10.5,
            epoch_losses=[3.0, 2.7, 2.5],
            epoch_perplexities=[20.0, 15.0, 12.18],
            epoch_times=[30.0, 35.0, 35.0],
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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


class TestWikiText2Benchmark:
    """Test WikiText2Benchmark class."""
    
    def test_create_benchmark(self):
        """Test creating benchmark instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = WikiText2Benchmark(output_dir=tmpdir)
            assert benchmark.output_dir.exists()
            assert len(benchmark.results) == 0
    
    def test_create_transformer_baseline(self):
        """Test creating Transformer baseline model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = WikiText2Benchmark(output_dir=tmpdir)
            
            config = BenchmarkConfig(
                model_name='transformer_test',
                d_model=32,
                n_layers=2,
                n_seq=64,
                batch_size=8,
                epochs=1,
                lr=1e-3,
                weight_decay=0.01,
                grad_clip=0.5,
                device='cpu',
                seed=42,
            )
            
            model = benchmark._create_transformer_baseline(config, vocab_size=1000)
            
            assert isinstance(model, nn.Module)
            
            # Test forward pass
            x = torch.randint(0, 1000, (8, 64))
            logits = model(x)
            
            assert logits.shape == (8, 64, 1000)
    
    def test_create_resnet_bk_model(self):
        """Test creating ResNet-BK model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = WikiText2Benchmark(output_dir=tmpdir)
            
            config = BenchmarkConfig(
                model_name='resnet_bk_test',
                d_model=32,
                n_layers=2,
                n_seq=64,
                batch_size=8,
                epochs=1,
                lr=1e-3,
                weight_decay=0.01,
                grad_clip=0.5,
                device='cpu',
                seed=42,
                use_analytic_gradient=True,
            )
            
            model = benchmark._create_model(config, vocab_size=1000)
            
            assert isinstance(model, nn.Module)
            
            # Test forward pass
            x = torch.randint(0, 1000, (8, 64))
            logits = model(x)
            
            assert logits.shape == (8, 64, 1000)
    
    @pytest.mark.slow
    def test_run_mini_benchmark(self):
        """Test running a minimal benchmark (slow test)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = WikiText2Benchmark(output_dir=tmpdir)
            
            # Very small configuration for fast testing
            config = BenchmarkConfig(
                model_name='mini_test',
                d_model=32,
                n_layers=2,
                n_seq=64,
                batch_size=8,
                epochs=1,
                lr=1e-3,
                weight_decay=0.01,
                grad_clip=0.5,
                device='cpu',
                seed=42,
                use_analytic_gradient=False,
            )
            
            try:
                results = benchmark.run_benchmark(config)
                
                assert results.model_name == 'mini_test'
                assert results.final_perplexity > 0
                assert results.training_time > 0
                assert results.total_training_flops > 0
                assert len(results.epoch_losses) == 1
                
                # Check that results file was created
                results_file = Path(tmpdir) / "mini_test_results.json"
                assert results_file.exists()
                
            except Exception as e:
                # If data loading fails (expected in test environment), that's okay
                if "Failed to load WikiText-2" in str(e):
                    pytest.skip("WikiText-2 dataset not available in test environment")
                else:
                    raise


def test_benchmark_imports():
    """Test that all required imports work."""
    from src.benchmarks.wikitext2_benchmark import (
        BenchmarkConfig,
        BenchmarkResults,
        WikiText2Benchmark,
    )
    
    assert BenchmarkConfig is not None
    assert BenchmarkResults is not None
    assert WikiText2Benchmark is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
