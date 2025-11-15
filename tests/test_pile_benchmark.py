"""
Tests for The Pile benchmark.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.pile_benchmark import (
    PileBenchmark,
    BenchmarkConfig,
    BenchmarkResults,
    load_pile_data
)


class TestPileBenchmark:
    """Test suite for The Pile benchmark."""
    
    def test_benchmark_config_creation(self):
        """Test creating benchmark configuration."""
        config = BenchmarkConfig(
            model_name='test_model',
            d_model=64,
            n_layers=2,
            n_seq=128,
            batch_size=4,
            epochs=1,
            lr=1e-3,
            weight_decay=0.01,
            grad_clip=0.5,
            device='cpu',
            seed=42,
            data_limit=10000,  # Small for testing
        )
        
        assert config.model_name == 'test_model'
        assert config.d_model == 64
        assert config.data_limit == 10000
    
    def test_benchmark_results_creation(self):
        """Test creating benchmark results."""
        results = BenchmarkResults(
            model_name='test_model',
            dataset_name='pile',
            config={'d_model': 64},
            final_loss=2.5,
            final_perplexity=12.18,
            best_perplexity=11.5,
            training_time=100.0,
            total_tokens=10000,
            vocab_size=5000,
            domain_perplexities={'Pile-CC': 15.0, 'Wikipedia': 10.0},
            forward_flops=1000000,
            backward_flops=2000000,
            optimizer_flops=500000,
            total_flops_per_step=3500000,
            total_training_flops=350000000,
            peak_memory_mb=100.0,
            model_size_mb=10.0,
            epoch_losses=[3.0, 2.5],
            epoch_perplexities=[20.0, 12.18],
            epoch_times=[50.0, 50.0],
        )
        
        assert results.model_name == 'test_model'
        assert results.dataset_name == 'pile'
        assert results.final_perplexity == 12.18
        assert len(results.domain_perplexities) == 2
    
    def test_benchmark_results_to_dict(self):
        """Test converting results to dictionary."""
        results = BenchmarkResults(
            model_name='test_model',
            dataset_name='pile',
            config={'d_model': 64},
            final_loss=2.5,
            final_perplexity=12.18,
            best_perplexity=11.5,
            training_time=100.0,
            total_tokens=10000,
            vocab_size=5000,
            domain_perplexities={'Pile-CC': 15.0},
            forward_flops=1000000,
            backward_flops=2000000,
            optimizer_flops=500000,
            total_flops_per_step=3500000,
            total_training_flops=350000000,
            peak_memory_mb=100.0,
            model_size_mb=10.0,
            epoch_losses=[2.5],
            epoch_perplexities=[12.18],
            epoch_times=[100.0],
        )
        
        result_dict = results.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['model_name'] == 'test_model'
        assert result_dict['final_perplexity'] == 12.18
    
    def test_benchmark_results_save_json(self, tmp_path):
        """Test saving results to JSON."""
        results = BenchmarkResults(
            model_name='test_model',
            dataset_name='pile',
            config={'d_model': 64},
            final_loss=2.5,
            final_perplexity=12.18,
            best_perplexity=11.5,
            training_time=100.0,
            total_tokens=10000,
            vocab_size=5000,
            domain_perplexities={'Pile-CC': 15.0},
            forward_flops=1000000,
            backward_flops=2000000,
            optimizer_flops=500000,
            total_flops_per_step=3500000,
            total_training_flops=350000000,
            peak_memory_mb=100.0,
            model_size_mb=10.0,
            epoch_losses=[2.5],
            epoch_perplexities=[12.18],
            epoch_times=[100.0],
        )
        
        filepath = tmp_path / "test_results.json"
        results.save_json(str(filepath))
        
        assert filepath.exists()
        
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['model_name'] == 'test_model'
        assert loaded['final_perplexity'] == 12.18
    
    def test_pile_benchmark_initialization(self, tmp_path):
        """Test initializing PileBenchmark."""
        benchmark = PileBenchmark(output_dir=str(tmp_path))
        
        assert benchmark.output_dir == tmp_path
        assert tmp_path.exists()
        assert len(benchmark.results) == 0
        assert len(benchmark.pile_domains) == 22
    
    def test_pile_domains_list(self):
        """Test that all 22 Pile domains are defined."""
        benchmark = PileBenchmark()
        
        expected_domains = [
            'Pile-CC', 'PubMed Central', 'Books3', 'OpenWebText2', 'ArXiv',
            'Github', 'FreeLaw', 'StackExchange', 'USPTO', 'PubMed Abstracts',
            'Gutenberg', 'OpenSubtitles', 'Wikipedia', 'DM Mathematics',
            'Ubuntu IRC', 'BookCorpus2', 'EuroParl', 'HackerNews',
            'YoutubeSubtitles', 'PhilPapers', 'NIH ExPorter', 'Enron Emails'
        ]
        
        assert len(benchmark.pile_domains) == 22
        for domain in expected_domains:
            assert domain in benchmark.pile_domains
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_benchmark_with_cuda(self, tmp_path):
        """Test benchmark configuration with CUDA."""
        config = BenchmarkConfig(
            model_name='test_cuda',
            d_model=32,
            n_layers=2,
            n_seq=64,
            batch_size=2,
            epochs=1,
            lr=1e-3,
            weight_decay=0.01,
            grad_clip=0.5,
            device='cuda',
            seed=42,
            data_limit=1000,
        )
        
        assert config.device == 'cuda'
    
    def test_benchmark_config_with_optimizations(self):
        """Test benchmark configuration with all optimizations enabled."""
        config = BenchmarkConfig(
            model_name='test_optimized',
            d_model=64,
            n_layers=4,
            n_seq=128,
            batch_size=8,
            epochs=2,
            lr=1e-3,
            weight_decay=0.01,
            grad_clip=0.5,
            device='cpu',
            seed=42,
            data_limit=100000,
            use_analytic_gradient=True,
            use_mixed_precision=True,
            use_act=True,
            use_multi_scale=True,
            use_sparse_bk=True,
        )
        
        assert config.use_analytic_gradient is True
        assert config.use_mixed_precision is True
        assert config.use_act is True
        assert config.use_multi_scale is True
        assert config.use_sparse_bk is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
